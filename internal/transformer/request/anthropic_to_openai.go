package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
)

type AnthropicToOpenAITransformer struct{}

func NewAnthropicToOpenAITransformer() *AnthropicToOpenAITransformer {
	return &AnthropicToOpenAITransformer{}
}

func (t *AnthropicToOpenAITransformer) Name() string {
	return "anthropic-to-openai"
}

func (t *AnthropicToOpenAITransformer) TransformRequest(body []byte, model string) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, err
	}

	messages, ok := req["messages"].([]interface{})
	if !ok {
		return body, nil
	}

	// Handle system prompt - can be string or array of content blocks
	var systemContent string
	if system, ok := req["system"]; ok && system != nil {
		systemContent = extractSystemContent(system)
		delete(req, "system")
	}

	transformedMessages := []interface{}{}

	// Add system message first if present
	if systemContent != "" {
		transformedMessages = append(transformedMessages, map[string]interface{}{
			"role":    "system",
			"content": systemContent,
		})
	}

	// Transform each message
	for _, msgIF := range messages {
		msg, ok := msgIF.(map[string]interface{})
		if !ok {
			continue
		}

		transformedMsgs := transformMessageToSlice(msg)
		for _, transformedMsg := range transformedMsgs {
			if transformedMsg != nil {
				transformedMessages = append(transformedMessages, transformedMsg)
			}
		}
	}

	// Remove empty assistant placeholder at the end (Claude Code sometimes adds this)
	if len(transformedMessages) > 0 {
		if lastMsg, ok := transformedMessages[len(transformedMessages)-1].(map[string]interface{}); ok {
			if role, _ := lastMsg["role"].(string); role == "assistant" {
				content := lastMsg["content"]
				toolCalls, hasToolCalls := lastMsg["tool_calls"]

				isEmpty := false
				if content == nil {
					isEmpty = true
				} else if str, ok := content.(string); ok && str == "" {
					isEmpty = true
				}

				hasNoToolCalls := !hasToolCalls || len(toolCalls.([]interface{})) == 0

				if isEmpty && hasNoToolCalls {
					transformedMessages = transformedMessages[:len(transformedMessages)-1]
				}
			}
		}
	}

	req["messages"] = transformedMessages

	// Handle thinking config - Claude uses {type: "enabled", budget_tokens: N}
	// OpenAI uses {type: "enabled", max_tokens: N}
	if thinking, ok := req["thinking"].(map[string]interface{}); ok {
		if thinkType, _ := thinking["type"].(string); thinkType == "enabled" {
			if budgetTokens, ok := thinking["budget_tokens"].(float64); ok {
				req["thinking"] = map[string]interface{}{
					"type":       "enabled",
					"max_tokens": int(budgetTokens),
				}
			}
		}
	}

	// Handle tool_choice conversion
	if toolChoice, ok := req["tool_choice"]; ok {
		req["tool_choice"] = transformToolChoice(toolChoice)
	}

	// Handle tools - convert Claude input_schema to OpenAI parameters
	if tools, ok := req["tools"].([]interface{}); ok {
		req["tools"] = transformTools(tools)
	}

	// Handle stop_sequences - truncate to max 4 items (OpenAI limit)
	if stopSeqs, ok := req["stop_sequences"].([]interface{}); ok && len(stopSeqs) > 4 {
		req["stop_sequences"] = stopSeqs[:4]
	}

	return json.Marshal(req)
}

// extractSystemContent handles both string and array system prompts
func extractSystemContent(system interface{}) string {
	if system == nil {
		return ""
	}

	// Handle string system prompt
	if str, ok := system.(string); ok {
		return str
	}

	// Handle array of content blocks
	if blocks, ok := system.([]interface{}); ok {
		var texts []string
		for _, block := range blocks {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockType, _ := blockMap["type"].(string); blockType == "text" {
					if text, ok := blockMap["text"].(string); ok {
						texts = append(texts, text)
					}
				}
			}
		}
		return strings.Join(texts, "\n")
	}

	return ""
}

func transformMessageToSlice(msg map[string]interface{}) []map[string]interface{} {
	role, _ := msg["role"].(string)
	content, _ := msg["content"]

	switch role {
	case "user":
		return transformUserMessageToSlice(msg, content)
	case "assistant":
		return []map[string]interface{}{transformAssistantMessage(msg, content)}
	default:
		return []map[string]interface{}{msg}
	}
}

func transformUserMessageToSlice(msg map[string]interface{}, content interface{}) []map[string]interface{} {
	if content == nil {
		return []map[string]interface{}{msg}
	}

	// Simple string passthrough
	if str, ok := content.(string); ok {
		msg["content"] = str
		return []map[string]interface{}{msg}
	}

	// Parse content blocks
	blocks, ok := content.([]interface{})
	if !ok {
		return []map[string]interface{}{msg}
	}

	var textParts []string
	var toolResults []map[string]interface{}
	var imageBlocks []interface{}

	for _, block := range blocks {
		b, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		btype, _ := b["type"].(string)
		switch btype {
		case "text":
			if text, ok := b["text"].(string); ok {
				textParts = append(textParts, text)
			}

		case "tool_result":
			toolUseID, _ := b["tool_use_id"].(string)
			resultContent := b["content"]

			toolResults = append(toolResults, map[string]interface{}{
				"role":         "tool",
				"content":      serializeToolResultContent(resultContent),
				"tool_call_id": toolUseID,
			})

		case "image":
			if source, ok := b["source"].(map[string]interface{}); ok {
				sourceType, _ := source["type"].(string)
				if sourceType == "base64" {
					mediaData, _ := source["data"].(string)
					mediaType, _ := source["media_type"].(string)
					imgURL := fmt.Sprintf("data:%s;base64,%s", mediaType, mediaData)
					imageBlocks = append(imageBlocks, map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": imgURL,
						},
					})
				}
			}
		}
	}

	result := make([]map[string]interface{}, 0)

	// First, add any text/image content as a user message
	if len(textParts) > 0 || len(imageBlocks) > 0 {
		userMsg := make(map[string]interface{})
		for k, v := range msg {
			userMsg[k] = v
		}

		if len(imageBlocks) > 0 {
			// Handle images - if we have text along with images, include it
			if len(textParts) > 0 {
				textBlock := map[string]interface{}{
					"type": "text",
					"text": strings.Join(textParts, "\n"),
				}
				imageBlocks = append([]interface{}{textBlock}, imageBlocks...)
			}
			userMsg["content"] = imageBlocks
		} else {
			// Simple text concatenation
			userMsg["content"] = strings.Join(textParts, "\n")
		}
		result = append(result, userMsg)
	}

	// Then add tool results as separate "tool" role messages
	for _, toolResult := range toolResults {
		result = append(result, toolResult)
	}

	return result
}

func transformAssistantMessage(msg map[string]interface{}, content interface{}) map[string]interface{} {
	if content == nil {
		return msg
	}

	// Simple string passthrough
	if str, ok := content.(string); ok {
		msg["content"] = str
		return msg
	}

	// Parse content blocks
	blocks, ok := content.([]interface{})
	if !ok {
		return msg
	}

	var thinkingParts []string
	var textParts []string
	var toolCalls []interface{}

	for _, block := range blocks {
		b, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		btype, _ := b["type"].(string)
		switch btype {
		case "thinking":
			if thinking, ok := b["thinking"].(string); ok {
				thinkingParts = append(thinkingParts, thinking)
			}

		case "text":
			if text, ok := b["text"].(string); ok {
				textParts = append(textParts, text)
			}

		case "tool_use":
			id, _ := b["id"].(string)
			name, _ := b["name"].(string)
			input := b["input"]

			args := "{}"
			if inputMap, ok := input.(map[string]interface{}); ok {
				if j, err := json.Marshal(inputMap); err == nil {
					args = string(j)
				}
			}

			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   id,
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": args,
				},
			})
		}
	}

	// Build content string with interleaved thinking blocks
	var combined strings.Builder

	// Add thinking content wrapped in <think> tags
	if len(thinkingParts) > 0 {
		for _, thinking := range thinkingParts {
			combined.WriteString("<think>")
			combined.WriteString(thinking)
			combined.WriteString("</think>\n")
		}
	}

	// Add regular text
	for i, text := range textParts {
		if i > 0 {
			combined.WriteString("\n")
		}
		combined.WriteString(text)
	}

	contentStr := combined.String()
	if contentStr != "" {
		msg["content"] = contentStr
	} else {
		msg["content"] = ""
	}

	// Add tool_calls if present
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}

	return msg
}

// serializeToolResultContent handles complex nested content in tool results
func serializeToolResultContent(content interface{}) string {
	if content == nil {
		return ""
	}

	// Simple string
	if str, ok := content.(string); ok {
		return str
	}

	// Array of content blocks
	if arr, ok := content.([]interface{}); ok {
		var parts []string
		for _, item := range arr {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if itemType, _ := itemMap["type"].(string); itemType == "text" {
					if text, ok := itemMap["text"].(string); ok {
						parts = append(parts, text)
					}
				} else {
					// Non-text block, serialize as JSON
					if j, err := json.Marshal(itemMap); err == nil {
						parts = append(parts, string(j))
					}
				}
			} else if str, ok := item.(string); ok {
				parts = append(parts, str)
			}
		}
		return strings.Join(parts, "\n")
	}

	// Object, serialize to JSON
	if obj, ok := content.(map[string]interface{}); ok {
		if j, err := json.Marshal(obj); err == nil {
			return string(j)
		}
	}

	return ""
}

func transformToolChoice(toolChoice interface{}) interface{} {
	switch choice := toolChoice.(type) {
	case string:
		switch strings.ToLower(choice) {
		case "any":
			return "required"
		default:
			return choice
		}
	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, v := range choice {
			if k == "disable_parallel_tool_use" {
				if disable, ok := v.(bool); ok {
					result["parallel_tool_calls"] = !disable
				}
			} else if k == "type" {
				if strings.ToLower(v.(string)) == "tool" {
					result["type"] = "function"
				} else {
					result[k] = v
				}
			} else {
				result[k] = v
			}
		}
		return result
	default:
		return toolChoice
	}
}

func transformTools(tools []interface{}) []interface{} {
	var result []interface{}
	for _, t := range tools {
		tool, ok := t.(map[string]interface{})
		if !ok {
			result = append(result, t)
			continue
		}

		newTool := map[string]interface{}{
			"type": "function",
		}

		funcData := map[string]interface{}{}
		if name, ok := tool["name"].(string); ok {
			funcData["name"] = name
		}
		if desc, ok := tool["description"].(string); ok {
			funcData["description"] = desc
		}
		if inputSchema, ok := tool["input_schema"]; ok {
			funcData["parameters"] = inputSchema
		}

		newTool["function"] = funcData
		result = append(result, newTool)
	}
	return result
}
