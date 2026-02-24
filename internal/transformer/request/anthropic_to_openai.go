package anthropic

import (
	"encoding/json"
	"fmt"
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

	var systemContent []interface{}

	if system, ok := req["system"].(string); ok && system != "" {
		systemContent = []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": system,
			},
		}
		delete(req, "system")
	}

	transformedMessages := []interface{}{}

	if len(systemContent) > 0 {
		transformedMessages = append(transformedMessages, map[string]interface{}{
			"role":    "system",
			"content": systemContent,
		})
	}

	for _, msgIF := range messages {
		msg, ok := msgIF.(map[string]interface{})
		if !ok {
			continue
		}

		transformedMsg := transformMessage(msg)
		transformedMessages = append(transformedMessages, transformedMsg)
	}

	req["messages"] = transformedMessages

	return json.Marshal(req)
}

func transformMessage(msg map[string]interface{}) map[string]interface{} {
	role, _ := msg["role"].(string)
	content, _ := msg["content"]

	switch role {
	case "user":
		if content == nil {
			return msg
		}
		if str, ok := content.(string); ok {
			msg["content"] = str
			return msg
		}
		if blocks, ok := content.([]interface{}); ok {
			var textContent string
			var toolResults []interface{}

			for _, block := range blocks {
				b, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				btype, _ := b["type"].(string)
				switch btype {
				case "text":
					if text, ok := b["text"].(string); ok {
						textContent += text
					}
				case "tool_result":
					toolResults = append(toolResults, map[string]interface{}{
						"type": "tool",
						"id":   b["tool_use_id"],
						"content": func() string {
							if c, ok := b["content"].(string); ok {
								return c
							}
							if c, ok := b["content"].(map[string]interface{}); ok {
								if j, err := json.Marshal(c); err == nil {
									return string(j)
								}
							}
							return ""
						}(),
					})
				case "image":
					if source, ok := b["source"].(map[string]interface{}); ok {
						sourceType, _ := source["type"].(string)
						if sourceType == "base64" {
							mediaData, _ := source["data"].(string)
							mediaType, _ := source["media_type"].(string)
							imgURL := fmt.Sprintf("data:%s;base64,%s", mediaType, mediaData)
							msg["content"] = []interface{}{
								map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]interface{}{
										"url": imgURL,
									},
								},
							}
							return msg
						}
					}
				}
			}

			if textContent != "" && len(toolResults) == 0 {
				msg["content"] = textContent
			} else if len(toolResults) > 0 {
				msg["content"] = toolResults
			}
		}

	case "assistant":
		if content == nil {
			return msg
		}
		if str, ok := content.(string); ok {
			msg["content"] = str
			return msg
		}
		if blocks, ok := content.([]interface{}); ok {
			var textContent string
			var toolCalls []interface{}

			for _, block := range blocks {
				b, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				btype, _ := b["type"].(string)
				switch btype {
				case "text":
					if text, ok := b["text"].(string); ok {
						textContent += text
					}
				case "tool_use":
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   b["id"],
						"type": "function",
						"function": map[string]interface{}{
							"name": b["name"],
							"arguments": func() string {
								if args, ok := b["input"].(map[string]interface{}); ok {
									if j, err := json.Marshal(args); err == nil {
										return string(j)
									}
								}
								return "{}"
							}(),
						},
					})
				}
			}

			if textContent != "" {
				msg["content"] = textContent
			}
			if len(toolCalls) > 0 {
				msg["tool_calls"] = toolCalls
			}
		}
	}

	return msg
}
