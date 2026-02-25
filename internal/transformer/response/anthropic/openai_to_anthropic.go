package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const MAX_STREAM_BUFFER_SIZE = 1024 * 1024 // 1MB buffer limit

// OpenAIToAnthropicTransformer converts OpenAI format to Anthropic format
// and implements StreamingTransformer for proper SSE handling
type OpenAIToAnthropicTransformer struct {
	streamState *streamState
}

// streamState tracks the current state of streaming transformation
type streamState struct {
	messageStarted  bool
	messageID       string
	model           string
	nextBlockIndex  int
	thinkingOpen    bool
	thinkingIndex   int
	textOpen        bool
	textIndex       int
	toolCallOpen    bool
	toolCallIndex   int
	toolCallBuffers map[string]*toolCallBuffer
	inputTokens     int
	outputTokens    int
	stopReason      string
	utf8Buffer      []byte
}

type toolCallBuffer struct {
	blockIndex int
	id         string
	name       string
	argsBuf    strings.Builder
}

type sseEvent struct {
	Event string
	Data  string
}

func NewOpenAIToAnthropicTransformer() *OpenAIToAnthropicTransformer {
	return &OpenAIToAnthropicTransformer{}
}

func (t *OpenAIToAnthropicTransformer) Reset() {
	t.streamState = nil
}

func (t *OpenAIToAnthropicTransformer) Name() string {
	return "openai-to-anthropic"
}

func (t *OpenAIToAnthropicTransformer) TransformResponse(body []byte, isStreaming bool) ([]byte, error) {
	if isStreaming {
		return body, nil
	}
	return t.transformNonStreaming(body)
}

func (t *OpenAIToAnthropicTransformer) transformNonStreaming(body []byte) ([]byte, error) {
	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, nil
	}

	if errorObj, ok := resp["error"].(map[string]interface{}); ok {
		return t.formatErrorResponse(errorObj, resp)
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return body, nil
	}

	choice := choices[0].(map[string]interface{})
	msg, ok := choice["message"].(map[string]interface{})
	if !ok {
		return body, nil
	}

	inputTokens := 0
	outputTokens := 0
	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			inputTokens = int(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			outputTokens = int(ct)
		} else if tt, ok := usage["total_tokens"].(float64); ok {
			outputTokens = int(tt) - inputTokens
		}
	}

	anthropicMsg := t.convertToAnthropicMessage(msg, inputTokens, outputTokens)

	if model, ok := resp["model"].(string); ok {
		anthropicMsg["model"] = model
	}

	result, err := json.Marshal(anthropicMsg)
	if err != nil {
		return body, nil
	}

	return result, nil
}

func (t *OpenAIToAnthropicTransformer) convertToAnthropicMessage(msg map[string]interface{}, inputTokens, outputTokens int) map[string]interface{} {
	content := t.extractAnthropicContent(msg)

	stopReason := "end_turn"
	if reason, ok := msg["finish_reason"].(string); ok {
		stopReason = translateFinishReason(reason)
	}

	return map[string]interface{}{
		"id":            msg["id"],
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}
}

func (t *OpenAIToAnthropicTransformer) extractAnthropicContent(msg map[string]interface{}) []interface{} {
	var content []interface{}

	reasoningContent := t.extractReasoningContent(msg)
	if reasoningContent != "" {
		content = append(content, map[string]interface{}{
			"type":     "thinking",
			"thinking": reasoningContent,
		})
	}

	if contentStr, ok := msg["content"].(string); ok && contentStr != "" {
		parts := splitThinkingContent(contentStr)
		for _, part := range parts {
			if part.isThinking {
				content = append(content, map[string]interface{}{
					"type":     "thinking",
					"thinking": part.text,
				})
			} else if part.text != "" {
				content = append(content, map[string]interface{}{
					"type": "text",
					"text": part.text,
				})
			}
		}
	}

	if toolCalls, ok := msg["tool_calls"].([]interface{}); ok {
		for _, tcIF := range toolCalls {
			tc, ok := tcIF.(map[string]interface{})
			if !ok {
				continue
			}

			funcData, ok := tc["function"].(map[string]interface{})
			if !ok {
				continue
			}

			name, _ := funcData["name"].(string)
			args, _ := funcData["arguments"].(string)
			id, _ := tc["id"].(string)

			var input map[string]interface{}
			json.Unmarshal([]byte(args), &input)

			content = append(content, map[string]interface{}{
				"type":  "tool_use",
				"id":    id,
				"name":  name,
				"input": input,
			})
		}
	}

	if len(content) == 0 {
		content = append(content, map[string]interface{}{
			"type": "text",
			"text": "",
		})
	}

	return content
}

func (t *OpenAIToAnthropicTransformer) extractReasoningContent(msg map[string]interface{}) string {
	reasoningFields := []string{
		"reasoning_content",
		"reasoning",
		"thinking_content",
		"thinking",
	}

	for _, field := range reasoningFields {
		if val, ok := msg[field].(string); ok && val != "" {
			return val
		}
	}

	return ""
}

type contentPart struct {
	text       string
	isThinking bool
}

func splitThinkingContent(content string) []contentPart {
	var parts []contentPart
	remaining := content

	for {
		startIdx := strings.Index(remaining, "<think>")
		if startIdx == -1 {
			if remaining != "" {
				parts = append(parts, contentPart{text: remaining, isThinking: false})
			}
			break
		}

		if startIdx > 0 {
			parts = append(parts, contentPart{text: remaining[:startIdx], isThinking: false})
		}

		endIdx := strings.Index(remaining[startIdx:], "</think>")
		if endIdx == -1 {
			parts = append(parts, contentPart{text: remaining, isThinking: false})
			break
		}

		thinkingStart := startIdx + len("<think>")
		thinkingEnd := startIdx + endIdx
		thinkingContent := remaining[thinkingStart:thinkingEnd]

		parts = append(parts, contentPart{text: thinkingContent, isThinking: true})

		remaining = remaining[thinkingEnd+len("</think>"):]
	}

	return parts
}

// TransformStream implements the StreamingTransformer interface
// Enhanced with buffer limits and non-streaming fallback
func (t *OpenAIToAnthropicTransformer) TransformStream(chunk []byte) (modified bool, newChunk []byte, keepChunk bool) {
	if t.streamState == nil {
		t.streamState = &streamState{
			toolCallBuffers: make(map[string]*toolCallBuffer),
		}
	}

	if string(chunk) == "[DONE]" {
		events := t.finalizeStream()
		return true, []byte(events), true
	}

	if t.streamState.utf8Buffer != nil && len(t.streamState.utf8Buffer)+len(chunk) > MAX_STREAM_BUFFER_SIZE {
		t.streamState = &streamState{
			toolCallBuffers: make(map[string]*toolCallBuffer),
		}
		errorEvent := t.createBufferOverflowError()
		return true, []byte(errorEvent), true
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(chunk, &parsed); err != nil {
		return false, chunk, true
	}

	if isNonStreamingResponse(parsed) {
		events := t.convertNonStreamingToSSE(parsed)
		return true, []byte(strings.Join(events, "")), true
	}

	if errorObj, ok := parsed["error"].(map[string]interface{}); ok {
		event := t.formatStreamError(errorObj)
		return true, []byte(event), true
	}

	events := t.processChunk(parsed)

	if len(events) == 0 {
		return false, chunk, true
	}

	return true, []byte(strings.Join(events, "")), true
}

func isNonStreamingResponse(parsed map[string]interface{}) bool {
	choices, ok := parsed["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return false
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return false
	}

	_, hasMessage := choice["message"]
	_, hasDelta := choice["delta"]

	return hasMessage && !hasDelta
}

func (t *OpenAIToAnthropicTransformer) convertNonStreamingToSSE(parsed map[string]interface{}) []string {
	var events []string

	if !t.streamState.messageStarted {
		t.streamState.messageStarted = true
		if model, ok := parsed["model"].(string); ok {
			t.streamState.model = model
		}
		if usage, ok := parsed["usage"].(map[string]interface{}); ok {
			if pt, ok := usage["prompt_tokens"].(float64); ok {
				t.streamState.inputTokens = int(pt)
			}
		}
		events = append(events, t.createMessageStartEvent())
	}

	choices := parsed["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	var content string
	if c, ok := message["content"].(string); ok {
		content = c
	}

	reasoning := t.extractReasoningContent(message)
	if reasoning != "" {
		events = append(events, t.processReasoningDelta(reasoning)...)
	}

	if content != "" {
		events = append(events, t.processTextDelta(content)...)
	}

	if toolCalls, ok := message["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
		events = append(events, t.processToolCallsComplete(toolCalls)...)
	}

	if finishReason, ok := choice["finish_reason"].(string); ok {
		t.streamState.stopReason = translateFinishReason(finishReason)
	}

	if usage, ok := parsed["usage"].(map[string]interface{}); ok {
		if ct, ok := usage["completion_tokens"].(float64); ok {
			t.streamState.outputTokens = int(ct)
		} else if tt, ok := usage["total_tokens"].(float64); ok {
			t.streamState.outputTokens = int(tt) - t.streamState.inputTokens
		}
	}

	events = append(events, t.finalizeStream())
	return events
}

func (t *OpenAIToAnthropicTransformer) processChunk(parsed map[string]interface{}) []string {
	var events []string

	if !t.streamState.messageStarted {
		t.streamState.messageStarted = true
		if model, ok := parsed["model"].(string); ok {
			t.streamState.model = model
		}
		if usage, ok := parsed["usage"].(map[string]interface{}); ok {
			if pt, ok := usage["prompt_tokens"].(float64); ok {
				t.streamState.inputTokens = int(pt)
			}
		}
		events = append(events, t.createMessageStartEvent())
	}

	choices, ok := parsed["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return events
	}

	choice := choices[0].(map[string]interface{})

	if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
		t.streamState.stopReason = translateFinishReason(finishReason)
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return events
	}

	if reasoning := t.extractReasoningContent(delta); reasoning != "" {
		events = append(events, t.processReasoningDelta(reasoning)...)
	}

	if content, ok := delta["content"].(string); ok && content != "" {
		events = append(events, t.processTextDelta(content)...)
	}

	if toolCalls, ok := delta["tool_calls"].([]interface{}); ok {
		events = append(events, t.processToolCallDeltas(toolCalls)...)
	}

	if usage, ok := parsed["usage"].(map[string]interface{}); ok {
		if ct, ok := usage["completion_tokens"].(float64); ok {
			t.streamState.outputTokens = int(ct)
		} else if tt, ok := usage["total_tokens"].(float64); ok {
			t.streamState.outputTokens = int(tt) - t.streamState.inputTokens
		}
	}

	return events
}

func (t *OpenAIToAnthropicTransformer) createMessageStartEvent() string {
	id := fmt.Sprintf("msg_%d", time.Now().UnixNano())
	t.streamState.messageID = id

	msg := map[string]interface{}{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       []interface{}{},
		"model":         t.streamState.model,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  t.streamState.inputTokens,
			"output_tokens": 0,
		},
	}

	start := map[string]interface{}{
		"type":    "message_start",
		"message": msg,
	}

	data, _ := json.Marshal(start)
	return fmt.Sprintf("event: message_start\ndata: %s\n\n", data)
}

func (t *OpenAIToAnthropicTransformer) processReasoningDelta(reasoning string) []string {
	var events []string

	if !t.streamState.thinkingOpen {
		t.streamState.thinkingOpen = true
		t.streamState.thinkingIndex = t.streamState.nextBlockIndex
		t.streamState.nextBlockIndex++

		start := map[string]interface{}{
			"type":  "content_block_start",
			"index": t.streamState.thinkingIndex,
			"content_block": map[string]interface{}{
				"type":     "thinking",
				"thinking": "",
			},
		}
		data, _ := json.Marshal(start)
		events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))
	}

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": t.streamState.thinkingIndex,
		"delta": map[string]interface{}{
			"type":     "thinking_delta",
			"thinking": reasoning,
		},
	}
	data, _ := json.Marshal(delta)
	events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))

	t.streamState.outputTokens += len(reasoning) / 4

	return events
}

func (t *OpenAIToAnthropicTransformer) processTextDelta(content string) []string {
	var events []string

	if t.streamState.thinkingOpen {
		t.streamState.thinkingOpen = false
		stop := map[string]interface{}{
			"type":  "content_block_stop",
			"index": t.streamState.thinkingIndex,
		}
		data, _ := json.Marshal(stop)
		events = append(events, fmt.Sprintf("event: content_block_stop\ndata: %s\n\n", data))
	}

	if !t.streamState.textOpen {
		t.streamState.textOpen = true
		t.streamState.textIndex = t.streamState.nextBlockIndex
		t.streamState.nextBlockIndex++

		start := map[string]interface{}{
			"type":  "content_block_start",
			"index": t.streamState.textIndex,
			"content_block": map[string]interface{}{
				"type": "text",
				"text": "",
			},
		}
		data, _ := json.Marshal(start)
		events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))
	}

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": t.streamState.textIndex,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": content,
		},
	}
	data, _ := json.Marshal(delta)
	events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))

	t.streamState.outputTokens += len(content) / 4

	return events
}

func (t *OpenAIToAnthropicTransformer) processToolCallDeltas(toolCalls []interface{}) []string {
	var events []string

	for _, tcIF := range toolCalls {
		tc, ok := tcIF.(map[string]interface{})
		if !ok {
			continue
		}

		id, _ := tc["id"].(string)

		buffer, exists := t.streamState.toolCallBuffers[id]
		if !exists {
			buffer = &toolCallBuffer{id: id}
			t.streamState.toolCallBuffers[id] = buffer
		}

		if funcData, ok := tc["function"].(map[string]interface{}); ok {
			if name, ok := funcData["name"].(string); ok && name != "" {
				buffer.name = name
			}
			if args, ok := funcData["arguments"].(string); ok && args != "" {
				buffer.argsBuf.WriteString(args)
			}
		}

		if !t.streamState.toolCallOpen && !exists {
			if t.streamState.textOpen {
				t.streamState.textOpen = false
				stop := map[string]interface{}{
					"type":  "content_block_stop",
					"index": t.streamState.textIndex,
				}
				data, _ := json.Marshal(stop)
				events = append(events, fmt.Sprintf("event: content_block_stop\ndata: %s\n\n", data))
			}

			t.streamState.toolCallOpen = true
			buffer.blockIndex = t.streamState.nextBlockIndex
			t.streamState.toolCallIndex = t.streamState.nextBlockIndex
			t.streamState.nextBlockIndex++

			start := map[string]interface{}{
				"type":  "content_block_start",
				"index": buffer.blockIndex,
				"content_block": map[string]interface{}{
					"type":  "tool_use",
					"id":    buffer.id,
					"name":  buffer.name,
					"input": map[string]interface{}{},
				},
			}
			data, _ := json.Marshal(start)
			events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))
		}

		if funcData, ok := tc["function"].(map[string]interface{}); ok {
			if args, ok := funcData["arguments"].(string); ok && args != "" {
				delta := map[string]interface{}{
					"type":  "content_block_delta",
					"index": buffer.blockIndex,
					"delta": map[string]interface{}{
						"type":         "input_json_delta",
						"partial_json": args,
					},
				}
				data, _ := json.Marshal(delta)
				events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))
			}
		}
	}

	return events
}

func (t *OpenAIToAnthropicTransformer) processToolCallsComplete(toolCalls []interface{}) []string {
	var events []string

	for _, tcIF := range toolCalls {
		tc, ok := tcIF.(map[string]interface{})
		if !ok {
			continue
		}

		id, _ := tc["id"].(string)
		funcData, _ := tc["function"].(map[string]interface{})
		name, _ := funcData["name"].(string)
		args, _ := funcData["arguments"].(string)

		buffer := &toolCallBuffer{
			id:         id,
			name:       name,
			blockIndex: t.streamState.nextBlockIndex,
		}
		if args != "" {
			buffer.argsBuf.WriteString(args)
		}

		if t.streamState.textOpen {
			t.streamState.textOpen = false
			stop := map[string]interface{}{
				"type":  "content_block_stop",
				"index": t.streamState.textIndex,
			}
			data, _ := json.Marshal(stop)
			events = append(events, fmt.Sprintf("event: content_block_stop\ndata: %s\n\n", data))
		}

		t.streamState.toolCallOpen = true
		t.streamState.toolCallIndex = t.streamState.nextBlockIndex
		t.streamState.nextBlockIndex++

		start := map[string]interface{}{
			"type":  "content_block_start",
			"index": buffer.blockIndex,
			"content_block": map[string]interface{}{
				"type":  "tool_use",
				"id":    buffer.id,
				"name":  buffer.name,
				"input": map[string]interface{}{},
			},
		}
		data, _ := json.Marshal(start)
		events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))

		if buffer.argsBuf.Len() > 0 {
			delta := map[string]interface{}{
				"type":  "content_block_delta",
				"index": buffer.blockIndex,
				"delta": map[string]interface{}{
					"type":         "input_json_delta",
					"partial_json": buffer.argsBuf.String(),
				},
			}
			data, _ = json.Marshal(delta)
			events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))
		}

		events = append(events, t.createBlockStopEvent(buffer.blockIndex))
	}

	if t.streamState.toolCallOpen {
		t.streamState.toolCallOpen = false
		events = append(events, t.createBlockStopEvent(t.streamState.toolCallIndex))
	}

	return events
}

func (t *OpenAIToAnthropicTransformer) finalizeStream() string {
	var events []string

	if t.streamState.thinkingOpen {
		t.streamState.thinkingOpen = false
		events = append(events, t.createBlockStopEvent(t.streamState.thinkingIndex))
	}

	if t.streamState.textOpen {
		t.streamState.textOpen = false
		events = append(events, t.createBlockStopEvent(t.streamState.textIndex))
	}

	if t.streamState.toolCallOpen {
		t.streamState.toolCallOpen = false
		events = append(events, t.createBlockStopEvent(t.streamState.toolCallIndex))
	}

	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   t.streamState.stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": t.streamState.outputTokens,
		},
	}
	data, _ := json.Marshal(msgDelta)
	events = append(events, fmt.Sprintf("event: message_delta\ndata: %s\n\n", data))

	stop := map[string]interface{}{
		"type": "message_stop",
	}
	data, _ = json.Marshal(stop)
	events = append(events, fmt.Sprintf("event: message_stop\ndata: %s\n\n", data))

	return strings.Join(events, "")
}

func (t *OpenAIToAnthropicTransformer) createBlockStopEvent(index int) string {
	stop := map[string]interface{}{
		"type":  "content_block_stop",
		"index": index,
	}
	data, _ := json.Marshal(stop)
	return fmt.Sprintf("event: content_block_stop\ndata: %s\n\n", data)
}

func (t *OpenAIToAnthropicTransformer) createBufferOverflowError() string {
	var events []string

	if t.streamState.thinkingOpen {
		events = append(events, t.createBlockStopEvent(t.streamState.thinkingIndex))
	}
	if t.streamState.textOpen {
		events = append(events, t.createBlockStopEvent(t.streamState.textIndex))
	}
	if t.streamState.toolCallOpen {
		events = append(events, t.createBlockStopEvent(t.streamState.toolCallIndex))
	}

	errorIndex := 0
	if t.streamState != nil {
		errorIndex = t.streamState.nextBlockIndex
	}

	start := map[string]interface{}{
		"type":  "content_block_start",
		"index": errorIndex,
		"content_block": map[string]interface{}{
			"type": "text",
			"text": "",
		},
	}
	data, _ := json.Marshal(start)
	events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": errorIndex,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": "Stream buffer overflow - please retry",
		},
	}
	data, _ = json.Marshal(delta)
	events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))

	events = append(events, t.createBlockStopEvent(errorIndex))

	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   "error",
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": 0,
		},
	}
	data, _ = json.Marshal(msgDelta)
	events = append(events, fmt.Sprintf("event: message_delta\ndata: %s\n\n", data))

	msgStop := map[string]interface{}{
		"type": "message_stop",
	}
	data, _ = json.Marshal(msgStop)
	events = append(events, fmt.Sprintf("event: message_stop\ndata: %s\n\n", data))

	return strings.Join(events, "")
}

func (t *OpenAIToAnthropicTransformer) formatErrorResponse(errorObj map[string]interface{}, originalResp map[string]interface{}) ([]byte, error) {
	message := "Unknown error"
	if msg, ok := errorObj["message"].(string); ok && msg != "" {
		message = msg
	} else if typ, ok := errorObj["type"].(string); ok && typ != "" {
		message = typ
	}

	content := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": fmt.Sprintf("Error: %s", message),
		},
	}

	result := map[string]interface{}{
		"id":            fmt.Sprintf("msg_error_%d", time.Now().UnixNano()),
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"stop_reason":   "error",
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}

	if model, ok := originalResp["model"].(string); ok {
		result["model"] = model
	}

	return json.Marshal(result)
}

func (t *OpenAIToAnthropicTransformer) formatStreamError(errorObj map[string]interface{}) string {
	var events []string

	if t.streamState.textOpen {
		t.streamState.textOpen = false
		events = append(events, t.createBlockStopEvent(t.streamState.textIndex))
	}

	message := "Unknown error"
	if msg, ok := errorObj["message"].(string); ok && msg != "" {
		message = msg
	}

	errorIndex := t.streamState.nextBlockIndex
	t.streamState.nextBlockIndex++

	start := map[string]interface{}{
		"type":  "content_block_start",
		"index": errorIndex,
		"content_block": map[string]interface{}{
			"type": "text",
			"text": "",
		},
	}
	data, _ := json.Marshal(start)
	events = append(events, fmt.Sprintf("event: content_block_start\ndata: %s\n\n", data))

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": errorIndex,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": fmt.Sprintf("Error: %s", message),
		},
	}
	data, _ = json.Marshal(delta)
	events = append(events, fmt.Sprintf("event: content_block_delta\ndata: %s\n\n", data))

	events = append(events, t.createBlockStopEvent(errorIndex))

	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   "error",
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": 0,
		},
	}
	data, _ = json.Marshal(msgDelta)
	events = append(events, fmt.Sprintf("event: message_delta\ndata: %s\n\n", data))

	msgStop := map[string]interface{}{
		"type": "message_stop",
	}
	data, _ = json.Marshal(msgStop)
	events = append(events, fmt.Sprintf("event: message_stop\ndata: %s\n\n", data))

	return strings.Join(events, "")
}

func translateFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "content_filter"
	default:
		return "end_turn"
	}
}
