package anthropic

import (
	"strings"
	"testing"
)

func TestIsNonStreamingResponse(t *testing.T) {
	tests := []struct {
		name     string
		parsed   map[string]interface{}
		expected bool
	}{
		{
			name: "streaming delta",
			parsed: map[string]interface{}{
				"choices": []interface{}{
					map[string]interface{}{
						"delta": map[string]interface{}{
							"content": "hello",
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "non-streaming message",
			parsed: map[string]interface{}{
				"choices": []interface{}{
					map[string]interface{}{
						"message": map[string]interface{}{
							"content": "hello",
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "empty choices",
			parsed:   map[string]interface{}{"choices": []interface{}{}},
			expected: false,
		},
		{
			name:     "no choices",
			parsed:   map[string]interface{}{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isNonStreamingResponse(tt.parsed)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestTransformStream_BasicStreaming(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"},"index":0}],"usage":{"prompt_tokens":10,"completion_tokens":1}}`)

	modified, output, _ := transformer.TransformStream(chunk1)

	if !modified {
		t.Fatal("Expected chunk to be modified")
	}

	if len(output) == 0 {
		t.Fatal("Expected output to not be empty")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "message_start") {
		t.Error("Expected message_start in output")
	}
	if !strings.Contains(outputStr, "content_block_start") {
		t.Error("Expected content_block_start in output")
	}
	if !strings.Contains(outputStr, "Hello") {
		t.Error("Expected 'Hello' content in output")
	}
	if !strings.Contains(outputStr, "content_block_delta") {
		t.Error("Expected content_block_delta in output")
	}
}

func TestTransformStream_DONE(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"},"index":0}]}`)
	transformer.TransformStream(chunk1)

	chunk2 := []byte("[DONE]")
	modified, output, _ := transformer.TransformStream(chunk2)

	if !modified {
		t.Fatal("Expected DONE to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "message_delta") {
		t.Error("Expected message_delta in output")
	}
	if !strings.Contains(outputStr, "message_stop") {
		t.Error("Expected message_stop in output")
	}
}

func TestTransformStream_Error(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"},"index":0}]}`)
	transformer.TransformStream(chunk1)

	chunk2 := []byte(`{"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}`)
	modified, output, _ := transformer.TransformStream(chunk2)

	if !modified {
		t.Fatal("Expected error to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "Error") {
		t.Error("Expected Error in output")
	}
	if !strings.Contains(outputStr, "stop_reason") {
		t.Error("Expected stop_reason in output")
	}
}

func TestTransformStream_NonStreamingFallback(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{
		"id": "chatcmpl-123",
		"model": "gpt-4",
		"choices": [{
			"message": {
				"role": "assistant",
				"content": "Hello there"
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5}
	}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected non-streaming response to be converted")
	}

	outputStr := string(output)

	requiredEvents := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	for _, event := range requiredEvents {
		if !strings.Contains(outputStr, event) {
			t.Errorf("Missing required event: %s", event)
		}
	}

	if !strings.Contains(outputStr, "Hello there") {
		t.Error("Expected content 'Hello there' in output")
	}
}

func TestTransformStream_MultipleChunks(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"model":"gpt-4","choices":[{"delta":{"content":"Hello "},"index":0}]}`)
	modified1, _, _ := transformer.TransformStream(chunk1)
	if !modified1 {
		t.Fatal("Expected first chunk to be modified")
	}

	chunk2 := []byte(`{"choices":[{"delta":{"content":"World"},"index":0}]}`)
	modified2, output2, _ := transformer.TransformStream(chunk2)
	if !modified2 {
		t.Fatal("Expected second chunk to be modified")
	}

	outputStr := string(output2)
	if !strings.Contains(outputStr, "World") {
		t.Error("Expected 'World' in second chunk output")
	}

	chunk3 := []byte("[DONE]")
	modified3, output3, _ := transformer.TransformStream(chunk3)
	if !modified3 {
		t.Fatal("Expected DONE to be modified")
	}

	outputStr3 := string(output3)
	if !strings.Contains(outputStr3, "message_stop") {
		t.Error("Expected message_stop in DONE output")
	}
}

func TestTransformStream_ThinkingBlocks(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{"model":"claude-3","choices":[{"delta":{"reasoning_content":"Let me think about this..."},"index":0}]}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected chunk to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "thinking") {
		t.Error("Expected thinking block in output")
	}
	if !strings.Contains(outputStr, "thinking_delta") {
		t.Error("Expected thinking_delta in output")
	}
	if !strings.Contains(outputStr, "Let me think about this") {
		t.Error("Expected reasoning content in output")
	}
}

func TestTransformStream_ThinkingThenText(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"model":"claude-3","choices":[{"delta":{"reasoning_content":"Thinking..."},"index":0}]}`)
	_, _, _ = transformer.TransformStream(chunk1)

	chunk2 := []byte(`{"choices":[{"delta":{"content":"Hello"}}]}`)
	modified, output, _ := transformer.TransformStream(chunk2)

	if !modified {
		t.Fatal("Expected second chunk to be modified")
	}

	outputStr := string(output)

	if !strings.Contains(outputStr, "content_block_stop") {
		t.Error("Expected content_block_stop (thinking block closed)")
	}
	if !strings.Contains(outputStr, "Hello") {
		t.Error("Expected Hello in output")
	}
}

func TestTransformStream_ToolCalls(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{"choices":[{"delta":{"tool_calls":[{"id":"call_123","function":{"name":"read_file","arguments":"{\"path\":\"/test.txt\"}"}}]},"index":0}]}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected chunk to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "tool_use") {
		t.Error("Expected tool_use in output")
	}
	if !strings.Contains(outputStr, "input_json_delta") {
		t.Error("Expected input_json_delta in output")
	}
	if !strings.Contains(outputStr, "read_file") {
		t.Error("Expected function name in output")
	}
}

func TestTransformStream_BufferOverflow(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk1 := []byte(`{"model":"gpt-4","choices":[{"delta":{"content":"Hello"},"index":0}]}`)
	transformer.TransformStream(chunk1)

	largeContent := strings.Repeat("x", 1024*1024)
	chunk2 := []byte(`{"choices":[{"delta":{"content":"` + largeContent + `"}}]}`)

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Panic on large chunk: %v", r)
		}
	}()

	modified, output, _ := transformer.TransformStream(chunk2)

	if modified && len(output) > 0 {
		outputStr := string(output)
		if !strings.Contains(outputStr, "content_block_delta") && !strings.Contains(outputStr, "error") {
			t.Error("Expected either content or error")
		}
	}
}

func TestTransformStream_Reset(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{"model":"gpt-4","choices":[{"delta":{"content":"Hello"},"index":0}]}`)
	transformer.TransformStream(chunk)

	transformer.Reset()

	chunk2 := []byte(`{"model":"gpt-4","choices":[{"delta":{"content":"World"},"index":0}]}`)
	modified, output, _ := transformer.TransformStream(chunk2)

	if !modified {
		t.Fatal("Expected chunk after reset to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "message_start") {
		t.Error("Expected message_start (new stream)")
	}
	if !strings.Contains(outputStr, "World") {
		t.Error("Expected 'World' content")
	}
}

func TestTransformStream_EmptyChunk(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	modified, output, keep := transformer.TransformStream([]byte(""))

	if modified {
		t.Error("Expected empty chunk to not be modified")
	}

	if len(output) != 0 {
		t.Error("Expected empty output")
	}

	_ = keep
}

func TestTransformStream_MalformedJSON(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{invalid json`)

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Panic on malformed JSON: %v", r)
		}
	}()

	modified, output, keep := transformer.TransformStream(chunk)

	if modified && len(output) > 0 {
		t.Error("Expected malformed JSON to pass through")
	}

	_ = keep
}

func TestTranslateFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "content_filter"},
		{"unknown", "end_turn"},
		{"", "end_turn"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := translateFinishReason(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestCreateBlockStopEvent(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()
	transformer.streamState = &streamState{
		nextBlockIndex: 5,
	}

	event := transformer.createBlockStopEvent(0)
	if !strings.Contains(event, "content_block_stop") {
		t.Error("Expected content_block_stop in event")
	}
}

func TestTransformStream_CompleteMessage(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{
		"id": "chatcmpl-123",
		"model": "gpt-4",
		"choices": [{
			"message": {
				"role": "assistant",
				"content": "Test response"
			},
			"finish_reason": "stop"
		}]
	}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected message to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "message_start") {
		t.Error("Expected message_start")
	}
	if !strings.Contains(outputStr, "message_stop") {
		t.Error("Expected message_stop")
	}
	if !strings.Contains(outputStr, "Test response") {
		t.Error("Expected response content")
	}
}

func TestTransformStream_EmptyContent(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{"model":"gpt-4","choices":[{"delta":{},"index":0}]}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected empty delta to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "message_start") {
		t.Error("Expected message_start even with empty delta")
	}
}

func TestTransformStream_ToolCallsWithNameAndArgs(t *testing.T) {
	transformer := NewOpenAIToAnthropicTransformer()

	chunk := []byte(`{
		"choices": [{
			"delta": {
				"tool_calls": [
					{
						"id": "call_abc",
						"function": {
							"name": "get_weather",
							"arguments": "{\"location\": \"San Francisco\"}"
						}
					}
				]
			}
		}]
	}`)

	modified, output, _ := transformer.TransformStream(chunk)

	if !modified {
		t.Fatal("Expected tool call to be modified")
	}

	outputStr := string(output)
	if !strings.Contains(outputStr, "get_weather") {
		t.Error("Expected function name in output")
	}
	if !strings.Contains(outputStr, "San Francisco") {
		t.Error("Expected arguments in output")
	}
}
