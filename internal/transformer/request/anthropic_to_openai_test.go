package anthropic

import (
	"encoding/json"
	"testing"
)

func TestTransformUserMessageWithToolResults(t *testing.T) {
	// Test case: User message with tool results (after Claude used a tool)
	input := map[string]interface{}{
		"role": "user",
		"content": []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "toolu_12345",
				"content":     "File contents here",
			},
		},
	}

	result := transformUserMessageToSlice(input, input["content"])

	if len(result) != 1 {
		t.Fatalf("Expected 1 message for tool_result only, got %d", len(result))
	}

	msg := result[0]
	if msg["role"] != "tool" {
		t.Errorf("Expected role 'tool', got %v", msg["role"])
	}
	if msg["tool_call_id"] != "toolu_12345" {
		t.Errorf("Expected tool_call_id 'toolu_12345', got %v", msg["tool_call_id"])
	}
	if msg["content"] != "File contents here" {
		t.Errorf("Expected content 'File contents here', got %v", msg["content"])
	}
}

func TestTransformUserMessageWithTextAndToolResults(t *testing.T) {
	// Test case: User message with both text and tool results
	input := map[string]interface{}{
		"role": "user",
		"content": []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Please analyze this file",
			},
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "toolu_67890",
				"content":     "console.log('hello')",
			},
		},
	}

	result := transformUserMessageToSlice(input, input["content"])

	if len(result) != 2 {
		t.Fatalf("Expected 2 messages (user + tool), got %d", len(result))
	}

	// First message should be the user message with text
	userMsg := result[0]
	if userMsg["role"] != "user" {
		t.Errorf("First message: expected role 'user', got %v", userMsg["role"])
	}
	if userMsg["content"] != "Please analyze this file" {
		t.Errorf("First message: expected text content, got %v", userMsg["content"])
	}

	// Second message should be the tool result
	toolMsg := result[1]
	if toolMsg["role"] != "tool" {
		t.Errorf("Second message: expected role 'tool', got %v", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "toolu_67890" {
		t.Errorf("Second message: expected tool_call_id 'toolu_67890', got %v", toolMsg["tool_call_id"])
	}
}

func TestTransformUserMessageWithMultipleToolResults(t *testing.T) {
	// Test case: Multiple tool results
	input := map[string]interface{}{
		"role": "user",
		"content": []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "toolu_111",
				"content":     "Result 1",
			},
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "toolu_222",
				"content":     "Result 2",
			},
		},
	}

	result := transformUserMessageToSlice(input, input["content"])

	if len(result) != 2 {
		t.Fatalf("Expected 2 tool messages, got %d", len(result))
	}

	if result[0]["tool_call_id"] != "toolu_111" {
		t.Errorf("First tool: expected tool_call_id 'toolu_111', got %v", result[0]["tool_call_id"])
	}
	if result[1]["tool_call_id"] != "toolu_222" {
		t.Errorf("Second tool: expected tool_call_id 'toolu_222', got %v", result[1]["tool_call_id"])
	}
}

func TestTransformFullRequestWithToolResults(t *testing.T) {
	// Full request transformation test
	reqBody := []byte(`{
		"model": "claude-3-opus-20240229",
		"messages": [
			{"role": "user", "content": "Read the file"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_123", "name": "read_file", "input": {"path": "/test.txt"}}]},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "toolu_123", "content": "Hello World"},
					{"type": "text", "text": "What does it say?"}
				]
			}
		],
		"max_tokens": 1024
	}`)

	transformer := NewAnthropicToOpenAITransformer()
	result, err := transformer.TransformRequest(reqBody, "test-model")
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var transformed map[string]interface{}
	if err := json.Unmarshal(result, &transformed); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	messages, ok := transformed["messages"].([]interface{})
	if !ok {
		t.Fatal("messages field not found or not an array")
	}

	// Expected: system (none), user, assistant, user (text), tool
	if len(messages) != 4 {
		t.Fatalf("Expected 4 messages, got %d", len(messages))
	}

	// Check the tool message is properly formatted
	toolMsg, ok := messages[3].(map[string]interface{})
	if !ok {
		t.Fatal("Tool message is not a map")
	}

	if toolMsg["role"] != "tool" {
		t.Errorf("Expected tool message role 'tool', got %v", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "toolu_123" {
		t.Errorf("Expected tool_call_id 'toolu_123', got %v", toolMsg["tool_call_id"])
	}
	if toolMsg["content"] != "Hello World" {
		t.Errorf("Expected content 'Hello World', got %v", toolMsg["content"])
	}
}

func TestTransformUserMessageSimpleString(t *testing.T) {
	input := map[string]interface{}{
		"role":    "user",
		"content": "Hello there",
	}

	result := transformUserMessageToSlice(input, input["content"])

	if len(result) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(result))
	}

	if result[0]["content"] != "Hello there" {
		t.Errorf("Expected content 'Hello there', got %v", result[0]["content"])
	}
}

func TestTransformUserMessageWithImage(t *testing.T) {
	input := map[string]interface{}{
		"role": "user",
		"content": []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "What's in this image?",
			},
			map[string]interface{}{
				"type": "image",
				"source": map[string]interface{}{
					"type":       "base64",
					"media_type": "image/png",
					"data":       "base64encodeddata",
				},
			},
		},
	}

	result := transformUserMessageToSlice(input, input["content"])

	if len(result) != 1 {
		t.Fatalf("Expected 1 message for image content, got %d", len(result))
	}

	content, ok := result[0]["content"].([]interface{})
	if !ok {
		t.Fatalf("Expected content to be an array for image, got %T", result[0]["content"])
	}

	if len(content) != 2 {
		t.Errorf("Expected 2 content blocks (text + image), got %d", len(content))
	}
}

func TestSerializeToolResultContent(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "simple string",
			input:    "Simple result",
			expected: "Simple result",
		},
		{
			name: "array of text blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Line 1"},
				map[string]interface{}{"type": "text", "text": "Line 2"},
			},
			expected: "Line 1\nLine 2",
		},
		{
			name:     "nil content",
			input:    nil,
			expected: "",
		},
		{
			name:     "complex object",
			input:    map[string]interface{}{"result": "success"},
			expected: `{"result":"success"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := serializeToolResultContent(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestTransformAssistantMessage(t *testing.T) {
	// Test assistant message with tool_use
	input := map[string]interface{}{
		"role": "assistant",
		"content": []interface{}{
			map[string]interface{}{
				"type":  "tool_use",
				"id":    "toolu_456",
				"name":  "read_file",
				"input": map[string]interface{}{"path": "/etc/hosts"},
			},
		},
	}

	result := transformAssistantMessage(input, input["content"])

	if result["content"] != "" {
		t.Errorf("Expected empty content for tool-only message, got %v", result["content"])
	}

	toolCalls, ok := result["tool_calls"].([]interface{})
	if !ok {
		t.Fatal("Expected tool_calls to be present")
	}
	if len(toolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(toolCalls))
	}
}
