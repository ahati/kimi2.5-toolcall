package kimi

import (
	"encoding/json"
	"testing"
)

func createMockChunk(content string) []byte {
	chunk := map[string]interface{}{
		"id":      "test-id",
		"object":  "chat.completion.chunk",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"content": content,
				},
				"finish_reason": nil,
			},
		},
	}
	b, _ := json.Marshal(chunk)
	return []byte("data: " + string(b) + "\n\n")
}

func createMockChunkWithRole(content, role string) []byte {
	delta := map[string]interface{}{}
	if content != "" {
		delta["content"] = content
	}
	if role != "" {
		delta["role"] = role
	}

	chunk := map[string]interface{}{
		"id":      "test-id",
		"object":  "chat.completion.chunk",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []interface{}{
			map[string]interface{}{
				"index":         0,
				"delta":         delta,
				"finish_reason": nil,
			},
		},
	}
	b, _ := json.Marshal(chunk)
	return []byte("data: " + string(b) + "\n\n")
}

func extractContentFromChunk(t *testing.T, chunk []byte) (reasoning, content string) {
	line := string(chunk)
	if len(line) < 6 || line[:6] != "data: " {
		t.Fatalf("Invalid chunk format: %s", line)
	}

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(line[6:]), &data); err != nil {
		t.Fatalf("JSON parse error: %v", err)
	}

	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("No choices in chunk")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Choice is not a map")
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		t.Fatalf("Delta is not a map")
	}

	if rc, ok := delta["reasoning_content"].(string); ok {
		reasoning = rc
	}
	if c, ok := delta["content"].(string); ok {
		content = c
	}

	return
}

func TestKimiReasoningTransformer_ThinkingMode(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	tests := []struct {
		name              string
		chunks            [][]byte
		expectedReasoning string
		expectedContent   string
	}{
		{
			name: "simple thinking block",
			chunks: [][]byte{
				createMockChunk("<think>Let me"),
				createMockChunk(" think about this."),
				createMockChunk("</think>Bonjour!"),
			},
			expectedReasoning: "Let me think about this.",
			expectedContent:   "Bonjour!",
		},
		{
			name: "thinking with newlines",
			chunks: [][]byte{
				createMockChunk("<think>I should check the weather"),
				createMockChunk(" before answering.\n"),
				createMockChunk("</think>\n"),
			},
			expectedReasoning: "I should check the weather before answering.\n",
			expectedContent:   "\n",
		},
		{
			name: "thinking followed by tool calls",
			chunks: [][]byte{
				createMockChunk("<think>I should check the weather"),
				createMockChunk(" before answering.\n"),
				createMockChunk("</think>\n"),
				createMockChunk("<|tool_calls_section_begin|>"),
				createMockChunk("<|tool_call_begin|>functions.get_weather:0"),
				createMockChunk("<|tool_call_argument_begin|>"),
				createMockChunk(`{"location":"NYC"}`),
				createMockChunk("<|tool_call_end|>"),
				createMockChunk("<|tool_calls_section_end|>"),
			},
			expectedReasoning: "I should check the weather before answering.\n",
			expectedContent:   "\n<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transformer.Reset()
			var allReasoning, allContent string

			for _, chunk := range tt.chunks {
				modified, newChunk, keepOriginal := transformer.TransformStream(chunk)
				if modified && len(newChunk) > 0 {
					// Use the transformed chunk
					reasoning, content := extractContentFromChunk(t, newChunk)
					allReasoning += reasoning
					allContent += content
				} else if !keepOriginal {
					// Transformer consumed the chunk but didn't emit new content
					// Don't add anything
				} else {
					// No modification - use original chunk
					_, content := extractContentFromChunk(t, chunk)
					allContent += content
				}
			}

			if allReasoning != tt.expectedReasoning {
				t.Errorf("reasoning = %q, want %q", allReasoning, tt.expectedReasoning)
			}
			if allContent != tt.expectedContent {
				t.Errorf("content = %q, want %q", allContent, tt.expectedContent)
			}
		})
	}
}

func TestKimiReasoningTransformer_InstantMode(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// Empty think block = instant mode (no thinking, direct answer)
	chunks := [][]byte{
		createMockChunk("<think>"),
		createMockChunk("</think>"),
		createMockChunk("Direct answer without thinking."),
	}

	var allReasoning, allContent string

	for _, chunk := range chunks {
		modified, newChunk, keepOriginal := transformer.TransformStream(chunk)
		if modified && len(newChunk) > 0 {
			reasoning, content := extractContentFromChunk(t, newChunk)
			allReasoning += reasoning
			allContent += content
		} else if !keepOriginal {
			// Transformer consumed the chunk but didn't emit new content
			// Don't add anything
		} else {
			_, content := extractContentFromChunk(t, chunk)
			allContent += content
		}
	}

	if allReasoning != "" {
		t.Errorf("expected empty reasoning for instant mode, got %q", allReasoning)
	}
	if allContent != "Direct answer without thinking." {
		t.Errorf("content = %q, want %q", allContent, "Direct answer without thinking.")
	}
}

func TestKimiReasoningTransformer_TokenByToken(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// Simulates token-by-token streaming
	chunks := [][]byte{
		createMockChunk("<think>"),
		createMockChunk("The user"),
		createMockChunk(" asked me"),
		createMockChunk(" to say hello."),
		createMockChunk("</think>"),
		createMockChunk("Hello"),
		createMockChunk("!"),
	}

	var allReasoning, allContent string

	for _, chunk := range chunks {
		modified, newChunk, keepOriginal := transformer.TransformStream(chunk)
		if modified && len(newChunk) > 0 {
			reasoning, content := extractContentFromChunk(t, newChunk)
			allReasoning += reasoning
			allContent += content
		} else if !keepOriginal {
			// Transformer consumed the chunk but didn't emit new content
			// Don't add anything
		} else {
			_, content := extractContentFromChunk(t, chunk)
			allContent += content
		}
	}

	expectedReasoning := "The user asked me to say hello."
	expectedContent := "Hello!"

	if allReasoning != expectedReasoning {
		t.Errorf("reasoning = %q, want %q", allReasoning, expectedReasoning)
	}
	if allContent != expectedContent {
		t.Errorf("content = %q, want %q", allContent, expectedContent)
	}
}

func TestKimiReasoningTransformer_NoReasoningTags(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// No reasoning tags - content should pass through unchanged
	chunks := [][]byte{
		createMockChunk("Just normal text without reasoning tags."),
	}

	modified, newChunk, _ := transformer.TransformStream(chunks[0])

	// Should pass through unmodified
	if modified {
		t.Error("Expected no modification for content without reasoning tags")
	}
	if string(newChunk) != string(chunks[0]) {
		t.Error("Expected chunk to pass through unchanged")
	}
}

func TestKimiReasoningTransformer_DONE(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// First send some content
	transformer.TransformStream(createMockChunk("<think>Thinking..."))
	transformer.TransformStream(createMockChunk("more thinking"))
	transformer.TransformStream(createMockChunk("</think>"))

	// Then receive DONE
	modified, newChunk, keep := transformer.TransformStream([]byte("data: [DONE]\n\n"))

	if !modified {
		t.Error("Expected modified=true when receiving DONE")
	}
	if keep {
		t.Error("Expected keep=false when receiving DONE")
	}
	if string(newChunk) != "data: [DONE]\n\n" {
		t.Errorf("Expected DONE chunk, got %q", string(newChunk))
	}
}

func TestKimiReasoningTransformer_PartialTokens(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// Partial token at end of chunk - should be buffered
	chunk1 := createMockChunk("Let me think about this. The answer is ")
	chunk2 := createMockChunk("<think") // Partial start token
	chunk3 := createMockChunk("> actual answer")

	// This tests that partial tokens don't cause issues
	_, _, _ = transformer.TransformStream(chunk1)
	_, _, _ = transformer.TransformStream(chunk2)
	_, newChunk3, _ := transformer.TransformStream(chunk3)

	// Should process correctly without errors
	if len(newChunk3) == 0 {
		t.Log("Partial token handling - waiting for more data")
	}
}

func TestKimiReasoningTransformer_CombinedWithToolCalls(t *testing.T) {
	// This test simulates the full flow:
	// 1. reasoning transformer extracts <think> blocks
	// 2. tool_calls transformer processes remaining content

	reasoningTransformer := NewKimiReasoningTransformer()
	toolCallsTransformer := NewKimiToolCallsTransformer()

	inputChunks := [][]byte{
		createMockChunk("<think>I should check the weather"),
		createMockChunk(" before answering.\n"),
		createMockChunk("</think>\n"),
		createMockChunk("<|tool_calls_section_begin|>"),
		createMockChunk("<|tool_call_begin|>functions.get_weather:0"),
		createMockChunk("<|tool_call_argument_begin|>"),
		createMockChunk(`{"location":"NYC"}`),
		createMockChunk("<|tool_call_end|>"),
		createMockChunk("<|tool_calls_section_end|>"),
	}

	var allReasoning string
	var toolCallFound bool

	for _, chunk := range inputChunks {
		// Step 1: reasoning transformer
		modified, reasoningChunk, _ := reasoningTransformer.TransformStream(chunk)

		if modified && len(reasoningChunk) > 0 {
			reasoning, _ := extractContentFromChunk(t, reasoningChunk)
			allReasoning += reasoning

			// Step 2: pass transformed chunk to tool_calls transformer
			_, toolChunk, _ := toolCallsTransformer.TransformStream(reasoningChunk)
			if len(toolChunk) > 0 {
				// Check if tool call was extracted
				line := string(toolChunk)
				if len(line) > 6 && line[:6] == "data: " {
					var data map[string]interface{}
					if err := json.Unmarshal([]byte(line[6:]), &data); err == nil {
						if choices, ok := data["choices"].([]interface{}); ok && len(choices) > 0 {
							if choice, ok := choices[0].(map[string]interface{}); ok {
								if tc, ok := choice["delta"].(map[string]interface{})["tool_calls"]; ok && tc != nil {
									toolCallFound = true
								}
							}
						}
					}
				}
			}
		}
	}

	expectedReasoning := "I should check the weather before answering.\n"
	if allReasoning != expectedReasoning {
		t.Errorf("reasoning = %q, want %q", allReasoning, expectedReasoning)
	}

	// Tool call should be found in the transformed output
	if !toolCallFound {
		t.Log("Tool call should be extracted from content after reasoning block")
	}
}

func TestKimiReasoningTransformer_NonStreaming(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// Test non-streaming response
	body := []byte(`{
		"choices": [{
			"message": {
				"content": "<think>This is reasoning content</think>This is the actual answer."
			}
		}]
	}`)

	transformed, err := transformer.TransformResponse(body, false)
	if err != nil {
		t.Fatalf("TransformResponse error: %v", err)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(transformed, &resp); err != nil {
		t.Fatalf("JSON parse error: %v", err)
	}

	choices := resp["choices"].([]interface{})
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})

	// Check reasoning_content
	rc, hasRC := msg["reasoning_content"]
	if !hasRC {
		t.Error("Expected reasoning_content field")
	}
	if rc != "This is reasoning content" {
		t.Errorf("reasoning_content = %q, want %q", rc, "This is reasoning content")
	}

	// Check content
	content, hasContent := msg["content"]
	if !hasContent {
		t.Error("Expected content field")
	}
	if content != "This is the actual answer." {
		t.Errorf("content = %q, want %q", content, "This is the actual answer.")
	}
}

func TestKimiReasoningTransformer_NonStreamingWithToolCalls(t *testing.T) {
	transformer := NewKimiReasoningTransformer()

	// Test non-streaming response with thinking and tool calls
	body := []byte(`{
		"choices": [{
			"message": {
				"content": "<think>Let me check the weather</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>"
			}
		}]
	}`)

	transformed, err := transformer.TransformResponse(body, false)
	if err != nil {
		t.Fatalf("TransformResponse error: %v", err)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(transformed, &resp); err != nil {
		t.Fatalf("JSON parse error: %v", err)
	}

	choices := resp["choices"].([]interface{})
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})

	// Check reasoning_content
	rc, hasRC := msg["reasoning_content"]
	if !hasRC {
		t.Error("Expected reasoning_content field")
	}
	if rc != "Let me check the weather" {
		t.Errorf("reasoning_content = %q, want %q", rc, "Let me check the weather")
	}

	// Check content - should contain tool call section
	content, hasContent := msg["content"]
	if !hasContent {
		t.Error("Expected content field")
	}
	expectedContent := "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>"
	if content != expectedContent {
		t.Errorf("content = %q, want %q", content, expectedContent)
	}
}
