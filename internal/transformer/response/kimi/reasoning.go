package kimi

import (
	"ai-proxy/internal/logger"
	"encoding/json"
	"strings"
)

const (
	REASONING_START = "<think>"
	REASONING_END   = "</think>"
)

// ReasoningAccumulator extracts content between   and   tags
// After the reasoning block ends, remaining content is passed through as normal content
type ReasoningAccumulator struct {
	buffer           strings.Builder
	inReasoning      bool
	justFinished     bool // true for one iteration after reasoning ends
	reasoningContent strings.Builder
}

func NewReasoningAccumulator() *ReasoningAccumulator {
	return &ReasoningAccumulator{}
}

func (a *ReasoningAccumulator) Reset() {
	a.buffer.Reset()
	a.inReasoning = false
	a.justFinished = false
	a.reasoningContent.Reset()
}

// feed processes incoming text and returns (reasoning, normal, isComplete)
// reasoning: content between   and
// normal: content after
// isComplete: true if we've seen the end tag
func (a *ReasoningAccumulator) feed(text string) (string, string, bool) {
	logger.Debugf("[kimi-reasoning] feed() called with %d chars, inReasoning=%v", len(text), a.inReasoning)

	a.buffer.WriteString(text)
	bufStr := a.buffer.String()

	var normalContent strings.Builder

	// Process complete tokens
	for len(bufStr) > 0 {
		if a.isPartialTokenPrefix(bufStr) {
			logger.Debugf("[kimi-reasoning] Breaking on partial token, buffer=%d chars", len(bufStr))
			break
		}

		if !a.inReasoning {
			// Not in reasoning block yet - check for both tokens
			// Check for END token first (since it can appear without START in some cases)
			if strings.HasPrefix(bufStr, REASONING_END) {
				logger.Debugf("[kimi-reasoning] Found reasoning end token (no start)")
				bufStr = bufStr[len(REASONING_END):]
				// Don't set inReasoning = false since we're already not in reasoning
				// Just consume the end token and continue
				if len(bufStr) == 0 {
					continue
				}
			}
			if strings.HasPrefix(bufStr, REASONING_START) {
				logger.Debugf("[kimi-reasoning] Found reasoning start token")
				a.inReasoning = true
				bufStr = bufStr[len(REASONING_START):]
				continue
			}
			// Content before   is normal content (not reasoning)
			ch := bufStr[0]
			bufStr = bufStr[1:]
			normalContent.WriteByte(ch)
		} else {
			// Inside reasoning block - look for end token
			if strings.HasPrefix(bufStr, REASONING_END) {
				logger.Debugf("[kimi-reasoning] Found reasoning end token")
				a.inReasoning = false
				a.justFinished = true
				bufStr = bufStr[len(REASONING_END):]
				continue
			}
			// Collect reasoning content
			ch := bufStr[0]
			bufStr = bufStr[1:]
			a.reasoningContent.WriteByte(ch)
		}
	}

	// Update buffer with remaining content
	a.buffer.Reset()
	a.buffer.WriteString(bufStr)

	reasoning := a.reasoningContent.String()
	normal := normalContent.String()

	// Clear the accumulators after returning the data
	// This is important because we call feed() multiple times for streaming
	a.reasoningContent.Reset()

	logger.Debugf("[kimi-reasoning] feed() returning: reasoning=%d chars, normal=%d chars, inReasoning=%v",
		len(reasoning), len(normal), a.inReasoning)

	return reasoning, normal, !a.inReasoning
}

func (a *ReasoningAccumulator) isPartialTokenPrefix(buf string) bool {
	if buf == "" {
		return false
	}

	tokens := []string{REASONING_START, REASONING_END}
	for _, tok := range tokens {
		if strings.HasPrefix(tok, buf) && buf != tok {
			return true
		}
	}
	return false
}

func (a *ReasoningAccumulator) hasIncompleteReasoning() bool {
	return a.inReasoning && a.reasoningContent.Len() > 0
}

type KimiReasoningTransformer struct {
	accumulator     *ReasoningAccumulator
	sawReasoning    bool
	chunkTemplate   map[string]interface{}
	lastChoiceIndex int
}

func NewKimiReasoningTransformer() *KimiReasoningTransformer {
	return &KimiReasoningTransformer{
		accumulator: NewReasoningAccumulator(),
	}
}

func (t *KimiReasoningTransformer) Reset() {
	t.accumulator.Reset()
	t.sawReasoning = false
	t.chunkTemplate = nil
	t.lastChoiceIndex = 0
}

func (t *KimiReasoningTransformer) Name() string {
	return "kimi-reasoning"
}

func (t *KimiReasoningTransformer) TransformResponse(body []byte, isStreaming bool) ([]byte, error) {
	if isStreaming {
		return body, nil
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, nil
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok {
		return body, nil
	}

	for i, choiceIF := range choices {
		choice, ok := choiceIF.(map[string]interface{})
		if !ok {
			continue
		}

		msgIF, ok := choice["message"]
		if !ok {
			continue
		}
		msg, ok := msgIF.(map[string]interface{})
		if !ok {
			continue
		}

		// Check content for reasoning tags
		if content, ok := msg["content"].(string); ok {
			if strings.Contains(content, REASONING_START) {
				logger.Debugf("[kimi-reasoning] Non-streaming: found reasoning tags in content")
				t.accumulator.Reset()
				reasoning, normal, _ := t.accumulator.feed(content)

				if reasoning != "" {
					msg["reasoning_content"] = reasoning
					t.sawReasoning = true
				}
				if normal != "" {
					msg["content"] = normal
				} else {
					delete(msg, "content")
				}
				choices[i] = choice
			}
		}
	}

	resp["choices"] = choices
	return json.Marshal(resp)
}

func (t *KimiReasoningTransformer) TransformStream(chunk []byte) (modified bool, newChunk []byte, keepChunk bool) {
	line := strings.TrimSpace(string(chunk))

	logger.Debugf("[kimi-reasoning] TransformStream: chunk_size=%d", len(chunk))

	if line == "" || line == "\n" {
		return false, chunk, true
	}

	if strings.HasPrefix(line, ":") {
		return false, chunk, true
	}

	if line == "data: [DONE]" {
		logger.Debugf("[kimi-reasoning] Received [DONE]")
		// Flush any remaining reasoning content
		if t.accumulator.hasIncompleteReasoning() {
			logger.Debugf("[kimi-reasoning] Flushing incomplete reasoning at [DONE]")
			remaining := t.accumulator.buffer.String()
			if remaining != "" {
				reasoningChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, map[string]interface{}{
					"reasoning_content": remaining,
				}, nil)
				if b, err := json.Marshal(reasoningChunk); err == nil {
					newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
				}
			}
		}
		return true, append(newChunk, []byte("data: [DONE]\n\n")...), false
	}

	jsonStr := line
	if strings.HasPrefix(line, "data: ") {
		jsonStr = line[len("data: "):]
	}

	var chunkData map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &chunkData); err != nil {
		logger.Debugf("[kimi-reasoning] JSON parse error: %v", err)
		return false, chunk, true
	}

	choices, ok := chunkData["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return false, chunk, true
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return false, chunk, true
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return false, chunk, true
	}

	if t.chunkTemplate == nil {
		t.chunkTemplate = map[string]interface{}{
			"id":                 chunkData["id"],
			"model":              chunkData["model"],
			"created":            chunkData["created"],
			"system_fingerprint": chunkData["system_fingerprint"],
		}
	}

	if idx, ok := choice["index"].(float64); ok {
		t.lastChoiceIndex = int(idx)
	}

	// Process content for reasoning tags
	if content, ok := delta["content"].(string); ok && content != "" {
		// Process if: content has tags, OR we're in reasoning mode, OR we just finished reasoning, OR there's buffered content
		shouldProcess := strings.Contains(content, REASONING_START) ||
			strings.Contains(content, REASONING_END) ||
			t.accumulator.inReasoning ||
			t.accumulator.justFinished ||
			t.accumulator.buffer.Len() > 0

		// Clear justFinished flag after checking (one-time only)
		wasJustFinished := t.accumulator.justFinished
		t.accumulator.justFinished = false

		logger.Debugf("[kimi-reasoning] Processing: content=%q, hasStart=%v, hasEnd=%v, inReasoning=%v, justFinished=%v, bufferLen=%d, shouldProcess=%v",
			content[:min(len(content), 30)], strings.Contains(content, REASONING_START),
			strings.Contains(content, REASONING_END), t.accumulator.inReasoning,
			wasJustFinished, t.accumulator.buffer.Len(), shouldProcess)

		if shouldProcess {
			logger.Debugf("[kimi-reasoning] Processing content with reasoning: %q", content[:min(len(content), 50)])

			reasoning, normal, _ := t.accumulator.feed(content)

			// Build output chunks
			if reasoning != "" {
				t.sawReasoning = true
				reasoningChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, map[string]interface{}{
					"reasoning_content": reasoning,
				}, nil)
				if b, err := json.Marshal(reasoningChunk); err == nil {
					newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
				}
				modified = true
			}

			if normal != "" {
				contentChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, map[string]interface{}{
					"content": normal,
				}, nil)
				if b, err := json.Marshal(contentChunk); err == nil {
					newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
				}
				modified = true
			}

			// Even if no reasoning/normal content, we still processed (e.g., just saw end tag)
			// This prevents the original content from passing through
			if modified || shouldProcess {
				return true, newChunk, false
			}
		}
	}

	return false, chunk, true
}

func (t *KimiReasoningTransformer) GetTokenUsage() (inputTokens, outputTokens int64) {
	return 0, 0
}
