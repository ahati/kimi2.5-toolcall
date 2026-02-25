package kimi

import (
	"ai-proxy/internal/logger"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
)

const (
	TOK_SECTION_BEGIN = "<|tool_calls_section_begin|>"
	TOK_SECTION_END   = "<|tool_calls_section_end|>"
	TOK_CALL_BEGIN    = "<|tool_call_begin|>"
	TOK_CALL_END      = "<|tool_call_end|>"
	TOK_ARG_BEGIN     = "<|tool_call_argument_begin|>"
)

var toolTokens = []string{
	TOK_SECTION_BEGIN,
	TOK_SECTION_END,
	TOK_CALL_BEGIN,
	TOK_CALL_END,
	TOK_ARG_BEGIN,
}

type ParsedToolCall struct {
	CallID       string
	FunctionName string
	Arguments    string
}

type ToolCallAccumulator struct {
	buffer         strings.Builder
	toolCalls      []ParsedToolCall
	inSection      bool
	finished       bool
	inCall         bool
	inArgs         bool
	currentIDBuf   strings.Builder
	currentArgsBuf strings.Builder
}

func (a *ToolCallAccumulator) feed(text string) string {
	a.buffer.WriteString(text)
	return a.consume()
}

func (a *ToolCallAccumulator) consume() string {
	var clean strings.Builder
	bufStr := a.buffer.String()

	for len(bufStr) > 0 {
		if isPartialTokenPrefix(bufStr) {
			break
		}

		matched := false

		if strings.HasPrefix(bufStr, TOK_SECTION_BEGIN) {
			a.inSection = true
			bufStr = bufStr[len(TOK_SECTION_BEGIN):]
			logger.Debugf("[kimi-tool-calls] Tool call section STARTED — intercepting raw tool tokens")
			matched = true
		} else if strings.HasPrefix(bufStr, TOK_SECTION_END) {
			a.inSection = false
			a.finished = true
			bufStr = bufStr[len(TOK_SECTION_END):]
			logger.Debugf("[kimi-tool-calls] Tool call section ENDED — %d tool call(s) parsed", len(a.toolCalls))
			matched = true
		} else if strings.HasPrefix(bufStr, TOK_CALL_BEGIN) {
			a.inCall = true
			a.inArgs = false
			a.currentIDBuf.Reset()
			a.currentArgsBuf.Reset()
			bufStr = bufStr[len(TOK_CALL_BEGIN):]
			logger.Debugf("[kimi-tool-calls] Parsing new tool call …")
			matched = true
		} else if strings.HasPrefix(bufStr, TOK_ARG_BEGIN) {
			a.inArgs = true
			bufStr = bufStr[len(TOK_ARG_BEGIN):]
			matched = true
		} else if strings.HasPrefix(bufStr, TOK_CALL_END) {
			a.finalizeCall()
			bufStr = bufStr[len(TOK_CALL_END):]
			matched = true
		}

		if !matched {
			ch := bufStr[0]
			bufStr = bufStr[1:]
			if a.inCall {
				if a.inArgs {
					a.currentArgsBuf.WriteByte(ch)
				} else {
					a.currentIDBuf.WriteByte(ch)
				}
			} else if a.inSection {
				// skip whitespace/noise between calls in section
			} else {
				clean.WriteByte(ch)
			}
		}
	}

	// Update buffer with remaining unprocessed content
	a.buffer.Reset()
	a.buffer.WriteString(bufStr)

	return clean.String()
}

func (a *ToolCallAccumulator) finalizeCall() {
	rawID := strings.TrimSpace(a.currentIDBuf.String())
	rawArgs := strings.TrimSpace(a.currentArgsBuf.String())

	funcName := rawID
	if strings.HasPrefix(funcName, "functions.") {
		funcName = strings.TrimPrefix(funcName, "functions.")
	}

	if idx := strings.Index(funcName, ":"); idx != -1 {
		funcName = funcName[:idx]
	}

	if funcName == "" {
		logger.Warnf("[kimi-tool-calls] Parsed tool call with EMPTY function name (raw_id=%q). This tool call will likely fail downstream.", rawID)
	}

	callID := fmt.Sprintf("call_%s", uuid.New().String()[:24])

	var argsStr string
	var argsObj map[string]interface{}
	if err := json.Unmarshal([]byte(rawArgs), &argsObj); err == nil {
		argsBytes, _ := json.Marshal(argsObj)
		argsStr = string(argsBytes)
	} else {
		argsPreview := rawArgs
		if len(argsPreview) > 200 {
			argsPreview = argsPreview[:200] + "..."
		}
		logger.Warnf("[kimi-tool-calls] Could not parse tool call arguments as valid JSON for '%s' (call_id=%s): %s", funcName, callID, argsPreview)
		argsStr = rawArgs
	}

	a.toolCalls = append(a.toolCalls, ParsedToolCall{
		CallID:       callID,
		FunctionName: funcName,
		Arguments:    argsStr,
	})

	argsPreview := argsStr
	if len(argsPreview) > 120 {
		argsPreview = argsPreview[:120] + "..."
	}
	logger.Debugf("[kimi-tool-calls] Parsed tool call: %s (call_id=%s) args=%s", funcName, callID, argsPreview)

	a.inCall = false
	a.inArgs = false
	a.currentIDBuf.Reset()
	a.currentArgsBuf.Reset()
}

func isPartialTokenPrefix(buf string) bool {
	if buf == "" {
		return false
	}
	for _, tok := range toolTokens {
		if strings.HasPrefix(tok, buf) && buf != tok {
			return true
		}
	}
	return false
}

func containsToolTokens(text string) bool {
	if text == "" {
		return false
	}
	return strings.Contains(text, TOK_SECTION_BEGIN) ||
		strings.Contains(text, TOK_SECTION_END) ||
		strings.Contains(text, TOK_CALL_BEGIN) ||
		strings.Contains(text, TOK_CALL_END) ||
		strings.Contains(text, TOK_ARG_BEGIN)
}

var toolSectionRegex = regexp.MustCompile(
	regexp.QuoteMeta(TOK_SECTION_BEGIN) + `(.*?)` + regexp.QuoteMeta(TOK_SECTION_END),
)

func init() {
	_ = toolSectionRegex
}

type KimiToolCallsTransformer struct {
	accumulator      ToolCallAccumulator
	sawToolTokens    bool
	emittedToolCalls []ParsedToolCall
	emittedFinish    bool
	chunkTemplate    map[string]interface{}
	lastChoiceIndex  int
	inputTokens      int64
	outputTokens     int64
}

func logf(format string, args ...interface{}) {
	logger.Debugf("[KIMI-TC] "+format, args...)
}

func NewKimiToolCallsTransformer() *KimiToolCallsTransformer {
	return &KimiToolCallsTransformer{
		accumulator:      ToolCallAccumulator{},
		sawToolTokens:    false,
		emittedToolCalls: []ParsedToolCall{},
		emittedFinish:    false,
	}
}

func (t *KimiToolCallsTransformer) Reset() {
	t.accumulator = ToolCallAccumulator{}
	t.sawToolTokens = false
	t.emittedToolCalls = []ParsedToolCall{}
	t.emittedFinish = false
	t.chunkTemplate = nil
	t.lastChoiceIndex = 0
	t.inputTokens = 0
	t.outputTokens = 0
}

func (t *KimiToolCallsTransformer) GetTokenUsage() (inputTokens, outputTokens int64) {
	return t.inputTokens, t.outputTokens
}

func (t *KimiToolCallsTransformer) Name() string {
	return "kimi-tool-calls"
}

func (t *KimiToolCallsTransformer) TransformResponse(body []byte, isStreaming bool) ([]byte, error) {
	logf("TransformResponse called (streaming=%v, body size=%d)", isStreaming, len(body))
	startTime := time.Now()

	if isStreaming {
		result, err := t.transformStreaming(body)
		logf("TransformResponse streaming completed in %v", time.Since(startTime))
		return result, err
	}
	result, err := t.transformNonStreaming(body)
	logf("TransformResponse non-streaming completed in %v", time.Since(startTime))
	return result, err
}

func (t *KimiToolCallsTransformer) transformNonStreaming(body []byte) ([]byte, error) {
	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, nil
	}

	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			t.inputTokens = int64(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			t.outputTokens = int64(ct)
		} else if tt, ok := usage["total_tokens"].(float64); ok {
			if t.inputTokens > 0 {
				t.outputTokens = int64(tt) - t.inputTokens
			}
		}
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok {
		logf("No choices found in response")
		return body, nil
	}
	logf("Processing %d choices", len(choices))

	for i, choiceIF := range choices {
		logf("Processing choice %d", i)
		choice, ok := choiceIF.(map[string]interface{})
		if !ok {
			logf("Choice %d is not a map, skipping", i)
			continue
		}

		msgIF, ok := choice["message"]
		if !ok {
			logf("Choice %d has no message, skipping", i)
			continue
		}
		msg, ok := msgIF.(map[string]interface{})
		if !ok {
			logf("Message in choice %d is not a map, skipping", i)
			continue
		}

		allToolCalls := []interface{}{}

		if rc, ok := msg["reasoning_content"].(string); ok && containsToolTokens(rc) {
			logf("Choice %d has tool tokens in reasoning_content, parsing...", i)
			cleanRC, tc := parseToolCallsFromText(rc)
			cleanRC = strings.TrimRight(cleanRC, " \n")
			if cleanRC != "" {
				msg["reasoning_content"] = cleanRC
			} else {
				delete(msg, "reasoning_content")
			}
			logf("Choice %d: parsed %d tool calls from reasoning_content", i, len(tc))
			for _, ptc := range tc {
				allToolCalls = append(allToolCalls, toolCallToJSON(ptc))
			}
		}

		if content, ok := msg["content"].(string); ok && containsToolTokens(content) {
			logf("Choice %d has tool tokens in content, parsing...", i)
			cleanC, tc := parseToolCallsFromText(content)
			cleanC = strings.TrimRight(cleanC, " \n")
			if cleanC != "" {
				msg["content"] = cleanC
			} else {
				delete(msg, "content")
			}
			logf("Choice %d: parsed %d tool calls from content", i, len(tc))
			for _, ptc := range tc {
				allToolCalls = append(allToolCalls, toolCallToJSON(ptc))
			}
		}

		if len(allToolCalls) > 0 {
			existing, _ := msg["tool_calls"].([]interface{})
			if existing == nil {
				existing = []interface{}{}
			}
			msg["tool_calls"] = append(existing, allToolCalls...)
			choice["finish_reason"] = "tool_calls"
			choices[i] = choice

			funcNames := make([]string, len(allToolCalls))
			for j, tc := range allToolCalls {
				if tcMap, ok := tc.(map[string]interface{}); ok {
					if fn, ok := tcMap["function"].(map[string]interface{}); ok {
						funcNames[j], _ = fn["name"].(string)
					}
				}
			}
			logf("Non-streaming fix: converted %d raw tool call token(s) → native tool_calls [%s]",
				len(allToolCalls), strings.Join(funcNames, ", "))
		}
	}

	resp["choices"] = choices
	return json.Marshal(resp)
}

func (t *KimiToolCallsTransformer) transformStreaming(body []byte) ([]byte, error) {
	// Streaming transformation is handled via TransformStream() method
	// This method is kept for interface compatibility
	return body, nil
}

func (t *KimiToolCallsTransformer) TransformStream(chunk []byte) (modified bool, newChunk []byte, keepChunk bool) {
	line := strings.TrimSpace(string(chunk))

	if line == "" || line == "\n" {
		return false, chunk, true
	}

	if strings.HasPrefix(line, ":") {
		return false, chunk, true
	}

	if line == "data: [DONE]" {
		if len(t.accumulator.toolCalls) > 0 {
			logf("Stream [DONE]: flushing %d remaining tool call(s)", len(t.accumulator.toolCalls))
			for _, tc := range t.accumulator.toolCalls {
				toolCallName := map[string]interface{}{
					"tool_calls": []interface{}{
						map[string]interface{}{
							"index": len(t.emittedToolCalls),
							"id":    tc.CallID,
							"type":  "function",
							"function": map[string]interface{}{
								"name":      tc.FunctionName,
								"arguments": "",
							},
						},
					},
				}
				nameChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, toolCallName, nil)
				if b, err := json.Marshal(nameChunk); err == nil {
					newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
				}

				toolCallArgs := map[string]interface{}{
					"tool_calls": []interface{}{
						map[string]interface{}{
							"index": len(t.emittedToolCalls),
							"function": map[string]interface{}{
								"arguments": tc.Arguments,
							},
						},
					},
				}
				argsChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, toolCallArgs, nil)
				if b, err := json.Marshal(argsChunk); err == nil {
					newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
				}

				t.emittedToolCalls = append(t.emittedToolCalls, tc)
			}

			finishChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, map[string]interface{}{}, "tool_calls")
			if b, err := json.Marshal(finishChunk); err == nil {
				newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
			}
			t.emittedFinish = true
		}
		if t.sawToolTokens {
			logf("Stream complete — tool call tokens were intercepted and converted")
		}
		return false, newChunk, false
	}

	if !strings.HasPrefix(line, "data: ") {
		return false, chunk, true
	}

	jsonStr := line[len("data: "):]
	var chunkData map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &chunkData); err != nil {
		return false, chunk, true
	}

	if usage, ok := chunkData["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			t.inputTokens = int64(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			t.outputTokens = int64(ct)
		} else if tt, ok := usage["total_tokens"].(float64); ok {
			if t.inputTokens > 0 {
				t.outputTokens = int64(tt) - t.inputTokens
			}
		}
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

	choiceIndex, _ := choice["index"].(int)
	t.lastChoiceIndex = choiceIndex

	modified = false

	if rc, ok := delta["reasoning_content"].(string); ok {
		t.outputTokens += int64(len(rc)) / 4
		if containsToolTokens(rc) || t.accumulator.inSection || t.accumulator.inCall {
			clean := t.accumulator.feed(rc)
			if clean != "" {
				delta["reasoning_content"] = clean
			} else {
				delete(delta, "reasoning_content")
			}
			modified = true
		}
	}

	if content, ok := delta["content"].(string); ok {
		t.outputTokens += int64(len(content)) / 4
		if containsToolTokens(content) || t.accumulator.inSection || t.accumulator.inCall {
			clean := t.accumulator.feed(content)
			if clean != "" {
				delta["content"] = clean
			} else {
				delete(delta, "content")
			}
			modified = true
		}
	}

	if t.accumulator.finished && len(t.accumulator.toolCalls) > 0 {
		t.sawToolTokens = true
		logf("Emitting %d parsed tool call(s) into stream", len(t.accumulator.toolCalls))

		if modified {
			hasContent := delta["reasoning_content"] != nil || delta["content"] != nil || delta["role"] != nil
			if hasContent {
				choice["finish_reason"] = nil
				newChunkData := map[string]interface{}{
					"id":                 t.chunkTemplate["id"],
					"object":             "chat.completion.chunk",
					"created":            t.chunkTemplate["created"],
					"model":              t.chunkTemplate["model"],
					"system_fingerprint": t.chunkTemplate["system_fingerprint"],
					"choices":            []interface{}{choice},
				}
				if b, err := json.Marshal(newChunkData); err == nil {
					newChunk = []byte("data: " + string(b) + "\n\n")
				}
			}
		}

		for _, tc := range t.accumulator.toolCalls {
			toolCallName := map[string]interface{}{
				"tool_calls": []interface{}{
					map[string]interface{}{
						"index": len(t.emittedToolCalls),
						"id":    tc.CallID,
						"type":  "function",
						"function": map[string]interface{}{
							"name":      tc.FunctionName,
							"arguments": "",
						},
					},
				},
			}
			nameChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, toolCallName, nil)
			if b, err := json.Marshal(nameChunk); err == nil {
				newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
			}

			toolCallArgs := map[string]interface{}{
				"tool_calls": []interface{}{
					map[string]interface{}{
						"index": len(t.emittedToolCalls),
						"function": map[string]interface{}{
							"arguments": tc.Arguments,
						},
					},
				},
			}
			argsChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, toolCallArgs, nil)
			if b, err := json.Marshal(argsChunk); err == nil {
				newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
			}

			t.emittedToolCalls = append(t.emittedToolCalls, tc)
		}

		finishChunk := buildChunk(t.chunkTemplate, t.lastChoiceIndex, map[string]interface{}{}, "tool_calls")
		if b, err := json.Marshal(finishChunk); err == nil {
			newChunk = append(newChunk, []byte("data: "+string(b)+"\n\n")...)
		}
		t.emittedFinish = true

		t.accumulator.toolCalls = nil
		t.emittedToolCalls = []ParsedToolCall{}
		t.accumulator.finished = false
		t.accumulator.buffer.Reset()
		t.accumulator.inSection = false
		t.accumulator.inCall = false
		t.accumulator.inArgs = false

		return true, newChunk, false
	}

	if modified {
		hasContent := delta["reasoning_content"] != nil || delta["content"] != nil || delta["role"] != nil
		if hasContent {
			choice["finish_reason"] = nil
			newChunkData := map[string]interface{}{
				"id":                 t.chunkTemplate["id"],
				"object":             "chat.completion.chunk",
				"created":            t.chunkTemplate["created"],
				"model":              t.chunkTemplate["model"],
				"system_fingerprint": t.chunkTemplate["system_fingerprint"],
				"choices":            []interface{}{choice},
			}
			if b, err := json.Marshal(newChunkData); err == nil {
				newChunk = []byte("data: " + string(b) + "\n\n")
			}
		}
		return true, newChunk, false
	}

	if t.sawToolTokens {
		if finishReason, ok := choice["finish_reason"].(string); ok && finishReason == "stop" {
			if t.emittedFinish {
				return true, nil, false
			}
			choice["finish_reason"] = "tool_calls"
			newChunkData := map[string]interface{}{
				"id":                 t.chunkTemplate["id"],
				"object":             "chat.completion.chunk",
				"created":            t.chunkTemplate["created"],
				"model":              t.chunkTemplate["model"],
				"system_fingerprint": t.chunkTemplate["system_fingerprint"],
				"choices":            []interface{}{choice},
			}
			if b, err := json.Marshal(newChunkData); err == nil {
				newChunk = []byte("data: " + string(b) + "\n\n")
			}
			return true, newChunk, false
		}
	}

	return false, chunk, true
}

func buildChunk(template map[string]interface{}, choiceIndex int, delta map[string]interface{}, finishReason interface{}) map[string]interface{} {
	created := int(time.Now().Unix())
	if c, ok := template["created"].(float64); ok {
		created = int(c)
	}

	result := map[string]interface{}{
		"id":                 template["id"],
		"object":             "chat.completion.chunk",
		"created":            created,
		"model":              template["model"],
		"system_fingerprint": template["system_fingerprint"],
		"choices": []interface{}{
			map[string]interface{}{
				"index":         choiceIndex,
				"delta":         delta,
				"finish_reason": finishReason,
			},
		},
	}
	return result
}

func toolCallToJSON(ptc ParsedToolCall) map[string]interface{} {
	return map[string]interface{}{
		"id":   ptc.CallID,
		"type": "function",
		"function": map[string]interface{}{
			"name":      ptc.FunctionName,
			"arguments": ptc.Arguments,
		},
	}
}

func parseToolCallsFromText(text string) (string, []ParsedToolCall) {
	logf("Parsing tool calls from text (%d chars)", len(text))
	startTime := time.Now()
	acc := ToolCallAccumulator{}
	clean := acc.feed(text)
	clean += acc.buffer.String()
	acc.buffer.Reset()
	logf("Parsed %d tool calls in %v", len(acc.toolCalls), time.Since(startTime))
	return clean, acc.toolCalls
}
