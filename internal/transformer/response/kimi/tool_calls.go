package kimi

import (
	"encoding/json"
	"fmt"
	"log"
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
	buffer         string
	toolCalls      []ParsedToolCall
	inSection      bool
	finished       bool
	inCall         bool
	inArgs         bool
	currentIDBuf   string
	currentArgsBuf string
}

func (a *ToolCallAccumulator) feed(text string) string {
	a.buffer += text
	return a.consume()
}

func (a *ToolCallAccumulator) consume() string {
	clean := ""

	for len(a.buffer) > 0 {
		if isPartialTokenPrefix(a.buffer) {
			break
		}

		matched := false

		if strings.HasPrefix(a.buffer, TOK_SECTION_BEGIN) {
			a.inSection = true
			a.buffer = a.buffer[len(TOK_SECTION_BEGIN):]
			log.Println("[kimi-tool-calls] Tool call section STARTED — intercepting raw tool tokens")
			matched = true
		} else if strings.HasPrefix(a.buffer, TOK_SECTION_END) {
			a.inSection = false
			a.finished = true
			a.buffer = a.buffer[len(TOK_SECTION_END):]
			log.Printf("[kimi-tool-calls] Tool call section ENDED — %d tool call(s) parsed", len(a.toolCalls))
			matched = true
		} else if strings.HasPrefix(a.buffer, TOK_CALL_BEGIN) {
			a.inCall = true
			a.inArgs = false
			a.currentIDBuf = ""
			a.currentArgsBuf = ""
			a.buffer = a.buffer[len(TOK_CALL_BEGIN):]
			log.Println("[kimi-tool-calls] Parsing new tool call …")
			matched = true
		} else if strings.HasPrefix(a.buffer, TOK_ARG_BEGIN) {
			a.inArgs = true
			a.buffer = a.buffer[len(TOK_ARG_BEGIN):]
			matched = true
		} else if strings.HasPrefix(a.buffer, TOK_CALL_END) {
			a.finalizeCall()
			a.buffer = a.buffer[len(TOK_CALL_END):]
			matched = true
		}

		if !matched {
			ch := a.buffer[0]
			a.buffer = a.buffer[1:]
			if a.inCall {
				if a.inArgs {
					a.currentArgsBuf += string(ch)
				} else {
					a.currentIDBuf += string(ch)
				}
			} else if a.inSection {
				// skip whitespace/noise between calls in section
			} else {
				clean += string(ch)
			}
		}
	}

	return clean
}

func (a *ToolCallAccumulator) finalizeCall() {
	rawID := strings.TrimSpace(a.currentIDBuf)
	rawArgs := strings.TrimSpace(a.currentArgsBuf)

	funcName := rawID
	if strings.HasPrefix(funcName, "functions.") {
		funcName = strings.TrimPrefix(funcName, "functions.")
	}

	if idx := strings.Index(funcName, ":"); idx != -1 {
		funcName = funcName[:idx]
	}

	if funcName == "" {
		log.Printf("[kimi-tool-calls] Parsed tool call with EMPTY function name (raw_id=%q). This tool call will likely fail downstream.", rawID)
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
		log.Printf("[kimi-tool-calls] Could not parse tool call arguments as valid JSON for '%s' (call_id=%s): %s", funcName, callID, argsPreview)
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
	log.Printf("[kimi-tool-calls] Parsed tool call: %s (call_id=%s) args=%s", funcName, callID, argsPreview)

	a.inCall = false
	a.inArgs = false
	a.currentIDBuf = ""
	a.currentArgsBuf = ""
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
	return strings.Contains(text, TOK_SECTION_BEGIN) || strings.Contains(text, TOK_CALL_BEGIN)
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
}

func (t *KimiToolCallsTransformer) Name() string {
	return "kimi-tool-calls"
}

func (t *KimiToolCallsTransformer) TransformResponse(body []byte, isStreaming bool) ([]byte, error) {
	if isStreaming {
		return t.transformStreaming(body)
	}
	return t.transformNonStreaming(body)
}

func (t *KimiToolCallsTransformer) transformNonStreaming(body []byte) ([]byte, error) {
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

		allToolCalls := []interface{}{}

		if rc, ok := msg["reasoning_content"].(string); ok && containsToolTokens(rc) {
			cleanRC, tc := parseToolCallsFromText(rc)
			cleanRC = strings.TrimRight(cleanRC, " \n")
			if cleanRC != "" {
				msg["reasoning_content"] = cleanRC
			} else {
				delete(msg, "reasoning_content")
			}
			for _, ptc := range tc {
				allToolCalls = append(allToolCalls, toolCallToJSON(ptc))
			}
		}

		if content, ok := msg["content"].(string); ok && containsToolTokens(content) {
			cleanC, tc := parseToolCallsFromText(content)
			cleanC = strings.TrimRight(cleanC, " \n")
			if cleanC != "" {
				msg["content"] = cleanC
			} else {
				delete(msg, "content")
			}
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
			log.Printf("[kimi-tool-calls] Non-streaming fix: converted %d raw tool call token(s) → native tool_calls [%s]",
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
			log.Printf("[kimi-tool-calls] Stream [DONE]: flushing %d remaining tool call(s)", len(t.accumulator.toolCalls))
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
			log.Println("[kimi-tool-calls] Stream complete — tool call tokens were intercepted and converted")
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
		log.Printf("[kimi-tool-calls] Emitting %d parsed tool call(s) into stream", len(t.accumulator.toolCalls))

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
		t.accumulator.buffer = ""
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
	acc := ToolCallAccumulator{}
	clean := acc.feed(text)
	clean += acc.buffer
	acc.buffer = ""
	return clean, acc.toolCalls
}
