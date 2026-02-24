package anthropic

import (
	"encoding/json"
)

type OpenAIToAnthropicTransformer struct{}

func NewOpenAIToAnthropicTransformer() *OpenAIToAnthropicTransformer {
	return &OpenAIToAnthropicTransformer{}
}

func (t *OpenAIToAnthropicTransformer) Reset() {
}

func (t *OpenAIToAnthropicTransformer) Name() string {
	return "openai-to-anthropic"
}

func (t *OpenAIToAnthropicTransformer) TransformResponse(body []byte, isStreaming bool) ([]byte, error) {
	if isStreaming {
		return t.transformStreaming(body)
	}
	return t.transformNonStreaming(body)
}

func (t *OpenAIToAnthropicTransformer) transformNonStreaming(body []byte) ([]byte, error) {
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

		anthropicMsg := convertToAnthropicMessage(msg)
		choice["message"] = anthropicMsg
		choices[i] = choice
	}

	resp["choices"] = choices
	return json.Marshal(resp)
}

func convertToAnthropicMessage(msg map[string]interface{}) map[string]interface{} {
	role, _ := msg["role"].(string)
	content, _ := msg["content"]

	var anthropicContent []interface{}

	if contentStr, ok := content.(string); ok && contentStr != "" {
		anthropicContent = []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": contentStr,
			},
		}
	}

	if toolCalls, ok := msg["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
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

			toolUse := map[string]interface{}{
				"type": "tool_use",
				"id":   tc["id"],
				"name": name,
				"input": func() map[string]interface{} {
					var argsMap map[string]interface{}
					if err := json.Unmarshal([]byte(args), &argsMap); err != nil {
						return map[string]interface{}{}
					}
					return argsMap
				}(),
			}

			anthropicContent = append(anthropicContent, toolUse)
		}
	}

	if len(anthropicContent) == 0 {
		anthropicContent = []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "",
			},
		}
	}

	result := map[string]interface{}{
		"role":    role,
		"content": anthropicContent,
	}

	if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
		result["reasoning"] = rc
	}

	return result
}

func (t *OpenAIToAnthropicTransformer) transformStreaming(body []byte) ([]byte, error) {
	return body, nil
}
