package provider

// MinimaxProvider implements the HardcodedProvider interface for minimax.io
type MinimaxProvider struct {
	models []ModelInfo
}

// NewMinimaxProvider creates a new minimax.io provider with hardcoded models
func NewMinimaxProvider() *MinimaxProvider {
	return &MinimaxProvider{
		models: []ModelInfo{
			{
				ID:                  "minimax.io/MiniMax-M2.5",
				Name:                "MiniMax-M2.5",
				Family:              "MiniMax-M2.5",
				ReleaseDate:         "2025-01",
				ContextLength:       196608,
				MaxOutputLength:     65536,
				InputModalities:     []string{"text"},
				OutputModalities:    []string{"text"},
				ConfidentialCompute: true,
				Pricing: PricingInfo{
					Prompt:         0.3,
					Completion:     1.1,
					InputCacheRead: 0.15,
				},
				Features: ModelFeatures{
					Attachment:  false,
					Reasoning:   true,
					Temperature: true,
					ToolCall:    true,
				},
			},
		},
	}
}

// Name returns the provider identifier
func (p *MinimaxProvider) Name() string {
	return "minimax.io"
}

// Models returns all models provided by minimax.io
func (p *MinimaxProvider) Models() []ModelInfo {
	return p.models
}

// SupportsModel checks if minimax.io supports the given model ID
func (p *MinimaxProvider) SupportsModel(modelID string) bool {
	for _, model := range p.models {
		if model.ID == modelID {
			return true
		}
	}
	return false
}

// GetModel returns the model info for a specific model ID
func (p *MinimaxProvider) GetModel(modelID string) (*ModelInfo, bool) {
	for i := range p.models {
		if p.models[i].ID == modelID {
			return &p.models[i], true
		}
	}
	return nil, false
}
