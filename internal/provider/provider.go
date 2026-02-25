package provider

// ModelInfo represents metadata for a model from a provider without a /v1/models endpoint
type ModelInfo struct {
	ID                  string
	Name                string
	Family              string
	ReleaseDate         string
	ContextLength       int
	MaxOutputLength     int
	InputModalities     []string
	OutputModalities    []string
	ConfidentialCompute bool
	Pricing             PricingInfo
	Features            ModelFeatures
}

// PricingInfo represents the pricing structure for a model
type PricingInfo struct {
	Prompt         float64
	Completion     float64
	InputCacheRead float64
}

// ModelFeatures represents the capabilities of a model
type ModelFeatures struct {
	Attachment  bool
	Reasoning   bool
	Temperature bool
	ToolCall    bool
}

// HardcodedProvider defines the interface for providers that don't have a /v1/models endpoint
type HardcodedProvider interface {
	// Name returns the provider identifier (e.g., "minimax.io")
	Name() string

	// Models returns all models provided by this provider
	Models() []ModelInfo

	// SupportsModel checks if this provider supports the given model ID
	SupportsModel(modelID string) bool

	// GetModel returns the model info for a specific model ID
	GetModel(modelID string) (*ModelInfo, bool)
}
