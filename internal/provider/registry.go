package provider

// Registry maintains a collection of hardcoded providers
type Registry struct {
	providers map[string]HardcodedProvider
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]HardcodedProvider),
	}
}

// Register adds a hardcoded provider to the registry
func (r *Registry) Register(p HardcodedProvider) {
	r.providers[p.Name()] = p
}

// GetProvider retrieves a provider by name
func (r *Registry) GetProvider(name string) (HardcodedProvider, bool) {
	p, ok := r.providers[name]
	return p, ok
}

// GetAllModels returns all models from all registered providers
func (r *Registry) GetAllModels() []ModelInfo {
	var models []ModelInfo
	for _, p := range r.providers {
		models = append(models, p.Models()...)
	}
	return models
}

// GetAllProviders returns all registered providers
func (r *Registry) GetAllProviders() []HardcodedProvider {
	var providers []HardcodedProvider
	for _, p := range r.providers {
		providers = append(providers, p)
	}
	return providers
}
