package transformer

import "ai-proxy/internal/config"

type RequestTransformer interface {
	Name() string
	TransformRequest(body []byte, model string) ([]byte, error)
}

type ResponseTransformer interface {
	Name() string
	TransformResponse(body []byte, isStreaming bool) ([]byte, error)
}

type StreamingTransformer interface {
	ResponseTransformer
	TransformStream(chunk []byte) (modified bool, newChunk []byte, keepChunk bool)
}

type TokenUsageTracker interface {
	GetTokenUsage() (inputTokens, outputTokens int64)
}

type Registry struct {
	requestTransformers  map[string]RequestTransformer
	responseTransformers map[string]ResponseTransformer
}

func NewRegistry() *Registry {
	return &Registry{
		requestTransformers:  make(map[string]RequestTransformer),
		responseTransformers: make(map[string]ResponseTransformer),
	}
}

func (r *Registry) RegisterRequest(t RequestTransformer) {
	r.requestTransformers[t.Name()] = t
}

func (r *Registry) RegisterResponse(t ResponseTransformer) {
	r.responseTransformers[t.Name()] = t
}

func (r *Registry) GetRequestTransformer(name string) RequestTransformer {
	return r.requestTransformers[name]
}

func (r *Registry) GetResponseTransformer(name string) ResponseTransformer {
	return r.responseTransformers[name]
}

func (r *Registry) GetTransformersForModel(modelCfg *config.ModelConfig) ([]RequestTransformer, []ResponseTransformer) {
	var reqTransformers []RequestTransformer
	var respTransformers []ResponseTransformer

	for _, name := range modelCfg.RequestTransformers {
		if t := r.GetRequestTransformer(name); t != nil {
			reqTransformers = append(reqTransformers, t)
		}
	}

	for _, name := range modelCfg.Transformers {
		if t := r.GetResponseTransformer(name); t != nil {
			respTransformers = append(respTransformers, t)
		}
	}

	return reqTransformers, respTransformers
}
