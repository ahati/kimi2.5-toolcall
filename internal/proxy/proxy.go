package proxy

import (
	"ai-proxy/internal/config"
	"ai-proxy/internal/transformer"
	anthropic "ai-proxy/internal/transformer/request"
	anthropicResp "ai-proxy/internal/transformer/response/anthropic"
	kimi "ai-proxy/internal/transformer/response/kimi"
	"net/http"
	"strings"
)

type Proxy struct {
	config      *config.Config
	transformer *transformer.Registry
	httpClient  *http.Client
}

func New(cfg *config.Config) *Proxy {
	return &Proxy{
		config: cfg,
		transformer: func() *transformer.Registry {
			r := transformer.NewRegistry()
			r.RegisterRequest(anthropic.NewAnthropicToOpenAITransformer())
			r.RegisterResponse(kimi.NewKimiToolCallsTransformer())
			r.RegisterResponse(anthropicResp.NewOpenAIToAnthropicTransformer())
			return r
		}(),
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

func (p *Proxy) HTTPClient() *http.Client {
	return p.httpClient
}

type RouteResult struct {
	Upstream         *config.UpstreamConfig
	ModelConfig      *config.ModelConfig
	StrippedModel    string
	ReqTransformers  []transformer.RequestTransformer
	RespTransformers []transformer.ResponseTransformer
}

func (p *Proxy) Route(model string) (*RouteResult, error) {
	parts := strings.SplitN(model, "/", 2)
	if len(parts) < 2 {
		return nil, ErrNoProviderInModel
	}

	provider := parts[0]
	remainingModel := parts[1]

	var matchedUpstream *config.UpstreamConfig
	for i := range p.config.Upstreams {
		if p.config.Upstreams[i].Provider == provider {
			matchedUpstream = &p.config.Upstreams[i]
			break
		}
	}

	if matchedUpstream == nil {
		for i := range p.config.Upstreams {
			for _, mc := range p.config.Upstreams[i].Models {
				if matchModelPattern(remainingModel, mc.Pattern) {
					matchedUpstream = &p.config.Upstreams[i]
					break
				}
			}
			if matchedUpstream != nil {
				break
			}
		}
	}

	if matchedUpstream == nil {
		for i := range p.config.Upstreams {
			for _, mc := range p.config.Upstreams[i].Models {
				if mc.Pattern == "*" {
					matchedUpstream = &p.config.Upstreams[i]
					break
				}
			}
			if matchedUpstream != nil {
				break
			}
		}
	}

	if matchedUpstream == nil {
		return nil, ErrUpstreamNotFound
	}

	var matchedModelCfg *config.ModelConfig
	for i := range matchedUpstream.Models {
		mc := &matchedUpstream.Models[i]
		if matchModelPattern(remainingModel, mc.Pattern) {
			matchedModelCfg = mc
			break
		}
	}

	if matchedModelCfg == nil {
		for i := range matchedUpstream.Models {
			mc := &matchedUpstream.Models[i]
			if mc.Pattern == "*" {
				matchedModelCfg = mc
				break
			}
		}
	}

	if matchedModelCfg == nil {
		return nil, ErrModelNotMatched
	}

	reqT, respT := p.transformer.GetTransformersForModel(matchedModelCfg)

	return &RouteResult{
		Upstream:         matchedUpstream,
		ModelConfig:      matchedModelCfg,
		StrippedModel:    remainingModel,
		ReqTransformers:  reqT,
		RespTransformers: respT,
	}, nil
}

func matchModelPattern(model, pattern string) bool {
	if pattern == "*" {
		return true
	}

	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(model, prefix)
	}

	return model == pattern
}

type ProxyError struct {
	Code    string
	Message string
}

func (e *ProxyError) Error() string {
	return e.Message
}

var (
	ErrNoProviderInModel = &ProxyError{Code: "no_provider", Message: "Model name must contain a provider prefix (e.g., provider/model)"}
	ErrUpstreamNotFound  = &ProxyError{Code: "upstream_not_found", Message: "No upstream configured for this provider"}
	ErrModelNotMatched   = &ProxyError{Code: "model_not_matched", Message: "Model does not match any configured pattern"}
)
