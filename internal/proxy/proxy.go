package proxy

import (
	"ai-proxy/internal/config"
	"ai-proxy/internal/logger"
	"ai-proxy/internal/transformer"
	anthropic "ai-proxy/internal/transformer/request"
	anthropicResp "ai-proxy/internal/transformer/response/anthropic"
	kimi "ai-proxy/internal/transformer/response/kimi"
	"net"
	"net/http"
	"strings"
	"time"
)

type Proxy struct {
	config      *config.Config
	transformer *transformer.Registry
	httpClient  *http.Client
}

func Logf(format string, args ...interface{}) {
	logger.Debugf("[PROXY] "+format, args...)
}

func New(cfg *config.Config) *Proxy {
	// Create transport with connection pooling for better performance
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

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
			Timeout:   cfg.Timeout,
			Transport: transport,
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
	Logf("Route called for model: %s", model)
	startTime := time.Now()

	parts := strings.SplitN(model, "/", 2)
	if len(parts) < 2 {
		Logf("Route failed: model format invalid (no provider prefix)")
		return nil, ErrNoProviderInModel
	}

	provider := parts[0]
	remainingModel := parts[1]

	Logf("Provider: %s, Model: %s", provider, remainingModel)

	var matchedUpstream *config.UpstreamConfig
	for i := range p.config.Upstreams {
		if p.config.Upstreams[i].Provider == provider {
			matchedUpstream = &p.config.Upstreams[i]
			Logf("Found upstream: %s", provider)
			break
		}
	}

	if matchedUpstream == nil {
		Logf("Upstream not found by provider, trying model pattern matching...")
		for i := range p.config.Upstreams {
			for _, mc := range p.config.Upstreams[i].Models {
				if matchModelPattern(remainingModel, mc.Pattern) {
					matchedUpstream = &p.config.Upstreams[i]
					Logf("Found upstream by pattern match: %s", p.config.Upstreams[i].Provider)
					break
				}
			}
			if matchedUpstream != nil {
				break
			}
		}
	}

	if matchedUpstream == nil {
		Logf("No upstream found for model %s", model)
		return nil, ErrUpstreamNotFound
	}

	var matchedModelCfg *config.ModelConfig
	for i := range matchedUpstream.Models {
		mc := &matchedUpstream.Models[i]
		if matchModelPattern(remainingModel, mc.Pattern) {
			matchedModelCfg = mc
			Logf("Found model config pattern: %s", mc.Pattern)
			break
		}
	}

	if matchedModelCfg == nil {
		Logf("Model config not found by pattern, trying wildcard...")
		for i := range matchedUpstream.Models {
			mc := &matchedUpstream.Models[i]
			if mc.Pattern == "*" {
				matchedModelCfg = mc
				Logf("Found wildcard model config")
				break
			}
		}
	}

	if matchedModelCfg == nil {
		Logf("Model config matching failed for %s", model)
		return nil, ErrModelNotMatched
	}

	reqT, respT := p.transformer.GetTransformersForModel(matchedModelCfg)
	Logf("Route completed in %v: %d request transformers, %d response transformers", time.Since(startTime), len(reqT), len(respT))

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
