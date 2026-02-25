package handler

import (
	"ai-proxy/internal/config"
	"ai-proxy/internal/logger"
	"ai-proxy/internal/metrics"
	"ai-proxy/internal/provider"
	"ai-proxy/internal/proxy"
	"ai-proxy/internal/transformer"
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

type Handler struct {
	proxy            *proxy.Proxy
	config           *config.Config
	metrics          *metrics.Metrics
	providerRegistry *provider.Registry
}

func New(p *proxy.Proxy, cfg *config.Config, m *metrics.Metrics, pr *provider.Registry) *Handler {
	return &Handler{
		proxy:            p,
		config:           cfg,
		metrics:          m,
		providerRegistry: pr,
	}
}

func logDebugf(requestID int64, component, format string, args ...interface{}) {
	logger.Debugf("[REQ-%d][%s] "+format, append([]interface{}{requestID, component}, args...)...)
}

func logInfof(requestID int64, component, format string, args ...interface{}) {
	logger.Infof("[REQ-%d][%s] "+format, append([]interface{}{requestID, component}, args...)...)
}

func logErrorf(requestID int64, component, format string, args ...interface{}) {
	logger.Errorf("[REQ-%d][%s] "+format, append([]interface{}{requestID, component}, args...)...)
}

func (h *Handler) ChatCompletions(c *gin.Context) {
	startTime := time.Now()
	requestID := time.Now().UnixNano()
	logInfof(requestID, "HTTP", "=== STARTING REQUEST ===")

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		logErrorf(requestID, "HTTP", "Failed to read request body: %v", err)
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "Failed to read request body"))
		return
	}
	logDebugf(requestID, "HTTP", "Read request body in %v", time.Since(startTime))

	defer c.Request.Body.Close()

	var req map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		logErrorf(requestID, "HTTP", "Failed to parse JSON: %v", err)
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "Invalid JSON"))
		return
	}
	logDebugf(requestID, "HTTP", "Parsed JSON in %v", time.Since(startTime))

	model, _ := req["model"].(string)
	if model == "" {
		logErrorf(requestID, "HTTP", "Missing model parameter")
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "model is required"))
		return
	}

	route, err := h.proxy.Route(model)
	if err != nil {
		if pe, ok := err.(*proxy.ProxyError); ok {
			logErrorf(requestID, "HTTP", "Route error: %v", err)
			c.JSON(http.StatusBadRequest, NewErrorResponse(pe.Code, pe.Message))
			return
		}
		logErrorf(requestID, "HTTP", "Route error: %v", err)
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", err.Error()))
		return
	}
	logDebugf(requestID, "HTTP", "Routed to %s in %v", route.Upstream.Provider, time.Since(startTime))

	for i, t := range route.ReqTransformers {
		t0 := time.Now()
		bodyBytes, err = t.TransformRequest(bodyBytes, route.StrippedModel)
		if err != nil {
			logErrorf(requestID, "HTTP", "Request transformer %d failed: %v", i, err)
			c.JSON(http.StatusInternalServerError, NewErrorResponse("transform_error", "Request transformation failed: "+err.Error()))
			return
		}
		logDebugf(requestID, "HTTP", "Request transformer %d (%s) took %v", i, t.Name(), time.Since(t0))
	}

	var transformedReq map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &transformedReq); err != nil {
		transformedReq = req
	}
	transformedReq["model"] = route.StrippedModel

	bodyBytes, err = json.Marshal(transformedReq)
	if err != nil {
		logErrorf(requestID, "HTTP", "Failed to serialize request: %v", err)
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to serialize request"))
		return
	}
	logDebugf(requestID, "HTTP", "Serialized request in %v", time.Since(startTime))

	inputTokens := h.countTokens(transformedReq)

	upstreamURL := route.Upstream.Endpoint + "/chat/completions"
	upstreamReq, err := http.NewRequest("POST", upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to create upstream request"))
		return
	}

	upstreamReq.Header = make(http.Header)
	for k, v := range c.Request.Header {
		if strings.ToLower(k) == "host" {
			continue
		}
		upstreamReq.Header[k] = v
	}
	upstreamReq.Header.Set("Authorization", "Bearer "+route.Upstream.APIKey)

	// Start timing
	reqStartTime := time.Now()

	isStreaming := false
	if s, ok := req["stream"].(bool); ok && s {
		isStreaming = true
	}

	hasStreamingTransformer := hasStreamingTransformer(route.RespTransformers)

	// For streaming with transformers, forward directly without buffering
	if isStreaming && hasStreamingTransformer {
		logDebugf(requestID, "HTTP", "Streaming request detected, using direct streaming mode")
		h.handleStreaming(c, upstreamURL, bodyBytes, route, reqStartTime, inputTokens)
		return
	}

	logDebugf(requestID, "HTTP", "Making upstream request to %s", upstreamURL)
	upstreamResp, err := h.proxy.HTTPClient().Do(upstreamReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, NewErrorResponse("upstream_error", "Upstream request failed: "+err.Error()))
		return
	}
	defer upstreamResp.Body.Close()

	logDebugf(requestID, "HTTP", "Reading response body...")
	respBody, _ := io.ReadAll(upstreamResp.Body)
	logDebugf(requestID, "HTTP", "Response body read (%d bytes) in %v", len(respBody), time.Since(reqStartTime))

	// Extract usage from original response before transformation
	usageInputTokens, usageOutputTokens := h.extractUsageFromResponse(respBody)

	// Calculate latency
	latencyMs := time.Since(reqStartTime).Milliseconds()
	logDebugf(requestID, "HTTP", "Upstream request completed in %vms", latencyMs)

	logDebugf(requestID, "HTTP", "Applying %d response transformers", len(route.RespTransformers))
	for i, t := range route.RespTransformers {
		t0 := time.Now()
		respBody, err = t.TransformResponse(respBody, isStreaming)
		if err != nil {
			logErrorf(requestID, "HTTP", "Transformer %d (%s) error: %v", i, t.Name(), err)
		} else {
			logDebugf(requestID, "HTTP", "Transformer %d (%s) completed in %v", i, t.Name(), time.Since(t0))
		}
	}

	// Use API usage data if available, otherwise fall back to estimation
	outputTokens := usageOutputTokens
	if outputTokens == 0 {
		t0 := time.Now()
		outputTokens = h.countResponseTokens(respBody)
		logDebugf(requestID, "HTTP", "Token counting took %v", time.Since(t0))
	}

	// Use API usage data for input tokens if available
	if usageInputTokens > 0 {
		inputTokens = usageInputTokens
	}

	// Record metrics
	h.metrics.RecordRequest(route.Upstream.Provider, route.StrippedModel, latencyMs, inputTokens, outputTokens)

	// Send response
	for k, v := range upstreamResp.Header {
		if strings.ToLower(k) == "transfer-encoding" {
			continue
		}
		c.Header(k, v[0])
	}

	c.Data(upstreamResp.StatusCode, upstreamResp.Header.Get("Content-Type"), respBody)
	logInfof(requestID, "HTTP", "=== REQUEST COMPLETE (total: %v) ===", time.Since(reqStartTime))
}

func (h *Handler) AnthropicMessages(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "Failed to read request body"))
		return
	}
	defer c.Request.Body.Close()

	var req map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "Invalid JSON"))
		return
	}

	model, _ := req["model"].(string)
	if model == "" {
		c.JSON(http.StatusBadRequest, NewErrorResponse("invalid_request_error", "model is required"))
		return
	}

	fullModel := "anthropic/" + model

	route, err := h.proxy.Route(fullModel)
	if err != nil {
		if pe, ok := err.(*proxy.ProxyError); ok {
			c.JSON(http.StatusBadRequest, NewErrorResponse(pe.Code, pe.Message))
			return
		}
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", err.Error()))
		return
	}

	for _, t := range route.ReqTransformers {
		bodyBytes, err = t.TransformRequest(bodyBytes, route.StrippedModel)
		if err != nil {
			c.JSON(http.StatusInternalServerError, NewErrorResponse("transform_error", "Request transformation failed: "+err.Error()))
			return
		}
	}

	req["model"] = route.StrippedModel

	bodyBytes, err = json.Marshal(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to serialize request"))
		return
	}

	upstreamURL := route.Upstream.Endpoint + "/chat/completions"
	upstreamReq, err := http.NewRequest("POST", upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to create upstream request"))
		return
	}

	upstreamReq.Header = make(http.Header)
	for k, v := range c.Request.Header {
		if strings.ToLower(k) == "host" {
			continue
		}
		upstreamReq.Header[k] = v
	}
	upstreamReq.Header.Set("Authorization", "Bearer "+route.Upstream.APIKey)

	upstreamResp, err := h.proxy.HTTPClient().Do(upstreamReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, NewErrorResponse("upstream_error", "Upstream request failed: "+err.Error()))
		return
	}
	defer upstreamResp.Body.Close()

	respBody, _ := io.ReadAll(upstreamResp.Body)

	isStreaming := false
	if s, ok := req["stream"].(bool); ok && s {
		isStreaming = true
	}

	for _, t := range route.RespTransformers {
		respBody, err = t.TransformResponse(respBody, isStreaming)
		if err != nil {
		}
	}

	for k, v := range upstreamResp.Header {
		if strings.ToLower(k) == "transfer-encoding" {
			continue
		}
		c.Header(k, v[0])
	}

	c.Data(upstreamResp.StatusCode, upstreamResp.Header.Get("Content-Type"), respBody)
}

func (h *Handler) ListModels(c *gin.Context) {
	models := []map[string]interface{}{}

	// Add hardcoded models from providers without /v1/models endpoint
	if h.providerRegistry != nil {
		for _, model := range h.providerRegistry.GetAllModels() {
			models = append(models, map[string]interface{}{
				"id":                   model.ID,
				"object":               "model",
				"created":              1700000000,
				"owned_by":             model.ID[:strings.Index(model.ID, "/")],
				"context_length":       model.ContextLength,
				"max_output_length":    model.MaxOutputLength,
				"input_modalities":     model.InputModalities,
				"output_modalities":    model.OutputModalities,
				"confidential_compute": model.ConfidentialCompute,
				"pricing": map[string]interface{}{
					"prompt":           model.Pricing.Prompt,
					"completion":       model.Pricing.Completion,
					"input_cache_read": model.Pricing.InputCacheRead,
				},
				"supported_features": func() []string {
					features := []string{}
					if model.Features.Reasoning {
						features = append(features, "reasoning")
					}
					if model.Features.Temperature {
						features = append(features, "temperature")
					}
					if model.Features.ToolCall {
						features = append(features, "tools")
					}
					return features
				}(),
			})
		}
	}

	for _, upstream := range h.config.Upstreams {
		upstreamURL := upstream.Endpoint + "/models"
		req, err := http.NewRequest("GET", upstreamURL, nil)
		if err != nil {
			continue
		}
		req.Header.Set("Authorization", "Bearer "+upstream.APIKey)

		resp, err := h.proxy.HTTPClient().Do(req)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			for _, mc := range upstream.Models {
				pattern := mc.Pattern
				if pattern == "*" {
					models = append(models, map[string]interface{}{
						"id":       upstream.Provider + "/*",
						"object":   "model",
						"created":  1700000000,
						"owned_by": upstream.Provider,
					})
				} else {
					models = append(models, map[string]interface{}{
						"id":       upstream.Provider + "/" + pattern,
						"object":   "model",
						"created":  1700000000,
						"owned_by": upstream.Provider,
					})
				}
			}
			continue
		}

		var upstreamResp map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&upstreamResp); err != nil {
			continue
		}

		if data, ok := upstreamResp["data"].([]interface{}); ok {
			for _, m := range data {
				if modelMap, ok := m.(map[string]interface{}); ok {
					if id, ok := modelMap["id"].(string); ok {
						modelMap["id"] = upstream.Provider + "/" + id
						models = append(models, modelMap)
					}
				}
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   models,
	})
}

func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

func NewErrorResponse(errType, message string) ErrorResponse {
	return ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errType,
		},
	}
}

func (h *Handler) countTokens(req map[string]interface{}) int64 {
	var count int64

	if messages, ok := req["messages"].([]interface{}); ok {
		for _, msg := range messages {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				count += h.countMessageTokens(msgMap)
			}
		}
	}

	return count
}

func (h *Handler) countMessageTokens(msgMap map[string]interface{}) int64 {
	var count int64

	if content := msgMap["content"]; content != nil {
		count += h.countContentTokens(content)
	}

	if toolCalls, ok := msgMap["tool_calls"].([]interface{}); ok {
		for _, tc := range toolCalls {
			if tcMap, ok := tc.(map[string]interface{}); ok {
				if fn, ok := tcMap["function"].(map[string]interface{}); ok {
					if name, ok := fn["name"].(string); ok {
						count += int64(len(name)) / 4
					}
					if args, ok := fn["arguments"].(string); ok {
						count += int64(len(args)) / 4
					}
				}
			}
		}
	}

	return count
}

func (h *Handler) countContentTokens(content interface{}) int64 {
	switch c := content.(type) {
	case string:
		return int64(len(c)) / 4
	case []interface{}:
		var count int64
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockType, ok := blockMap["type"].(string); ok {
					switch blockType {
					case "text":
						if text, ok := blockMap["text"].(string); ok {
							count += int64(len(text)) / 4
						}
					case "thinking":
						if thinking, ok := blockMap["thinking"].(string); ok {
							count += int64(len(thinking)) / 4
						}
					case "image_url":
						if imgURL, ok := blockMap["image_url"].(map[string]interface{}); ok {
							if url, ok := imgURL["url"].(string); ok {
								count += int64(len(url)) / 4
							}
						}
					}
				}
			}
		}
		return count
	default:
		return 0
	}
}

func (h *Handler) countResponseTokens(respBody []byte) int64 {

	var count int64

	var resp map[string]interface{}
	if err := json.Unmarshal(respBody, &resp); err == nil {
		if choices, ok := resp["choices"].([]interface{}); ok {
			for _, choice := range choices {
				if choiceMap, ok := choice.(map[string]interface{}); ok {
					if message, ok := choiceMap["message"].(map[string]interface{}); ok {
						if content, ok := message["content"].(string); ok {
							count += int64(len(content)) / 4
						}
					}
				}
			}
		}
		if usage, ok := resp["usage"].(map[string]interface{}); ok {
			if completionTokens, ok := usage["completion_tokens"].(float64); ok {
				count = int64(completionTokens)
			}
		}
	}

	return count
}

func (h *Handler) extractUsageFromResponse(respBody []byte) (inputTokens, outputTokens int64) {
	var resp map[string]interface{}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return 0, 0
	}

	usage, ok := resp["usage"].(map[string]interface{})
	if !ok {
		return 0, 0
	}

	if promptTokens, ok := usage["prompt_tokens"].(float64); ok {
		inputTokens = int64(promptTokens)
	} else if totalTokens, ok := usage["total_tokens"].(float64); ok {
		if completionTokens, ok := usage["completion_tokens"].(float64); ok {
			inputTokens = int64(totalTokens) - int64(completionTokens)
		}
	}

	if completionTokens, ok := usage["completion_tokens"].(float64); ok {
		outputTokens = int64(completionTokens)
	} else if totalTokens, ok := usage["total_tokens"].(float64); ok {
		if promptTokens, ok := usage["prompt_tokens"].(float64); ok {
			outputTokens = int64(totalTokens) - int64(promptTokens)
		}
	}

	return inputTokens, outputTokens
}

func (h *Handler) MetricsHandler(c *gin.Context) {
	data, err := h.metrics.GetJSON()
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to get metrics"))
		return
	}
	c.Data(http.StatusOK, "application/json", data)
}

func hasStreamingTransformer(transformers []transformer.ResponseTransformer) bool {
	for _, t := range transformers {
		if _, ok := t.(transformer.StreamingTransformer); ok {
			return true
		}
	}
	return false
}

func getStreamingTransformers(transformers []transformer.ResponseTransformer) []transformer.StreamingTransformer {
	var result []transformer.StreamingTransformer
	for _, t := range transformers {
		if st, ok := t.(transformer.StreamingTransformer); ok {
			result = append(result, st)
		}
	}
	return result
}

func (h *Handler) handleStreaming(c *gin.Context, upstreamURL string, bodyBytes []byte, route *proxy.RouteResult, startTime time.Time, inputTokens int64) {
	logInfof(0, "HTTP", "=== STARTING STREAMING REQUEST ===")
	logDebugf(0, "HTTP", "Creating upstream request to %s", upstreamURL)
	upstreamReq, err := http.NewRequest("POST", upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		logErrorf(0, "HTTP", "Failed to create upstream request: %v", err)
		c.JSON(http.StatusInternalServerError, NewErrorResponse("internal_error", "Failed to create upstream request"))
		return
	}

	upstreamReq.Header = make(http.Header)
	for k, v := range c.Request.Header {
		if strings.ToLower(k) == "host" {
			continue
		}
		upstreamReq.Header[k] = v
	}
	upstreamReq.Header.Set("Authorization", "Bearer "+route.Upstream.APIKey)
	upstreamReq.Header.Set("Accept", "text/event-stream")
	upstreamReq.Header.Set("Cache-Control", "no-cache")
	upstreamReq.Header.Set("Connection", "keep-alive")

	logDebugf(0, "HTTP", "Sending upstream request...")
	reqStart := time.Now()
	upstreamResp, err := h.proxy.HTTPClient().Do(upstreamReq)
	logDebugf(0, "HTTP", "Upstream request completed in %v", time.Since(reqStart))
	if err != nil {
		logErrorf(0, "HTTP", "Upstream request failed: %v", err)
		c.JSON(http.StatusBadGateway, NewErrorResponse("upstream_error", "Upstream request failed: "+err.Error()))
		return
	}
	defer upstreamResp.Body.Close()

	logDebugf(0, "HTTP", "Starting to read streaming response...")

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Status(http.StatusOK)

	streamingTransformers := getStreamingTransformers(route.RespTransformers)
	logDebugf(0, "HTTP", "Found %d streaming transformers", len(streamingTransformers))

	scanner := bufio.NewReaderSize(upstreamResp.Body, 64*1024) // 64KB buffer for better throughput
	chunkCount := 0
	for {
		chunkStart := time.Now()
		line, err := scanner.ReadString('\n')
		readTime := time.Since(chunkStart)

		if readTime > 100*time.Millisecond {
			logDebugf(0, "HTTP", "Slow read: %v for line %d", readTime, chunkCount)
		}

		if err != nil {
			logDebugf(0, "HTTP", "Stream ended after %d chunks", chunkCount)
			break
		}

		chunkCount++
		line = strings.TrimSpace(line)

		if line == "" {
			continue
		}

		if strings.HasPrefix(line, ":") {
			continue
		}

		if line == "data: [DONE]" {
			logDebugf(0, "HTTP", "Received [DONE], processing final chunks")
			for _, st := range streamingTransformers {
				_, newChunk, _ := st.TransformStream([]byte(line))
				if len(newChunk) > 0 {
					c.Writer.Write(newChunk)
				}
			}
			c.Writer.Write([]byte("data: [DONE]\n\n"))
			c.Writer.Flush()
			logInfof(0, "HTTP", "=== STREAMING REQUEST COMPLETE (total chunks: %d) ===", chunkCount)
			break
		}

		if !strings.HasPrefix(line, "data: ") {
			c.Writer.Write([]byte(line + "\n"))
			c.Writer.Flush()
			continue
		}

		jsonStr := line[len("data: "):]
		chunkSize := len(jsonStr)

		for _, st := range streamingTransformers {
			t0 := time.Now()
			modified, newChunk, keepChunk := st.TransformStream([]byte(jsonStr))
			transformTime := time.Since(t0)

			if transformTime > 50*time.Millisecond {
				logger.Warnf("[REQ-%d][%s] Slow transform (%s): %v for chunk %d (size: %d)", 0, "HTTP", st.Name(), transformTime, chunkCount, chunkSize)
			}

			if modified && len(newChunk) > 0 {
				c.Writer.Write(newChunk)
				c.Writer.Flush()
				if !keepChunk {
					continue
				}
			}
			if !modified || keepChunk {
				c.Writer.Write([]byte(line + "\n\n"))
				c.Writer.Flush()
			}
		}
	}

	// Drain any remaining bytes from backend stream to prevent backend from seeing connection reset
	// This ensures the backend doesn't see a connection reset/cancellation
	logDebugf(0, "HTTP", "Draining remaining stream bytes...")
	io.Copy(io.Discard, upstreamResp.Body)

	outputTokens := int64(0)
	for _, st := range streamingTransformers {
		if tracker, ok := st.(transformer.TokenUsageTracker); ok {
			_, outTokens := tracker.GetTokenUsage()
			if outTokens > 0 {
				outputTokens = outTokens
			}
		}
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logDebugf(0, "HTTP", "Recording metrics: latency=%vms, inputTokens=%d, outputTokens=%d", latencyMs, inputTokens, outputTokens)
	h.metrics.RecordRequest(route.Upstream.Provider, route.StrippedModel, latencyMs, inputTokens, outputTokens)
	logger.Infof("[REQ-%d][%s] === STREAMING REQUEST COMPLETE ===", 0, "HTTP")
}
