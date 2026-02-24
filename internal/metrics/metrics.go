package metrics

import (
	"encoding/json"
	"sync"
	"time"
)

type LatencyStats struct {
	Current    int64   `json:"current_ms"`
	RollingAvg float64 `json:"rolling_avg_ms"`
	Last10Avg  float64 `json:"last_10_avg_ms"`
	TotalCount int64   `json:"total_count"`
}

type TPSCounter struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	Requests     int64 `json:"requests"`
	LastUpdated  int64 `json:"last_updated_unix"`
}

type ModelMetrics struct {
	Model    string       `json:"model"`
	Provider string       `json:"provider"`
	Latency  LatencyStats `json:"latency"`
	TPS      TPSCounter   `json:"tps"`
}

type ProviderMetrics struct {
	Provider string       `json:"provider"`
	Latency  LatencyStats `json:"latency"`
	TPS      TPSCounter   `json:"tps"`
}

type Metrics struct {
	mu              sync.RWMutex
	globalLatency   *latencyTracker
	modelMetrics    map[string]*modelTracker
	providerMetrics map[string]*providerTracker
}

type latencyTracker struct {
	totalLatency int64
	totalCount   int64
	last10       []int64
	last10Index  int
}

type modelTracker struct {
	latency *latencyTracker
	tps     *tpsTracker
}

type providerTracker struct {
	latency *latencyTracker
	tps     *tpsTracker
}

type tpsTracker struct {
	inputTokens  int64
	outputTokens int64
	requests     int64
	lastUpdated  int64
}

func New() *Metrics {
	return &Metrics{
		globalLatency: &latencyTracker{
			last10: make([]int64, 0, 10),
		},
		modelMetrics:    make(map[string]*modelTracker),
		providerMetrics: make(map[string]*providerTracker),
	}
}

func (m *Metrics) RecordRequest(provider, model string, latencyMs int64, inputTokens, outputTokens int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update global latency
	m.globalLatency.totalLatency += latencyMs
	m.globalLatency.totalCount++

	// Update last 10 rolling window
	if len(m.globalLatency.last10) < 10 {
		m.globalLatency.last10 = append(m.globalLatency.last10, latencyMs)
	} else {
		m.globalLatency.last10[m.globalLatency.last10Index] = latencyMs
		m.globalLatency.last10Index = (m.globalLatency.last10Index + 1) % 10
	}

	// Update model metrics
	modelKey := provider + "/" + model
	if _, exists := m.modelMetrics[modelKey]; !exists {
		m.modelMetrics[modelKey] = &modelTracker{
			latency: &latencyTracker{last10: make([]int64, 0, 10)},
			tps:     &tpsTracker{},
		}
	}
	mt := m.modelMetrics[modelKey]
	mt.latency.totalLatency += latencyMs
	mt.latency.totalCount++
	if len(mt.latency.last10) < 10 {
		mt.latency.last10 = append(mt.latency.last10, latencyMs)
	} else {
		mt.latency.last10[mt.latency.last10Index] = latencyMs
		mt.latency.last10Index = (mt.latency.last10Index + 1) % 10
	}
	mt.tps.inputTokens += inputTokens
	mt.tps.outputTokens += outputTokens
	mt.tps.requests++
	mt.tps.lastUpdated = time.Now().Unix()

	// Update provider metrics
	if _, exists := m.providerMetrics[provider]; !exists {
		m.providerMetrics[provider] = &providerTracker{
			latency: &latencyTracker{last10: make([]int64, 0, 10)},
			tps:     &tpsTracker{},
		}
	}
	pt := m.providerMetrics[provider]
	pt.latency.totalLatency += latencyMs
	pt.latency.totalCount++
	if len(pt.latency.last10) < 10 {
		pt.latency.last10 = append(pt.latency.last10, latencyMs)
	} else {
		pt.latency.last10[pt.latency.last10Index] = latencyMs
		pt.latency.last10Index = (pt.latency.last10Index + 1) % 10
	}
	pt.tps.inputTokens += inputTokens
	pt.tps.outputTokens += outputTokens
	pt.tps.requests++
	pt.tps.lastUpdated = time.Now().Unix()
}

func (m *Metrics) GetGlobalLatency() LatencyStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.calculateLatencyStats(m.globalLatency)
}

func (m *Metrics) GetModelMetrics(modelKey string) (ModelMetrics, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	mt, exists := m.modelMetrics[modelKey]
	if !exists {
		return ModelMetrics{}, false
	}

	parts := splitModelKey(modelKey)
	return ModelMetrics{
		Model:    parts[1],
		Provider: parts[0],
		Latency:  m.calculateLatencyStats(mt.latency),
		TPS: TPSCounter{
			InputTokens:  mt.tps.inputTokens,
			OutputTokens: mt.tps.outputTokens,
			Requests:     mt.tps.requests,
			LastUpdated:  mt.tps.lastUpdated,
		},
	}, true
}

func (m *Metrics) GetAllModelMetrics() []ModelMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]ModelMetrics, 0, len(m.modelMetrics))
	for key, mt := range m.modelMetrics {
		parts := splitModelKey(key)
		result = append(result, ModelMetrics{
			Model:    parts[1],
			Provider: parts[0],
			Latency:  m.calculateLatencyStats(mt.latency),
			TPS: TPSCounter{
				InputTokens:  mt.tps.inputTokens,
				OutputTokens: mt.tps.outputTokens,
				Requests:     mt.tps.requests,
				LastUpdated:  mt.tps.lastUpdated,
			},
		})
	}
	return result
}

func (m *Metrics) GetProviderMetrics(provider string) (ProviderMetrics, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pt, exists := m.providerMetrics[provider]
	if !exists {
		return ProviderMetrics{}, false
	}

	return ProviderMetrics{
		Provider: provider,
		Latency:  m.calculateLatencyStats(pt.latency),
		TPS: TPSCounter{
			InputTokens:  pt.tps.inputTokens,
			OutputTokens: pt.tps.outputTokens,
			Requests:     pt.tps.requests,
			LastUpdated:  pt.tps.lastUpdated,
		},
	}, true
}

func (m *Metrics) GetAllProviderMetrics() []ProviderMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]ProviderMetrics, 0, len(m.providerMetrics))
	for provider, pt := range m.providerMetrics {
		result = append(result, ProviderMetrics{
			Provider: provider,
			Latency:  m.calculateLatencyStats(pt.latency),
			TPS: TPSCounter{
				InputTokens:  pt.tps.inputTokens,
				OutputTokens: pt.tps.outputTokens,
				Requests:     pt.tps.requests,
				LastUpdated:  pt.tps.lastUpdated,
			},
		})
	}
	return result
}

func (m *Metrics) calculateLatencyStats(lt *latencyTracker) LatencyStats {
	if lt.totalCount == 0 {
		return LatencyStats{}
	}

	rollingAvg := float64(lt.totalLatency) / float64(lt.totalCount)

	var last10Avg float64
	if len(lt.last10) > 0 {
		var sum int64
		for _, v := range lt.last10 {
			sum += v
		}
		last10Avg = float64(sum) / float64(len(lt.last10))
	}

	var current int64
	if len(lt.last10) > 0 {
		idx := (lt.last10Index - 1 + len(lt.last10)) % len(lt.last10)
		current = lt.last10[idx]
	}

	return LatencyStats{
		Current:    current,
		RollingAvg: rollingAvg,
		Last10Avg:  last10Avg,
		TotalCount: lt.totalCount,
	}
}

func (m *Metrics) GetJSON() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	data := map[string]interface{}{
		"global_latency": m.calculateLatencyStats(m.globalLatency),
		"models":         m.GetAllModelMetrics(),
		"providers":      m.GetAllProviderMetrics(),
	}

	return json.MarshalIndent(data, "", "  ")
}

func splitModelKey(key string) []string {
	parts := make([]string, 2)
	idx := 0
	for i, c := range key {
		if c == '/' && idx == 0 {
			parts[0] = key[:i]
			parts[1] = key[i+1:]
			idx = 1
			break
		}
	}
	if idx == 0 {
		parts[0] = key
		parts[1] = ""
	}
	return parts
}

func (m *Metrics) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.globalLatency = &latencyTracker{
		last10: make([]int64, 0, 10),
	}
	m.modelMetrics = make(map[string]*modelTracker)
	m.providerMetrics = make(map[string]*providerTracker)
}
