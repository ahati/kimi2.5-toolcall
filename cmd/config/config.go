package config

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"ai-proxy/internal/provider"

	"github.com/spf13/cobra"
)

type OpenAIFetcher struct {
	BaseURL string
	Client  *http.Client
}

type ModelListResponse struct {
	Data []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID                  string   `json:"id"`
	ContextLength       int      `json:"context_length"`
	MaxOutputLength     int      `json:"max_output_length"`
	InputModalities     []string `json:"input_modalities"`
	OutputModalities    []string `json:"output_modalities"`
	ConfidentialCompute bool     `json:"confidential_compute"`
	Pricing             Pricing  `json:"pricing"`
	SupportedFeatures   []string `json:"supported_features"`
}

type Pricing struct {
	Prompt         float64 `json:"prompt"`
	Completion     float64 `json:"completion"`
	InputCacheRead float64 `json:"input_cache_read"`
}

type OpenCodeProviderConfig struct {
	API     string                 `json:"api"`
	Options map[string]interface{} `json:"options"`
	Models  map[string]interface{} `json:"models"`
}

type OpenCodeModelConfig struct {
	ID          string             `json:"id"`
	Name        string             `json:"name"`
	Family      string             `json:"family"`
	ReleaseDate string             `json:"release_date"`
	Attachment  bool               `json:"attachment"`
	Reasoning   bool               `json:"reasoning"`
	Temperature bool               `json:"temperature"`
	ToolCall    bool               `json:"tool_call"`
	Interleaved *map[string]string `json:"interleaved,omitempty"`
	Limit       LimitConfig        `json:"limit"`
	Cost        CostConfig         `json:"cost"`
	Modalities  ModalitiesConfig   `json:"modalities"`
}

type LimitConfig struct {
	Context int `json:"context"`
	Input   int `json:"input"`
	Output  int `json:"output"`
}

type CostConfig struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
}

type ModalitiesConfig struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

type OpenCodeConfig struct {
	Schema     string                            `json:"$schema"`
	Provider   map[string]OpenCodeProviderConfig `json:"provider"`
	Model      string                            `json:"model"`
	SmallModel string                            `json:"small_model"`
}

var ConfigCmd = &cobra.Command{
	Use:   "config",
	Short: "Generate opencode configuration",
	Long:  `Fetches models from the AI proxy and generates opencode configuration file.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		outputPath, _ := cmd.Flags().GetString("output")
		if outputPath == "" {
			homeDir, err := os.UserHomeDir()
			if err != nil {
				return fmt.Errorf("failed to get home directory: %w", err)
			}
			outputPath = filepath.Join(homeDir, ".config", "opencode", "opencode.json")
		}

		proxyURL, _ := cmd.Flags().GetString("proxy-url")
		if proxyURL == "" {
			return fmt.Errorf("proxy-url is required")
		}

		providerName, _ := cmd.Flags().GetString("provider-name")
		if providerName == "" {
			providerName = "chutes-tee"
		}

		defaultModel, _ := cmd.Flags().GetString("default-model")

		fetcher := &OpenAIFetcher{
			BaseURL: proxyURL,
			Client:  &http.Client{Timeout: 30 * 1000000000},
		}

		models, err := fetcher.FetchModels()
		if err != nil {
			return fmt.Errorf("failed to fetch models: %w", err)
		}

		// Inject hardcoded models from providers without /v1/models endpoint
		providerRegistry := provider.NewRegistry()
		providerRegistry.Register(provider.NewMinimaxProvider())
		for _, hardcodedModel := range providerRegistry.GetAllModels() {
			models = append(models, ModelInfo{
				ID:                  hardcodedModel.ID,
				ContextLength:       hardcodedModel.ContextLength,
				MaxOutputLength:     hardcodedModel.MaxOutputLength,
				InputModalities:     hardcodedModel.InputModalities,
				OutputModalities:    hardcodedModel.OutputModalities,
				ConfidentialCompute: hardcodedModel.ConfidentialCompute,
				Pricing: Pricing{
					Prompt:         hardcodedModel.Pricing.Prompt,
					Completion:     hardcodedModel.Pricing.Completion,
					InputCacheRead: hardcodedModel.Pricing.InputCacheRead,
				},
				SupportedFeatures: func() []string {
					features := []string{}
					if hardcodedModel.Features.Reasoning {
						features = append(features, "reasoning")
					}
					if hardcodedModel.Features.Temperature {
						features = append(features, "temperature")
					}
					if hardcodedModel.Features.ToolCall {
						features = append(features, "tools")
					}
					return features
				}(),
			})
		}

		teeModels := make(map[string]interface{})
		var firstTEEModel string

		for _, model := range models {
			if !model.ConfidentialCompute {
				continue
			}

			modelID := model.ID
			if firstTEEModel == "" {
				firstTEEModel = modelID
			}

			// Parse family from ID
			family := "unknown"
			parts := strings.Split(modelID, "/")
			if len(parts) >= 2 {
				family = parts[len(parts)-1]
				familyParts := strings.Split(family, "-TEE")
				family = familyParts[0]
			}

			// Check capabilities
			hasAttachment := contains(model.InputModalities, "image") || contains(model.InputModalities, "video")
			hasReasoning := contains(model.SupportedFeatures, "reasoning")
			hasToolCall := contains(model.SupportedFeatures, "tools")
			hasTemperature := true

			var interleaved *map[string]string
			if hasReasoning && (hasAttachment || strings.Contains(modelID, "Kimi-K2")) {
				interleaved = &map[string]string{
					"field": "reasoning_content",
				}
			}

			modelConfig := OpenCodeModelConfig{
				ID:          modelID,
				Name:        family + " TEE",
				Family:      family,
				ReleaseDate: "2025-01",
				Attachment:  hasAttachment,
				Reasoning:   hasReasoning,
				Temperature: hasTemperature,
				ToolCall:    hasToolCall,
				Interleaved: interleaved,
				Limit: LimitConfig{
					Context: model.ContextLength,
					Input:   model.ContextLength,
					Output:  model.MaxOutputLength,
				},
				Cost: CostConfig{
					Input:      model.Pricing.Prompt,
					Output:     model.Pricing.Completion,
					CacheRead:  model.Pricing.InputCacheRead,
					CacheWrite: model.Pricing.Prompt,
				},
				Modalities: ModalitiesConfig{
					Input:  model.InputModalities,
					Output: model.OutputModalities,
				},
			}

			teeModels[modelID] = modelConfig
		}

		if len(teeModels) == 0 {
			return fmt.Errorf("no TEE models found")
		}

		// Set defaults
		if defaultModel == "" {
			defaultModel = firstTEEModel
		}
		smallModel := defaultModel

		config := OpenCodeConfig{
			Schema: "https://opencode.ai/config.json",
			Provider: map[string]OpenCodeProviderConfig{
				providerName: {
					API: "openai-compatible",
					Options: map[string]interface{}{
						"baseURL": proxyURL + "/v1",
						"timeout": 600000,
					},
					Models: teeModels,
				},
			},
			Model:      providerName + "/" + defaultModel,
			SmallModel: providerName + "/" + smallModel,
		}

		outputDir := filepath.Dir(outputPath)
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			return fmt.Errorf("failed to create output directory: %w", err)
		}

		data, err := json.MarshalIndent(config, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		if err := os.WriteFile(outputPath, data, 0644); err != nil {
			return fmt.Errorf("failed to write config file: %w", err)
		}

		fmt.Printf("Generated opencode config at: %s\n", outputPath)
		fmt.Printf("Provider: %s\n", providerName)
		fmt.Printf("Proxy URL: %s\n", proxyURL)
		fmt.Printf("Default Model: %s\n", defaultModel)
		fmt.Printf("Small Model: %s\n", smallModel)
		fmt.Printf("Total TEE Models: %d\n", len(teeModels))

		return nil
	},
}

func (f *OpenAIFetcher) FetchModels() ([]ModelInfo, error) {
	resp, err := f.Client.Get(f.BaseURL + "/v1/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var modelResp ModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelResp); err != nil {
		return nil, err
	}

	return modelResp.Data, nil
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func init() {
	ConfigCmd.Flags().StringP("output", "o", "", "Output path for config file (default: ~/.config/opencode/opencode.json)")
	ConfigCmd.Flags().String("proxy-url", "http://localhost:8080", "URL of the AI proxy")
	ConfigCmd.Flags().String("provider-name", "local-proxy", "Name for the provider in config")
	ConfigCmd.Flags().String("default-model", "", "Default model to use (auto-selected if not specified)")
}

func GetCommand() *cobra.Command {
	return ConfigCmd
}
