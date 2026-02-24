package config

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type Config struct {
	Port         int              `mapstructure:"port"`
	LogLevel     string           `mapstructure:"log_level"`
	Timeout      time.Duration    `mapstructure:"timeout"`
	Transformers map[string]any   `mapstructure:"transformers"`
	Upstreams    []UpstreamConfig `mapstructure:"upstreams"`
}

type UpstreamConfig struct {
	Name     string        `mapstructure:"name"`
	Provider string        `mapstructure:"provider"`
	Endpoint string        `mapstructure:"endpoint"`
	APIKey   string        `mapstructure:"api_key"`
	Models   []ModelConfig `mapstructure:"models"`
}

type ModelConfig struct {
	Pattern             string   `mapstructure:"pattern"`
	Transformers        []string `mapstructure:"transformers"`
	RequestTransformers []string `mapstructure:"request_transformers"`
}

func DefaultConfig() *Config {
	return &Config{
		Port:     8080,
		LogLevel: "info",
		Timeout:  10 * time.Minute,
		Transformers: map[string]any{
			"kimi-tool-calls": map[string]any{
				"enabled": true,
			},
		},
		Upstreams: nil,
	}
}

func LoadConfig(cmd *cobra.Command) (*Config, error) {
	v := viper.New()

	v.SetDefault("port", 8080)
	v.SetDefault("log_level", "info")
	v.SetDefault("timeout", "600s")

	if configFile, _ := cmd.Flags().GetString("config-file"); configFile != "" {
		v.SetConfigFile(configFile)
	} else {
		v.SetConfigName("ai-proxy-config")
		v.SetConfigType("json")

		searchPaths := []string{}

		if xdgData := os.Getenv("XDG_DATA_DIR"); xdgData != "" {
			searchPaths = append(searchPaths, filepath.Join(xdgData, "ai-proxy-config.json"))
		}

		searchPaths = append(searchPaths, "./ai-proxy-config.json")

		for _, p := range searchPaths {
			if _, err := os.Stat(p); err == nil {
				v.SetConfigFile(p)
				break
			}
		}

		if len(searchPaths) > 0 && v.ConfigFileUsed() == "" {
			v.AddConfigPath(filepath.Dir(searchPaths[0]))
		}
	}

	if err := v.ReadInConfig(); err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	if port, _ := cmd.Flags().GetInt("port"); port > 0 {
		v.Set("port", port)
	}

	if logLevel, _ := cmd.Flags().GetString("log-level"); logLevel != "" {
		v.Set("log_level", logLevel)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	if cfg.Timeout == 0 {
		cfg.Timeout = 10 * time.Minute
	}

	// Resolve environment variables for API keys
	for i := range cfg.Upstreams {
		cfg.Upstreams[i].APIKey = ResolveEnvVar(cfg.Upstreams[i].APIKey)
	}

	return &cfg, nil
}

func ResolveEnvVar(value string) string {
	if strings.HasPrefix(value, "ENV:") {
		envVar := strings.TrimPrefix(value, "ENV:")
		if envValue := os.Getenv(envVar); envValue != "" {
			return envValue
		}
	}
	return value
}
