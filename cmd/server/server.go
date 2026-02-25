package server

import (
	ai_proxy_config "ai-proxy/internal/config"
	"ai-proxy/internal/handler"
	"ai-proxy/internal/logger"
	"ai-proxy/internal/metrics"
	"ai-proxy/internal/provider"
	"ai-proxy/internal/proxy"
	"fmt"

	"github.com/gin-gonic/gin"
	"github.com/spf13/cobra"
)

var ServeCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the AI proxy server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := ai_proxy_config.LoadConfig(cmd)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		// Update logger level from config
		logger.SetLevel(cfg.LogLevel)

		m := metrics.New()
		p := proxy.New(cfg)

		// Initialize provider registry and register hardcoded providers
		providerRegistry := provider.NewRegistry()
		providerRegistry.Register(provider.NewMinimaxProvider())

		h := handler.New(p, cfg, m, providerRegistry)

		r := gin.Default()

		r.POST("/v1/chat/completions", h.ChatCompletions)
		r.POST("/v1/anthropic/messages", h.AnthropicMessages)
		r.GET("/v1/models", h.ListModels)
		r.GET("/health", h.Health)
		r.GET("/metrics", h.MetricsHandler)

		addr := fmt.Sprintf(":%d", cfg.Port)
		logger.Infof("Starting AI Proxy on %s", addr)
		return r.Run(addr)
	},
}

func GetCommand() *cobra.Command {
	return ServeCmd
}
