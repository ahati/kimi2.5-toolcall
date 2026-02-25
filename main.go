package main

import (
	"log"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"ai-proxy/cmd/config"
	"ai-proxy/cmd/server"
	"ai-proxy/internal/logger"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "ai-proxy",
		Short: "AI Proxy - Multi-provider AI API proxy with protocol conversion",
		Long:  `AI-Proxy is a generic protocol converter and proxy server for AI providers.`,
	}

	rootCmd.PersistentFlags().IntP("port", "p", 8080, "Port to listen on")
	rootCmd.PersistentFlags().String("config-file", "", "Path to config file")
	rootCmd.PersistentFlags().String("log-level", "info", "Log level (debug, info, warn, error)")

	rootCmd.AddCommand(server.ServeCmd)
	rootCmd.AddCommand(config.ConfigCmd)

	viper.BindPFlags(rootCmd.PersistentFlags())

	// Initialize logger before executing commands
	logger.Init(viper.GetString("log-level"))

	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
}
