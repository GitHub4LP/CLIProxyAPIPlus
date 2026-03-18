// Package kaggle provides Kaggle Models Proxy integration for CLIProxyAPIPlus.
// This package enables running CLIProxyAPIPlus inside Kaggle Notebooks,
// proxying requests to Kaggle's internal Models API.
package kaggle

import (
	"os"
	"strings"
	"time"
)

// Config holds Kaggle proxy configuration from environment variables.
type Config struct {
	// ProxyURL is the Kaggle Models Proxy base URL (for OpenAI/Claude)
	ProxyURL string
	// GenAIURL is the Kaggle GenAI base URL (for Gemini)
	GenAIURL string
	// APIKey is the Kaggle Models Proxy API key
	APIKey string
	// Timeout is the request timeout duration
	Timeout time.Duration
	// AvailableModels is the list of available models
	AvailableModels []string
}

// LoadConfig loads configuration from environment variables.
func LoadConfig() *Config {
	proxyURL := os.Getenv("MODEL_PROXY_URL")
	if proxyURL == "" {
		proxyURL = "https://mp-staging.kaggle.net/models/openapi"
	}

	apiKey := os.Getenv("MODEL_PROXY_API_KEY")
	if apiKey == "" {
		panic("MODEL_PROXY_API_KEY environment variable is required")
	}

	timeout := 300 * time.Second
	if timeoutStr := os.Getenv("REQUEST_TIMEOUT"); timeoutStr != "" {
		if d, err := time.ParseDuration(timeoutStr + "s"); err == nil {
			timeout = d
		}
	}

	var models []string
	if modelsStr := os.Getenv("LLMS_AVAILABLE"); modelsStr != "" {
		for _, m := range strings.Split(modelsStr, ",") {
			if m = strings.TrimSpace(m); m != "" {
				models = append(models, m)
			}
		}
	}

	return &Config{
		ProxyURL:        proxyURL,
		GenAIURL:        strings.Replace(proxyURL, "/openapi", "/genai", 1),
		APIKey:          apiKey,
		Timeout:         timeout,
		AvailableModels: models,
	}
}
