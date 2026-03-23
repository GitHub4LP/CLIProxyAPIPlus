package kaggle

import (
	"bytes"
	"context"
	"io"
	"net/http"

	log "github.com/sirupsen/logrus"
)

// Client wraps HTTP client for Kaggle API requests.
type Client struct {
	config     *Config
	httpClient *http.Client
}

// NewClient creates a new Kaggle API client.
func NewClient(cfg *Config) *Client {
	return &Client{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// ProxyRequest sends a request to Kaggle Models Proxy.
func (c *Client) ProxyRequest(ctx context.Context, method, path string, body []byte, headers map[string]string) (*http.Response, error) {
	url := c.config.ProxyURL + path
	
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Set default headers
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	req.Header.Set("Content-Type", "application/json")

	// Copy additional headers
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	log.Debugf("Kaggle proxy: %s %s", method, url)
	
	return c.httpClient.Do(req)
}

// ProxyStreamRequest sends a streaming request to Kaggle Models Proxy.
func (c *Client) ProxyStreamRequest(ctx context.Context, method, path string, body []byte, headers map[string]string) (*http.Response, error) {
	url := c.config.ProxyURL + path
	
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Set default headers
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	req.Header.Set("Content-Type", "application/json")

	// Copy additional headers
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	log.Debugf("Kaggle proxy stream: %s %s", method, url)
	
	// Use a client without timeout for streaming
	client := &http.Client{
		Timeout: 0, // No timeout for streaming
	}
	
	return client.Do(req)
}

// ProxyGeminiRequest sends a request to Kaggle GenAI endpoint (for Gemini models).
func (c *Client) ProxyGeminiRequest(ctx context.Context, method, path string, body []byte, headers map[string]string) (*http.Response, error) {
	url := c.config.GenAIURL + path
	
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Set default headers (Gemini uses x-goog-api-key, not Authorization)
	req.Header.Set("Content-Type", "application/json")

	// Copy additional headers
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	log.Debugf("Kaggle GenAI proxy: %s %s", method, url)
	
	return c.httpClient.Do(req)
}

// ProxyGeminiStreamRequest sends a streaming request to Kaggle GenAI endpoint (for Gemini models).
func (c *Client) ProxyGeminiStreamRequest(ctx context.Context, method, path string, body []byte, headers map[string]string) (*http.Response, error) {
	url := c.config.GenAIURL + path
	
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Set default headers (Gemini uses x-goog-api-key, not Authorization)
	req.Header.Set("Content-Type", "application/json")

	// Copy additional headers
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	log.Debugf("Kaggle GenAI proxy stream: %s %s", method, url)
	
	// Use a client without timeout for streaming
	client := &http.Client{
		Timeout: 0, // No timeout for streaming
	}
	
	return client.Do(req)
}

// ReadResponseBody reads the full response body and closes it.
func ReadResponseBody(resp *http.Response) ([]byte, error) {
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}
