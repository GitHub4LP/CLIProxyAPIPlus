package kaggle

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Handler handles Kaggle proxy HTTP requests.
type Handler struct {
	client        *Client
	config        *Config
	geminiAdapter *GeminiAdapter
	claudeAdapter *ClaudeAdapter
}

// NewHandler creates a new Kaggle handler.
func NewHandler(cfg *Config) *Handler {
	return &Handler{
		client:        NewClient(cfg),
		config:        cfg,
		geminiAdapter: NewGeminiAdapter(cfg.AvailableModels),
		claudeAdapter: NewClaudeAdapter(cfg.AvailableModels),
	}
}

// RegisterRoutes registers Kaggle proxy routes.
func (h *Handler) RegisterRoutes(router gin.IRouter) {
	router.POST("/v1/chat/completions", h.ChatCompletions)
	router.POST("/v1/messages", h.ClaudeMessages)
	router.POST("/v1beta/models/*action", h.GeminiHandler)
	
	log.Debug("Kaggle proxy routes registered")
}

// ChatCompletions handles OpenAI chat completions (direct proxy).
func (h *Handler) ChatCompletions(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "Failed to read request body", "type": "invalid_request_error"}})
		return
	}

	// Check if streaming
	var reqData map[string]interface{}
	if err := json.Unmarshal(body, &reqData); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "Invalid JSON", "type": "invalid_request_error"}})
		return
	}

	stream, _ := reqData["stream"].(bool)

	if stream {
		h.proxyStream(c, "/chat/completions", body, nil)
	} else {
		h.proxyNonStream(c, "/chat/completions", body, nil)
	}
}

// ClaudeMessages handles Claude messages API (with translation).
func (h *Handler) ClaudeMessages(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"type": "error", "error": gin.H{"type": "invalid_request_error", "message": "Failed to read request body"}})
		return
	}

	// Parse request
	var reqData map[string]interface{}
	if err := json.Unmarshal(body, &reqData); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"type": "error", "error": gin.H{"type": "invalid_request_error", "message": "Invalid JSON"}})
		return
	}

	model, _ := reqData["model"].(string)
	stream, _ := reqData["stream"].(bool)

	// Map model name for Kaggle
	kaggleModel := h.claudeAdapter.MapModelName(model)
	if model != kaggleModel {
		log.Debugf("Claude model mapping: %s → %s", model, kaggleModel)
	}

	// Translate Claude → OpenAI
	translatedBody := translator.Request("claude", "openai", kaggleModel, body, stream)
	
	// Clean tools parameters for Kaggle compatibility
	translatedBody = cleanOpenAIToolsParameters(translatedBody)

	if stream {
		h.claudeStreamResponse(c, "/chat/completions", translatedBody, body, model)
	} else {
		h.claudeNonStreamResponse(c, "/chat/completions", translatedBody, body, model)
	}
}

// GeminiHandler handles Gemini API (with Kaggle adaptations).
func (h *Handler) GeminiHandler(c *gin.Context) {
	action := c.Param("action")
	
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "Failed to read request body"}})
		return
	}

	// Parse action: model:method
	parts := strings.Split(strings.TrimPrefix(action, "/"), ":")
	if len(parts) < 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "Invalid action format"}})
		return
	}

	modelName := parts[0]
	method := parts[1]

	// Map model name
	kaggleModel := h.geminiAdapter.MapModelName(modelName)
	
	// Log model mapping
	if modelName != kaggleModel {
		log.Debugf("Gemini request: %s → %s:%s", modelName, kaggleModel, method)
	}
	
	// Adapt request body for Kaggle limitations
	adaptedBody, err := h.geminiAdapter.AdaptRequest(body, kaggleModel)
	if err != nil {
		log.Warnf("Failed to adapt Gemini request: %v", err)
		adaptedBody = body
	}

	// Build path - Gemini uses /genai instead of /openapi
	path := "/v1/models/" + kaggleModel + ":" + method
	
	// Use x-goog-api-key header for Gemini
	headers := map[string]string{
		"x-goog-api-key": h.config.APIKey,
	}

	// Check if streaming
	isStream := method == "streamGenerateContent"
	
	// Check if SSE format requested
	needSSE := c.Query("alt") == "sse"
	
	if isStream {
		if needSSE {
			h.geminiStreamSSE(c, path, adaptedBody, headers)
		} else {
			h.geminiProxyStream(c, path, adaptedBody, headers)
		}
	} else {
		h.geminiProxyNonStream(c, path, adaptedBody, headers)
	}
}

// proxyStream proxies a streaming request.
func (h *Handler) proxyStream(c *gin.Context, path string, body []byte, headers map[string]string) {
	resp, err := h.client.ProxyStreamRequest(c.Request.Context(), "POST", path, body, headers)
	if err != nil {
		log.Errorf("Kaggle proxy stream error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		if len(v) > 0 {
			c.Header(k, v[0])
		}
	}
	c.Status(resp.StatusCode)

	// Stream response
	c.Stream(func(w io.Writer) bool {
		buf := make([]byte, 4096)
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
		}
		return err == nil
	})
}

// proxyNonStream proxies a non-streaming request.
func (h *Handler) proxyNonStream(c *gin.Context, path string, body []byte, headers map[string]string) {
	resp, err := h.client.ProxyRequest(c.Request.Context(), "POST", path, body, headers)
	if err != nil {
		log.Errorf("Kaggle proxy error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}

	respBody, err := ReadResponseBody(resp)
	if err != nil {
		log.Errorf("Kaggle proxy read error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}

	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

// claudeStreamResponse handles Claude streaming response with translation.
func (h *Handler) claudeStreamResponse(c *gin.Context, path string, translatedBody, originalBody []byte, model string) {
	resp, err := h.client.ProxyStreamRequest(c.Request.Context(), "POST", path, translatedBody, nil)
	if err != nil {
		log.Errorf("Kaggle Claude stream error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"type": "error", "error": gin.H{"type": "api_error", "message": err.Error()}})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		convertedErr := convertOpenAIErrorToClaude(respBody)
		c.Data(resp.StatusCode, "application/json", convertedErr)
		return
	}

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(http.StatusOK)

	// Translate OpenAI SSE → Claude SSE
	var param any
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		
		if len(line) == 0 {
			continue
		}

		// Translate response
		translated := translator.Response("openai", "claude", context.Background(), model, originalBody, translatedBody, line, &param)
		for _, t := range translated {
			c.Writer.Write(t)
			c.Writer.Flush()
		}
	}
	
	// Ensure response is properly finalized by sending [DONE]
	translated := translator.Response("openai", "claude", context.Background(), model, originalBody, translatedBody, []byte("data: [DONE]"), &param)
	for _, t := range translated {
		c.Writer.Write(t)
		c.Writer.Flush()
	}
}

// claudeNonStreamResponse handles Claude non-streaming response with translation.
func (h *Handler) claudeNonStreamResponse(c *gin.Context, path string, translatedBody, originalBody []byte, model string) {
	resp, err := h.client.ProxyRequest(c.Request.Context(), "POST", path, translatedBody, nil)
	if err != nil {
		log.Errorf("Kaggle Claude error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"type": "error", "error": gin.H{"type": "api_error", "message": err.Error()}})
		return
	}

	respBody, err := ReadResponseBody(resp)
	if err != nil {
		log.Errorf("Kaggle Claude read error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"type": "error", "error": gin.H{"type": "api_error", "message": err.Error()}})
		return
	}

	if resp.StatusCode != http.StatusOK {
		convertedErr := convertOpenAIErrorToClaude(respBody)
		c.Data(resp.StatusCode, "application/json", convertedErr)
		return
	}

	// Translate OpenAI → Claude
	var param any
	translated := translator.ResponseNonStream("openai", "claude", context.Background(), model, originalBody, translatedBody, respBody, &param)
	c.Data(http.StatusOK, "application/json", translated)
}

// geminiStreamSSE handles Gemini streaming with SSE format conversion.
func (h *Handler) geminiStreamSSE(c *gin.Context, path string, body []byte, headers map[string]string) {
	resp, err := h.client.ProxyGeminiStreamRequest(c.Request.Context(), "POST", path, body, headers)
	if err != nil {
		log.Errorf("Kaggle Gemini SSE error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}
	defer resp.Body.Close()

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		// Return error in SSE format
		respBody, _ := io.ReadAll(resp.Body)
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": string(respBody),
			},
		}
		errorJSON, _ := json.Marshal(errorData)
		c.Writer.WriteString("data: ")
		c.Writer.Write(errorJSON)
		c.Writer.WriteString("\n\n")
		c.Writer.Flush()
		return
	}

	// Convert JSON Lines to SSE in real-time
	reader := bufio.NewReader(resp.Body)
	if err := StreamConvertJSONLinesToSSE(reader, c.Writer); err != nil {
		log.Errorf("Kaggle Gemini SSE conversion error: %v", err)
	}
}

// geminiProxyStream proxies a Gemini streaming request.
func (h *Handler) geminiProxyStream(c *gin.Context, path string, body []byte, headers map[string]string) {
	resp, err := h.client.ProxyGeminiStreamRequest(c.Request.Context(), "POST", path, body, headers)
	if err != nil {
		log.Errorf("Kaggle Gemini proxy stream error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		if len(v) > 0 {
			c.Header(k, v[0])
		}
	}
	c.Status(resp.StatusCode)

	// Stream response
	c.Stream(func(w io.Writer) bool {
		buf := make([]byte, 4096)
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
		}
		return err == nil
	})
}

// geminiProxyNonStream proxies a Gemini non-streaming request.
func (h *Handler) geminiProxyNonStream(c *gin.Context, path string, body []byte, headers map[string]string) {
	resp, err := h.client.ProxyGeminiRequest(c.Request.Context(), "POST", path, body, headers)
	if err != nil {
		log.Errorf("Kaggle Gemini proxy error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}

	respBody, err := ReadResponseBody(resp)
	if err != nil {
		log.Errorf("Kaggle Gemini proxy read error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "api_error"}})
		return
	}

	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

// convertOpenAIErrorToClaude converts OpenAI error format to Claude error format.
// OpenAI: {"error": {"message": "...", "type": "..."}}
// Claude: {"type": "error", "error": {"type": "...", "message": "..."}}
func convertOpenAIErrorToClaude(body []byte) []byte {
	var openaiErr map[string]interface{}
	if err := json.Unmarshal(body, &openaiErr); err != nil {
		// 无法解析，返回通用 Claude 错误
		claudeErr := map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "api_error",
				"message": string(body),
			},
		}
		result, _ := json.Marshal(claudeErr)
		return result
	}
	
	// 检查是否有 error 字段
	if errorObj, ok := openaiErr["error"]; ok {
		// 包装成 Claude 格式
		claudeErr := map[string]interface{}{
			"type":  "error",
			"error": errorObj,
		}
		result, _ := json.Marshal(claudeErr)
		return result
	}
	
	// 没有 error 字段，可能已经是其他格式，包装成 Claude 错误
	claudeErr := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "api_error",
			"message": string(body),
		},
	}
	result, _ := json.Marshal(claudeErr)
	return result
}

// cleanOpenAIToolsParameters cleans OpenAI tools parameters for Kaggle compatibility.
// Removes $schema field and ensures type field exists.
func cleanOpenAIToolsParameters(body []byte) []byte {
	root := gjson.ParseBytes(body)
	
	// Check if tools exist
	tools := root.Get("tools")
	if !tools.Exists() || !tools.IsArray() {
		return body
	}
	
	// Parse and clean each tool
	result := string(body)
	tools.ForEach(func(key, tool gjson.Result) bool {
		params := tool.Get("function.parameters")
		if !params.Exists() {
			return true
		}
		
		// Clean parameters
		cleaned := params.Raw
		
		// Remove $schema if present
		if gjson.Get(cleaned, "$schema").Exists() {
			cleaned, _ = sjson.Delete(cleaned, "$schema")
		}
		
		// Ensure type field exists
		if !gjson.Get(cleaned, "type").Exists() {
			cleaned, _ = sjson.Set(cleaned, "type", "object")
		}
		
		// Update the result
		path := "tools." + key.String() + ".function.parameters"
		result, _ = sjson.SetRaw(result, path, cleaned)
		
		return true
	})
	
	return []byte(result)
}
