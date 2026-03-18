package kaggle

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// GeminiAdapter handles Kaggle-specific Gemini API adaptations.
type GeminiAdapter struct {
	availableModels []string
}

// NewGeminiAdapter creates a new Gemini adapter.
func NewGeminiAdapter(availableModels []string) *GeminiAdapter {
	return &GeminiAdapter{
		availableModels: availableModels,
	}
}

// MapModelName maps client model name to Kaggle model name.
// Example: gemini-2.5-flash → google/gemini-2.5-flash
func (a *GeminiAdapter) MapModelName(clientModel string) string {
	// Already has provider prefix
	if strings.HasPrefix(clientModel, "google/") {
		return clientModel
	}

	// Try with google/ prefix
	fullName := "google/" + clientModel
	for _, m := range a.availableModels {
		if m == fullName {
			return fullName
		}
	}

	// Return original if not found
	return clientModel
}

// IsGemmaModel checks if the model is a Gemma model.
// Gemma models don't support systemInstruction.
func (a *GeminiAdapter) IsGemmaModel(modelName string) bool {
	return strings.Contains(strings.ToLower(modelName), "gemma")
}

// AdaptRequest adapts Gemini request for Kaggle limitations.
func (a *GeminiAdapter) AdaptRequest(body []byte, modelName string) ([]byte, error) {
	var modified bool
	result := body

	// Remove systemInstruction for Gemma models
	if a.IsGemmaModel(modelName) {
		if gjson.GetBytes(result, "systemInstruction").Exists() {
			var err error
			result, err = sjson.DeleteBytes(result, "systemInstruction")
			if err != nil {
				return body, err
			}
			modified = true
		}
	}

	// Remove thinkingConfig (Kaggle doesn't support it)
	if gjson.GetBytes(result, "generationConfig.thinkingConfig").Exists() {
		var err error
		result, err = sjson.DeleteBytes(result, "generationConfig.thinkingConfig")
		if err != nil {
			return body, err
		}
		modified = true
	}

	// Convert tools type fields to uppercase
	if gjson.GetBytes(result, "tools").Exists() {
		toolsJSON := gjson.GetBytes(result, "tools").Raw
		uppercased := convertTypeToUppercase([]byte(toolsJSON))
		var err error
		result, err = sjson.SetRawBytes(result, "tools", uppercased)
		if err != nil {
			return body, err
		}
		modified = true
	}

	if modified {
		return result, nil
	}
	return body, nil
}

// convertTypeToUppercase recursively converts all "type" field values to uppercase.
func convertTypeToUppercase(data []byte) []byte {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return data
	}

	converted := convertTypeRecursive(obj)
	result, err := json.Marshal(converted)
	if err != nil {
		return data
	}
	return result
}

func convertTypeRecursive(obj interface{}) interface{} {
	switch v := obj.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, val := range v {
			if k == "type" {
				if str, ok := val.(string); ok {
					result[k] = strings.ToUpper(str)
				} else {
					result[k] = val
				}
			} else {
				result[k] = convertTypeRecursive(val)
			}
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(v))
		for i, val := range v {
			result[i] = convertTypeRecursive(val)
		}
		return result
	default:
		return v
	}
}

// StreamConvertJSONLinesToSSE converts Gemini JSON Lines stream to SSE format in real-time.
// This is used when client requests alt=sse parameter.
func StreamConvertJSONLinesToSSE(reader *bufio.Reader, writer io.Writer) error {
	lineBuffer := ""
	buf := make([]byte, 4096) // 按块读取，不等待换行符

	for {
		n, err := reader.Read(buf)
		if n > 0 {
			// 立即处理读到的数据
			lineBuffer += string(buf[:n])
		}
		
		if err != nil {
			if err == io.EOF {
				// EOF 时处理剩余数据
				break
			}
			return err
		}

		// Inner loop to process all content in lineBuffer
		for len(lineBuffer) > 0 {
			lineBuffer = strings.TrimSpace(lineBuffer)

			if lineBuffer == "" {
				break
			}

			// Skip array start
			if strings.HasPrefix(lineBuffer, "[") {
				lineBuffer = lineBuffer[1:]
				continue
			}

			// Skip commas
			if strings.HasPrefix(lineBuffer, ",") {
				lineBuffer = lineBuffer[1:]
				continue
			}

			// End of JSON array - stop processing
			if strings.HasPrefix(lineBuffer, "]") {
				return nil
			}

			// Try to parse as JSON object
			if strings.HasPrefix(lineBuffer, "{") {
				// Find complete JSON object
				bracketCount := 0
				inString := false
				escape := false
				endPos := -1

				for i, char := range lineBuffer {
					if escape {
						escape = false
						continue
					}
					if char == '\\' {
						escape = true
						continue
					}
					if char == '"' && !escape {
						inString = !inString
					}
					if !inString {
						if char == '{' {
							bracketCount++
						} else if char == '}' {
							bracketCount--
							if bracketCount == 0 {
								endPos = i + 1
								break
							}
						}
					}
				}

				if endPos > 0 {
					jsonStr := lineBuffer[:endPos]
					// Validate JSON
					var obj map[string]interface{}
					if err := json.Unmarshal([]byte(jsonStr), &obj); err == nil {
						// Write SSE format immediately
						writer.Write([]byte("data: "))
						writer.Write([]byte(jsonStr))
						writer.Write([]byte("\n\n"))

						// Flush if possible
						if flusher, ok := writer.(http.Flusher); ok {
							flusher.Flush()
						}
					}
					lineBuffer = lineBuffer[endPos:]
				} else {
					// Incomplete JSON, break inner loop to read more data
					break
				}
			} else {
				// Unknown character, skip it
				lineBuffer = lineBuffer[1:]
			}
		}
	}

	// 外层循环结束后，处理 lineBuffer 中剩余的内容
	for len(lineBuffer) > 0 {
		lineBuffer = strings.TrimSpace(lineBuffer)

		if lineBuffer == "" {
			break
		}

		// Skip array start
		if strings.HasPrefix(lineBuffer, "[") {
			lineBuffer = lineBuffer[1:]
			continue
		}

		// Skip commas
		if strings.HasPrefix(lineBuffer, ",") {
			lineBuffer = lineBuffer[1:]
			continue
		}

		// End of JSON array - stop processing
		if strings.HasPrefix(lineBuffer, "]") {
			return nil
		}

		// Try to parse as JSON object
		if strings.HasPrefix(lineBuffer, "{") {
			// Find complete JSON object
			bracketCount := 0
			inString := false
			escape := false
			endPos := -1

			for i, char := range lineBuffer {
				if escape {
					escape = false
					continue
				}
				if char == '\\' {
					escape = true
					continue
				}
				if char == '"' && !escape {
					inString = !inString
				}
				if !inString {
					if char == '{' {
						bracketCount++
					} else if char == '}' {
						bracketCount--
						if bracketCount == 0 {
							endPos = i + 1
							break
						}
					}
				}
			}

			if endPos > 0 {
				jsonStr := lineBuffer[:endPos]
				// Validate JSON
				var obj map[string]interface{}
				if err := json.Unmarshal([]byte(jsonStr), &obj); err == nil {
					// Write SSE format immediately
					writer.Write([]byte("data: "))
					writer.Write([]byte(jsonStr))
					writer.Write([]byte("\n\n"))

					// Flush if possible
					if flusher, ok := writer.(http.Flusher); ok {
						flusher.Flush()
					}
				}
				lineBuffer = lineBuffer[endPos:]
			} else {
				// Incomplete JSON, no more data to read
				break
			}
		} else {
			// Unknown character, skip it
			lineBuffer = lineBuffer[1:]
		}
	}

	return nil
}
