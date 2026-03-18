package kaggle

import (
	"strings"
)

// ClaudeAdapter handles Claude-specific adaptations for Kaggle.
type ClaudeAdapter struct {
	availableModels []string
}

// NewClaudeAdapter creates a new Claude adapter.
func NewClaudeAdapter(availableModels []string) *ClaudeAdapter {
	return &ClaudeAdapter{
		availableModels: availableModels,
	}
}

// MapModelName maps Claude model names to Kaggle format.
// Examples:
//   claude-opus-4-6 → anthropic/claude-opus-4-6@default
//   claude-opus-4-5-20251101 → anthropic/claude-opus-4-5@20251101
func (a *ClaudeAdapter) MapModelName(clientModel string) string {
	providerPrefix := "anthropic/"
	
	// Already in Kaggle format
	if strings.HasPrefix(clientModel, providerPrefix) && strings.Contains(clientModel, "@") {
		return clientModel
	}
	
	// Exact match with date version
	for _, model := range a.availableModels {
		if strings.HasPrefix(model, providerPrefix) {
			// anthropic/claude-opus-4-6@default → claude-opus-4-6-default
			kaggleName := strings.Replace(model, providerPrefix, "", 1)
			kaggleName = strings.Replace(kaggleName, "@", "-", 1)
			if kaggleName == clientModel {
				return model
			}
		}
	}
	
	// Latest match without date
	prefix := providerPrefix + clientModel + "@"
	var candidates []string
	for _, model := range a.availableModels {
		if strings.HasPrefix(model, prefix) {
			candidates = append(candidates, model)
		}
	}
	
	if len(candidates) > 0 {
		// Prefer @default
		for _, candidate := range candidates {
			if strings.HasSuffix(candidate, "@default") {
				return candidate
			}
		}
		// Return the last one (lexicographically largest)
		return candidates[len(candidates)-1]
	}
	
	return clientModel
}
