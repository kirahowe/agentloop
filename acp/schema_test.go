package acpagent

import "github.com/kirahowe/agentloop/internal/acpschema"

// validate checks v, marshalled to JSON, against one of the vendored
// schema's definitions.
func validate(def string, v any) error { return acpschema.Validate(def, v) }

// validateUpdate checks an update as the session/update notification that
// carries it.
func validateUpdate(u Update) error {
	return validate("SessionNotification", map[string]any{"sessionId": "s1", "update": u})
}
