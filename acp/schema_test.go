package acpagent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// testdata/schema.json is ACP's schema.json at schema-v1.25.0, as the
// caelis SDK v1.5.0 vendors it (schema/schema.json, upstream commit
// 4cf3dd858c819fc3ab99ae53f76883ede0345a10).
const schemaURL = "https://agentclientprotocol.com/schema-v1.25.0.json"

var compiled = sync.OnceValues(func() (map[string]*jsonschema.Schema, error) {
	raw, err := os.ReadFile("testdata/schema.json")
	if err != nil {
		return nil, err
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	c := jsonschema.NewCompiler()
	if err := c.AddResource(schemaURL, doc); err != nil {
		return nil, err
	}
	out := map[string]*jsonschema.Schema{}
	for _, def := range []string{"SessionNotification", "PromptResponse", "ContentBlock", "InitializeResponse", "NewSessionResponse"} {
		s, err := c.Compile(schemaURL + "#/$defs/" + def)
		if err != nil {
			return nil, err
		}
		out[def] = s
	}
	return out, nil
})

// validate checks v, marshalled to JSON, against one of the schema's
// definitions.
func validate(def string, v any) error {
	schemas, err := compiled()
	if err != nil {
		return fmt.Errorf("the schema does not compile: %w", err)
	}
	s, ok := schemas[def]
	if !ok {
		return fmt.Errorf("no compiled definition %s", def)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	inst, err := jsonschema.UnmarshalJSON(bytes.NewReader(b))
	if err != nil {
		return err
	}
	if err := s.Validate(inst); err != nil {
		return fmt.Errorf("%s does not validate as %s: %v", b, def, err)
	}
	return nil
}

// validateUpdate checks an update as the session/update notification that
// carries it.
func validateUpdate(u Update) error {
	return validate("SessionNotification", map[string]any{"sessionId": "s1", "update": u})
}
