// Package acpschema validates JSON against the ACP schema vendored in
// acp/testdata/schema.json. It is for tests.
package acpschema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// schema.json is ACP's schema at schema-v1.25.0, as the caelis SDK v1.5.0
// vendors it (schema/schema.json, upstream commit
// 4cf3dd858c819fc3ab99ae53f76883ede0345a10).
const schemaURL = "https://agentclientprotocol.com/schema-v1.25.0.json"

var (
	mu       sync.Mutex
	compiler *jsonschema.Compiler
	schemas  = map[string]*jsonschema.Schema{}
)

func schemaFile() string {
	_, here, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(here), "..", "..", "acp", "testdata", "schema.json")
}

func compile(def string) (*jsonschema.Schema, error) {
	mu.Lock()
	defer mu.Unlock()
	if s, ok := schemas[def]; ok {
		return s, nil
	}
	if compiler == nil {
		raw, err := os.ReadFile(schemaFile())
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
		compiler = c
	}
	s, err := compiler.Compile(schemaURL + "#/$defs/" + def)
	if err != nil {
		return nil, fmt.Errorf("the schema does not compile %s: %w", def, err)
	}
	schemas[def] = s
	return s, nil
}

// Validate checks v, marshalled to JSON, against one of the schema's
// definitions.
func Validate(def string, v any) error {
	s, err := compile(def)
	if err != nil {
		return err
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
