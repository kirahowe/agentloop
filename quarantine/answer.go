package quarantine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// Compile compiles a schema the way a reader's answer to it must be
// checked: as JSON Schema 2020-12, with format asserted (so the formats
// in Formats close a string) and no reference loaded from anywhere.
func Compile(schema []byte) (*jsonschema.Schema, error) {
	if err := strictjson.Input(schema); err != nil {
		return nil, fmt.Errorf("the schema is not acceptable JSON: %w", err)
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(schema))
	if err != nil {
		return nil, fmt.Errorf("the schema is not JSON: %w", err)
	}
	c := jsonschema.NewCompiler()
	c.DefaultDraft(jsonschema.Draft2020)
	c.AssertFormat()
	c.UseLoader(noLoader{})
	const url = "quarantine:schema"
	if err := c.AddResource(url, doc); err != nil {
		return nil, err
	}
	s, err := c.Compile(url)
	if err != nil {
		return nil, fmt.Errorf("the schema is not valid: %w", err)
	}
	return s, nil
}

type noLoader struct{}

func (noLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("schemas may not load %s", url)
}

// Answer checks a reader's answer against a compiled schema and returns
// it as canonical JSON (keys sorted, no whitespace, no HTML escaping,
// numbers as written). The caller must be given these bytes and never
// the reader's own: an answer that names a key twice is refused here, but
// whitespace, escapes and key order are the reader's to choose, and only
// the parsed value has been validated.
func Answer(s *jsonschema.Schema, answer []byte) ([]byte, error) {
	if err := strictjson.Input(answer); err != nil {
		return nil, fmt.Errorf("the answer is not acceptable JSON: %w", err)
	}
	v, err := decode(answer)
	if err != nil {
		return nil, fmt.Errorf("the answer is not JSON: %w", err)
	}
	if err := s.Validate(v); err != nil {
		return nil, fmt.Errorf("the answer does not match the schema: %w", err)
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return []byte(strings.TrimSuffix(b.String(), "\n")), nil
}
