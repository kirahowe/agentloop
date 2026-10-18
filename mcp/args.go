package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// narrowing is what a tool's FixedArgs and HiddenArgs take away from the
// model: the arguments it may not give, and the values set for it.
type narrowing struct {
	fixed  map[string]json.RawMessage
	hidden []string
	// full is the server's own input schema, which a call is checked
	// against once the fixed arguments are in.
	full *jsonschema.Schema
}

// narrowSchema returns the input schema the model is shown, without the
// fixed and hidden arguments, and the narrowing to apply to each call. It
// returns a nil narrowing when the tool has neither. Each name must be a
// property of the schema, a hidden one cannot be required (fix it
// instead), and no name can be both fixed and hidden.
func narrowSchema(name string, params json.RawMessage, cfg ToolConfig) (json.RawMessage, *narrowing, error) {
	if len(cfg.FixedArgs) == 0 && len(cfg.HiddenArgs) == 0 {
		return params, nil, nil
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(params, &schema); err != nil {
		return nil, nil, fmt.Errorf("inputSchema is not an object: %w", err)
	}
	var props map[string]json.RawMessage
	if raw, ok := schema["properties"]; ok {
		if err := json.Unmarshal(raw, &props); err != nil {
			return nil, nil, fmt.Errorf("inputSchema's properties is not an object: %w", err)
		}
	}
	var required []string
	if raw, ok := schema["required"]; ok {
		if err := json.Unmarshal(raw, &required); err != nil {
			return nil, nil, fmt.Errorf("inputSchema's required is not a list of names: %w", err)
		}
	}
	for _, h := range cfg.HiddenArgs {
		if _, both := cfg.FixedArgs[h]; both {
			return nil, nil, fmt.Errorf("the argument %s is both fixed and hidden", h)
		}
		if slices.Contains(required, h) {
			return nil, nil, fmt.Errorf("the argument %s is required, so it cannot be hidden; fix it instead", h)
		}
	}
	names := slices.Concat(slices.Collect(maps.Keys(cfg.FixedArgs)), cfg.HiddenArgs)
	sort.Strings(names)
	for _, n := range names {
		if _, ok := props[n]; !ok {
			return nil, nil, fmt.Errorf("the tool has no argument %s", n)
		}
		delete(props, n)
	}
	required = slices.DeleteFunc(required, func(r string) bool { return slices.Contains(names, r) })
	p, err := json.Marshal(props)
	if err != nil {
		return nil, nil, err
	}
	schema["properties"] = p
	if len(required) == 0 {
		delete(schema, "required")
	} else if schema["required"], err = json.Marshal(required); err != nil {
		return nil, nil, err
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, nil, err
	}
	if raw, err = Canonical(raw); err != nil {
		return nil, nil, err
	}
	full, err := compileSchema("mcp-input-full:"+name, params)
	if err != nil {
		return nil, nil, err
	}
	fixed := map[string]json.RawMessage{}
	for k, v := range cfg.FixedArgs {
		c, err := Canonical(v)
		if err != nil {
			return nil, nil, fmt.Errorf("fixed argument %s is not JSON: %w", k, err)
		}
		fixed[k] = c
	}
	return raw, &narrowing{fixed: fixed, hidden: slices.Clone(cfg.HiddenArgs), full: full}, nil
}

// apply returns the arguments the server is sent: the model's, which may
// name no fixed or hidden argument, with the fixed ones added. refused is
// what the model is told when it named one, and bad why the arguments
// cannot be sent otherwise.
func (n *narrowing) apply(input json.RawMessage) (out json.RawMessage, refused, bad string) {
	if err := strictjson.DuplicateKeys(input); err != nil {
		return nil, "", "the arguments name a key twice"
	}
	var args map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(input))
	if err := dec.Decode(&args); err != nil || args == nil {
		return nil, "", "the arguments are not a JSON object"
	}
	given := slices.Sorted(maps.Keys(args))
	for _, k := range given {
		if _, ok := n.fixed[k]; ok || slices.Contains(n.hidden, k) {
			return nil, k, ""
		}
	}
	for k, v := range n.fixed {
		args[k] = v
	}
	raw, err := json.Marshal(args)
	if err != nil {
		return nil, "", err.Error()
	}
	v, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, "", err.Error()
	}
	if err := n.full.Validate(v); err != nil {
		return nil, "", "with the fixed arguments, the call does not match the server's input schema: " + err.Error()
	}
	return raw, "", ""
}

// NotTaken is what the model is told of a call that gave an argument the
// operator fixed or hid.
func NotTaken(tool, arg string) string {
	return fmt.Sprintf("The tool %s does not take the argument %s. Call it without that argument.", tool, arg)
}
