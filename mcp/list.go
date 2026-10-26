package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// A Listed tool is one a server lists, with its digest, for an operator
// choosing which tools to allow.
type Listed struct {
	Name         string               `json:"name"`
	Title        string               `json:"title,omitempty"`
	Description  string               `json:"description"`
	Annotations  *sdk.ToolAnnotations `json:"annotations,omitempty"`
	InputSchema  json.RawMessage      `json:"input_schema"`
	OutputSchema json.RawMessage      `json:"output_schema,omitempty"`
	Digest       string               `json:"digest"`
}

// List connects to a server, lists every tool it has, by name, and
// closes. srv.Tools is not used. A tool whose schema cannot be read
// fails the listing, as it would fail Connect.
func List(ctx context.Context, srv Server) ([]Listed, error) {
	probe := srv
	probe.Tools = map[string]ToolConfig{"-": {}}
	if err := checkServer(probe); err != nil {
		return nil, err
	}
	cn, listed, err := dial(ctx, srv)
	if err != nil {
		return nil, err
	}
	defer cn.close()
	out := make([]Listed, 0, len(listed))
	for name, def := range listed {
		d, err := Digest(def)
		if err != nil {
			return nil, fmt.Errorf("mcp server %s, tool %s: %w", srv.Name, name, err)
		}
		in, err := canonicalValue(def.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("mcp server %s, tool %s: inputSchema: %w", srv.Name, name, err)
		}
		l := Listed{Name: name, Title: def.Title, Description: def.Description, Annotations: def.Annotations,
			InputSchema: in, Digest: d}
		if def.Title == "" && def.Annotations != nil {
			l.Title = def.Annotations.Title
		}
		if def.OutputSchema != nil {
			if l.OutputSchema, err = canonicalValue(def.OutputSchema); err != nil {
				return nil, fmt.Errorf("mcp server %s, tool %s: outputSchema: %w", srv.Name, name, err)
			}
		}
		out = append(out, l)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}
