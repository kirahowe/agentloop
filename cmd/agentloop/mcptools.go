package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/kirahowe/agentloop/internal/request"
	"github.com/kirahowe/agentloop/internal/strictjson"
	"github.com/kirahowe/agentloop/internal/usage"
	"github.com/kirahowe/agentloop/mcp"
)

// mcpToolsInput is what agentloop mcp-tools reads on stdin: a remote
// server's URL and the bearer token to send it, if any. The token comes
// on stdin so that it is never in an argument list.
type mcpToolsInput struct {
	URL   string `json:"url"`
	Token string `json:"token"`
}

// mcpToolsCmd lists every tool of a remote MCP server, with each
// definition's digest, as JSON on stdout: {"tools": [...]} (mcp.Listed).
// It is for an operator choosing which tools to allow, and for pinning
// what they chose.
func mcpToolsCmd(ctx context.Context, stdin io.Reader, stdout io.Writer) error {
	raw, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil {
		return err
	}
	if err := strictjson.Check(raw, reflect.TypeFor[mcpToolsInput]()); err != nil {
		return usage.Errorf("mcp-tools input: %v", err)
	}
	var in mcpToolsInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return usage.Errorf("mcp-tools input: %v", err)
	}
	if err := mcp.CheckURL(in.URL); err != nil {
		return usage.Errorf("mcp-tools: %v", err)
	}
	srv := mcp.Server{Name: "remote", URL: in.URL}
	if tok := strings.TrimSpace(in.Token); tok != "" {
		srv.Token = func(context.Context, bool) (string, error) { return tok, nil }
	}
	ctx, cancel := context.WithTimeout(ctx, request.MCPStartTimeout)
	defer cancel()
	listed, err := mcp.List(ctx, srv)
	if err != nil {
		return fmt.Errorf("listing the tools: %w", err)
	}
	return json.NewEncoder(stdout).Encode(map[string]any{"tools": listed})
}
