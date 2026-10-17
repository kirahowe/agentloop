// Command mcpserver is a small MCP stdio server for agentloop's tests.
//
//	-desc FILE    the lookup tool's description is FILE's content
//	-creds FILE   the credentials file whoami reports on
//
// Its tools:
//
//	lookup {q, labels}  returns {"q", "meta"} (the request's _meta) as text,
//	                    and labels, when given, as the result's labels
//	picture             returns text and an image
//	doc                 returns an embedded text resource
//	whoami              reports the credentials file: its path, mode and
//	                    the SHA-256 of what it holds, never the content
//	wipe                says it wiped; its annotations say destructive
//	crash               exits during the call
//	unlisted            on no allowlist
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const labelsKey = "com.kirahowe.agentloop/labels"

func main() {
	desc := flag.String("desc", "", "")
	creds := flag.String("creds", "", "")
	flag.Parse()
	fmt.Fprintln(os.Stderr, "mcpserver up")
	lookupDesc := "Look something up."
	if *desc != "" {
		b, err := os.ReadFile(*desc)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		lookupDesc = string(b)
	}
	obj := func(props string) json.RawMessage {
		return json.RawMessage(`{"type":"object","properties":{` + props + `}}`)
	}
	text := func(s string) []sdk.Content { return []sdk.Content{&sdk.TextContent{Text: s}} }
	s := sdk.NewServer(&sdk.Implementation{Name: "mcpserver", Version: "1"}, nil)
	s.AddTool(&sdk.Tool{Name: "lookup", Description: lookupDesc, InputSchema: obj(`"q":{"type":"string"},"labels":{}`)},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			var in map[string]any
			json.Unmarshal(req.Params.Arguments, &in)
			b, _ := json.Marshal(map[string]any{"q": in["q"], "meta": req.Params.Meta})
			r := &sdk.CallToolResult{Content: text(string(b))}
			if l, ok := in["labels"]; ok {
				r.Meta = sdk.Meta{labelsKey: l}
			}
			return r, nil
		})
	s.AddTool(&sdk.Tool{Name: "picture", Description: "Draw.", InputSchema: obj("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{
				&sdk.TextContent{Text: "a picture"},
				&sdk.ImageContent{Data: []byte("PIXELS"), MIMEType: "image/png"},
			}}, nil
		})
	s.AddTool(&sdk.Tool{Name: "doc", Description: "Read a document.", InputSchema: obj("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{
				&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///doc", MIMEType: "text/plain", Text: "the document"}},
			}}, nil
		})
	s.AddTool(&sdk.Tool{Name: "whoami", Description: "Report the credentials file.", InputSchema: obj("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			b, err := os.ReadFile(*creds)
			if err != nil {
				return nil, err
			}
			st, err := os.Stat(*creds)
			if err != nil {
				return nil, err
			}
			sum := sha256.Sum256(b)
			out, _ := json.Marshal(map[string]string{
				"path": *creds, "mode": fmt.Sprintf("%o", st.Mode().Perm()), "sha256": hex.EncodeToString(sum[:]),
			})
			return &sdk.CallToolResult{Content: text(string(out))}, nil
		})
	yes := true
	s.AddTool(&sdk.Tool{Name: "wipe", Description: "Wipe.", InputSchema: obj(""),
		Annotations: &sdk.ToolAnnotations{DestructiveHint: &yes}},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: text("wiped")}, nil
		})
	s.AddTool(&sdk.Tool{Name: "crash", Description: "Exit mid-call.", InputSchema: obj("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			os.Exit(3)
			return nil, nil
		})
	s.AddTool(&sdk.Tool{Name: "unlisted", Description: "Not allowed.", InputSchema: obj("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: text("should not run")}, nil
		})
	if err := s.Run(context.Background(), &sdk.StdioTransport{}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
