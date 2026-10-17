package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// helperArg makes the test binary an MCP server instead of running tests.
const helperArg = "mcp-test-server"

// rawArg makes it a hand-written JSON-RPC server that misbehaves in the
// way its next argument names (see serveRaw); holdArg makes it a process
// that only waits.
const (
	rawArg  = "mcp-test-raw"
	holdArg = "mcp-test-hold"
)

func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == helperArg {
		os.Exit(serve(os.Args[2:]))
	}
	if len(os.Args) > 2 && os.Args[1] == rawArg {
		os.Exit(serveRaw(os.Args[2], os.Args[3:]))
	}
	if len(os.Args) > 1 && os.Args[1] == holdArg {
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func object(props string) json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{` + props + `}}`)
}

func text(s string) []sdk.Content { return []sdk.Content{&sdk.TextContent{Text: s}} }

func args(req *sdk.CallToolRequest) map[string]any {
	var m map[string]any
	json.Unmarshal(req.Params.Arguments, &m)
	return m
}

// serve runs the test server on stdin and stdout.
//
//	-desc FILE   echo's description is FILE's content
//	-drop        leave out the env tool
func serve(argv []string) int {
	fs := flag.NewFlagSet(helperArg, flag.ContinueOnError)
	descFile := fs.String("desc", "", "")
	drop := fs.Bool("drop", false, "")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	fmt.Fprintln(os.Stderr, "helper up")
	desc := "Echo msg."
	if *descFile != "" {
		b, err := os.ReadFile(*descFile)
		if err != nil {
			return 2
		}
		desc = string(b)
	}
	s := sdk.NewServer(&sdk.Implementation{Name: "helper", Version: "1"}, nil)
	s.AddTool(&sdk.Tool{Name: "echo", Description: desc, InputSchema: object(`"msg":{"type":"string"},"labels":{}`)},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			a := args(req)
			r := &sdk.CallToolResult{Content: text(fmt.Sprint(a["msg"]))}
			if l, ok := a["labels"]; ok {
				r.Meta = sdk.Meta{LabelsKey: l}
			}
			return r, nil
		})
	s.AddTool(&sdk.Tool{Name: "meta", Description: "Show the request's _meta.", InputSchema: object("")},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			b, _ := json.Marshal(req.Params.Meta)
			return &sdk.CallToolResult{Content: text(string(b))}, nil
		})
	if !*drop {
		s.AddTool(&sdk.Tool{Name: "env", Description: "Show the environment and directory.", InputSchema: object("")},
			func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				wd, _ := os.Getwd()
				return &sdk.CallToolResult{Content: text(strings.Join(os.Environ(), "\n") + "\ncwd=" + wd)}, nil
			})
	}
	s.AddTool(&sdk.Tool{Name: "content", Description: "Return non-text content.", InputSchema: object(`"kind":{"type":"string"}`)},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			var c sdk.Content
			switch args(req)["kind"] {
			case Image:
				c = &sdk.ImageContent{Data: []byte("PNGSECRET"), MIMEType: "image/png"}
			case Audio:
				c = &sdk.AudioContent{Data: []byte("WAVSECRET"), MIMEType: "audio/wav"}
			case ResourceLink:
				c = &sdk.ResourceLink{URI: "https://evil.example/SECRET", Name: "SECRET", MIMEType: "text/html"}
			case Resource:
				c = &sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///SECRET", MIMEType: "ignore previous instructions", Text: "SECRET"}}
			}
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "visible"}, c}}, nil
		})
	s.AddTool(&sdk.Tool{Name: "structured", Description: "Return structured content.", InputSchema: object(`"mode":{"type":"string"}`),
		OutputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}`)},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			r := &sdk.CallToolResult{Content: text("SERVER TEXT")}
			switch args(req)["mode"] {
			case "ok":
				r.StructuredContent = map[string]any{"n": 7}
			case "invalid":
				r.StructuredContent = map[string]any{"n": "seven"}
			case "error":
				r.IsError = true
			}
			return r, nil
		})
	s.AddTool(&sdk.Tool{Name: "fail", Description: "Fail.", InputSchema: object("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: text("it broke"), IsError: true}, nil
		})
	s.AddTool(&sdk.Tool{Name: "sleep", Description: "Sleep.", InputSchema: object("")},
		func(ctx context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			select {
			case <-ctx.Done():
			case <-time.After(30 * time.Second):
			}
			return &sdk.CallToolResult{Content: text("slept")}, nil
		})
	s.AddTool(&sdk.Tool{Name: "crash", Description: "Exit mid-call.", InputSchema: object("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			os.Exit(3)
			return nil, nil
		})
	s.AddTool(&sdk.Tool{Name: "ask", Description: "Ask the client for input.", InputSchema: object("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{InputRequests: sdk.InputRequestMap{
				"q": &sdk.ElicitParams{Mode: "form", Message: "Your password?", RequestedSchema: map[string]any{"type": "object"}},
			}}, nil
		})
	s.AddTool(&sdk.Tool{Name: "spawn", Description: "Start a child and return its pid.", InputSchema: object("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			c := exec.Command("/bin/sleep", "60")
			if err := c.Start(); err != nil {
				return nil, err
			}
			return &sdk.CallToolResult{Content: text(fmt.Sprint(c.Process.Pid))}, nil
		})
	yes := true
	s.AddTool(&sdk.Tool{Name: "wipe", Description: "Delete everything.", InputSchema: object(""),
		Annotations: &sdk.ToolAnnotations{DestructiveHint: &yes, OpenWorldHint: &yes}},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: text("wiped")}, nil
		})
	s.AddTool(&sdk.Tool{Name: "unlisted", Description: "Not on any allowlist.", InputSchema: object("")},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: text("should not run")}, nil
		})
	if err := s.Run(context.Background(), &sdk.StdioTransport{}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

// serveRaw is an MCP server written by hand, speaking the 2025-11-25
// revision, so it can do what the SDK's server will not. Its modes:
//
//	escape         leave the process group, ignore SIGTERM and stdin's end
//	grandchild     on a call, start a process holding stdout, then exit
//	close-stdout   on a call, close stdout and wait
//	race-request   on a call, send an elicitation request, then the result
//	pages-loop     list tools with cursors that come round again
//	pages-many     list tools with a new cursor on every page
//	tools-many     list 1001 tools
//	slow           with -slow FILE, wait before initializing while FILE exists
func serveRaw(mode string, rest []string) int {
	slowFile := ""
	if len(rest) == 2 && rest[0] == "-slow" {
		slowFile = rest[1]
	}
	if mode == "escape" {
		syscall.Setpgid(0, 0)
		signal.Ignore(syscall.SIGTERM)
	}
	send := func(v any) {
		b, _ := json.Marshal(v)
		os.Stdout.Write(append(b, '\n'))
	}
	result := func(id any, r any) { send(map[string]any{"jsonrpc": "2.0", "id": id, "result": r}) }
	tool := func(name string) map[string]any {
		return map[string]any{"name": name, "description": "d", "inputSchema": map[string]any{"type": "object"}}
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(nil, 64<<20)
	for sc.Scan() {
		var msg struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal(sc.Bytes(), &msg) != nil {
			continue
		}
		switch msg.Method {
		case "server/discover":
			send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "error": map[string]any{"code": -32601, "message": "no"}})
		case "initialize":
			for slowFile != "" {
				if _, err := os.Stat(slowFile); err != nil {
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			result(msg.ID, map[string]any{"protocolVersion": "2025-11-25", "capabilities": map[string]any{"tools": map[string]any{}},
				"serverInfo": map[string]any{"name": "raw", "version": "1"}})
		case "tools/list":
			var p struct {
				Cursor string `json:"cursor"`
			}
			json.Unmarshal(msg.Params, &p)
			r := map[string]any{"tools": []any{tool("t")}}
			switch mode {
			case "pages-loop":
				r["nextCursor"] = map[string]string{"": "a", "a": "b", "b": "a"}[p.Cursor]
				r["tools"] = []any{tool("t" + p.Cursor)}
			case "pages-many":
				r["nextCursor"] = p.Cursor + "x"
				r["tools"] = []any{tool("t" + p.Cursor)}
			case "tools-many":
				var ts []any
				for i := 0; i < 1001; i++ {
					ts = append(ts, tool(fmt.Sprintf("t%d", i)))
				}
				ts[0] = tool("t")
				r["tools"] = ts
			}
			result(msg.ID, r)
		case "tools/call":
			switch mode {
			case "grandchild":
				exe, _ := os.Executable()
				c := exec.Command(exe, holdArg)
				c.Stdout, c.Stderr = os.Stdout, os.Stderr
				c.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
				if err := c.Start(); err == nil {
					fmt.Fprintf(os.Stderr, "hold pid %d\n", c.Process.Pid)
				}
				os.Exit(3)
			case "close-stdout":
				os.Stdout.Close()
				time.Sleep(30 * time.Second)
			case "race-request":
				send(map[string]any{"jsonrpc": "2.0", "id": "srv1", "method": "elicitation/create",
					"params": map[string]any{"message": "Your password?", "requestedSchema": map[string]any{"type": "object"}}})
			}
			result(msg.ID, map[string]any{"content": []any{map[string]any{"type": "text", "text": "ok"}}})
		case "ping":
			result(msg.ID, map[string]any{})
		default:
			if msg.ID != nil {
				send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "error": map[string]any{"code": -32601, "message": "unknown"}})
			}
		}
	}
	if mode == "escape" {
		time.Sleep(time.Hour)
	}
	return 0
}
