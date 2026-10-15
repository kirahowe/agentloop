package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// No request ever carries a provider-side tool (web search, code
// execution, MCP servers and the like): every tool sent is one of the
// agent's own, as a custom tool, whatever it is called. A server-side
// tool would be egress the policy never sees.
func TestNoServerSideTools(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "hi") })
	m := model(s, Config{RefusalFallback: true})
	rapid.Check(t, func(rt *rapid.T) {
		names := rapid.SliceOfNDistinct(rapid.SampledFrom([]string{
			"web_search", "web_fetch", "code_execution", "bash", "text_editor", "computer", "memory", "mcp", "get", "x",
		}), 0, 4, rapid.ID).Draw(rt, "names")
		req := userReq("hello")
		for _, n := range names {
			req.Tools = append(req.Tools, agentloop.ToolSpec{Name: n, Description: "d"})
		}
		if _, err := m.Generate(context.Background(), req, nil); err != nil {
			rt.Fatal(err)
		}
		var b map[string]json.RawMessage
		if err := json.Unmarshal(s.last(t).body, &b); err != nil {
			rt.Fatal(err)
		}
		for _, k := range []string{"mcp_servers", "container", "tool_choice"} {
			if _, ok := b[k]; ok {
				rt.Fatalf("the request has %s", k)
			}
		}
		var tools []map[string]json.RawMessage
		json.Unmarshal(b["tools"], &tools)
		if len(tools) != len(names) {
			rt.Fatalf("%d tools sent for %d", len(tools), len(names))
		}
		for i, tool := range tools {
			var name string
			json.Unmarshal(tool["name"], &name)
			if name != names[i] {
				rt.Fatalf("tool %d is %q, want %q", i, name, names[i])
			}
			if typ, ok := tool["type"]; ok && string(typ) != `"custom"` {
				rt.Fatalf("tool %s has type %s", name, typ)
			}
			for k := range tool {
				switch k {
				case "name", "description", "input_schema", "type", "cache_control":
				default:
					rt.Fatalf("tool %s has the field %s", name, k)
				}
			}
		}
	})
}
