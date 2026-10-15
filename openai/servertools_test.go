package openai

import (
	"encoding/json"
	"net/http"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// No request ever carries a provider-side tool (web search, file search,
// code interpreter, MCP and the like): every tool sent is one of the
// agent's own, as a function, whatever it is called.
func TestNoServerSideTools(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w, completed(`[]`)...)
	})
	m := model(s)
	rapid.Check(t, func(rt *rapid.T) {
		names := rapid.SliceOfNDistinct(rapid.SampledFrom([]string{
			"web_search", "web_search_preview", "file_search", "code_interpreter", "mcp", "image_generation", "computer_use", "get",
		}), 0, 4, rapid.ID).Draw(rt, "names")
		req := &agentloop.Request{Messages: []agentloop.Message{agentloop.Text(agentloop.User, "hi")}}
		for _, n := range names {
			req.Tools = append(req.Tools, agentloop.ToolSpec{Name: n})
		}
		gen(t, m, req)
		b := s.lastBody(t)
		raw, _ := json.Marshal(b["tools"])
		var tools []map[string]json.RawMessage
		json.Unmarshal(raw, &tools)
		if len(tools) != len(names) {
			rt.Fatalf("%d tools sent for %d", len(tools), len(names))
		}
		for i, tool := range tools {
			if string(tool["type"]) != `"function"` || string(tool["name"]) != `"`+names[i]+`"` {
				rt.Fatalf("tool %d: %v", i, tool)
			}
		}
		if _, ok := b["tool_choice"]; ok {
			rt.Fatal("the request sets tool_choice")
		}
	})
}
