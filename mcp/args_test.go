package mcp

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

func narrowed() ToolConfig {
	cfg := read()
	cfg.FixedArgs = map[string]json.RawMessage{"send_now": json.RawMessage(`false`)}
	cfg.HiddenArgs = []string{"list"}
	return cfg
}

func TestNarrowedToolHidesItsArgumentsFromTheModel(t *testing.T) {
	r := newRemote(t, false)
	c := connect(t, r.server(map[string]ToolConfig{"echo": narrowed()}, staticToken("good")))
	spec := c.Tools()[0].Spec()
	var schema struct {
		Properties map[string]any `json:"properties"`
		Required   []string       `json:"required"`
	}
	if err := json.Unmarshal(spec.Parameters, &schema); err != nil {
		t.Fatal(err)
	}
	if _, ok := schema.Properties["send_now"]; ok {
		t.Error("the model is shown the fixed argument")
	}
	if _, ok := schema.Properties["list"]; ok {
		t.Error("the model is shown the hidden argument")
	}
	if _, ok := schema.Properties["msg"]; !ok || len(schema.Required) != 1 {
		t.Errorf("the rest of the schema is gone: %s", spec.Parameters)
	}
	// The digest is the server's own definition's.
	listed, _ := List(context.Background(), Server{Name: "remote", URL: r.URL + "/mcp", Token: staticToken("good")})
	if listed[1].Digest != c.Digests()["echo"] {
		t.Error("narrowing changed the pinned digest")
	}
}

func TestNarrowedToolRefusesAndFixesBeforeTheServerSeesTheCall(t *testing.T) {
	r := newRemote(t, false)
	c := connect(t, r.server(map[string]ToolConfig{"echo": narrowed()}, staticToken("good")))
	for _, in := range []string{`{"msg":"a","list":"everyone"}`, `{"msg":"a","send_now":true}`, `{"msg":"a","send_now":false}`,
		`{"msg":"a","LIST":"everyone"}`, `{"msg":"a","Send_Now":true}`, "{\"msg\":\"a\",\"li\u017ft\":\"x\"}"} {
		res := callTool(t, c, "echo", in)
		if !res.IsError || !strings.Contains(res.Content, "does not take the argument") {
			t.Errorf("%s: %+v", in, res)
		}
	}
	for _, in := range []string{`[1]`, `"x"`, `{"msg":"a","msg":"b"}`} {
		if res := callTool(t, c, "echo", in); !res.IsError || res.Content != Failed("echo") {
			t.Errorf("%s: %+v", in, res)
		}
	}
	res := callTool(t, c, "echo", `{"msg":"ok"}`)
	if res.IsError || !strings.Contains(res.Content, `"send_now":false`) {
		t.Fatalf("%+v", res)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) != 1 {
		t.Fatalf("the server saw %d calls: %v", len(r.calls), r.calls)
	}
}

func TestNarrowingThatCannotWorkFailsConnect(t *testing.T) {
	r := newRemote(t, false)
	for why, cfg := range map[string]ToolConfig{
		"no such argument":    {Labels: read().Labels, HiddenArgs: []string{"nope"}},
		"hidden and required": {Labels: read().Labels, HiddenArgs: []string{"msg"}},
		"fixed and hidden": {Labels: read().Labels, HiddenArgs: []string{"list"},
			FixedArgs: map[string]json.RawMessage{"list": json.RawMessage(`"a"`)}},
		"repeated hidden":  {Labels: read().Labels, HiddenArgs: []string{"list", "list"}},
		"fixed not JSON":   {Labels: read().Labels, FixedArgs: map[string]json.RawMessage{"list": json.RawMessage(`{`)}},
		"fixed wrong type": {Labels: read().Labels, FixedArgs: map[string]json.RawMessage{"send_now": json.RawMessage(`"no"`)}},
	} {
		c, err := Connect(context.Background(), r.server(map[string]ToolConfig{"echo": cfg}, staticToken("good")))
		if why == "fixed wrong type" {
			// Found at the call, against the server's schema.
			if err != nil {
				t.Fatal(err)
			}
			res := callTool(t, c, "echo", `{"msg":"a"}`)
			c.Close()
			if !res.IsError || !strings.Contains(res.Log, "input schema") {
				t.Errorf("%s: %+v", why, res)
			}
			continue
		}
		if err == nil {
			c.Close()
			t.Errorf("%s: connected", why)
		}
	}
}

// Whatever the model sends, the server never gets a hidden argument or a
// fixed one other than as fixed, and a call that names neither reaches
// it with exactly the model's arguments plus the fixed ones.
func TestPropertyNarrowing(t *testing.T) {
	params := json.RawMessage(`{"type":"object","properties":{"a":{},"b":{},"c":{},"d":{}}}`)
	rapid.Check(t, func(t *rapid.T) {
		names := []string{"a", "b", "c", "d"}
		fixed := map[string]json.RawMessage{}
		var hidden []string
		for _, n := range names {
			switch rapid.IntRange(0, 2).Draw(t, "role-"+n) {
			case 1:
				fixed[n] = json.RawMessage(rapid.SampledFrom([]string{`1`, `"x"`, `null`, `{"k":[true]}`}).Draw(t, "v-"+n))
			case 2:
				hidden = append(hidden, n)
			}
		}
		_, nw, err := narrowSchema("t", params, ToolConfig{FixedArgs: fixed, HiddenArgs: hidden})
		if err != nil {
			t.Fatal(err)
		}
		if nw == nil {
			return
		}
		in := map[string]any{}
		for _, n := range append(names, "e") {
			if rapid.Bool().Draw(t, "give-"+n) {
				in[n] = rapid.SampledFrom([]any{1.0, "y", nil, true}).Draw(t, "g-"+n)
			}
		}
		raw, _ := json.Marshal(in)
		out, refused, bad := nw.apply(raw)
		if bad != "" {
			t.Fatal(bad)
		}
		named := false
		for k := range in {
			if _, f := fixed[k]; f || slices.Contains(hidden, k) {
				named = true
			}
		}
		if named != (refused != "") {
			t.Fatalf("refused %q for %v", refused, in)
		}
		if named {
			return
		}
		var sent map[string]json.RawMessage
		json.Unmarshal(out, &sent)
		for _, h := range hidden {
			if _, ok := sent[h]; ok {
				t.Fatalf("hidden %s sent", h)
			}
		}
		for k, v := range fixed {
			c, _ := Canonical(v)
			if string(sent[k]) != string(c) {
				t.Fatalf("fixed %s sent as %s", k, sent[k])
			}
		}
		if len(sent) != len(in)+len(fixed) {
			t.Fatalf("sent %v for %v", sent, in)
		}
	})
}

// The agent asks before the policy decides, so a call that names a fixed
// or hidden argument is denied rather than held for the operator.
func TestNarrowedToolChecksInputBeforeThePolicy(t *testing.T) {
	r := newRemote(t, false)
	c := connect(t, r.server(map[string]ToolConfig{"echo": narrowed()}, staticToken("good")))
	ic := c.Tools()[0].(agentloop.InputChecker)
	if why := ic.CheckInput(json.RawMessage(`{"msg":"a","list":"all"}`)); !strings.Contains(why, "does not take the argument list") {
		t.Fatalf("%q", why)
	}
	if why := ic.CheckInput(json.RawMessage(`{"msg":"a"}`)); why != "" {
		t.Fatalf("%q", why)
	}
}
