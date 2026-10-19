package quarantine

import (
	"encoding/json"
	"maps"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// narrowedInner is an inner tool whose "dest" the operator fixed.
type narrowedInner struct {
	inner
	dest string
}

func (t *narrowedInner) EffectiveInput(in json.RawMessage) (json.RawMessage, map[string]json.RawMessage, bool) {
	var args map[string]json.RawMessage
	if json.Unmarshal(in, &args) != nil || args == nil {
		return nil, nil, false
	}
	fixed := map[string]json.RawMessage{"dest": json.RawMessage(`"` + t.dest + `"`)}
	maps.Copy(args, fixed)
	out, _ := json.Marshal(args)
	return out, fixed, true
}

// Regression (Codex review): a call through the reader to an inner tool
// with fixed arguments was held and approved on the outer input only, so
// a changed destination ran under the old approval.
func TestReaderBindsTheInnerToolsFixedArguments(t *testing.T) {
	in := &narrowedInner{inner: inner{name: "fetch"}, dest: "a.example"}
	r := newTestReader(t, fake.Script(), in)
	call := agentloop.ToolCall{ID: "c1", Name: ToolName, Input: readerInput("fetch", closedSchema)}
	before := agentloop.HeldCallOf([]agentloop.Tool{r}, call)
	if string(before.FixedArgs["dest"]) != `"a.example"` {
		t.Fatalf("held %+v", before)
	}
	var eff struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	if json.Unmarshal(before.EffectiveInput, &eff) != nil || eff.Tool != "fetch" ||
		agentloop.InputDigest(eff.Input) != agentloop.InputDigest(json.RawMessage(`{"dest":"a.example","id":"m-1"}`)) {
		t.Fatalf("effective input %s", before.EffectiveInput)
	}
	in.dest = "b.example"
	if after := agentloop.HeldCallOf([]agentloop.Tool{r}, call); after.InputSHA256 == before.InputSHA256 {
		t.Fatal("a changed inner destination keeps the approval's digest")
	}
	plain := newTestReader(t, fake.Script(), &inner{name: "fetch"})
	if h := agentloop.HeldCallOf([]agentloop.Tool{plain}, call); h.FixedArgs != nil || h.InputSHA256 != agentloop.InputDigest(call.Input) {
		t.Fatalf("a reader over a plain tool: %+v", h)
	}
}
