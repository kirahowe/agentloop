package acpagent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// --- the pure parts ---

// The model is one select in the "model" category: its id is the only
// value and the current one, and its display name names it, falling back
// to the id.
func TestConfigOptions(t *testing.T) {
	opts := ConfigOptions(agentloop.ModelInfo{ID: "claude-opus-5-5", DisplayName: "Claude Opus 5.5", ContextWindow: 1000000})
	b, _ := json.Marshal(opts)
	want := `[{"category":"model","currentValue":"claude-opus-5-5","id":"model","name":"Model",` +
		`"options":[{"name":"Claude Opus 5.5","value":"claude-opus-5-5"}],"type":"select"}]`
	if string(b) != want {
		t.Fatalf("got  %s\nwant %s", b, want)
	}
	b, _ = json.Marshal(ConfigOptions(agentloop.ModelInfo{ID: "gpt-6"}))
	if !strings.Contains(string(b), `"options":[{"name":"gpt-6","value":"gpt-6"}]`) {
		t.Fatalf("no display name: %s", b)
	}
	if ConfigOptions(agentloop.ModelInfo{}) != nil {
		t.Fatal("options for a model with no id")
	}
}

// The request's id is kept even when the provider resolves it to another
// (an alias to a dated id, say), and a negative window is unknown.
func TestDescribed(t *testing.T) {
	got := Described("claude-opus-5-5", agentloop.ModelInfo{ID: "claude-opus-5-5-20260801", DisplayName: "Claude Opus 5.5", ContextWindow: 1000000})
	if got != (agentloop.ModelInfo{ID: "claude-opus-5-5", DisplayName: "Claude Opus 5.5", ContextWindow: 1000000}) {
		t.Fatalf("%+v", got)
	}
	if got := Described("m", agentloop.ModelInfo{ContextWindow: -5}); got.ContextWindow != 0 {
		t.Fatalf("%+v", got)
	}
}

// For any model and any set_config_option request, the request succeeds
// exactly when it sets the model option to the model it already is, and
// then answers with the complete, unchanged list. The options always
// validate against ACP's schema, and the current value is always one of
// the values offered (what the TCK's ACP-CONFIG-001 checks).
func TestPropertySetConfigOption(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		info := agentloop.ModelInfo{
			ID:            rapid.SampledFrom([]string{"", "claude-opus-5-5", "gpt-6", "mock"}).Draw(t, "id"),
			DisplayName:   rapid.SampledFrom([]string{"", "Claude Opus 5.5", "é \U0001F600"}).Draw(t, "name"),
			ContextWindow: rapid.Int64Range(-1, 2000000).Draw(t, "window"),
		}
		opts := ConfigOptions(info)
		if err := validate("NewSessionResponse", acp.NewSessionResponse{SessionId: "s1", ConfigOptions: opts}); err != nil {
			t.Fatal(err)
		}
		for _, o := range opts {
			if o.Select == nil || o.Select.Options.Ungrouped == nil {
				t.Fatalf("not a flat select: %+v", o)
			}
			found := false
			for _, v := range *o.Select.Options.Ungrouped {
				found = found || v.Value == o.Select.CurrentValue
			}
			if !found {
				t.Fatalf("current value %q is not offered", o.Select.CurrentValue)
			}
		}

		configID := acp.SessionConfigId(rapid.SampledFrom([]string{"model", "mode", ""}).Draw(t, "configId"))
		var req acp.SetSessionConfigOptionRequest
		var accept bool
		if rapid.Bool().Draw(t, "boolean") {
			req.Boolean = &acp.SetSessionConfigOptionBoolean{SessionId: "s1", ConfigId: configID, Type: "boolean", Value: rapid.Bool().Draw(t, "b")}
		} else {
			value := rapid.SampledFrom([]string{info.ID, "claude-fable-5-1", ""}).Draw(t, "value")
			req.ValueId = &acp.SetSessionConfigOptionValueId{SessionId: "s1", ConfigId: configID, Value: acp.SessionConfigValueId(value)}
			accept = info.ID != "" && configID == ModelConfigID && value == info.ID
		}
		resp, rpcErr := SetConfigOption(info, req)
		if accept {
			if rpcErr != nil {
				t.Fatalf("refused setting the model to itself: %+v", rpcErr)
			}
			if !reflect.DeepEqual(resp.ConfigOptions, opts) {
				t.Fatalf("answered %+v, want the options as they are", resp.ConfigOptions)
			}
			if err := validate("SetSessionConfigOptionResponse", resp); err != nil {
				t.Fatal(err)
			}
			return
		}
		if rpcErr == nil || rpcErr.Code != -32602 || rpcErr.Message == "" {
			t.Fatalf("accepted %+v for %+v: %+v", req, info, resp)
		}
	})
}

// --- through the server ---

// countingModel is a describedModel that counts its descriptions.
type countingModel struct {
	describedModel
	n *atomic.Int32
}

func (c countingModel) Describe(ctx context.Context, id string) (agentloop.ModelInfo, error) {
	c.n.Add(1)
	return c.describedModel.Describe(ctx, id)
}

// The provider's description reaches the client: session/new, load and
// resume carry the model as a config option, usage_update's size is the
// window the provider gave, and the model is described once however many
// sessions are opened. Setting the model to itself answers the options;
// switching is refused.
func TestServerDescribesTheModel(t *testing.T) {
	var n atomic.Int32
	info := agentloop.ModelInfo{ID: "claude-opus-5-5", DisplayName: "Claude Opus 5.5", ContextWindow: 1000000}
	model := countingModel{describedModel{fake.Script(fake.Text("One."), fake.Text("Two.")), info, nil}, &n}
	h := serve(t, &agentloop.Agent{Model: model, ModelID: "claude-opus-5-5"}, Options{}, false)
	want := ConfigOptions(info)

	resp, err := h.client.NewSession(h.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp.ConfigOptions, want) {
		t.Fatalf("session/new options %+v", resp.ConfigOptions)
	}
	id := resp.SessionId
	if _, err := h.prompt(id, "hi"); err != nil {
		t.Fatal(err)
	}
	var sizes []uint64
	for _, u := range h.sessionUpdates(id) {
		if u.UsageUpdate != nil {
			sizes = append(sizes, u.UsageUpdate.Size)
		}
	}
	if len(sizes) != 1 || sizes[0] != 1000000 {
		t.Fatalf("usage sizes %v, want the provider's window", sizes)
	}

	set, err := h.client.SetSessionConfigOption(h.ctx(), acp.SetSessionConfigOptionRequest{ValueId: &acp.SetSessionConfigOptionValueId{
		SessionId: id, ConfigId: ModelConfigID, Value: "claude-opus-5-5"}})
	if err != nil || !reflect.DeepEqual(set.ConfigOptions, want) {
		t.Fatalf("setting the model to itself: %+v, %v", set, err)
	}
	_, err = h.client.SetSessionConfigOption(h.ctx(), acp.SetSessionConfigOptionRequest{ValueId: &acp.SetSessionConfigOptionValueId{
		SessionId: id, ConfigId: ModelConfigID, Value: "claude-fable-5-1"}})
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32602 || !strings.Contains(re.Message, "not supported") {
		t.Fatalf("switching models: %v", err)
	}
	_, err = h.client.SetSessionConfigOption(h.ctx(), acp.SetSessionConfigOptionRequest{ValueId: &acp.SetSessionConfigOptionValueId{
		SessionId: "00000000-0000-4000-8000-000000000000", ConfigId: ModelConfigID, Value: "claude-opus-5-5"}})
	if !isRPC(err, -32002, "Resource not found") {
		t.Fatalf("an unknown session: %v", err)
	}

	if err := h.close(id); err != nil {
		t.Fatal(err)
	}
	load, err := h.client.LoadSession(h.ctx(), acp.LoadSessionRequest{SessionId: id, Cwd: "/", McpServers: []acp.McpServer{}})
	if err != nil || !reflect.DeepEqual(load.ConfigOptions, want) {
		t.Fatalf("session/load: %+v, %v", load, err)
	}
	if err := h.close(id); err != nil {
		t.Fatal(err)
	}
	resume, err := h.client.ResumeSession(h.ctx(), acp.ResumeSessionRequest{SessionId: id, Cwd: "/"})
	if err != nil || !reflect.DeepEqual(resume.ConfigOptions, want) {
		t.Fatalf("session/resume: %+v, %v", resume, err)
	}
	h.newSession()
	if got := n.Load(); got != 1 {
		t.Fatalf("described %d times, want once", got)
	}
}

// A model the provider cannot describe is still served: one line goes to
// the log, the client sees the model by its id, and no usage_update is
// sent, since its size is unknown.
func TestServerUndescribedModel(t *testing.T) {
	model := describedModel{fake.Script(fake.Text("One.")), agentloop.ModelInfo{}, &agentloop.ModelError{Kind: agentloop.NotFound, Status: 404, Message: "no such model"}}
	h := serve(t, &agentloop.Agent{Model: model, ModelID: "claude-x"}, Options{}, false)
	resp, err := h.client.NewSession(h.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp.ConfigOptions, ConfigOptions(agentloop.ModelInfo{ID: "claude-x"})) {
		t.Fatalf("options %+v", resp.ConfigOptions)
	}
	if _, err := h.prompt(resp.SessionId, "hi"); err != nil {
		t.Fatal(err)
	}
	for _, u := range h.sessionUpdates(resp.SessionId) {
		if u.UsageUpdate != nil {
			t.Fatalf("usage_update with no known window: %+v", u.UsageUpdate)
		}
	}
	if log := h.logged(); strings.Count(log, "cannot describe model claude-x") != 1 || !strings.Contains(log, "no such model") {
		t.Fatalf("log %q", log)
	}
}
