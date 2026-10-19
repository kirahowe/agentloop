package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/fakeremote"
)

// startRemote serves the fake remote MCP server, which takes the token
// "good", and returns its MCP URL and where it records calls.
func startRemote(t *testing.T) (string, string) {
	t.Helper()
	calls := filepath.Join(t.TempDir(), "calls.jsonl")
	f := fakeremote.New(fakeremote.Options{StaticToken: "good", CallsFile: calls})
	srv := httptest.NewUnstartedServer(nil)
	f.Base = "http://" + srv.Listener.Addr().String()
	srv.Config.Handler = f.Handler()
	srv.Start()
	t.Cleanup(srv.Close)
	return f.Base + "/mcp", calls
}

// remoteFixture is a request whose tools come from the fake remote
// server. Its credentials command prints "stale" the first time and
// "good" after, so the first request is refused and the token renewed.
func remoteFixture(t *testing.T, api *fakeAPI, url string) *fixture {
	f := newFixture(t, api, "")
	delete(f.req, "tool_command")
	delete(f.req, "tool_env")
	flag := filepath.Join(f.dir, "renewed")
	f.req["mcp_servers"] = map[string]any{"remote": map[string]any{
		"url":                 url,
		"credentials_command": []string{"/bin/sh", "-c", `if [ -e "$1" ]; then echo good; else : > "$1"; echo stale; fi`, "sh", flag},
	}}
	f.req["tools"] = []map[string]any{
		{"name": "send", "protocol": "mcp-http", "server": "remote", "tool": "send_campaign",
			"fixed_args": map[string]any{"send_now": false}, "hidden_args": []string{"subject"}},
	}
	f.req["policy"] = map[string]any{
		"labels": []string{"private"},
		"tools":  map[string]any{"send": map[string]any{"class": "read"}},
	}
	return f
}

func TestRemoteMCPRunNarrowsArguments(t *testing.T) {
	url, calls := startRemote(t)
	api := newFakeAPI(t,
		toolReply("toolu_1", "send", `{"group":"Kira only","subject":"Hi"}`),
		toolReply("toolu_2", "send", `{"group":"Kira only"}`),
		textReply("done"))
	f := remoteFixture(t, api, url)
	res := f.run()
	out := res.outcome(t)
	if res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d %v\n%s", res.code, out, res.stderr)
	}
	trs := toolResults(res)
	if len(trs) != 2 {
		t.Fatalf("results %v", trs)
	}
	first := trs[0]["result"].(map[string]any)
	if first["is_error"] != true || !strings.Contains(first["content"].(string), "does not take the argument subject") {
		t.Fatalf("hidden argument: %v", first)
	}
	b, err := os.ReadFile(calls)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 1 {
		t.Fatalf("the server saw %d calls: %s", len(lines), b)
	}
	var call struct {
		Args map[string]any `json:"args"`
	}
	json.Unmarshal([]byte(lines[0]), &call)
	if call.Args["send_now"] != false || call.Args["group"] != "Kira only" || len(call.Args) != 2 {
		t.Fatalf("the server got %v", call.Args)
	}
	// The model was shown neither the fixed nor the hidden argument.
	for _, body := range api.requests() {
		var req struct {
			Tools json.RawMessage `json:"tools"`
		}
		json.Unmarshal([]byte(body), &req)
		if s := string(req.Tools); !strings.Contains(s, "group") || strings.Contains(s, "send_now") || strings.Contains(s, "subject") {
			t.Fatalf("the model was shown %s", s)
		}
	}
	sess, _ := os.ReadFile(f.session)
	for what, text := range map[string]string{"stderr": res.stderr, "stdout": res.stdout, "session": string(sess)} {
		if strings.Contains(text, `"good"`) || strings.Contains(text, "Bearer") {
			t.Fatalf("the token reached the %s", what)
		}
	}
}

func TestMCPToolsListsARemoteServer(t *testing.T) {
	url, _ := startRemote(t)
	in, _ := json.Marshal(map[string]string{"url": url, "token": "good"})
	var out, errb bytes.Buffer
	if code := realMain([]string{"mcp-tools"}, bytes.NewReader(in), &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	var got struct {
		Tools []struct {
			Name        string         `json:"name"`
			Digest      string         `json:"digest"`
			Annotations map[string]any `json:"annotations"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil || len(got.Tools) != 4 {
		t.Fatalf("%v: %s", err, out.String())
	}
	if got.Tools[0].Name != "add_note" || len(got.Tools[0].Digest) != 64 || got.Tools[2].Annotations["readOnlyHint"] != true {
		t.Fatalf("%+v", got.Tools)
	}
	for _, bad := range []string{`{"url":"http://example.com/mcp"}`, `{"url":"` + url + `","extra":1}`, `{"url":"` + url + `","token":"bad"}`} {
		out.Reset()
		errb.Reset()
		if code := realMain([]string{"mcp-tools"}, strings.NewReader(bad), &out, &errb); code == 0 {
			t.Errorf("%s: listed %s", bad, out.String())
		}
	}
}

// fixedGroupFixture is a request whose act tool send has its group fixed
// by the operator to group.
func fixedGroupFixture(t *testing.T, api *fakeAPI, url, group string, consent []map[string]any) *fixture {
	f := remoteFixture(t, api, url)
	f.req["tools"] = []map[string]any{
		{"name": "send", "protocol": "mcp-http", "server": "remote", "tool": "send_campaign",
			"fixed_args": map[string]any{"group": group, "send_now": false}},
	}
	entry := map[string]any{"class": "act"}
	if consent != nil {
		entry["consent"] = consent
	}
	f.req["policy"].(map[string]any)["tools"] = map[string]any{"send": entry}
	return f
}

func serverCalls(t *testing.T, path string) []map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if l == "" {
			continue
		}
		var call struct {
			Args map[string]any `json:"args"`
		}
		json.Unmarshal([]byte(l), &call)
		out = append(out, call.Args)
	}
	return out
}

// A consent rule can name an argument the operator fixed: consent sees
// the input the call runs on, the model's with the fixed arguments.
func TestConsentSeesFixedArguments(t *testing.T) {
	url, calls := startRemote(t)
	api := newFakeAPI(t, toolReply("toolu_1", "send", `{"subject":"Hi"}`), textReply("done"))
	f := fixedGroupFixture(t, api, url, "Kira only", []map[string]any{{"group": "Kira only"}})
	res := f.run()
	if out := res.outcome(t); res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d %v\n%s", res.code, out, res.stderr)
	}
	got := serverCalls(t, calls)
	if len(got) != 1 || got[0]["group"] != "Kira only" {
		t.Fatalf("the server saw %v", got)
	}
	// With another group fixed, the same rule does not cover the call.
	url, calls = startRemote(t)
	api = newFakeAPI(t, toolReply("toolu_1", "send", `{"subject":"Hi"}`), textReply("done"))
	f = fixedGroupFixture(t, api, url, "Public list", []map[string]any{{"group": "Kira only"}})
	if out := f.run().outcome(t); out["reason"] != "held" {
		t.Fatalf("outcome %v", out)
	}
	if got := serverCalls(t, calls); len(got) != 0 {
		t.Fatalf("the server saw %v", got)
	}
}

// A held call shows, and its approval is bound to, the input it would
// run on. When the operator changes a fixed argument before the approved
// call resumes, it is not run, and the model is told why.
func TestApprovalIsBoundToFixedArguments(t *testing.T) {
	url, calls := startRemote(t)
	api := newFakeAPI(t, toolReply("toolu_1", "send", `{"subject":"Hi"}`), textReply("done"), textReply("done"))
	f := fixedGroupFixture(t, api, url, "Kira only", nil)
	out := f.run().outcome(t)
	held, _ := out["held"].([]any)
	if out["reason"] != "held" || len(held) != 1 {
		t.Fatalf("outcome %v", out)
	}
	h := held[0].(map[string]any)
	want := json.RawMessage(`{"group":"Kira only","send_now":false,"subject":"Hi"}`)
	if h["input_sha256"] != agentloop.InputDigest(want) {
		t.Fatalf("the digest is not of the effective input: %v", h)
	}
	if b, _ := json.Marshal(h["effective_input"]); agentloop.InputDigest(b) != agentloop.InputDigest(want) {
		t.Fatalf("effective input %s", b)
	}
	if fx, _ := h["fixed_args"].(map[string]any); fx["group"] != "Kira only" || fx["send_now"] != false || len(fx) != 2 {
		t.Fatalf("fixed args %v", h["fixed_args"])
	}
	// The operator changes the group, then the approval is carried back.
	f.req["tools"].([]map[string]any)[0]["fixed_args"] = map[string]any{"group": "Public list", "send_now": false}
	f.req["prompt"] = ""
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_1", "verdict": "allow", "input_sha256": h["input_sha256"]}}
	res := f.run()
	if out := res.outcome(t); res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d %v\n%s", res.code, out, res.stderr)
	}
	if got := serverCalls(t, calls); len(got) != 0 {
		t.Fatalf("the call ran with changed settings: %v", got)
	}
	var reason any
	for _, l := range res.lines {
		if d, ok := l["decision"].(map[string]any); ok && d["call_id"] == "toolu_1" {
			reason = d["reason"]
		}
	}
	if reason != agentloop.SettingsChanged {
		t.Fatalf("the deny says %v", reason)
	}
	// The same resolution again is the same deny, not an error.
	if res := f.run(); res.code != 0 {
		t.Fatalf("exit %d %s", res.code, res.stderr)
	}
	if got := serverCalls(t, calls); len(got) != 0 {
		t.Fatalf("the server saw %v", got)
	}
}

// With the settings unchanged, the approved call runs on what was shown.
func TestApprovedCallRunsWithItsFixedArguments(t *testing.T) {
	url, calls := startRemote(t)
	api := newFakeAPI(t, toolReply("toolu_1", "send", `{"subject":"Hi"}`), textReply("done"))
	f := fixedGroupFixture(t, api, url, "Kira only", nil)
	out := f.run().outcome(t)
	h := out["held"].([]any)[0].(map[string]any)
	f.req["prompt"] = ""
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_1", "verdict": "allow", "input_sha256": h["input_sha256"]}}
	res := f.run()
	if out := res.outcome(t); res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d %v\n%s", res.code, out, res.stderr)
	}
	got := serverCalls(t, calls)
	if len(got) != 1 || got[0]["group"] != "Kira only" || got[0]["subject"] != "Hi" {
		t.Fatalf("the server saw %v", got)
	}
	// The digest of the model's input alone does not approve it.
	url, _ = startRemote(t)
	api = newFakeAPI(t, toolReply("toolu_1", "send", `{"subject":"Hi"}`), textReply("done"))
	f = fixedGroupFixture(t, api, url, "Kira only", nil)
	f.run()
	f.req["prompt"] = ""
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_1", "verdict": "allow", "input_sha256": agentloop.InputDigest(json.RawMessage(`{"subject":"Hi"}`))}}
	if res := f.run(); res.code != 2 {
		t.Fatalf("exit %d", res.code)
	}
}
