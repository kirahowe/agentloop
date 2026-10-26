package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
		"tools":  map[string]any{"send": map[string]any{}},
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
