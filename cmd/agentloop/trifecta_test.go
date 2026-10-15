package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
)

const envelopeScript = `#!/bin/sh
in=$(cat)
printf '%s %s\n' "$1" "$in" >> "$CALLS"
case "$1" in
read_mail)
	echo "STDERR-FROM-MAIL" >&2
	printf '{"content":"mail body","is_error":false,"labels":["contaminated"]}' ;;
web_post)
	printf '{"content":"posted","labels":[]}' ;;
broken)
	echo "STDERR-FROM-BROKEN" >&2
	echo "STDOUT-FROM-BROKEN not an envelope" ;;
esac
`

// trifectaFixture is a request with a real policy: the session starts
// private, reading mail contaminates it, and posting to the web is egress.
func trifectaFixture(t *testing.T, api *fakeAPI) (*fixture, string) {
	f := newFixture(t, api, envelopeScript)
	calls := filepath.Join(f.dir, "calls.log")
	f.req["tool_env"] = map[string]string{"PATH": "/usr/bin:/bin", "CALLS": calls}
	f.req["tools"] = []map[string]any{
		{"name": "read_mail", "parameters": map[string]any{"type": "object"}},
		{"name": "web_post", "parameters": map[string]any{"type": "object"}},
		{"name": "broken", "parameters": map[string]any{"type": "object"}},
	}
	f.req["policy"] = map[string]any{
		"labels": []string{"private"},
		"tools": map[string]any{
			"read_mail": map[string]any{"class": "read", "labels": []string{"contaminated"}},
			"web_post":  map[string]any{"class": "egress"},
			"broken":    map[string]any{"class": "read"},
		},
	}
	f.req["tool_protocol"] = "envelope"
	return f, calls
}

func readCalls(t *testing.T, path string) string {
	b, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	return string(b)
}

func TestTrifectaHoldThenResolveAllow(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_1", "read_mail", `{}`),
		toolReply("toolu_2", "web_post", `{"body":"secrets"}`),
		textReply("done"),
	)
	f, calls := trifectaFixture(t, api)
	res := f.run()
	out := res.outcome(t)
	if res.code != 0 || out["reason"] != "held" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	held := out["held"].([]any)
	if len(held) != 1 || held[0].(map[string]any)["id"] != "toolu_2" {
		t.Fatalf("held %v", held)
	}
	digest := held[0].(map[string]any)["input_sha256"].(string)
	if want := agentloop.InputDigest(json.RawMessage(`{"body":"secrets"}`)); digest != want {
		t.Fatalf("digest %s, want %s", digest, want)
	}
	if labels := fmt.Sprint(out["labels"]); labels != "[contaminated private]" {
		t.Fatalf("outcome labels %s", labels)
	}
	got := readCalls(t, calls)
	if strings.Contains(got, "web_post") {
		t.Fatalf("the held egress call ran: %s", got)
	}
	if !strings.Contains(got, `read_mail {"input":{},"labels":["contaminated","private"],"call_id":"toolu_1"}`) {
		t.Fatalf("read_mail's stdin: %s", got)
	}
	// The mail tool's stderr went to the log entry, not to the model.
	if !strings.Contains(res.stdout, "STDERR-FROM-MAIL") {
		t.Fatal("stderr is not in the operator's log")
	}
	for _, body := range api.requests() {
		if strings.Contains(body, "STDERR-FROM-MAIL") {
			t.Fatal("stderr reached the model")
		}
	}

	// A run with the same session and no resolve is still held.
	f.req["prompt"] = ""
	if out := f.run().outcome(t); out["reason"] != "held" {
		t.Fatalf("outcome %v", out)
	}
	// An unknown call id is refused before anything is written.
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_nope", "verdict": "allow", "input_sha256": digest}}
	if res := f.run(); res.code != 2 || res.stdout != "" {
		t.Fatalf("exit %d stdout %q", res.code, res.stdout)
	}
	// So is a call that is not held.
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_1", "verdict": "allow", "input_sha256": agentloop.InputDigest(json.RawMessage(`{}`))}}
	if res := f.run(); res.code != 2 || res.stdout != "" {
		t.Fatalf("exit %d stdout %q", res.code, res.stdout)
	}
	// So is a digest that is not the held input's.
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_2", "verdict": "allow", "input_sha256": agentloop.InputDigest(json.RawMessage(`{"body":"other"}`))}}
	if res := f.run(); res.code != 2 || res.stdout != "" {
		t.Fatalf("exit %d stdout %q", res.code, res.stdout)
	}

	f.req["resolve"] = []map[string]any{{"call_id": "toolu_2", "verdict": "allow", "input_sha256": digest}}
	res = f.run()
	out = res.outcome(t)
	if res.code != 0 || out["reason"] != "completed" || out["text"] != "done" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	if first := res.lines[0]; first["type"] != "decision" || first["decision"].(map[string]any)["resolved"] != true {
		t.Fatalf("first line %v", first)
	}
	if got := readCalls(t, calls); !strings.Contains(got, `web_post {"input":{"body":"secrets"},"labels":["contaminated","private"],"call_id":"toolu_2"}`) {
		t.Fatalf("calls: %s", got)
	}
	// Run again with the same resolve, as after a crash: it is a no-op.
	res = f.run()
	if out := res.outcome(t); res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	if n := strings.Count(readCalls(t, calls), "web_post"); n != 1 {
		t.Fatalf("web_post ran %d times", n)
	}
}

func TestTrifectaResolveDeny(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_1", "read_mail", `{}`),
		toolReply("toolu_2", "web_post", `{}`),
		textReply("ok"),
	)
	f, calls := trifectaFixture(t, api)
	out := f.run().outcome(t)
	if out["reason"] != "held" {
		t.Fatalf("outcome %v", out)
	}
	digest := out["held"].([]any)[0].(map[string]any)["input_sha256"]
	f.req["prompt"] = ""
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_2", "verdict": "deny", "reason": "not now", "input_sha256": digest}}
	res := f.run()
	if out := res.outcome(t); res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d outcome %v %s", res.code, out, res.stderr)
	}
	if strings.Contains(readCalls(t, calls), "web_post") {
		t.Fatal("a denied call ran")
	}
	reqs := api.requests()
	if last := reqs[len(reqs)-1]; !strings.Contains(last, "not now") {
		t.Fatalf("the model was not told the refusal: %s", last)
	}
}

// Egress before anything contaminated is not held.
func TestTrifectaEgressWhileOnlyPrivate(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "web_post", `{}`), textReply("done"))
	f, calls := trifectaFixture(t, api)
	if out := f.run().outcome(t); out["reason"] != "completed" {
		t.Fatalf("outcome %v", out)
	}
	if !strings.Contains(readCalls(t, calls), `web_post {"input":{},"labels":["private"],"call_id":"toolu_1"}`) {
		t.Fatalf("calls: %s", readCalls(t, calls))
	}
}

// A tool that breaks the protocol shows the model a fixed text, and
// nothing of what it wrote.
func TestTrifectaBrokenEnvelope(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "broken", `{}`), textReply("done"))
	f, _ := trifectaFixture(t, api)
	res := f.run()
	if out := res.outcome(t); out["reason"] != "completed" {
		t.Fatalf("outcome %v", out)
	}
	reqs := api.requests()
	last := reqs[len(reqs)-1]
	if !strings.Contains(last, "The tool broken failed.") {
		t.Fatalf("request: %s", last)
	}
	for _, leak := range []string{"STDERR-FROM-BROKEN", "STDOUT-FROM-BROKEN"} {
		if strings.Contains(last, leak) {
			t.Fatalf("%s reached the model", leak)
		}
	}
	if !strings.Contains(res.stdout, "STDERR-FROM-BROKEN") {
		t.Fatal("the operator's log lacks stderr")
	}
}
