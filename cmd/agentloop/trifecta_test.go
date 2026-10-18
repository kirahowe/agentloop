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

// Regression: an unrestricted request could carry on a session held under
// a policy, and drop everything the policy stood for.
func TestUnrestrictedRequestRefusesAPolicedSession(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "read_mail", `{}`), toolReply("toolu_2", "web_post", `{}`), textReply("done"))
	f, _ := trifectaFixture(t, api)
	if out := f.run().outcome(t); out["reason"] != "held" {
		t.Fatalf("outcome %v", out)
	}
	f.req["prompt"] = ""
	f.req["policy"] = map[string]any{"unrestricted": true}
	delete(f.req, "tool_protocol")
	res := f.run()
	if res.code != 2 || res.stdout != "" || !strings.Contains(res.stderr, "policy") {
		t.Fatalf("exit %d stdout %q stderr %q", res.code, res.stdout, res.stderr)
	}
}

// Regression: an allow for a call the policy now denies outright (its
// tool was removed) made run exit 2 with no outcome, every time it was
// sent. It is now recorded as a deny and the run carries on; sending it
// again is the same deny. A wrong digest still exits 2.
func TestResolveAllowForAToolThatIsGone(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_1", "read_mail", `{}`),
		toolReply("toolu_2", "web_post", `{}`),
		textReply("ok"), textReply("ok again"),
	)
	f, calls := trifectaFixture(t, api)
	out := f.run().outcome(t)
	if out["reason"] != "held" {
		t.Fatalf("outcome %v", out)
	}
	digest := out["held"].([]any)[0].(map[string]any)["input_sha256"]
	f.req["prompt"] = ""
	f.req["tools"] = f.req["tools"].([]map[string]any)[:1]
	delete(f.req["policy"].(map[string]any)["tools"].(map[string]any), "web_post")
	delete(f.req["policy"].(map[string]any)["tools"].(map[string]any), "broken")
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_2", "verdict": "allow", "input_sha256": "0000"}}
	if res := f.run(); res.code != 2 || res.stdout != "" {
		t.Fatalf("a wrong digest: exit %d stdout %q", res.code, res.stdout)
	}
	f.req["resolve"] = []map[string]any{{"call_id": "toolu_2", "verdict": "allow", "input_sha256": digest}}
	for i := range 2 {
		res := f.run()
		out := res.outcome(t)
		if res.code != 0 || out["reason"] != "completed" {
			t.Fatalf("run %d: exit %d outcome %v stderr %s", i, res.code, out, res.stderr)
		}
	}
	if strings.Contains(readCalls(t, calls), "web_post") {
		t.Fatal("the call ran")
	}
	reqs := api.requests()
	if !strings.Contains(reqs[2], "The tool is no longer allowed.") {
		t.Fatalf("the model was not told: %s", reqs[2])
	}
}

// An outcome with no key still reports the session's labels; when the
// session cannot be read it leaves labels out instead of claiming none.
func TestNoKeyOutcomeLabels(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "read_mail", `{}`), textReply("done"))
	f, _ := trifectaFixture(t, api)
	if out := f.run().outcome(t); out["reason"] != "completed" {
		t.Fatalf("outcome %v", out)
	}
	f.req["api_key_command"] = []string{"/bin/sh", "-c", "exit 1"}
	if out := f.run().outcome(t); fmt.Sprint(out["labels"]) != "[contaminated private]" {
		t.Fatalf("outcome %v", out)
	}
	b, _ := os.ReadFile(f.session)
	os.WriteFile(f.session, append([]byte("not json\n"), b...), 0o600)
	out := f.run().outcome(t)
	if _, ok := out["labels"]; ok || out["reason"] != "model_error" {
		t.Fatalf("an unreadable session gave %v", out)
	}
	f.req["session"] = filepath.Join(f.dir, "none", "s.jsonl")
	if out := f.run().outcome(t); fmt.Sprint(out["labels"]) != "[]" {
		t.Fatalf("a missing session gave %v", out)
	}
}

const forbidScript = `#!/bin/sh
cat >/dev/null
case "$1" in
read_secret) printf '{"content":"SECRET-TEXT","labels":["private"]}' ;;
read_web) printf '{"content":"a web page","labels":["contaminated"]}' ;;
esac
`

// A session that may not hold private: a result carrying it never reaches
// the model or the session, and the outcome says which call it was.
func TestForbiddenLabelsRefuseAResult(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_1", "read_secret", `{}`),
		toolReply("toolu_2", "read_web", `{}`),
		textReply("done"),
	)
	f := newFixture(t, api, forbidScript)
	f.req["tools"] = []map[string]any{
		{"name": "read_secret", "parameters": map[string]any{"type": "object"}},
		{"name": "read_web", "parameters": map[string]any{"type": "object"}},
	}
	f.req["policy"] = map[string]any{
		"labels":        []string{},
		"forbid_labels": []string{"private"},
		"tools": map[string]any{
			"read_secret": map[string]any{"class": "read"},
			"read_web":    map[string]any{"class": "read"},
		},
	}
	f.req["tool_protocol"] = "envelope"
	res := f.run()
	out := res.outcome(t)
	if res.code != 0 || out["reason"] != "completed" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	if labels := fmt.Sprint(out["labels"]); labels != "[contaminated]" {
		t.Fatalf("outcome labels %s", labels)
	}
	if refused := fmt.Sprint(out["refused"]); refused != "[map[call_id:toolu_1 labels:[private] tool:read_secret]]" {
		t.Fatalf("refused %s", refused)
	}
	if strings.Contains(res.stdout, "SECRET-TEXT") {
		t.Fatal("the refused result is in the session")
	}
	for _, body := range api.requests() {
		if strings.Contains(body, "SECRET-TEXT") {
			t.Fatal("the refused result reached the model")
		}
	}
	if !strings.Contains(api.requests()[2], "This result was refused") {
		t.Fatal("the model was not told the result was refused")
	}

	// A later run of a session that somehow holds private does not start.
	f.req["policy"].(map[string]any)["forbid_labels"] = []string{"contaminated"}
	f.req["prompt"] = "more"
	res = f.run()
	out = res.outcome(t)
	if res.code != 0 || out["reason"] != "forbidden" || fmt.Sprint(out["forbidden"]) != "[contaminated]" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	if n := len(api.requests()); n != 3 {
		t.Fatalf("%d model requests", n)
	}
}

// A request's policy carries the operator's standing consent: an act call
// whose input matches a rule runs on a contaminated session, and is
// recorded with the rule's index; a near miss is held. An edit call runs
// on a session that is private but clean.
func TestConsentAndEditFromTheRequest(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_3", "broken", `{}`),
		toolReply("toolu_1", "read_mail", `{}`),
		toolReply("toolu_2", "web_post", `{"group":"Kira only","body":"hi"}`),
		toolReply("toolu_4", "web_post", `{"group":"Kira only ","body":"hi"}`),
		textReply("done"),
	)
	f, calls := trifectaFixture(t, api)
	pol := f.req["policy"].(map[string]any)["tools"].(map[string]any)
	pol["web_post"] = map[string]any{"class": "act", "consent": []map[string]any{{"group": "Everyone"}, {"group": "Kira only"}}}
	pol["broken"] = map[string]any{"class": "edit"}
	res := f.run()
	out := res.outcome(t)
	if res.code != 0 || out["reason"] != "held" {
		t.Fatalf("exit %d outcome %v stderr %s", res.code, out, res.stderr)
	}
	held := out["held"].([]any)
	if len(held) != 1 || held[0].(map[string]any)["id"] != "toolu_4" {
		t.Fatalf("held %v", held)
	}
	got := readCalls(t, calls)
	if !strings.Contains(got, `web_post {"input":{"group":"Kira only","body":"hi"}`) || !strings.Contains(got, "broken ") {
		t.Fatalf("calls: %s", got)
	}
	var rule any = "none"
	for _, l := range res.lines {
		if d, ok := l["decision"].(map[string]any); ok && d["call_id"] == "toolu_2" {
			rule = d["consent"]
		}
	}
	if rule != float64(1) {
		t.Fatalf("toolu_2's decision has consent %v, want rule 1", rule)
	}
}

// An edit call from a contaminated session is held: what it writes would
// come back from outside relay without its labels.
func TestEditIsHeldOnceContaminated(t *testing.T) {
	api := newFakeAPI(t,
		toolReply("toolu_1", "read_mail", `{}`),
		toolReply("toolu_2", "broken", `{"text":"injected"}`),
		textReply("done"),
	)
	f, calls := trifectaFixture(t, api)
	f.req["policy"].(map[string]any)["tools"].(map[string]any)["broken"] = map[string]any{"class": "edit"}
	res := f.run()
	out := res.outcome(t)
	held, _ := out["held"].([]any)
	if out["reason"] != "held" || len(held) != 1 || held[0].(map[string]any)["id"] != "toolu_2" {
		t.Fatalf("outcome %v stderr %s", out, res.stderr)
	}
	if strings.Contains(readCalls(t, calls), "broken") {
		t.Fatal("the held edit call ran")
	}
}

func TestConsentRequestsAreChecked(t *testing.T) {
	for _, c := range []struct {
		entry map[string]any
		want  string
	}{
		{map[string]any{"class": "egress", "consent": []map[string]any{{"group": "x"}}}, "only a tool of class act"},
		{map[string]any{"class": "act", "always_hold": true, "consent": []map[string]any{{"group": "x"}}}, "both consent and always_hold"},
		{map[string]any{"class": "act", "consent": []map[string]any{{}}}, "rule 0 is empty"},
	} {
		f, _ := trifectaFixture(t, newFakeAPI(t))
		f.req["policy"].(map[string]any)["tools"].(map[string]any)["web_post"] = c.entry
		res := f.run()
		if res.code != 2 || !strings.Contains(res.stderr, c.want) {
			t.Errorf("%v: exit %d stderr %s", c.entry, res.code, res.stderr)
		}
	}
}
