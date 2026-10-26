package command

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

func envTool(script string) *Tool {
	t := sh(script)
	t.Def = agentloop.ToolSpec{Name: "fetch"}
	t.Envelope = true
	return t
}

func callEnv(t *testing.T, tool *Tool, c agentloop.Call) agentloop.Result {
	t.Helper()
	r, err := tool.CallWith(context.Background(), c)
	if err != nil {
		t.Fatalf("Go error %v", err)
	}
	return r
}

func TestEnvelopeStdin(t *testing.T) {
	tool := envTool(`cat >&2; echo '{"content":"ok","labels":[]}'`)
	r := callEnv(t, tool, agentloop.Call{ID: "toolu_1", Input: json.RawMessage(`{"q":"x"}`), Labels: []string{"private"}})
	want := `stderr: {"input":{"q":"x"},"labels":["private"],"call_id":"toolu_1"}`
	if r.Log != want {
		t.Fatalf("log %q, want %q", r.Log, want)
	}
	// No labels is an empty array, and no input is an empty object.
	r = callEnv(t, tool, agentloop.Call{ID: "c"})
	if r.Log != `stderr: {"input":{},"labels":[],"call_id":"c"}` {
		t.Fatalf("log %q", r.Log)
	}
	if r.Content != "ok" || r.IsError || r.Labels != nil {
		t.Fatalf("%+v", r)
	}
}

func TestEnvelopeResult(t *testing.T) {
	tool := envTool(`echo 'warning: slow' >&2; printf '%s\n' '{"content":"page text","is_error":true,"labels":["contaminated"]}'`)
	r := callEnv(t, tool, agentloop.Call{ID: "c"})
	if r.Content != "page text" || !r.IsError || !reflect.DeepEqual(r.Labels, []string{"contaminated"}) {
		t.Fatalf("%+v", r)
	}
	if r.Log != "stderr: warning: slow" {
		t.Fatalf("log %q", r.Log)
	}
}

func TestEnvelopeFailuresShowNothing(t *testing.T) {
	cases := map[string]string{
		"exit non-zero":    `echo '{"content":"SECRET","labels":[]}'; echo SECRET >&2; exit 3`,
		"not json":         `echo SECRET; echo SECRET >&2`,
		"two envelopes":    `echo '{"content":"SECRET","labels":[]}{"content":"x","labels":[]}'`,
		"trailing text":    `echo '{"content":"SECRET","labels":[]} SECRET'`,
		"empty label":      `echo '{"content":"SECRET","labels":[""]}'`,
		"unknown field":    `echo '{"content":"SECRET","labels":[],"extra":1}'`,
		"no labels":        `echo '{"content":"SECRET"}'`,
		"null labels":      `echo '{"content":"SECRET","labels":null}'`,
		"no content":       `echo '{"labels":[]}' ; echo SECRET >&2`,
		"content a number": `echo '{"content":7,"labels":[]}'`,
		"array":            `echo '["SECRET"]'`,
		"empty":            `echo SECRET >&2`,
		"killed":           `echo SECRET; kill -9 $$`,
		"too much":         `printf '{"content":"SECRET SECRET SECRET SECRET","labels":[]}'`,
	}
	for name, script := range cases {
		t.Run(name, func(t *testing.T) {
			tool := envTool(script)
			if name == "too much" {
				tool.MaxOutputBytes = 20
			}
			r := callEnv(t, tool, agentloop.Call{ID: "c"})
			if r.Content != "The tool fetch failed." || !r.IsError || r.Labels != nil {
				t.Fatalf("%+v", r)
			}
			if r.Log == "" {
				t.Fatal("the operator is not told why")
			}
		})
	}
	// A program that cannot start fails the same way.
	tool := envTool("")
	tool.Argv = []string{"/no/such/program"}
	if r := callEnv(t, tool, agentloop.Call{}); r.Content != "The tool fetch failed." || !strings.Contains(r.Log, "no such") {
		t.Fatalf("%+v", r)
	}
}

func TestEnvelopeTimeoutIsTheAgentsToReport(t *testing.T) {
	tool := envTool(`echo SECRET >&2; echo '{"content":"SECRET","labels":[]}'; sleep 5`)
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	r, err := tool.CallWith(ctx, agentloop.Call{})
	if err == nil {
		t.Fatal("no error after the deadline")
	}
	if r.Content != "" || r.Labels != nil || !strings.Contains(r.Log, "SECRET") {
		t.Fatalf("%+v", r)
	}
}

// Regression: encoding/json reads keys without regard to case and lets a
// repeated key win, so a second "labels" or a "Labels" could replace the
// labels the envelope declared.
func TestParseEnvelopeRefusesRepeatedAndMisspelledKeys(t *testing.T) {
	for _, out := range []string{
		`{"content":"page text","labels":["contaminated"],"labels":[]}`,
		`{"content":"page text","labels":["contaminated"],"Labels":[]}`,
		`{"content":"page text","LABELS":["contaminated"],"labels":[]}`,
		`{"content":"page text","labels":["private","contaminated"],"lAbElS":["private"]}`,
		`{"CONTENT":"x","Is_Error":true,"LABELS":["private"]}`,
		`{"content":"x","content":"y","labels":[]}`,
		`{"content":"x","is_error":true,"is_error":false,"labels":[]}`,
		// The rest of the strict dialect (relay#58).
		"{\"content\":\"a\xffb\",\"labels\":[]}",
		"\xef\xbb\xbf{\"content\":\"x\",\"labels\":[]}",
		`{"content":"\ud800","labels":[]}`,
		`{"content":"x","labels":[]} null`,
		`{"content":"x","labels":[],"display":{"title":"t","body":"b","labels":[],"n":` + strings.Repeat("9", 1001) + `}}`,
	} {
		if res, why := ParseEnvelope([]byte(out)); why == "" {
			t.Errorf("accepted %s as %+v", out, res)
		}
	}
}

// ParseEnvelope accepts exactly one envelope object with valid labels and
// nothing else after it.
func TestParseEnvelopeProperties(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		content := rapid.SampledFrom([]string{"", "x", "a\nb", "café", `{"content":"y"}`}).Draw(t, "content")
		labels := rapid.SliceOfN(rapid.SampledFrom([]string{"private", "contaminated", "", "a\nb", "from the web"}), 0, 3).Draw(t, "labels")
		isErr := rapid.Bool().Draw(t, "is_error")
		obj := map[string]any{"content": content, "is_error": isErr, "labels": labels}
		mut := rapid.IntRange(0, 6).Draw(t, "mutation")
		switch mut {
		case 1:
			delete(obj, "content")
		case 2:
			delete(obj, "labels")
		case 3:
			delete(obj, "is_error")
		case 4:
			obj["more"] = true
		}
		b, _ := json.Marshal(obj)
		pad := rapid.SampledFrom([]string{"", "\n", " \n\t"}).Draw(t, "pad")
		tail := ""
		if mut == 5 {
			tail = rapid.SampledFrom([]string{"x", "{}", "null", `"s"`}).Draw(t, "tail")
		}
		if mut == 6 {
			b = b[:rapid.IntRange(0, len(b)-1).Draw(t, "cut")]
		}
		stdout := []byte(pad + string(b) + pad + tail)
		res, why := ParseEnvelope(stdout)
		allKnown := !slices.ContainsFunc(labels, func(l string) bool { return agentloop.CheckLabel(l) != nil })
		valid := (mut == 0 || mut == 3) && allKnown
		if valid != (why == "") {
			t.Fatalf("%q: valid=%v why=%q", stdout, valid, why)
		}
		if !valid {
			if !reflect.DeepEqual(res, agentloop.Result{}) {
				t.Fatalf("a rejected envelope gave %+v", res)
			}
			return
		}
		wantErr := isErr && mut != 3
		if res.Content != content || res.IsError != wantErr || len(res.Labels) != len(labels) {
			t.Fatalf("%q gave %+v", stdout, res)
		}
	})
}

// For random stdout, stderr and exit status, an envelope tool shows the
// model the envelope's content only when the program exited 0 and wrote
// one valid envelope within the cap; otherwise the fixed text. Stderr
// never reaches the content.
func TestEnvelopeOutputProperties(t *testing.T) {
	dir := t.TempDir()
	outFile, errFile := filepath.Join(dir, "out"), filepath.Join(dir, "err")
	n := 0
	rapid.Check(t, func(t *rapid.T) {
		n++
		content := rapid.SampledFrom([]string{"fine", "", "line\nline"}).Draw(t, "content")
		var out []byte
		switch rapid.IntRange(0, 3).Draw(t, "stdout") {
		case 0, 1:
			b, _ := json.Marshal(map[string]any{"content": content, "labels": []string{"private"}})
			out = b
		case 2:
			out = []byte("STDERRMARK? no: plain text " + content)
		case 3:
			out = []byte(`{"content":"x","labels":["bogus"]}`)
		}
		errb := []byte("STDERRMARK " + strconv.Itoa(n))
		code := rapid.SampledFrom([]int{0, 0, 0, 1, 2, -1}).Draw(t, "code")
		max := rapid.SampledFrom([]int{0, 0, 8, len(out), len(out) + 1}).Draw(t, "max")
		os.WriteFile(outFile, out, 0o644)
		os.WriteFile(errFile, errb, 0o644)
		script := `cat "$1"; cat "$2" >&2; if [ "$3" = -1 ]; then kill -9 $$; fi; exit "$3"`
		tool := &Tool{
			Def:            agentloop.ToolSpec{Name: "t"},
			Argv:           []string{"/bin/sh", "-c", script, "sh", outFile, errFile, strconv.Itoa(code)},
			Env:            []string{"PATH=/usr/bin:/bin"},
			MaxOutputBytes: max,
			Envelope:       true,
		}
		r, err := tool.CallWith(context.Background(), agentloop.Call{ID: "c"})
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(r.Content, "STDERRMARK") && !strings.Contains(string(out), "STDERRMARK") {
			t.Fatalf("stderr reached the model: %+v", r)
		}
		eff := max
		if eff == 0 {
			eff = DefaultMaxOutputBytes
		}
		parsed, why := ParseEnvelope(out)
		ok := code == 0 && len(out) <= eff && why == ""
		if ok {
			if r.Content != parsed.Content || r.IsError || !reflect.DeepEqual(r.Labels, parsed.Labels) {
				t.Fatalf("code %d max %d: %+v, want %+v", code, max, r, parsed)
			}
		} else if r.Content != Failed("t") || !r.IsError || r.Labels != nil {
			t.Fatalf("code %d max %d out %q: %+v", code, max, out, r)
		}
		if want := string(clip(errb, eff)); !strings.Contains(r.Log, want) {
			t.Fatalf("stderr is not in the operator's log: %q", r.Log)
		}
	})
}
