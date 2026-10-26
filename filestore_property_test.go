package agentloop_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
	"github.com/kirahowe/agentloop/internal/scenario"
)

// randomLog runs a random scenario and returns the log it left, line by
// line.
func randomLog(t *rapid.T) [][]byte {
	sc := scenario.New(t, scenario.GenConfig(t), scenario.Faults{})
	sc.Execute()
	return sc.Store.Snapshot()
}

func writeLines(t fataler, path string, lines [][]byte) []byte {
	t.Helper()
	raw := bytes.Join(lines, []byte("\n"))
	raw = append(raw, '\n')
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return raw
}

func entriesJSON(t fataler, es []agentloop.Entry) [][]byte {
	t.Helper()
	var out [][]byte
	for _, e := range es {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, b)
	}
	return out
}

// resumeAgent has no tools and a model that always finishes. Resuming a
// session with it must work whatever state the log is in.
func resumeAgent() *agentloop.Agent {
	return &agentloop.Agent{Model: fake.New(func(*agentloop.Request) (*agentloop.Response, error) { return fake.Text("again"), nil })}
}

// TestPropertyFileStoreTorn is property 12: a file cut at any byte loads as
// a prefix of its entries, and appending afterwards leaves a loadable file
// that still starts with that prefix.
func TestPropertyFileStoreTorn(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := randomLog(t)
		raw := bytes.Join(lines, []byte("\n"))
		raw = append(raw, '\n')

		// Offsets: random ones, and the neighbourhood of a random line end.
		offs := []int{rapid.IntRange(0, len(raw)).Draw(t, "off1")}
		end := 0
		for i := 0; i < rapid.IntRange(0, len(lines)-1).Draw(t, "line"); i++ {
			end += len(lines[i]) + 1
		}
		offs = append(offs, end+rapid.IntRange(-1, 1).Draw(t, "near"))
		for _, off := range offs {
			off = min(max(off, 0), len(raw))
			path := filepath.Join(tmpDir(t), "s.jsonl")
			if err := os.WriteFile(path, raw[:off], 0o600); err != nil {
				t.Fatal(err)
			}
			prefix := raw[:off]
			want := bytes.Count(prefix, []byte("\n"))
			if tail := prefix[bytes.LastIndexByte(prefix, '\n')+1:]; len(bytes.TrimSpace(tail)) > 0 && json.Valid(tail) {
				want++
			}

			store := agentloop.NewFileStore(path)
			got, err := store.Load()
			if err != nil {
				t.Fatalf("cut at %d of %d: Load: %v", off, len(raw), err)
			}
			if len(got) != want {
				t.Fatalf("cut at %d: loaded %d entries, want %d", off, len(got), want)
			}
			for i, b := range entriesJSON(t, got) {
				if !jsonEqual(t, b, lines[i]) {
					t.Fatalf("cut at %d: entry %d is not the original", off, i+1)
				}
			}
			s, err := agentloop.OpenSession(store)
			if err != nil {
				t.Fatalf("cut at %d: a prefix of a valid log does not replay: %v", off, err)
			}
			out, err := resumeAgent().Run(context.Background(), s, "again")
			if err != nil {
				t.Fatalf("cut at %d: resume: %v", off, err)
			}
			if out.Reason != agentloop.ReasonCompleted {
				t.Fatalf("cut at %d: %+v", off, out)
			}

			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			for i, l := range bytes.Split(bytes.TrimRight(after, "\n"), []byte("\n")) {
				if !json.Valid(l) {
					t.Fatalf("cut at %d: line %d of the file is not JSON: %q", off, i+1, l)
				}
			}
			reloaded, err := agentloop.NewFileStore(path).Load()
			if err != nil {
				t.Fatalf("cut at %d: the file no longer loads: %v", off, err)
			}
			if len(reloaded) <= len(got) {
				t.Fatalf("cut at %d: %d entries before, %d after a run", off, len(got), len(reloaded))
			}
			for i, b := range entriesJSON(t, reloaded[:len(got)]) {
				if !jsonEqual(t, b, entriesJSON(t, got)[i]) {
					t.Fatalf("cut at %d: entry %d changed by appending", off, i+1)
				}
			}
			if _, err := agentloop.OpenSession(agentloop.NewFileStore(path)); err != nil {
				t.Fatalf("cut at %d: the file no longer replays: %v", off, err)
			}
		}
	})
}

func jsonEqual(t fataler, a, b []byte) bool {
	t.Helper()
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	ab, _ := json.Marshal(x)
	bb, _ := json.Marshal(y)
	return bytes.Equal(ab, bb)
}

// TestFileStoreEveryOffset cuts one fixed log at every byte.
func TestFileStoreEveryOffset(t *testing.T) {
	var lines [][]byte
	rapid.Check(t, func(rt *rapid.T) {
		if lines != nil {
			return
		}
		cfg := scenario.Config{Steps: []scenario.Step{
			{Kind: scenario.StToolUse, Stop: agentloop.StopToolUse, Text: "café <x>", In: 3, Out: 4, Calls: []scenario.CallSpec{{Name: "echo"}, {Name: "any"}, {Name: "echo", Input: 4}}},
			{Kind: scenario.StErr, ErrKind: agentloop.Overloaded},
		}, Behaviors: map[string]scenario.Behavior{"echo": {}, "any": {Kind: scenario.BOK, Labels: []string{"private"}}, "lab": {}},
			Rules: map[string]scenario.Rule{"echo": {Base: agentloop.Deny}, "any": {Base: agentloop.Allow}, "lab": {Base: agentloop.Allow}},
			Retry: &agentloop.RetryPolicy{Retries: 2}, Iterations: 1}
		sc := scenario.New(rt, cfg, scenario.Faults{})
		sc.Execute()
		lines = sc.Store.Snapshot()
	})
	raw := bytes.Join(lines, []byte("\n"))
	raw = append(raw, '\n')
	if len(lines) < 15 {
		t.Fatalf("only %d entries", len(lines))
	}
	dir := t.TempDir()
	for off := 0; off <= len(raw); off++ {
		path := filepath.Join(dir, "s.jsonl")
		if err := os.WriteFile(path, raw[:off], 0o600); err != nil {
			t.Fatal(err)
		}
		prefix := raw[:off]
		want := bytes.Count(prefix, []byte("\n"))
		if tail := prefix[bytes.LastIndexByte(prefix, '\n')+1:]; json.Valid(tail) && len(tail) > 0 {
			want++
		}
		store := agentloop.NewFileStore(path)
		got, err := store.Load()
		if err != nil || len(got) != want {
			t.Fatalf("cut at %d: %d entries (want %d), %v", off, len(got), want, err)
		}
		s, err := agentloop.OpenSession(store)
		if err != nil {
			t.Fatalf("cut at %d: %v", off, err)
		}
		// Appending syncs to disk, so resume only near line ends and at
		// every few bytes elsewhere.
		nearEnd := false
		for d := -2; d <= 2; d++ {
			if i := off + d; i >= 0 && i < len(raw) && raw[i] == '\n' {
				nearEnd = true
			}
		}
		if !nearEnd && off%13 != 0 {
			continue
		}
		if _, err := resumeAgent().Run(context.Background(), s, "again"); err != nil {
			t.Fatalf("cut at %d: %v", off, err)
		}
		if _, err := agentloop.OpenSession(agentloop.NewFileStore(path)); err != nil {
			t.Fatalf("cut at %d: after the run: %v", off, err)
		}
	}
}

// TestPropertyFileStoreCorruption: a bad line anywhere but last is an
// error; a bad last line is a torn write and is ignored.
func TestPropertyFileStoreCorruption(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := randomLog(t)
		if len(lines) < 3 {
			t.Skip()
		}
		bad := rapid.SampledFrom([]string{"garbage{", "{\"seq\":1", "[1,2]", "\"x\"", "{\"seq\":"}).Draw(t, "bad")
		i := rapid.IntRange(0, len(lines)-1).Draw(t, "index")
		mutated := make([][]byte, len(lines))
		copy(mutated, lines)
		mutated[i] = []byte(bad)
		path := filepath.Join(tmpDir(t), "s.jsonl")
		writeLines(t, path, mutated)
		got, err := agentloop.NewFileStore(path).Load()
		if i == len(lines)-1 {
			if err != nil || len(got) != len(lines)-1 {
				t.Fatalf("torn last line: %d entries, %v", len(got), err)
			}
			return
		}
		if err == nil {
			t.Fatalf("a bad line %d of %d was accepted (%d entries)", i+1, len(lines), len(got))
		}
		if !strings.Contains(err.Error(), "line") {
			t.Fatalf("the error does not say where: %v", err)
		}
	})
}

// FuzzLoadAndOpen: arbitrary bytes in a session file never panic, in
// loading, replaying or resuming.
func FuzzLoadAndOpen(f *testing.F) {
	seed := func(entries ...string) { f.Add([]byte(strings.Join(entries, "\n") + "\n")) }
	user := `{"seq":2,"time":"2026-01-01T00:00:00Z","type":"message","message":{"message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}}`
	asst := `{"seq":3,"time":"2026-01-01T00:00:00Z","type":"message","message":{"message":{"role":"assistant","parts":[{"type":"tool_call","tool_call":{"id":"a","name":"x","input":{}}}]},"stop":"tool_use"}}`
	resUser := `{"seq":4,"time":"2026-01-01T00:00:00Z","type":"message","message":{"message":{"role":"user","parts":[{"type":"tool_result"}]}}}`
	resUnknown := `{"seq":4,"time":"2026-01-01T00:00:00Z","type":"message","message":{"message":{"role":"user","parts":[{"type":"tool_result","tool_result":{"call_id":"zz","content":"x"}}]}}}`
	result := `{"seq":5,"time":"2026-01-01T00:00:00Z","type":"tool_result","tool_result":{"result":{"call_id":"a","content":"ok"}}}`
	dec := `{"seq":4,"time":"2026-01-01T00:00:00Z","type":"decision","decision":{"call_id":"a","tool":"x","verdict":"allow"}}`
	started := `{"seq":5,"time":"2026-01-01T00:00:00Z","type":"call_started","call_started":{"call_id":"a"}}`
	seed(user, asst, dec, started, result)
	seed(user, asst, resUser, result)
	seed(user, asst, resUnknown, result)
	seed(asst, result, user)
	seed(`{}`, `null`, `[]`, user)
	seed(user, asst, `{"seq":4,"type":"decision","decision":{"call_id":"a","verdict":"bogus"}}`, result)
	f.Add([]byte(""))
	f.Add([]byte("\n\n  \n"))
	f.Add([]byte(`{"seq":1,"type":"run_started"`))

	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "s.jsonl")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		store := agentloop.NewFileStore(path)
		if _, err := store.Load(); err != nil {
			return
		}
		s, err := agentloop.OpenSession(store)
		if err != nil {
			return
		}
		touch(s)
		a := resumeAgent()
		a.Tools = []agentloop.Tool{agentloop.Func(agentloop.ToolSpec{Name: "x"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "ok", Labels: []string{"l"}}, nil
		})}
		for _, prompt := range []string{"", "p"} {
			if _, err := a.Run(context.Background(), s, prompt); err != nil {
				return
			}
			touch(s)
			if _, err := agentloop.OpenSession(store); err != nil {
				t.Fatalf("a log written by Run does not replay: %v", err)
			}
		}
	})
}

// FuzzOpenAndResume is the same without the disk, so it runs faster:
// arbitrary entries in a store never panic OpenSession or a resumed Run.
func FuzzOpenAndResume(f *testing.F) {
	f.Add([]byte(`{"seq":1,"type":"message","message":{"message":{"role":"assistant","parts":[{"type":"tool_call","tool_call":{"id":"a","name":"x","input":{}}}]},"stop":"tool_use"}}
{"seq":2,"type":"decision","decision":{"call_id":"a","tool":"x","verdict":"allow"}}
{"seq":3,"type":"call_started","call_started":{"call_id":"a"}}
{"seq":4,"type":"tool_result","tool_result":{"result":{"call_id":"a","content":"ok"},"labels":["l"]}}`))
	f.Add([]byte(`{"type":"message","message":{"message":{"role":"user","parts":[{"type":"tool_result"}]}}}`))
	f.Add([]byte(`{"type":"message","message":{"message":{"role":"assistant","parts":[{"type":"tool_call"},{"type":"tool_call","tool_call":{"id":"","name":""}}]},"stop":"pause"}}`))
	f.Add([]byte(`{"type":"run_ended"}` + "\n" + `{"type":"model_error","model_error":{}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var entries []agentloop.Entry
		for _, line := range bytes.Split(data, []byte("\n")) {
			var e agentloop.Entry
			if json.Unmarshal(line, &e) == nil {
				entries = append(entries, e)
			}
		}
		store := agentloop.NewMemoryStore(entries...)
		s, err := agentloop.OpenSession(store)
		if err != nil {
			return
		}
		touch(s)
		a := resumeAgent()
		a.Tools = []agentloop.Tool{agentloop.Func(agentloop.ToolSpec{Name: "x"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "ok", Labels: []string{"l"}}, nil
		})}
		a.Policy = agentloop.PolicyFunc(func(_ context.Context, c agentloop.ToolCall, v agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Verdict([]string{"allow", "deny", "hold", ""}[len(c.ID)%4])}
		})
		for _, prompt := range []string{"", "p"} {
			if _, err := a.Run(context.Background(), s, prompt); err != nil {
				return
			}
			touch(s)
			if _, err := agentloop.OpenSession(store); err != nil {
				t.Fatalf("a log written by Run does not replay: %v", err)
			}
		}
	})
}

func touch(s *agentloop.Session) {
	s.Messages()
	s.Labels()
	s.Usage()
	s.Entries()
	s.View()
	s.Pending()
	for _, turn := range s.Turns() {
		for _, id := range turn.Calls {
			s.Call(id)
		}
	}
}

// tmpDir is a directory removed when the check ends.
func tmpDir(t *rapid.T) string {
	dir, err := os.MkdirTemp("", "agentloop-prop")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

type fataler interface {
	Helper()
	Fatal(args ...any)
	Fatalf(format string, args ...any)
}
