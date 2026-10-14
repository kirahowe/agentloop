package agentloop_test

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"agentloop"
	"agentloop/fake"
)

func sampleEntries(t *testing.T) []agentloop.Entry {
	t.Helper()
	a, _ := crashScenario(false)
	s, _ := newSession(t)
	finish(t, a, s, "go")
	return s.Entries()
}

func TestFileStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.jsonl")
	store := agentloop.NewFileStore(path)
	if got, err := store.Load(); err != nil || len(got) != 0 {
		t.Fatalf("missing file: %v, %v", got, err)
	}
	entries := sampleEntries(t)
	for _, e := range entries {
		if err := store.Append(e); err != nil {
			t.Fatal(err)
		}
	}
	got, err := agentloop.NewFileStore(path).Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(entryJSON(t, got), entryJSON(t, entries)) {
		t.Fatal("entries changed on the way through the file")
	}
	raw, _ := os.ReadFile(path)
	if n := bytes.Count(raw, []byte("\n")); n != len(entries) {
		t.Fatalf("%d lines for %d entries", n, len(entries))
	}
}

func entryJSON(t *testing.T, entries []agentloop.Entry) []string {
	t.Helper()
	var out []string
	for _, e := range entries {
		b, err := jsonMarshal(e)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, string(b))
	}
	return out
}

func TestFileStoreSessionRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.jsonl")
	a, _ := crashScenario(false)
	s := reopen(t, agentloop.NewFileStore(path))
	finish(t, a, s, "go")
	s2 := reopen(t, agentloop.NewFileStore(path))
	if !reflect.DeepEqual(s.Messages(), s2.Messages()) {
		t.Fatal("conversation differs after reopening the file")
	}
}

func TestFileStoreSnakeCaseFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.jsonl")
	m := fake.Script(fake.ToolUse("", fake.Call("a", "echo", input("x"))), fake.Text("done"))
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", newCounter())}}
	s := reopen(t, agentloop.NewFileStore(path))
	run(t, a, s, "go")
	raw, _ := os.ReadFile(path)
	for _, want := range []string{`"type":"run_started"`, `"call_id":"a"`, `"tool_call":`, `"tool_result":`, `"call_started":`, `"run_ended":`, `"input_tokens":10`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("log has no %s", want)
		}
	}
}

func TestFileStoreTornFinalLine(t *testing.T) {
	entries := sampleEntries(t)
	for _, tail := range []string{`{"seq":99,"type":"mess`, `{"seq":`, `{`, "garbage"} {
		path := filepath.Join(t.TempDir(), "s.jsonl")
		store := agentloop.NewFileStore(path)
		for _, e := range entries[:5] {
			store.Append(e)
		}
		f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		f.WriteString(tail)
		f.Close()

		got, err := agentloop.NewFileStore(path).Load()
		if err != nil || len(got) != 5 {
			t.Fatalf("tail %q: %d entries, %v", tail, len(got), err)
		}

		// Appending after the tear drops the torn fragment instead of
		// gluing the next entry to it.
		store = agentloop.NewFileStore(path)
		if err := store.Append(entries[5]); err != nil {
			t.Fatal(err)
		}
		got, err = agentloop.NewFileStore(path).Load()
		if err != nil || len(got) != 6 {
			t.Fatalf("tail %q after append: %d entries, %v", tail, len(got), err)
		}
	}
}

func TestFileStoreTornLineWithNewline(t *testing.T) {
	entries := sampleEntries(t)
	path := filepath.Join(t.TempDir(), "s.jsonl")
	store := agentloop.NewFileStore(path)
	for _, e := range entries[:3] {
		store.Append(e)
	}
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString("{\"seq\":\n")
	f.Close()
	got, err := agentloop.NewFileStore(path).Load()
	if err != nil || len(got) != 3 {
		t.Fatalf("%d entries, %v", len(got), err)
	}
	if err := agentloop.NewFileStore(path).Append(entries[3]); err != nil {
		t.Fatal(err)
	}
	got, err = agentloop.NewFileStore(path).Load()
	if err != nil || len(got) != 4 {
		t.Fatalf("%d entries, %v", len(got), err)
	}
}

func TestFileStoreCompleteLastLineWithoutNewline(t *testing.T) {
	entries := sampleEntries(t)
	path := filepath.Join(t.TempDir(), "s.jsonl")
	store := agentloop.NewFileStore(path)
	store.Append(entries[0])
	raw, _ := os.ReadFile(path)
	os.WriteFile(path, bytes.TrimRight(raw, "\n"), 0o600)

	got, err := agentloop.NewFileStore(path).Load()
	if err != nil || len(got) != 1 {
		t.Fatalf("%d entries, %v", len(got), err)
	}
	if err := agentloop.NewFileStore(path).Append(entries[1]); err != nil {
		t.Fatal(err)
	}
	got, err = agentloop.NewFileStore(path).Load()
	if err != nil || len(got) != 2 {
		t.Fatalf("%d entries, %v", len(got), err)
	}
}

func TestFileStoreBadLineInTheMiddle(t *testing.T) {
	entries := sampleEntries(t)
	path := filepath.Join(t.TempDir(), "s.jsonl")
	store := agentloop.NewFileStore(path)
	store.Append(entries[0])
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString("not json\n")
	f.Close()
	store.Append(entries[1])
	if _, err := agentloop.NewFileStore(path).Load(); err == nil {
		t.Fatal("a bad line in the middle was accepted")
	}
	if _, err := agentloop.OpenSession(agentloop.NewFileStore(path)); err == nil {
		t.Fatal("a session opened on a damaged log")
	}
}

// A session on a torn file resumes: the run that died is settled.
func TestResumeFromTornFile(t *testing.T) {
	full := sampleEntries(t)
	cut := 0
	for i, e := range full {
		if e.Type == agentloop.EntryCallStarted && e.CallStarted.CallID == "c2" {
			cut = i + 1
		}
	}
	path := filepath.Join(t.TempDir(), "s.jsonl")
	store := agentloop.NewFileStore(path)
	for _, e := range full[:cut] {
		store.Append(e)
	}
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString(`{"seq":1000,"type":"tool_res`)
	f.Close()

	a, c := crashScenario(false)
	s := reopen(t, agentloop.NewFileStore(path))
	wantReason(t, run(t, a, s, ""), agentloop.ReasonCompleted)
	if c.count("read") != 0 {
		t.Fatal("the started call ran again")
	}
	checkConversation(t, s.Messages())
	s2 := reopen(t, agentloop.NewFileStore(path))
	if !reflect.DeepEqual(s.Messages(), s2.Messages()) {
		t.Fatal("the repaired file reads back differently")
	}
}

func TestMemoryStoreCopiesEntries(t *testing.T) {
	m := agentloop.NewMemoryStore()
	e := agentloop.Entry{Type: agentloop.EntryRunStarted, RunStarted: &agentloop.RunStartedEntry{Prompt: "one"}}
	m.Append(e)
	e.RunStarted.Prompt = "changed"
	got, _ := m.Load()
	if got[0].RunStarted.Prompt != "one" {
		t.Fatal("the store shares pointers with the caller")
	}
	got[0].RunStarted.Prompt = "also changed"
	again, _ := m.Load()
	if again[0].RunStarted.Prompt != "one" {
		t.Fatal("Load hands out the store's own entries")
	}
}
