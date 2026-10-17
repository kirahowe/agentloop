package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runOtel(t *testing.T, args ...string) (string, string, int) {
	t.Helper()
	var out, errb bytes.Buffer
	code := realMain(append([]string{"otel"}, args...), strings.NewReader(""), &out, &errb)
	return out.String(), errb.String(), code
}

func TestOtelCommand(t *testing.T) {
	const fixture = "../../otel/testdata/session.jsonl"
	out, _, code := runOtel(t, fixture)
	if code != 0 || strings.Count(out, "\n") != 1 || !strings.HasSuffix(out, "\n") {
		t.Fatalf("code %d, output %q", code, out)
	}
	var req struct {
		ResourceSpans []struct {
			ScopeSpans []struct{ Spans []map[string]any }
		}
	}
	if err := json.Unmarshal([]byte(out), &req); err != nil || len(req.ResourceSpans) != 1 {
		t.Fatalf("not an OTLP request: %v", err)
	}
	// The conversation id defaults to the file's base name.
	if !strings.Contains(out, `"stringValue":"session"`) {
		t.Fatal("conversation id is not the file's base name")
	}
	// No content by default.
	for _, w := range []string{"quarterly numbers", "boss@example.com"} {
		if strings.Contains(out, w) {
			t.Fatalf("content %q in the output", w)
		}
	}
	again, _, _ := runOtel(t, fixture)
	if again != out {
		t.Fatal("re-export differs")
	}

	// --out writes the same line to a file, flags in either order.
	file := filepath.Join(t.TempDir(), "o.json")
	stdout, _, code := runOtel(t, fixture, "--conversation-id", "x", "--out", file)
	if code != 0 || stdout != "" {
		t.Fatalf("code %d stdout %q", code, stdout)
	}
	b, _ := os.ReadFile(file)
	if !strings.Contains(string(b), `"stringValue":"x"`) || strings.Contains(string(b), `"stringValue":"session"`) {
		t.Fatal("--conversation-id ignored")
	}

	// A parent puts the runs in the parent's trace.
	tr, sp := strings.Repeat("ab", 16), strings.Repeat("cd", 8)
	p, _, code := runOtel(t, "--parent-trace", tr, "--parent-span", sp, fixture)
	if code != 0 || !strings.Contains(p, `"parentSpanId":"`+sp+`"`) || strings.Contains(p, `"traceId":"c3`) {
		t.Fatalf("parent not applied: %d", code)
	}
}

func TestOtelUsageErrors(t *testing.T) {
	for _, args := range [][]string{
		{}, {"a", "b"}, {"nope.jsonl"}, {"--bogus", "x"},
		{"--parent-trace", strings.Repeat("a", 32), "../../otel/testdata/session.jsonl"},
		{"--parent-trace", "zz", "--parent-span", "yy", "../../otel/testdata/session.jsonl"},
	} {
		if _, _, code := runOtel(t, args...); code != 2 {
			t.Errorf("%v: exit %d, want 2", args, code)
		}
	}
}
