// Package stricttest loads the table of lines that relay, agentloop and
// relay-tui must read alike (relay#58), from testdata/strict-json/cases.jsonl.
// The file is a byte-identical copy of the shared table; change the
// table there and copy it to all three repos.
package stricttest

import (
	"bytes"
	"encoding/base64"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// A Case is one line of the table.
type Case struct {
	Name string
	// Line is the exact bytes of one message, without its newline.
	Line []byte
	// Reject says the line must be refused. When it is false, Method is
	// the value of the exact member "method", or nil when it has none.
	Reject bool
	Method *string
}

type row struct {
	Name   string `json:"name"`
	B64    string `json:"b64"`
	Expect any    `json:"expect"`
}

// Path is where the table is in the repo.
func Path() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "testdata", "strict-json", "cases.jsonl")
}

// Cases reads the table, failing t if it cannot.
func Cases(t testing.TB) []Case {
	t.Helper()
	raw, err := os.ReadFile(Path())
	if err != nil {
		t.Fatal(err)
	}
	var out []Case
	for i, line := range bytes.Split(raw, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var r row
		if err := strictjson.UnmarshalKnown(line, &r); err != nil {
			t.Fatalf("cases.jsonl line %d: %v", i+1, err)
		}
		b, err := base64.StdEncoding.DecodeString(r.B64)
		if err != nil {
			t.Fatalf("cases.jsonl line %d: %v", i+1, err)
		}
		c := Case{Name: r.Name, Line: b}
		switch e := r.Expect.(type) {
		case string:
			if e != "reject" {
				t.Fatalf("cases.jsonl line %d: expect %q", i+1, e)
			}
			c.Reject = true
		case map[string]any:
			m, ok := e["method"]
			if !ok || len(e) != 1 {
				t.Fatalf("cases.jsonl line %d: expect %v", i+1, e)
			}
			switch m := m.(type) {
			case nil:
			case string:
				c.Method = &m
			default:
				t.Fatalf("cases.jsonl line %d: method %v", i+1, m)
			}
		default:
			t.Fatalf("cases.jsonl line %d: expect %v", i+1, e)
		}
		out = append(out, c)
	}
	if len(out) == 0 {
		t.Fatal("cases.jsonl is empty")
	}
	return out
}
