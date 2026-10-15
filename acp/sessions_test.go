package acpagent

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"
)

// genSessionIDString is a string a client might send as a sessionId:
// arbitrary text, valid ids, and valid ids bent towards a path elsewhere.
func genSessionIDString() *rapid.Generator[string] {
	return rapid.Custom(func(t *rapid.T) string {
		var b [16]byte
		copy(b[:], rapid.SliceOfN(rapid.Byte(), 16, 16).Draw(t, "bytes"))
		id := string(NewSessionID(b))
		switch rapid.IntRange(0, 9).Draw(t, "shape") {
		case 0:
			return id
		case 1:
			return strings.ToUpper(id)
		case 2:
			return "../" + id
		case 3:
			return id + "/../../" + id
		case 4:
			return "/" + id
		case 5:
			return id + "\x00"
		case 6:
			return rapid.SampledFrom([]string{"", ".", "..", "/", "../outside", "..\\x", "~", id[:35], id + "0"}).Draw(t, "odd")
		case 7:
			return strings.Replace(id, "-", "/", 1)
		default:
			return rapid.String().Draw(t, "any")
		}
	})
}

// TestPropertySessionFileIsInside is the pure half of P4.4: for any
// string, SessionFile either refuses it as an unknown session or names a
// plain file directly inside the session directory, for an id of the
// shape agentloop makes.
func TestPropertySessionFileIsInside(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		dir := rapid.SampledFrom([]string{"/sessions", "/a b/sessions", "rel/sessions"}).Draw(t, "dir")
		id := acp.SessionId(genSessionIDString().Draw(t, "id"))
		path, err := SessionFile(dir, id)
		if err != nil {
			if err.Code != -32002 || path != "" {
				t.Fatalf("%q: %v, %q", id, err, path)
			}
			if sessionID.MatchString(string(id)) {
				t.Fatalf("refused the valid id %q", id)
			}
			return
		}
		rel, relErr := filepath.Rel(dir, path)
		if relErr != nil || rel != string(id)+".jsonl" || filepath.Dir(path) != filepath.Clean(dir) || strings.ContainsAny(rel, "/\\\x00") {
			t.Fatalf("%q gives %q (rel %q)", id, path, rel)
		}
		if got, ok := SessionFileID(filepath.Base(path)); !ok || got != id {
			t.Fatalf("the file %q does not name %q back", path, id)
		}
	})
}

func TestSessionFileID(t *testing.T) {
	id := NewSessionID([16]byte{1})
	for name, want := range map[string]bool{
		string(id) + ".jsonl":       true,
		string(id) + ".jsonl.lock":  false,
		string(id):                  false,
		"x" + string(id) + ".jsonl": false,
		".jsonl":                    false,
		"notes.txt":                 false,
	} {
		if _, ok := SessionFileID(name); ok != want {
			t.Errorf("%q: %v", name, ok)
		}
	}
}

func TestSessionCwd(t *testing.T) {
	for _, c := range []struct{ toolDir, wd, want string }{
		{"", "/home/k", "/home/k"},
		{"/srv/tools/", "/home/k", "/srv/tools"},
		{"work", "/home/k", "/home/k/work"},
		{"../w", "/home/k", "/home/w"},
	} {
		if got := SessionCwd(c.toolDir, c.wd); got != c.want {
			t.Errorf("SessionCwd(%q, %q) = %q, want %q", c.toolDir, c.wd, got, c.want)
		}
	}
}

// A session can be taken up again on its own connection only when no turn
// is in flight.
func TestReattach(t *testing.T) {
	for _, st := range []TurnState{Idle, Running, Cancelling, Closing, Closed} {
		err := Reattach("s", Turn{State: st})
		if (err == nil) != (st == Idle) {
			t.Errorf("%s: %v", st, err)
		}
		if err != nil && (err.Code != -32603 || err.Message != "session in use") {
			t.Errorf("%s: %v", st, err)
		}
	}
}

// TestPropertyListResponse: every stored session is listed once, with
// the server's cwd, newest first, with its last entry's time when it has
// one; a cwd filter matches all or nothing; a cursor is refused; and the
// answer validates against the schema.
func TestPropertyListResponse(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		cwd := "/work/tools"
		n := rapid.IntRange(0, 8).Draw(t, "n")
		base := time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
		var found []StoredSession
		seen := map[acp.SessionId]bool{}
		for i := 0; i < n; i++ {
			id := NewSessionID([16]byte{byte(i), byte(rapid.IntRange(0, 255).Draw(t, "b"))})
			if seen[id] {
				continue
			}
			seen[id] = true
			st := StoredSession{ID: id}
			if rapid.IntRange(0, 3).Draw(t, "empty") > 0 {
				st.Updated = base.Add(time.Duration(rapid.IntRange(0, 5).Draw(t, "min")) * time.Minute)
			}
			found = append(found, st)
		}
		req := acp.ListSessionsRequest{}
		filter := rapid.IntRange(0, 3).Draw(t, "filter")
		switch filter {
		case 1:
			req.Cwd = acp.Ptr(cwd + "/")
		case 2:
			req.Cwd = acp.Ptr("/elsewhere")
		case 3:
			req.Cursor = acp.Ptr("c1")
		}
		resp, err := ListResponse(found, cwd, req)
		if filter == 3 {
			if err == nil || err.Code != -32602 {
				t.Fatalf("a cursor gave %v", err)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if verr := validate("ListSessionsResponse", resp); verr != nil {
			t.Fatal(verr)
		}
		if filter == 2 {
			if resp.Sessions == nil || len(resp.Sessions) != 0 {
				t.Fatalf("an unmatched cwd listed %v", resp.Sessions)
			}
			return
		}
		if len(resp.Sessions) != len(found) {
			t.Fatalf("%d listed of %d", len(resp.Sessions), len(found))
		}
		want := map[acp.SessionId]time.Time{}
		for _, f := range found {
			want[f.ID] = f.Updated
		}
		var prev time.Time
		for i, s := range resp.Sessions {
			upd, ok := want[s.SessionId]
			if !ok || s.Cwd != cwd {
				t.Fatalf("listed %+v", s)
			}
			delete(want, s.SessionId)
			if (s.UpdatedAt == nil) != upd.IsZero() {
				t.Fatalf("%s: updatedAt %v for a log updated %v", s.SessionId, s.UpdatedAt, upd)
			}
			if s.UpdatedAt != nil && *s.UpdatedAt != upd.Format(time.RFC3339Nano) {
				t.Fatalf("%s: updatedAt %s, want %s", s.SessionId, *s.UpdatedAt, upd.Format(time.RFC3339Nano))
			}
			if i > 0 && upd.After(prev) {
				t.Fatalf("not newest first: %v", resp.Sessions)
			}
			prev = upd
		}
		if len(want) != 0 {
			t.Fatalf("not listed: %v", want)
		}
	})
}

func ExampleListResponse() {
	id := NewSessionID([16]byte{})
	resp, _ := ListResponse([]StoredSession{{ID: id}}, "/work", acp.ListSessionsRequest{})
	fmt.Println(resp.Sessions[0].SessionId, resp.Sessions[0].Cwd)
	// Output: 00000000-0000-4000-8000-000000000000 /work
}
