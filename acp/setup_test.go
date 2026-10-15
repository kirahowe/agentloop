package acpagent

import (
	"encoding/json"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop/internal/request"
)

var uuidV4 = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

// TestPropertySessionIDs: every id is a version 4 UUID, which keeps it a
// plain file name inside session_dir, and ids from different bytes differ
// unless the bytes differ only in the version and variant bits.
func TestPropertySessionIDs(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		var a, b [16]byte
		copy(a[:], rapid.SliceOfN(rapid.Byte(), 16, 16).Draw(t, "a"))
		copy(b[:], rapid.SliceOfN(rapid.Byte(), 16, 16).Draw(t, "b"))
		id := NewSessionID(a)
		if !uuidV4.MatchString(string(id)) {
			t.Fatalf("%q is not a v4 UUID", id)
		}
		dir := "/sessions"
		if p := SessionPath(dir, id); filepath.Dir(p) != dir || filepath.Base(p) != string(id)+".jsonl" {
			t.Fatalf("path %q", p)
		}
		mask := func(x [16]byte) [16]byte { x[6] &= 0x0f; x[8] &= 0x3f; return x }
		if (NewSessionID(b) == id) != (mask(a) == mask(b)) {
			t.Fatalf("%x and %x give ids %s and %s", a, b, id, NewSessionID(b))
		}
	})
}

// initialize advertises what the server implements and nothing more:
// loading, resuming, listing and closing sessions, but not deleting them,
// modes, images, audio or MCP. The answer validates against the schema.
func TestInitializeResponse(t *testing.T) {
	r := InitializeResponse("1.2.3", "")
	if err := validate("InitializeResponse", r); err != nil {
		t.Fatal(err)
	}
	c := r.AgentCapabilities
	if r.ProtocolVersion != 1 || r.AgentInfo.Name != "agentloop" || r.AgentInfo.Version != "1.2.3" ||
		!c.LoadSession || c.SessionCapabilities.List == nil || c.SessionCapabilities.Resume == nil ||
		c.SessionCapabilities.Close == nil || c.SessionCapabilities.Delete != nil ||
		c.SessionCapabilities.AdditionalDirectories != nil || c.PromptCapabilities.Image || c.PromptCapabilities.Audio ||
		!c.PromptCapabilities.EmbeddedContext || c.McpCapabilities.Http || c.McpCapabilities.Sse || len(r.AuthMethods) != 0 {
		t.Fatalf("%+v", r)
	}
	if err := validate("NewSessionResponse", acp.NewSessionResponse{SessionId: NewSessionID([16]byte{})}); err != nil {
		t.Fatal(err)
	}
}

func TestClientNotices(t *testing.T) {
	for _, c := range []struct {
		caps acp.ClientCapabilities
		want bool
	}{
		{acp.ClientCapabilities{}, false},
		{acp.ClientCapabilities{Session: &acp.ClientSessionCapabilities{}}, false},
		{acp.ClientCapabilities{Session: &acp.ClientSessionCapabilities{Notices: &acp.NoticeCapabilities{}}}, true},
	} {
		if got := ClientNotices(acp.InitializeRequest{ClientCapabilities: c.caps}); got != c.want {
			t.Errorf("%+v: %v", c.caps, got)
		}
	}
}

func TestMCPNotes(t *testing.T) {
	if n := MCPNotes(0); n != nil {
		t.Errorf("notes for no servers: %v", n)
	}
	n := MCPNotes(2)
	if len(n) != 1 || !strings.Contains(n[0], "ignoring 2 MCP server(s)") {
		t.Errorf("%v", n)
	}
}

// The kinds a request may give a tool are exactly ACP's, so Kind keeps
// every one the request accepted.
func TestRequestToolKindsAreACPs(t *testing.T) {
	all := []acp.ToolKind{acp.ToolKindRead, acp.ToolKindEdit, acp.ToolKindDelete, acp.ToolKindMove, acp.ToolKindSearch,
		acp.ToolKindExecute, acp.ToolKindThink, acp.ToolKindFetch, acp.ToolKindSwitchMode, acp.ToolKindOther}
	var want []string
	for _, k := range all {
		want = append(want, string(k))
	}
	if !slices.Equal(slices.Sorted(slices.Values(request.ToolKinds)), slices.Sorted(slices.Values(want))) {
		t.Fatalf("request.ToolKinds %v, ACP %v", request.ToolKinds, want)
	}
	for _, k := range request.ToolKinds {
		if got := Kind("t", map[string]acp.ToolKind{"t": acp.ToolKind(k)}); string(got) != k {
			t.Errorf("Kind(%s) = %s", k, got)
		}
	}
}

// TestInitializeTitle: a title becomes agentInfo.title, and no title
// leaves the field out of the JSON.
func TestInitializeTitle(t *testing.T) {
	r := InitializeResponse("1.2.3", "relay · hub")
	if err := validate("InitializeResponse", r); err != nil {
		t.Fatal(err)
	}
	if r.AgentInfo.Title == nil || *r.AgentInfo.Title != "relay · hub" {
		t.Fatalf("%+v", r.AgentInfo)
	}
	b, err := json.Marshal(InitializeResponse("1.2.3", ""))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"title"`) {
		t.Fatalf("empty title was sent: %s", b)
	}
}
