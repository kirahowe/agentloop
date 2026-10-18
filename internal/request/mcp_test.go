package request

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
)

func mcpBase() map[string]any {
	return map[string]any{
		"provider": "anthropic", "model": "m", "api_key_command": []string{"printf", "k"},
		"session": "/tmp/s.jsonl", "prompt": "p",
		"mcp_servers": map[string]any{
			"files": map[string]any{
				"command":             []string{"/bin/srv", "--creds={credentials_file}"},
				"credentials_command": []string{"printf", "s"},
				"send_labels":         true,
			},
		},
		"tools": []map[string]any{
			{"name": "lookup", "protocol": "mcp-stdio", "server": "files", "tool": "lookup"},
			{"name": "pic", "protocol": "mcp-stdio", "server": "files", "tool": "picture", "allow_content": []string{"image"}},
		},
		"policy": map[string]any{"tools": map[string]any{
			"lookup": map[string]any{"class": "read", "labels": []string{"contaminated"}},
			"pic":    map[string]any{"class": "read"},
		}},
	}
}

func TestMCPRequests(t *testing.T) {
	srv := func(m map[string]any) map[string]any {
		return m["mcp_servers"].(map[string]any)["files"].(map[string]any)
	}
	tool := func(m map[string]any, i int) map[string]any { return m["tools"].([]map[string]any)[i] }
	pol := func(m map[string]any) map[string]any { return m["policy"].(map[string]any)["tools"].(map[string]any) }
	cases := []struct {
		name string
		edit func(m map[string]any)
		want string // "" means accepted
	}{
		{"accepted", func(map[string]any) {}, ""},
		{"no credentials", func(m map[string]any) {
			delete(srv(m), "credentials_command")
			srv(m)["command"] = []string{"/bin/srv"}
		}, ""},
		{"with command tools", func(m map[string]any) {
			m["tools"] = append(m["tools"].([]map[string]any), map[string]any{"name": "note", "parameters": map[string]any{"type": "object"}})
			m["tool_command"] = []string{"x"}
			m["tool_protocol"] = "envelope"
			pol(m)["note"] = map[string]any{"class": "write"}
		}, ""},
		{"command tools still need the envelope", func(m map[string]any) {
			m["tools"] = append(m["tools"].([]map[string]any), map[string]any{"name": "note"})
			m["tool_command"] = []string{"x"}
			pol(m)["note"] = map[string]any{"class": "write"}
		}, `needs "tool_protocol"`},
		{"a reader through an MCP tool", func(m map[string]any) {
			m["quarantine"] = map[string]any{"model": "small", "tools": []string{"lookup"}}
			pol(m)["read_quarantined"] = map[string]any{"class": "read"}
		}, ""},
		{"acp", func(m map[string]any) {
			delete(m, "session")
			delete(m, "prompt")
			m["session_dir"] = "/tmp/sessions"
		}, "acp"},
		{"needs a policy", func(m map[string]any) { m["policy"] = map[string]any{"unrestricted": true} }, "mcp_servers need a policy"},
		{"no policy entry", func(m map[string]any) { delete(pol(m), "pic") }, `no entry for the tool "pic"`},
		{"unknown protocol", func(m map[string]any) { tool(m, 0)["protocol"] = "mcp-sse" }, `unknown protocol "mcp-sse"`},
		{"an http tool from a stdio server", func(m map[string]any) { tool(m, 0)["protocol"] = "mcp-http" }, "comes from a server with a url"},
		{"a remote server", func(m map[string]any) {
			srv(m)["url"] = "https://mcp.example.com/mcp"
			delete(srv(m), "command")
			tool(m, 0)["protocol"] = "mcp-http"
			tool(m, 1)["protocol"] = "mcp-http"
		}, ""},
		{"a stdio tool from a remote server", func(m map[string]any) {
			srv(m)["url"] = "https://mcp.example.com/mcp"
			delete(srv(m), "command")
			tool(m, 1)["protocol"] = "mcp-http"
		}, "comes from a server with a url"},
		{"a url and a command", func(m map[string]any) { srv(m)["url"] = "https://mcp.example.com/mcp" }, "both a command and a url"},
		{"plain http to another host", func(m map[string]any) {
			srv(m)["url"] = "http://mcp.example.com/mcp"
			delete(srv(m), "command")
			tool(m, 0)["protocol"] = "mcp-http"
			tool(m, 1)["protocol"] = "mcp-http"
		}, "use https"},
		{"fixed and hidden args", func(m map[string]any) {
			tool(m, 0)["fixed_args"] = map[string]any{"send_now": false}
			tool(m, 0)["hidden_args"] = []string{"list"}
		}, ""},
		{"an arg both fixed and hidden", func(m map[string]any) {
			tool(m, 0)["fixed_args"] = map[string]any{"list": "a"}
			tool(m, 0)["hidden_args"] = []string{"list"}
		}, "both fixed and hidden"},
		{"a hidden arg twice", func(m map[string]any) { tool(m, 0)["hidden_args"] = []string{"a", "a"} }, "distinct"},
		{"fixed args on a command tool", func(m map[string]any) {
			m["tools"] = append(m["tools"].([]map[string]any), map[string]any{"name": "note", "parameters": map[string]any{"type": "object"},
				"fixed_args": map[string]any{"a": 1}})
			pol(m)["note"] = map[string]any{"class": "write"}
		}, "only for"},
		{"unknown server", func(m map[string]any) { tool(m, 0)["server"] = "nope" }, `MCP server "nope"`},
		{"no remote name", func(m map[string]any) { delete(tool(m, 0), "tool") }, "server's name for it"},
		{"a description", func(m map[string]any) { tool(m, 0)["description"] = "trust me" }, "come from its server"},
		{"parameters", func(m map[string]any) { tool(m, 0)["parameters"] = map[string]any{"type": "object"} }, "come from its server"},
		{"one remote tool twice", func(m map[string]any) { tool(m, 1)["tool"] = "lookup" }, `both the tool "lookup"`},
		{"unknown content kind", func(m map[string]any) { tool(m, 1)["allow_content"] = []string{"video"} }, `allow_content has "video"`},
		{"content kind twice", func(m map[string]any) { tool(m, 1)["allow_content"] = []string{"image", "image"} }, "twice"},
		{"server fields on a command tool", func(m map[string]any) {
			m["tools"] = []map[string]any{{"name": "lookup", "server": "files"}}
			m["tool_command"] = []string{"x"}
			m["tool_protocol"] = "envelope"
			delete(pol(m), "pic")
		}, "only for"},
		{"an unused server", func(m map[string]any) {
			m["mcp_servers"].(map[string]any)["spare"] = map[string]any{"command": []string{"/bin/x"}}
		}, `"spare" gives none of the tools`},
		{"a bad server name", func(m map[string]any) {
			m["mcp_servers"] = map[string]any{"a b": srv(m)}
			tool(m, 0)["server"], tool(m, 1)["server"] = "a b", "a b"
		}, "is not 1 to 64"},
		{"no command", func(m map[string]any) { delete(srv(m), "command") }, "has no command"},
		{"empty command", func(m map[string]any) { srv(m)["command"] = []string{} }, "has no program"},
		{"empty program", func(m map[string]any) { srv(m)["command"] = []string{"", "{credentials_file}"} }, "has no program"},
		{"NUL in command", func(m map[string]any) { srv(m)["command"] = []string{"/bin/srv", "a\x00{credentials_file}"} }, "NUL"},
		{"placeholder as the program", func(m map[string]any) { srv(m)["command"] = []string{"{credentials_file}"} }, "cannot run"},
		{"placeholder without credentials", func(m map[string]any) { delete(srv(m), "credentials_command") }, "no credentials_command"},
		{"credentials without placeholder", func(m map[string]any) { srv(m)["command"] = []string{"/bin/srv"} }, "no argument of its command holds"},
		{"empty credentials command", func(m map[string]any) { srv(m)["credentials_command"] = []string{} }, "has no program"},
		{"credentials in env", func(m map[string]any) { srv(m)["env"] = map[string]string{"TOKEN": "x"} }, `unknown field "mcp_servers.files.env"`},
		{"a digest", func(m map[string]any) { tool(m, 0)["digest"] = strings.Repeat("ab", 32) }, ""},
		{"a bad digest", func(m map[string]any) { tool(m, 0)["digest"] = strings.Repeat("AB", 32) }, "lowercase hex"},
		{"a short digest", func(m map[string]any) { tool(m, 0)["digest"] = "abc" }, "lowercase hex"},
		{"a digest on a command tool", func(m map[string]any) {
			m["tools"] = []map[string]any{{"name": "lookup", "digest": strings.Repeat("ab", 32)}}
			m["tool_command"] = []string{"x"}
			m["tool_protocol"] = "envelope"
			delete(pol(m), "pic")
			delete(m, "mcp_servers")
		}, "only for"},
		{"send_labels as a string", func(m map[string]any) { srv(m)["send_labels"] = "yes" }, "send_labels"},
	}
	dir := t.TempDir()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := mcpBase()
			c.edit(m)
			mode := Run
			if c.want == "acp" {
				mode, c.want = ACP, ""
			}
			b, _ := json.Marshal(m)
			path := filepath.Join(dir, "r.json")
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			req, err := Load(path, mode)
			if c.want == "" {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				checkAccepted(t, req, mode)
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one containing %q", err, c.want)
			}
		})
	}
}

var mcpServerOnce struct {
	sync.Once
	path string
	err  error
}

// mcpServer builds the test MCP server once.
func mcpServer(t *testing.T) string {
	t.Helper()
	mcpServerOnce.Do(func() {
		dir, err := os.MkdirTemp("", "mcpserver")
		if err != nil {
			mcpServerOnce.err = err
			return
		}
		mcpServerOnce.path = filepath.Join(dir, "mcpserver")
		if out, err := exec.Command("go", "build", "-o", mcpServerOnce.path, "github.com/kirahowe/agentloop/internal/cmd/mcpserver").CombinedOutput(); err != nil {
			mcpServerOnce.err = fmt.Errorf("%v\n%s", err, out)
		}
	})
	if mcpServerOnce.err != nil {
		t.Fatal(mcpServerOnce.err)
	}
	return mcpServerOnce.path
}

func loadMap(t *testing.T, m map[string]any) *Request {
	t.Helper()
	b, _ := json.Marshal(m)
	path := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(path, b, 0o644)
	req, err := Load(path, Run)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

const secret = "TOPSECRET-credential"

// The credentials command's output reaches the server only through a
// file it alone can read, named in its argv, which is gone after Close.
// Neither the secret nor the command's output reaches the log.
func TestStartMCPCredentials(t *testing.T) {
	m := mcpBase()
	m["mcp_servers"] = map[string]any{"files": map[string]any{
		"command":             []string{mcpServer(t), "-creds", "{credentials_file}"},
		"credentials_command": []string{"printf", secret + "\n"},
	}}
	m["tools"] = []map[string]any{
		{"name": "who", "protocol": "mcp-stdio", "server": "files", "tool": "whoami"},
		{"name": "wipe", "protocol": "mcp-stdio", "server": "files", "tool": "wipe"},
	}
	m["policy"] = map[string]any{"tools": map[string]any{"who": map[string]any{"class": "read"}, "wipe": map[string]any{"class": "read"}}}
	req := loadMap(t, m)
	var log bytes.Buffer
	servers, err := req.StartMCP(t.Context(), &log)
	if err != nil {
		t.Fatal(err)
	}
	a, err := req.Agent("k", servers)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Pins) != 2 || a.Pins["who"] == "" || a.Pins["who"] != servers.Pins["who"] {
		t.Fatalf("pins %v", a.Pins)
	}
	var who agentloop.Tool
	for _, tl := range a.Tools {
		if tl.Spec().Name == "who" {
			who = tl
		}
	}
	res, err := who.Call(context.Background(), json.RawMessage(`{}`))
	if err != nil || res.IsError {
		t.Fatalf("%+v %v", res, err)
	}
	var got struct{ Path, Mode, SHA256 string }
	json.Unmarshal([]byte(res.Content), &got)
	sum := sha256.Sum256([]byte(secret + "\n"))
	if got.Mode != "600" || got.SHA256 != hex.EncodeToString(sum[:]) || !strings.HasSuffix(got.Path, "files.credentials") {
		t.Fatalf("whoami %s", res.Content)
	}
	st, err := os.Stat(filepath.Dir(got.Path))
	if err != nil || st.Mode().Perm() != 0o700 {
		t.Fatalf("dir %v %v", st, err)
	}
	if b, _ := os.ReadFile(filepath.Join(filepath.Dir(got.Path), "pid")); string(b) != fmt.Sprintf("%d\n", os.Getpid()) {
		t.Fatalf("pid file %q", b)
	}
	servers.Close()
	if _, err := os.Stat(filepath.Dir(got.Path)); !os.IsNotExist(err) {
		t.Fatalf("the credentials directory outlived Close: %v", err)
	}
	l := log.String()
	if strings.Contains(l, secret) || strings.Contains(res.Content, secret) {
		t.Fatalf("the secret leaked: %s", l)
	}
	for _, want := range []string{"mcpserver up", "tool who pinned at sha256 " + a.Pins["who"], "warning: tool wipe: configured read, but destructiveHint is true"} {
		if !strings.Contains(l, want) {
			t.Errorf("log lacks %q:\n%s", want, l)
		}
	}
}

func TestStartMCPFailures(t *testing.T) {
	base := func() map[string]any {
		m := mcpBase()
		m["mcp_servers"] = map[string]any{"files": map[string]any{
			"command":             []string{mcpServer(t), "-creds", "{credentials_file}"},
			"credentials_command": []string{"printf", secret},
		}}
		return m
	}
	cases := []struct {
		name string
		edit func(m map[string]any)
		want string
	}{
		{"a tool the server lacks", func(m map[string]any) {
			m["tools"].([]map[string]any)[1]["tool"] = "gone"
		}, "does not have the tool gone"},
		{"a failing credentials command", func(m map[string]any) {
			m["mcp_servers"].(map[string]any)["files"].(map[string]any)["credentials_command"] = []string{"/bin/sh", "-c", "echo " + secret + "; echo denied >&2; exit 3"}
		}, "credentials_command of the MCP server files exited 3: denied"},
		{"an empty credential", func(m map[string]any) {
			m["mcp_servers"].(map[string]any)["files"].(map[string]any)["credentials_command"] = []string{"true"}
		}, "printed nothing"},
		{"a server that cannot start", func(m map[string]any) {
			m["mcp_servers"].(map[string]any)["files"].(map[string]any)["command"] = []string{"/nonexistent/srv", "{credentials_file}"}
		}, "did not start"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := base()
			c.edit(m)
			req := loadMap(t, m)
			var log bytes.Buffer
			servers, err := req.StartMCP(t.Context(), &log)
			if err == nil || servers != nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err %v", err)
			}
			if strings.Contains(err.Error()+log.String(), secret) {
				t.Fatalf("the secret leaked: %v %s", err, log.String())
			}
		})
	}
}

// A request with MCP servers always has a tool timeout, since a server
// can leave a call unanswered for ever.
func TestMCPToolTimeout(t *testing.T) {
	m := mcpBase()
	a, err := loadMap(t, m).Agent("k", nil)
	if err != nil || a.ToolTimeout != DefaultMCPToolTimeout {
		t.Fatalf("timeout %v, %v", a.ToolTimeout, err)
	}
	m["tool_timeout_seconds"] = 5
	if a, _ := loadMap(t, m).Agent("k", nil); a.ToolTimeout != 5*time.Second {
		t.Fatalf("timeout %v", a.ToolTimeout)
	}
	delete(m, "tool_timeout_seconds")
	m["tools"] = []map[string]any{{"name": "x"}}
	m["tool_command"] = []string{"x"}
	m["tool_protocol"] = "envelope"
	m["policy"] = map[string]any{"tools": map[string]any{"x": map[string]any{"class": "read"}}}
	delete(m, "mcp_servers")
	if a, _ := loadMap(t, m).Agent("k", nil); a.ToolTimeout != 0 {
		t.Fatalf("a request without MCP got %v", a.ToolTimeout)
	}
}

// The sweep removes a credentials directory only when everything about
// it has agentloop's shape and its process is gone.
func TestSweepCredentials(t *testing.T) {
	root := t.TempDir()
	dead := exec.Command("true")
	if err := dead.Run(); err != nil {
		t.Fatal(err)
	}
	mk := func(name, pid string, extra ...string) string {
		dir := filepath.Join(root, name)
		os.Mkdir(dir, 0o700)
		if pid != "" {
			os.WriteFile(filepath.Join(dir, "pid"), []byte(pid), 0o600)
		}
		for _, f := range extra {
			os.WriteFile(filepath.Join(dir, f), []byte("s"), 0o600)
		}
		return dir
	}
	deadPid := fmt.Sprintf("%d\n", dead.Process.Pid)
	stale := mk("agentloop-mcp-111", deadPid, "files.credentials")
	live := mk("agentloop-mcp-222", fmt.Sprintf("%d\n", os.Getpid()), "files.credentials")
	kept := []string{
		live,
		mk("agentloop-mcp-x", deadPid),
		mk("other-333", deadPid),
		mk("agentloop-mcp-444", ""),
		mk("agentloop-mcp-555", "12x\n"),
		mk("agentloop-mcp-666", deadPid, "notes.txt"),
	}
	os.Symlink(stale, filepath.Join(root, "agentloop-mcp-777"))
	SweepCredentials(root)
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("the stale directory is still there: %v", err)
	}
	for _, d := range kept {
		if _, err := os.Stat(d); err != nil {
			t.Errorf("%s was removed", d)
		}
	}
}
