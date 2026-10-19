package request

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/workbench"
)

func workbenchBase() map[string]any {
	m := quarantineBase()
	m["workbench"] = map[string]any{"deny_read": []string{"/Users/k/private", "/Users/k/.secret*"}}
	m["policy"].(map[string]any)["tools"].(map[string]any)["workbench"] = map[string]any{"class": "read"}
	return m
}

func TestWorkbenchRequests(t *testing.T) {
	w := func(m map[string]any) map[string]any { return m["workbench"].(map[string]any) }
	pol := func(m map[string]any) map[string]any { return m["policy"].(map[string]any)["tools"].(map[string]any) }
	cases := []struct {
		name string
		edit func(m map[string]any)
		want string
	}{
		{"accepted", func(map[string]any) {}, ""},
		{"its own model and limits", func(m map[string]any) {
			w(m)["model"] = "other"
			w(m)["effort"] = "high"
			w(m)["max_turns"] = 5
			w(m)["max_tokens"] = 1000
			w(m)["retries"] = 0
			w(m)["command_timeout_seconds"] = 30
			w(m)["output_bytes"] = 4096
			w(m)["processes"] = 8
			w(m)["scratch_bytes"] = 1000000
		}, ""},
		{"no quarantine block, its own model", func(m map[string]any) {
			delete(m, "quarantine")
			delete(pol(m), "read_quarantined")
			w(m)["model"] = "other"
		}, ""},
		{"no model anywhere", func(m map[string]any) {
			delete(m, "quarantine")
			delete(pol(m), "read_quarantined")
		}, "workbench has no model"},
		{"needs a policy", func(m map[string]any) {
			m["policy"] = map[string]any{"unrestricted": true}
			delete(m, "tool_protocol")
			delete(m, "quarantine")
		}, "workbench needs a policy"},
		{"no policy entry", func(m map[string]any) { delete(pol(m), "workbench") }, `no entry for the tool "workbench"`},
		{"not class read", func(m map[string]any) { pol(m)["workbench"] = map[string]any{"class": "egress"} }, "give workbench the class read"},
		{"bad effort", func(m map[string]any) { w(m)["effort"] = "huge" }, "workbench.effort"},
		{"negative numbers", func(m map[string]any) { w(m)["max_turns"] = -1 }, "cannot be negative"},
		{"too many retries", func(m map[string]any) { w(m)["retries"] = MaxQuarantineRetries + 1 }, "workbench.retries"},
		{"a relative deny path", func(m map[string]any) { w(m)["deny_read"] = []string{"private"} }, "not absolute"},
		{"a deny path that would escape its string", func(m map[string]any) { w(m)["deny_read"] = []string{`/a"b`} }, "deny_read"},
		{"a tool named like it", func(m map[string]any) {
			m["tools"] = append(m["tools"].([]map[string]any), map[string]any{"name": "workbench"})
		}, "is built in"},
		{"it cannot be an inner reader tool", func(m map[string]any) {
			m["quarantine"].(map[string]any)["tools"] = []string{"workbench"}
		}, `"workbench", which is not one of the tools`},
		{"unknown field", func(m map[string]any) { w(m)["sandbox"] = "off" }, "unknown field"},
	}
	dir := t.TempDir()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := workbench.Available(); err != nil {
				t.Skip(err)
			}
			m := workbenchBase()
			c.edit(m)
			b, _ := json.Marshal(m)
			path := filepath.Join(dir, "r.json")
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			req, err := Load(path, Run)
			if c.want == "" {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				checkAccepted(t, req, Run)
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one containing %q", err, c.want)
			}
		})
	}
}

func TestWorkbenchIsBuiltFromTheRequest(t *testing.T) {
	if err := workbench.Available(); err != nil {
		t.Skip(err)
	}
	m := workbenchBase()
	m["workbench"].(map[string]any)["max_turns"] = 7
	m["workbench"].(map[string]any)["command_timeout_seconds"] = 12
	b, _ := json.Marshal(m)
	path := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(path, b, 0o644)
	req, err := Load(path, Run)
	if err != nil {
		t.Fatal(err)
	}
	a, err := req.Agent("k", req.stubMCP())
	if err != nil {
		t.Fatal(err)
	}
	var bench *workbench.Workbench
	for _, tool := range a.Tools {
		if w, ok := tool.(*workbench.Workbench); ok {
			bench = w
		}
	}
	if bench == nil {
		t.Fatal("no workbench tool")
	}
	// The model is the reader's, since the block names none.
	if bench.ModelID != "small" || bench.Effort != agentloop.EffortLow || bench.MaxTurns != 7 ||
		bench.MaxTokens != WorkbenchLimits.MaxTokens || bench.Limits.Timeout != 12*1e9 {
		t.Errorf("workbench = %+v", bench)
	}
	deny := bench.Deny()
	if !slices.Contains(deny, "/Users/k/private") || !slices.Contains(deny, "/Users/k/.secret*") {
		t.Errorf("deny list %v lacks the request's paths", deny)
	}
	// The ones it always denies are there too.
	home, _ := os.UserHomeDir()
	if !slices.Contains(deny, workbench.Resolve(home)+"/.ssh") {
		t.Errorf("deny list %v lacks the default paths", deny)
	}
}
