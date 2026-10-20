package acpagent

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/kirahowe/agentloop/internal/strictjson/stricttest"
	"pgregory.net/rapid"
)

// screened runs the client's input through the screen and returns what
// the SDK would read and what the client was sent.
func screened(t tb, in []byte) (sdk, answers []byte, err error) {
	t.Helper()
	var out bytes.Buffer
	s := newStrictInput(bytes.NewReader(in), &serialWriter{w: &out}, func(string, ...any) {})
	sdk, err = io.ReadAll(s)
	return sdk, out.Bytes(), err
}

// Every line of the shared table that relay and acp-tui accept reaches
// the SDK, and the SDK reads the method the table expects from it: the
// table's case variants ("Method", "paramſ") arrive without the folded
// member, which the dialect ignores and v1 would have read. A line with
// nothing to remove arrives byte for byte. A line the table refuses never
// reaches the SDK. None of them is a request, so none is answered.
func TestStrictInputCases(t *testing.T) {
	for _, c := range stricttest.Cases(t) {
		t.Run(c.Name, func(t *testing.T) {
			for _, end := range []string{"\n", "\r\n", ""} {
				line := append(bytes.Clone(c.Line), end...)
				sdk, answers, err := screened(t, line)
				if err != nil || len(answers) > 0 {
					t.Fatalf("err %v, answers %q", err, answers)
				}
				if c.Reject {
					if len(sdk) > 0 {
						t.Fatalf("passed %q", sdk)
					}
					continue
				}
				// A rewritten line loses whitespace, a \r included.
				if !bytes.HasSuffix(sdk, []byte(strings.TrimPrefix(end, "\r"))) {
					t.Fatalf("passed %q without its newline", sdk)
				}
				// What the SDK's own envelope, decoded with v1, reads.
				var env struct {
					Method *string `json:"method"`
				}
				if err := json.Unmarshal(sdk, &env); err != nil {
					t.Fatalf("v1 cannot read %q: %v", sdk, err)
				}
				if (env.Method == nil) != (c.Method == nil) || (env.Method != nil && *env.Method != *c.Method) {
					t.Fatalf("the SDK reads method %v from %q, want %v", show(env.Method), sdk, show(c.Method))
				}
				if _, removed := withoutFolded(c.Line); len(removed) == 0 && !bytes.Equal(sdk, line) {
					t.Fatalf("passed %q, want %q", sdk, line)
				}
			}
		})
	}
}

func show(s *string) string {
	if s == nil {
		return "none"
	}
	return strconv.Quote(*s)
}

func TestStrictInputAnswersRequests(t *testing.T) {
	cases := []struct {
		name, line, id string // id "" means no answer
	}{
		{"repeated method", `{"jsonrpc":"2.0","id":7,"method":"session/prompt","method":"_relay/x"}`, `7`},
		{"invalid UTF-8", "{\"jsonrpc\":\"2.0\",\"id\":\"a\",\"method\":\"m\",\"params\":{\"s\":\"\xff\"}}", `"a"`},
		{"trailing value", `{"jsonrpc":"2.0","id":1,"method":"m"} null`, `1`},
		{"escaped id", `{"jsonrpc":"2.0","id":"xé","method":"m","params":{"a":1,"a":2}}`, `"xé"`},
		{"too deep", `{"jsonrpc":"2.0","id":2,"method":"m","params":` + strings.Repeat("[", 200) + strings.Repeat("]", 200) + `}`, `2`},
		{"repeated id", `{"jsonrpc":"2.0","id":1,"id":2,"method":"m"}`, ""},
		{"no id", `{"jsonrpc":"2.0","method":"m","method":"n"}`, ""},
		{"null id", `{"jsonrpc":"2.0","id":null,"method":"m","method":"n"}`, ""},
		{"object id", `{"jsonrpc":"2.0","id":{},"method":"m","method":"n"}`, ""},
		{"id with invalid UTF-8", "{\"jsonrpc\":\"2.0\",\"id\":\"\xff\",\"method\":\"m\"}", ""},
		{"response", `{"jsonrpc":"2.0","id":3,"result":{},"result":{"outcome":"allow"}}`, ""},
		{"not JSON", `{"jsonrpc":"2.0","id":4,"method":"m"`, ""},
		{"capitalised id is not the id", `{"jsonrpc":"2.0","ID":5,"method":"m","method":"n"}`, ""},
	}
	for _, c := range cases {
		sdk, answers, err := screened(t, []byte(c.line+"\n"))
		if err != nil || len(sdk) > 0 {
			t.Errorf("%s: err %v, passed %q", c.name, err, sdk)
			continue
		}
		if c.id == "" {
			if len(answers) > 0 {
				t.Errorf("%s: answered %q", c.name, answers)
			}
			continue
		}
		want := `{"jsonrpc":"2.0","id":` + c.id + `,"error":{"code":-32700,"message":"Parse error"}}` + "\n"
		if string(answers) != want {
			t.Errorf("%s: answered %q, want %q", c.name, answers, want)
		}
	}
}

// A member whose name v1 would fold onto an SDK field is removed, at any
// depth, and the rest of the line reaches the SDK with its values as
// written. To relay and acp-tui such a member is unknown, so after the
// removal the SDK reads what they read.
func TestStrictInputRemovesFoldedNames(t *testing.T) {
	cases := []struct{ name, line, want string }{
		{"capitalised method", `{"jsonrpc":"2.0","id":6,"Method":"session/new"}`, `{"jsonrpc":"2.0","id":6}`},
		{"capitalised id", `{"jsonrpc":"2.0","ID":6,"method":"session/new"}`, `{"jsonrpc":"2.0","method":"session/new"}`},
		{"method and a folded method", `{"jsonrpc":"2.0","id":8,"method":"session/new","Method":"session/list"}`, `{"jsonrpc":"2.0","id":8,"method":"session/new"}`},
		{"folded name in params", `{"jsonrpc":"2.0","id":9,"method":"session/new","params":{"cwd":"/","Cwd":"/etc","mcpServers":[]}}`, `{"jsonrpc":"2.0","id":9,"method":"session/new","params":{"cwd":"/","mcpServers":[]}}`},
		{"long s in params", "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"session/prompt\",\"params\":{\"ſessionId\":\"x\"}}", `{"jsonrpc":"2.0","id":10,"method":"session/prompt","params":{}}`},
		{"escaped folded name", `{"jsonrpc":"2.0","method":"x","\u004dethod":"y"}`, `{"jsonrpc":"2.0","method":"x"}`},
		{"values kept as written", `{"jsonrpc":"2.0", "id":1,"method":"m","params":{"n":123456789012345678901234567890,"d":0.1000000000000000055511151231257827,"s":"\u00e9\/","Cwd":1}}`,
			`{"jsonrpc":"2.0","id":1,"method":"m","params":{"n":123456789012345678901234567890,"d":0.1000000000000000055511151231257827,"s":"\u00e9\/"}}`},
		{"in an array", `{"jsonrpc":"2.0","method":"m","params":{"l":[{"Cwd":1,"x":[{"Method":2}]}]}}`, `{"jsonrpc":"2.0","method":"m","params":{"l":[{"x":[{}]}]}}`},
		{"_meta kept whole", `{"jsonrpc":"2.0","method":"m","params":{"_meta":{"Cwd":1},"Cwd":2}}`, `{"jsonrpc":"2.0","method":"m","params":{"_meta":{"Cwd":1}}}`},
	}
	for _, c := range cases {
		sdk, answers, err := screened(t, []byte(c.line+"\n"))
		if err != nil || len(answers) > 0 || string(sdk) != c.want+"\n" {
			t.Errorf("%s: err %v, answers %q, passed %q, want %q", c.name, err, answers, sdk, c.want)
		}
	}
}

func TestStrictInputLineTooLong(t *testing.T) {
	ok := append(append([]byte(`{"s":"`), bytes.Repeat([]byte("a"), MaxLine-8)...), "\"}\n"...)
	if sdk, _, err := screened(t, ok); err != nil || !bytes.Equal(sdk, ok) {
		t.Fatalf("a line of MaxLine bytes: err %v, %d bytes passed", err, len(sdk))
	}
	long := append(append([]byte(`{"s":"`), bytes.Repeat([]byte("a"), MaxLine)...), "\"}\n{}\n"...)
	if sdk, _, err := screened(t, long); !errors.Is(err, errLineTooLong) || len(sdk) > 0 {
		t.Fatalf("err %v, %d bytes passed", err, len(sdk))
	}
}

// Whatever lines the client writes, the SDK reads exactly the ones that
// pass the strict reader, in order, with no folded name left in them, and
// unchanged when they had none.
func TestPropertyStrictInput(t *testing.T) {
	cases := stricttest.Cases(t)
	rapid.Check(t, func(t *rapid.T) {
		var in, want []byte
		for range rapid.IntRange(0, 6).Draw(t, "lines") {
			var line []byte
			pass := true
			switch rapid.IntRange(0, 2).Draw(t, "kind") {
			case 0:
				c := rapid.SampledFrom(cases).Draw(t, "case")
				line, pass = c.Line, !c.Reject
			case 1:
				m := rapid.MapOfN(rapid.String(), rapid.String(), 0, 4).Draw(t, "object")
				line, _ = json.Marshal(m)
			default:
				line = []byte(rapid.SampledFrom([]string{"", " ", "\t", "\r"}).Draw(t, "blank"))
				pass = false
			}
			in = append(in, append(bytes.Clone(line), '\n')...)
			if pass {
				out, removed := withoutFolded(line)
				if len(removed) == 0 && !bytes.Equal(out, line) {
					t.Fatalf("changed %q to %q with nothing to remove", line, out)
				}
				if again, more := withoutFolded(out); len(more) > 0 || !bytes.Equal(again, out) {
					t.Fatalf("%q still holds folded names %q", out, more)
				}
				want = append(want, append(bytes.Clone(out), '\n')...)
			}
		}
		sdk, answers, err := screened(t, in)
		if err != nil || !bytes.Equal(sdk, want) || len(answers) > 0 {
			t.Fatalf("err %v: passed %q, want %q; answers %q", err, sdk, want, answers)
		}
	})
}

// Through the server: a request that is not strict JSON gets a parse
// error for its id, and the connection goes on.
func TestServerRefusesALooseRequest(t *testing.T) {
	h := serve(t, textAgent(), Options{}, false)
	if _, err := h.toAgent.Write([]byte(`{"jsonrpc":"2.0","id":"loose","method":"session/new","method":"session/list","params":{"cwd":"/","mcpServers":[]}}` + "\n")); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		var got *message
		for _, m := range parseLines(t, h.toClient.snapshot()) {
			if string(m.ID) == `"loose"` {
				got = &m
			}
		}
		if got != nil {
			if !strings.Contains(string(got.Error), `-32700`) || got.Result != nil {
				t.Fatalf("answer %+v", got)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no answer")
		}
		time.Sleep(10 * time.Millisecond)
	}
	h.newSession()
}

// The screen's list of the names the SDK decodes covers every json tag
// in the SDK's source, and no two of them differ only in case, which a
// check by name alone could not tell apart.
func TestSDKNamesCoverTheSDK(t *testing.T) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/caelis-labs/acp-go-sdk").Output()
	if err != nil {
		t.Fatalf("go list: %v", err)
	}
	dir := strings.TrimSpace(string(out))
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no SDK source in %s: %v", dir, err)
	}
	tag := regexp.MustCompile("json:\"([^\",]+)")
	found := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range tag.FindAllSubmatch(src, -1) {
			name := string(m[1])
			if name == "-" {
				continue
			}
			found++
			if !slices.Contains(sdkNames[fold(name)], name) {
				t.Errorf("%s: json name %q is not in sdkNames; add its type to sdkRoots", filepath.Base(f), name)
			}
		}
	}
	if found < 100 {
		t.Fatalf("only %d json tags found in %s", found, dir)
	}
	for k, names := range sdkNames {
		if len(names) > 1 {
			t.Errorf("names %q differ only in case (fold %q)", names, k)
		}
	}
}

// Through the server: a folded name is read as an unknown member, as
// relay and acp-tui read it, whether it is the method, the id or a
// parameter, and the shared table's lines are never refused.
func TestServerIgnoresFoldedNames(t *testing.T) {
	h := serve(t, textAgent(), Options{}, false)
	lines := []string{
		// No method: a response to nothing, which the SDK drops.
		`{"jsonrpc":"2.0","id":"fold-method","Method":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","id":"fold-upper","METHOD":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		// session/list, not session/new.
		`{"jsonrpc":"2.0","id":"fold-both","method":"session/list","Method":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		// cwd "/", not "/etc".
		`{"jsonrpc":"2.0","id":"fold-param","method":"session/new","params":{"cwd":"/","mcpServers":[],"Cwd":"/etc"}}`,
	}
	for _, c := range stricttest.Cases(t) {
		if !c.Reject {
			lines = append(lines, string(c.Line))
		}
	}
	for _, l := range lines {
		if _, err := h.toAgent.Write([]byte(l + "\n")); err != nil {
			t.Fatal(err)
		}
	}
	// The screen reads lines in order and the SDK answers session/new
	// after the requests above, so they have all been read by now.
	h.newSession()
	got := map[string]message{}
	for _, m := range parseLines(t, h.toClient.snapshot()) {
		if strings.Contains(string(m.Error), "-32700") {
			t.Errorf("refused a line: %+v", m)
		}
		got[string(m.ID)] = m
	}
	for _, id := range []string{`"fold-method"`, `"fold-upper"`} {
		if m, ok := got[id]; ok {
			t.Errorf("answered %s: %+v", id, m)
		}
	}
	if m := got[`"fold-both"`]; !strings.Contains(string(m.Result), `"sessions"`) {
		t.Errorf("fold-both was not read as session/list: %+v", m)
	}
	if m := got[`"fold-param"`]; !strings.Contains(string(m.Result), `"sessionId"`) {
		t.Errorf("fold-param was not read as session/new: %+v", m)
	}
}

// fold agrees with bytes.EqualFold, the rule encoding/json v1 matches
// names by, on strings that differ only in the case of some letters.
func TestPropertyFoldIsEqualFold(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := rapid.String().Draw(t, "a")
		var b strings.Builder
		for _, r := range a {
			for range rapid.IntRange(0, 3).Draw(t, "turns") {
				r = unicode.SimpleFold(r)
			}
			b.WriteRune(r)
		}
		other := rapid.String().Draw(t, "other")
		for _, s := range []string{b.String(), other} {
			if (fold(a) == fold(s)) != bytes.EqualFold([]byte(a), []byte(s)) {
				t.Fatalf("fold(%q)=%q, fold(%q)=%q, EqualFold %v", a, fold(a), s, fold(s), bytes.EqualFold([]byte(a), []byte(s)))
			}
		}
	})
}

// Every SDK field named in freeForm holds raw JSON, an any or a string,
// so v1 matches no name inside its value, and withoutFolded may skip it.
func TestFreeFormNamesAreFreeForm(t *testing.T) {
	ok := map[reflect.Type]bool{
		reflect.TypeFor[map[string]json.RawMessage](): true,
		reflect.TypeFor[json.RawMessage]():            true,
		reflect.TypeFor[any]():                        true,
		reflect.TypeFor[string]():                     true,
	}
	found := map[string]int{}
	seen := map[reflect.Type]bool{}
	var walk func(reflect.Type)
	walk = func(rt reflect.Type) {
		for rt.Kind() == reflect.Pointer || rt.Kind() == reflect.Slice || rt.Kind() == reflect.Array || rt.Kind() == reflect.Map {
			rt = rt.Elem()
		}
		if rt.Kind() != reflect.Struct || seen[rt] {
			return
		}
		seen[rt] = true
		for i := range rt.NumField() {
			f := rt.Field(i)
			name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			if freeForm[name] {
				found[name]++
				if !ok[f.Type] {
					t.Errorf("%s.%s, json name %q, is a %s, which v1 may decode names into", rt, f.Name, name, f.Type)
				}
			}
			walk(f.Type)
		}
	}
	for _, r := range sdkRoots {
		walk(reflect.TypeOf(r))
	}
	for name := range freeForm {
		if found[name] == 0 {
			t.Errorf("no SDK field is named %q", name)
		}
	}
}

// Names inside a free-form value pass: _meta and an error's data are
// kept as raw JSON or decoded into any, so v1 folds nothing there. The
// same names elsewhere, and a folded _meta itself, are removed.
func TestStrictInputFreeFormNames(t *testing.T) {
	pass := []string{
		`{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{"Title":"hello","Cwd":{"Method":1}}}}`,
		`{"jsonrpc":"2.0","id":3,"error":{"code":-32603,"message":"m","data":{"Message":"x","Path":"/"}}}`,
		`{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s","_meta":{"SessionId":"t"}}}`,
	}
	for _, l := range pass {
		line := []byte(l + "\n")
		if sdk, answers, err := screened(t, line); err != nil || !bytes.Equal(sdk, line) || len(answers) > 0 {
			t.Errorf("%s: err %v, passed %q, answered %q", l, err, sdk, answers)
		}
	}
	removed := map[string]string{
		`{"jsonrpc":"2.0","id":4,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_Meta":{"x":1},"Cwd":"/etc"}}`: `{"jsonrpc":"2.0","id":4,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","id":5,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{},"Cwd":"/etc"}}`:      `{"jsonrpc":"2.0","id":5,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{}}}`,
		`{"jsonrpc":"2.0","id":6,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{"x":{}}},"Params":{}}`: `{"jsonrpc":"2.0","id":6,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{"x":{}}}}`,
	}
	for l, want := range removed {
		if sdk, _, err := screened(t, []byte(l+"\n")); err != nil || string(sdk) != want+"\n" {
			t.Errorf("%s: err %v, passed %q", l, err, sdk)
		}
	}
}
