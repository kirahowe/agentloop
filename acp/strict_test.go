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
// the SDK byte for byte, unless it holds a member name the SDK would fold
// onto another (the table's case variants), which is dropped; a line they
// refuse never reaches it. None of them is a request, so none is answered.
func TestStrictInputCases(t *testing.T) {
	for _, c := range stricttest.Cases(t) {
		t.Run(c.Name, func(t *testing.T) {
			for _, end := range []string{"\n", "\r\n", ""} {
				line := append(bytes.Clone(c.Line), end...)
				sdk, answers, err := screened(t, line)
				if err != nil || len(answers) > 0 {
					t.Fatalf("err %v, answers %q", err, answers)
				}
				if c.Reject && len(sdk) > 0 {
					t.Fatalf("passed %q", sdk)
				}
				if !c.Reject && foldedName(c.Line) != nil && len(sdk) > 0 {
					t.Fatalf("passed a folded name: %q", sdk)
				}
				if !c.Reject && foldedName(c.Line) == nil && !bytes.Equal(sdk, line) {
					t.Fatalf("passed %q, want %q", sdk, line)
				}
			}
		})
	}
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
		{"capitalised method", `{"jsonrpc":"2.0","id":6,"Method":"session/new"}`, ""},
		{"capitalised id", `{"jsonrpc":"2.0","ID":6,"method":"session/new"}`, ""},
		{"method and a folded method", `{"jsonrpc":"2.0","id":8,"method":"session/new","Method":"session/list"}`, `8`},
		{"folded name in params", `{"jsonrpc":"2.0","id":9,"method":"session/new","params":{"cwd":"/","Cwd":"/etc","mcpServers":[]}}`, `9`},
		{"long s in params", "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"session/prompt\",\"params\":{\"ſessionId\":\"x\"}}", `10`},
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
// pass the strict reader and hold no folded name, in order and unchanged.
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
				line, pass = c.Line, !c.Reject && foldedName(c.Line) == nil
			case 1:
				m := rapid.MapOfN(rapid.String(), rapid.String(), 0, 4).Draw(t, "object")
				line, _ = json.Marshal(m)
				pass = foldedName(line) == nil
			default:
				line = []byte(rapid.SampledFrom([]string{"", " ", "\t", "\r"}).Draw(t, "blank"))
				pass = false
			}
			line = append(bytes.Clone(line), '\n')
			in = append(in, line...)
			if pass {
				want = append(want, line...)
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

// Through the server: a line whose names the SDK would fold onto its
// own is never dispatched and never answered as a request, whether the
// folded name is the method, the id or a parameter.
func TestServerDoesNotFoldNames(t *testing.T) {
	h := serve(t, textAgent(), Options{}, false)
	lines := []string{
		`{"jsonrpc":"2.0","id":"fold-method","Method":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","id":"fold-upper","METHOD":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","ID":"fold-id","method":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","id":"fold-both","method":"session/list","Method":"session/new","params":{"cwd":"/","mcpServers":[]}}`,
		`{"jsonrpc":"2.0","id":"fold-param","method":"session/new","params":{"cwd":"/","mcpServers":[],"Cwd":"/etc"}}`,
		"{\"jsonrpc\":\"2.0\",\"id\":\"fold-s\",\"method\":\"session/new\",\"paramſ\":{\"cwd\":\"/\",\"mcpServers\":[]}}",
	}
	for _, c := range stricttest.Cases(t) {
		if !c.Reject && foldedName(c.Line) != nil {
			lines = append(lines, string(c.Line))
		}
	}
	for _, l := range lines {
		if _, err := h.toAgent.Write([]byte(l + "\n")); err != nil {
			t.Fatal(err)
		}
	}
	// The screen reads lines in order, so once a later request is
	// answered every line above has been dropped or answered.
	h.newSession()
	for _, m := range parseLines(t, h.toClient.snapshot()) {
		id := string(m.ID)
		if !strings.HasPrefix(id, `"fold-`) {
			continue
		}
		// A line with an exact id and method is answered with a parse
		// error; one whose id or method is folded is not a request.
		answered := id == `"fold-both"` || id == `"fold-param"` || id == `"fold-s"`
		if m.Result != nil || !answered || !strings.Contains(string(m.Error), "-32700") {
			t.Errorf("answer to a folded line: %+v", m)
		}
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
// so v1 matches no name inside its value, and foldedName may skip it.
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
// same names elsewhere, and a folded _meta itself, are still refused.
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
	refuse := []string{
		`{"jsonrpc":"2.0","id":4,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_Meta":{"x":1},"Cwd":"/etc"}}`,
		`{"jsonrpc":"2.0","id":5,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{},"Cwd":"/etc"}}`,
		`{"jsonrpc":"2.0","id":6,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"_meta":{"x":{}}},"Params":{}}`,
	}
	for _, l := range refuse {
		if sdk, _, err := screened(t, []byte(l+"\n")); err != nil || len(sdk) > 0 {
			t.Errorf("%s: err %v, passed %q", l, err, sdk)
		}
	}
}
