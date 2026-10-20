package acpagent

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

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

// Every line of the shared table reaches the SDK byte for byte exactly
// when relay and acp-tui accept it; none of them is a request, so none is
// answered.
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
				if !c.Reject && !bytes.Equal(sdk, line) {
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
// pass the strict reader, in order and unchanged.
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
