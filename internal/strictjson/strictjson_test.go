package strictjson_test

import (
	"bytes"
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/kirahowe/agentloop/internal/strictjson"
	"github.com/kirahowe/agentloop/internal/strictjson/stricttest"
	"pgregory.net/rapid"
)

// Every line of the shared table is read as relay and relay-tui read it.
func TestCases(t *testing.T) {
	for _, c := range stricttest.Cases(t) {
		t.Run(c.Name, func(t *testing.T) {
			var m struct {
				Method *string `json:"method"`
			}
			err := strictjson.Object(c.Line)
			if err == nil {
				err = strictjson.Unmarshal(c.Line, &m)
			}
			if c.Reject {
				if err == nil {
					t.Fatalf("accepted %q", c.Line)
				}
				return
			}
			if err != nil {
				t.Fatalf("refused: %v", err)
			}
			if !reflect.DeepEqual(m.Method, c.Method) {
				t.Fatalf("method %v, want %v", deref(m.Method), deref(c.Method))
			}
		})
	}
}

func deref(s *string) any {
	if s == nil {
		return nil
	}
	return *s
}

type inner struct {
	Class  string   `json:"class"`
	Labels []string `json:"labels,omitempty"`
}

type outer struct {
	Name   string            `json:"name"`
	Tools  map[string]inner  `json:"tools"`
	List   []inner           `json:"list"`
	Raw    jsonv1.RawMessage `json:"raw"`
	Ptr    *inner            `json:"ptr"`
	Wait   time.Duration     `json:"wait"`
	hidden int
}

func TestUnmarshalKnown(t *testing.T) {
	good := []string{
		`{}`, `{"name":"x"}`,
		`{"tools":{"a":{"class":"read"},"A":{"class":"read","labels":[]}}}`,
		`{"list":[{"class":"x"},{"labels":["p"]}],"raw":{"Any":1,"any":2},"ptr":null}`,
		`{"ptr":{"class":"c"}}`, `{"wait":1500000000}`,
	}
	for _, g := range good {
		var o outer
		if err := strictjson.UnmarshalKnown([]byte(g), &o); err != nil {
			t.Errorf("%s: %v", g, err)
		}
	}
	bad := []string{
		`{"name":"x","name":"y"}`,
		`{"Name":"x"}`,
		`{"NAME":"x"}`,
		`{"tools":{"a":{"class":"read"},"a":{"class":"act"}}}`,
		`{"tools":{"a":{"Class":"read"}}}`,
		`{"list":[{"labels":["p"],"labels":[]}]}`,
		`{"ptr":{"labels":[],"Labels":[]}}`,
		`{"raw":{"a":1,"a":2}}`,
		`{"hidden":1}`,
		`{"name":"x"} {}`,
		`[1]`,
	}
	for _, b := range bad {
		var o outer
		if strictjson.UnmarshalKnown([]byte(b), &o) == nil {
			t.Errorf("%s passed", b)
		}
	}
	var o outer
	if err := strictjson.UnmarshalKnown([]byte(`{"wait":1500000000,"raw":{"a" : 1}}`), &o); err != nil || o.Wait != 1500*time.Millisecond || string(o.Raw) != `{"a" : 1}` {
		t.Errorf("%+v %v", o, err)
	}
}

// A name that differs from a field's only in case is another member:
// Unmarshal ignores it where encoding/json would fill the field.
func TestNamesAreExact(t *testing.T) {
	for _, in := range []string{`{"Class":"act"}`, `{"CLASS":"act"}`, "{\"cKass\":\"act\"}"} {
		var v inner
		if err := strictjson.Unmarshal([]byte(in), &v); err != nil || v.Class != "" {
			t.Errorf("%s: %+v %v", in, v, err)
		}
		var v1 inner
		if jsonv1.Unmarshal([]byte(in), &v1); in != "{\"cKass\":\"act\"}" && v1.Class != "act" {
			t.Errorf("%s: encoding/json no longer folds case; the test is stale", in)
		}
	}
}

func nest(n int) []byte {
	return append(bytes.Repeat([]byte{'['}, n), bytes.Repeat([]byte{']'}, n)...)
}

func num(n int) []byte {
	return append(append([]byte(`{"n":`), bytes.Repeat([]byte{'9'}, n)...), '}')
}

func TestBounds(t *testing.T) {
	for _, ok := range [][]byte{nest(strictjson.MaxNesting), num(strictjson.MaxNumber), []byte("-0.5e+10")} {
		if err := strictjson.Valid(ok); err != nil {
			t.Errorf("%.40s: %v", ok, err)
		}
	}
	for _, bad := range [][]byte{nest(strictjson.MaxNesting + 1), num(strictjson.MaxNumber + 1), nil, []byte(" "), []byte("1 2")} {
		if strictjson.Valid(bad) == nil {
			t.Errorf("%.40q passed", bad)
		}
	}
	for _, ok := range [][]byte{nest(strictjson.MaxDepth), num(strictjson.MaxNumberLength), []byte(`{"a":{"b":[1,2.5e10,"` + strings.Repeat("9", 500) + `"]}}`)} {
		if err := strictjson.Input(ok); err != nil {
			t.Errorf("%.40s: %v", ok, err)
		}
	}
	for _, bad := range [][]byte{nest(strictjson.MaxDepth + 1), nest(1001), num(strictjson.MaxNumberLength + 1), num(1001), []byte(`{"a":1,"a":2}`), []byte("\"\xff\""), []byte(`{} {}`)} {
		if strictjson.Input(bad) == nil || strictjson.Bounded(bad) == nil {
			t.Errorf("%.40q passed", bad)
		}
	}
}

// Whatever the keys, an object passes Valid exactly when it names no key
// twice.
func TestPropertyDuplicateKeys(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		keys := rapid.SliceOfN(rapid.SampledFrom([]string{"a", "A", "b", "labels", "Labels", "ſ", "s"}), 0, 5).Draw(t, "keys")
		escape := rapid.Bool().Draw(t, "escape")
		b := []byte("{")
		seen, dup := map[string]bool{}, false
		for i, k := range keys {
			if i > 0 {
				b = append(b, ',')
			}
			kb, _ := jsonv1.Marshal(k)
			if escape && k == "a" {
				kb = []byte(`"\u0061"`)
			}
			b = append(append(b, kb...), `:[{"x":1}]`...)
			dup = dup || seen[k]
			seen[k] = true
		}
		b = append(b, '}')
		if err := strictjson.Valid(b); (err != nil) != dup {
			t.Fatalf("%s: %v", b, err)
		}
	})
}

// A JSON value drawn at random, with the text it is written as.
func genValue(depth int) *rapid.Generator[[]byte] {
	return rapid.Custom(func(t *rapid.T) []byte {
		kinds := 5
		if depth > 0 {
			kinds = 7
		}
		switch rapid.IntRange(0, kinds-1).Draw(t, "kind") {
		case 0:
			return []byte("null")
		case 1:
			return []byte(strconv.FormatBool(rapid.Bool().Draw(t, "bool")))
		case 2:
			return []byte(genNumber().Draw(t, "number"))
		case 3, 4:
			b, _ := jsonv1.Marshal(rapid.String().Draw(t, "string"))
			return b
		case 5:
			n := rapid.IntRange(0, 4).Draw(t, "len")
			parts := make([][]byte, n)
			for i := range parts {
				parts[i] = genValue(depth-1).Draw(t, "elem")
			}
			return append(append([]byte("["), bytes.Join(parts, []byte(","))...), ']')
		default:
			keys := rapid.SliceOfNDistinct(rapid.String(), 0, 4, rapid.ID[string]).Draw(t, "keys")
			parts := make([][]byte, len(keys))
			for i, k := range keys {
				kb, _ := jsonv1.Marshal(k)
				parts[i] = append(append(kb, ':'), genValue(depth-1).Draw(t, "member")...)
			}
			return append(append([]byte("{"), bytes.Join(parts, []byte(","))...), '}')
		}
	})
}

// A number literal of any length the dialect allows.
func genNumber() *rapid.Generator[string] {
	return rapid.Custom(func(t *rapid.T) string {
		s := ""
		if rapid.Bool().Draw(t, "neg") {
			s = "-"
		}
		s += rapid.StringMatching(`0|[1-9][0-9]{0,60}`).Draw(t, "int")
		if rapid.Bool().Draw(t, "frac") {
			s += "." + rapid.StringMatching(`[0-9]{1,40}`).Draw(t, "frac")
		}
		if rapid.Bool().Draw(t, "exp") {
			s += rapid.StringMatching(`[eE][+-]?[0-9]{1,4}`).Draw(t, "exp")
		}
		return s
	})
}

// Every value written by a correct encoder passes, and reads back with
// its strings and its number literals exactly as written.
func TestPropertyRoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		b := genValue(4).Draw(t, "json")
		if err := strictjson.Valid(b); err != nil {
			t.Fatalf("%s: %v", b, err)
		}
		v, err := strictjson.Value(b)
		if err != nil {
			t.Fatalf("%s: %v", b, err)
		}
		if want := reference(b); !reflect.DeepEqual(v, want) {
			t.Fatalf("%s read as %#v, want %#v", b, v, want)
		}
	})
}

// reference reads valid JSON with no repeated names the way encoding/json
// does with UseNumber, which keeps every string and number literal.
func reference(b []byte) any {
	dec := jsonv1.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		panic(err)
	}
	return v
}

// Valid never accepts what encoding/json or jsontext would call invalid,
// and accepts nothing with invalid UTF-8.
func FuzzValid(f *testing.F) {
	for _, c := range stricttest.Cases(f) {
		f.Add(c.Line)
	}
	f.Add([]byte(`{"a":[1,{"b":"é"}]}`))
	f.Fuzz(func(t *testing.T, b []byte) {
		err := strictjson.Valid(b)
		if err != nil {
			return
		}
		if !jsonv1.Valid(b) || !jsontext.Value(b).IsValid() || !utf8.Valid(b) {
			t.Fatalf("Valid accepted %q", b)
		}
		v, err := strictjson.Value(b)
		if err != nil {
			t.Fatalf("Valid accepted %q but Value refused it: %v", b, err)
		}
		if want := reference(b); !reflect.DeepEqual(v, want) {
			t.Fatalf("%q read as %#v, want %#v", b, v, want)
		}
	})
}

// An exponent passes exactly when its value is within ±MaxExponent,
// however many leading zeros it has.
func TestPropertyExponentBound(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		n := rapid.IntRange(-3*strictjson.MaxExponent, 3*strictjson.MaxExponent).Draw(t, "exp")
		zeros := strings.Repeat("0", rapid.IntRange(0, 30).Draw(t, "zeros"))
		sign := rapid.SampledFrom([]string{"", "+"}).Draw(t, "sign")
		if n < 0 {
			sign = "-"
		}
		abs := max(n, -n)
		lit := "1" + rapid.SampledFrom([]string{"e", "E"}).Draw(t, "e") + sign + zeros + strconv.Itoa(abs)
		err := strictjson.Valid([]byte(`{"n":` + lit + `}`))
		if want := abs <= strictjson.MaxExponent; (err == nil) != want {
			t.Fatalf("%s: err %v, want accepted %v", lit, err, want)
		}
	})
}
