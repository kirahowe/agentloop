package trifecta

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// An edit call is held exactly when the session holds contaminated,
// whatever else it holds and whatever the input: what it writes leaves
// relay without its labels, so a contaminated session must not write
// unseen where a clean one would read it back.
func TestEditIsHeldExactlyWhenContaminated(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tool := Tool{Class: Edit, Labels: genLabels().Draw(t, "tool labels")}
		labels := genLabels().Draw(t, "session")
		v, rule := DecideCall(labels, tool, genInput().Draw(t, "input"))
		want := agentloop.Allow
		if slices.Contains(labels, Contaminated) {
			want = agentloop.Hold
		}
		if v != want || rule != -1 {
			t.Fatalf("edit call on %v: %s, rule %d, want %s", labels, v, rule, want)
		}
	})
}

// Write, relay's own data, which keeps labels, runs whatever the labels.
func TestWriteIsNeverHeld(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		labels := genLabels().Draw(t, "session")
		if v, _ := DecideCall(labels, Tool{Class: Write}, genInput().Draw(t, "input")); v != agentloop.Allow {
			t.Fatalf("write call on %v: %s", labels, v)
		}
	})
}

// An act call is held for every set of labels unless a consent rule
// matches its input; always_hold holds it even then.
func TestActIsHeldWithoutConsent(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		rules := rapid.SliceOfN(genRule(), 0, 3).Draw(t, "rules")
		tool := Tool{Class: Act, Labels: genLabels().Draw(t, "tool labels"), Consent: rules,
			AlwaysHold: rapid.Bool().Draw(t, "always")}
		input := genInput().Draw(t, "input")
		v, rule := DecideCall(genLabels().Draw(t, "session"), tool, input)
		matched := Consented(rules, input)
		switch {
		case tool.AlwaysHold || matched < 0:
			if v != agentloop.Hold || rule != -1 {
				t.Fatalf("%+v on %s: %s, rule %d", tool, input, v, rule)
			}
		default:
			if v != agentloop.Allow || rule != matched {
				t.Fatalf("%+v on %s: %s, rule %d, want rule %d", tool, input, v, rule, matched)
			}
		}
	})
}

// Consent belongs to act alone: on any other class it changes nothing.
func TestConsentOnlyActs(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tool := genTool().Draw(t, "tool")
		if tool.Class == Act {
			return
		}
		rule := genRule().Draw(t, "rule")
		tool.Consent = []map[string]json.RawMessage{rule}
		labels := genLabels().Draw(t, "session")
		v, i := DecideCall(labels, tool, encode(t, toAny(rule), false))
		if v != Decide(labels, tool) || i != -1 {
			t.Fatalf("consent on %q changed %s to %s", tool.Class, Decide(labels, tool), v)
		}
	})
}

// A class the rule does not know is held, whatever else is true.
func TestUnknownClassIsHeld(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		c := Class(rapid.StringMatching(`[a-zA-Z]{0,8}`).Draw(t, "class"))
		if KnownClass(c) {
			return
		}
		rule := genRule().Draw(t, "rule")
		tool := Tool{Class: c, Consent: []map[string]json.RawMessage{rule}}
		if v, _ := DecideCall(genLabels().Draw(t, "session"), tool, encode(t, toAny(rule), false)); v != agentloop.Hold {
			t.Fatalf("class %q got %s", c, v)
		}
	})
}

// A rule matches any input that holds its pairs, however the input is
// spelled (key order, spacing, string escapes) and whatever else it
// holds; and changing any one of the rule's values, or leaving out one
// of its keys, makes it not match.
func TestConsentMatchIsExact(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		rule := genRule().Draw(t, "rule")
		in := toAny(rule)
		for k, v := range rapid.MapOfN(rapid.StringMatching(`X[a-z]{0,3}`), genValue(), 0, 3).Draw(t, "extra") {
			in[k] = v
		}
		rules := []map[string]json.RawMessage{rule}
		if got := Consented(rules, encode(t, in, rapid.Bool().Draw(t, "escape"))); got != 0 {
			t.Fatalf("rule %v did not match %v", rule, in)
		}
		keys := slices.Sorted(maps.Keys(rule))
		k := rapid.SampledFrom(keys).Draw(t, "key")
		near := maps.Clone(in)
		if rapid.Bool().Draw(t, "drop") {
			delete(near, k)
		} else {
			near[k] = nearMiss(t, in[k])
		}
		if got := Consented(rules, encode(t, near, false)); got != -1 {
			t.Fatalf("rule %v matched the near miss %v", rule, near)
		}
	})
}

func TestConsentTable(t *testing.T) {
	rule := map[string]json.RawMessage{"group": json.RawMessage(`"Kira only"`), "n": json.RawMessage(`1`)}
	cases := []struct {
		input string
		want  int
	}{
		{`{"group":"Kira only","n":1}`, 0},
		{`{ "n" : 1 , "group" : "Kira only", "other": [1] }`, 0},
		{`{"group":"Kira only","n":1.0}`, -1},
		{`{"group":"Kira only ","n":1}`, -1},
		{`{"group":"kira only","n":1}`, -1},
		{`{"group":"Kira only","n":1}`, -1},
		{`{"group":"Kira only"}`, -1},
		{`{"group":["Kira only"],"n":1}`, -1},
		{`{"group":"Everyone","group":"Kira only","n":1}`, -1},
		{`{"group":"Kira only","n":1,"x":{"a":1,"a":2}}`, -1},
		{`{"group":"Kira only","n":1} {}`, -1},
		{`[{"group":"Kira only","n":1}]`, -1},
		{`null`, -1},
		{``, -1},
	}
	for _, c := range cases {
		if got := Consented([]map[string]json.RawMessage{rule}, json.RawMessage(c.input)); got != c.want {
			t.Errorf("%s: got %d, want %d", c.input, got, c.want)
		}
	}
	if Consented([]map[string]json.RawMessage{{}}, json.RawMessage(`{}`)) != -1 {
		t.Error("an empty rule matched")
	}
	two := []map[string]json.RawMessage{{"a": json.RawMessage(`1`)}, {"b": json.RawMessage(`{"y":[true,null],"x":"s"}`)}}
	if got := Consented(two, json.RawMessage(`{"b":{"x":"s","y":[true,null]}}`)); got != 1 {
		t.Errorf("second rule: got %d", got)
	}
	if got := Consented(two, json.RawMessage(`{"b":{"x":"s","y":[null,true]}}`)); got != -1 {
		t.Errorf("reordered array: got %d", got)
	}
}

// The decision records which rule allowed a call.
func TestPolicyRecordsTheRule(t *testing.T) {
	p := Policy{Tools: map[string]Tool{"send": {Class: Act, Consent: []map[string]json.RawMessage{
		{"group": json.RawMessage(`"A"`)}, {"group": json.RawMessage(`"B"`)}}}}}
	d := p.Decide(t.Context(), agentloop.ToolCall{ID: "c", Name: "send", Input: json.RawMessage(`{"group":"B"}`)}, agentloop.View{Labels: Labels})
	if d.Verdict != agentloop.Allow || d.Consent == nil || *d.Consent != 1 {
		t.Fatalf("%+v", d)
	}
	d = p.Decide(t.Context(), agentloop.ToolCall{ID: "c", Name: "send", Input: json.RawMessage(`{"group":"C"}`)}, agentloop.View{})
	if d.Verdict != agentloop.Hold || d.Consent != nil || !strings.Contains(d.Reason, "consent") {
		t.Fatalf("%+v", d)
	}
}

func genValue() *rapid.Generator[any] {
	scalar := rapid.OneOf(
		rapid.Map(rapid.String(), func(s string) any { return s }),
		rapid.Map(rapid.Int64(), func(n int64) any { return json.Number(fmt.Sprint(n)) }),
		rapid.Map(rapid.Bool(), func(b bool) any { return b }),
		rapid.Just[any](nil),
	)
	return rapid.OneOf(scalar,
		rapid.Map(rapid.SliceOfN(scalar, 0, 3), func(xs []any) any { return xs }),
		rapid.Map(rapid.MapOfN(rapid.StringMatching(`[a-c]`), scalar, 0, 3), func(m map[string]any) any { return m }),
	)
}

func genRule() *rapid.Generator[map[string]json.RawMessage] {
	return rapid.Custom(func(t *rapid.T) map[string]json.RawMessage {
		m := rapid.MapOfN(rapid.StringMatching(`[a-z]{1,4}`), genValue(), 1, 3).Draw(t, "pairs")
		out := map[string]json.RawMessage{}
		for k, v := range m {
			b, _ := json.Marshal(v)
			out[k] = b
		}
		return out
	})
}

func genInput() *rapid.Generator[json.RawMessage] {
	return rapid.Custom(func(t *rapid.T) json.RawMessage {
		m := rapid.MapOfN(rapid.StringMatching(`[a-z]{1,4}`), genValue(), 0, 4).Draw(t, "input")
		b, _ := json.Marshal(m)
		return b
	})
}

// toAny is a rule as an input object.
func toAny(rule map[string]json.RawMessage) map[string]any {
	out := map[string]any{}
	for k, v := range rule {
		out[k] = normal(v)
	}
	return out
}

// encode writes v as JSON with keys in a drawn order, spaces between
// tokens, and, with escape, every string character as \uXXXX.
func encode(t *rapid.T, v any, escape bool) json.RawMessage {
	var b strings.Builder
	var w func(any)
	w = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			keys := slices.Sorted(maps.Keys(x))
			if len(keys) > 1 {
				keys = rapid.Permutation(keys).Draw(t, "order")
			}
			b.WriteString("{ ")
			for i, k := range keys {
				if i > 0 {
					b.WriteString(" , ")
				}
				w(k)
				b.WriteString(" : ")
				w(x[k])
			}
			b.WriteString(" }")
		case []any:
			b.WriteString("[")
			for i, e := range x {
				if i > 0 {
					b.WriteString(", ")
				}
				w(e)
			}
			b.WriteString("]")
		case string:
			if !escape {
				s, _ := json.Marshal(x)
				b.Write(s)
				return
			}
			b.WriteString(`"`)
			for _, u := range utf16Units(x) {
				fmt.Fprintf(&b, `\u%04x`, u)
			}
			b.WriteString(`"`)
		default:
			s, _ := json.Marshal(x)
			b.Write(s)
		}
	}
	w(v)
	return json.RawMessage(b.String())
}

func utf16Units(s string) []uint16 {
	var out []uint16
	for _, r := range strings.ToValidUTF8(s, "�") {
		if r >= 0x10000 {
			r -= 0x10000
			out = append(out, uint16(0xD800+(r>>10)), uint16(0xDC00+(r&0x3FF)))
		} else {
			out = append(out, uint16(r))
		}
	}
	return out
}

// nearMiss is a value close to v that is not equal to it.
func nearMiss(t *rapid.T, v any) any {
	switch x := v.(type) {
	case string:
		return rapid.SampledFrom([]string{x + " ", " " + x, x + "​", strings.ToUpper(x) + "x", ""}).Filter(func(s string) bool { return s != x }).Draw(t, "near string")
	case json.Number:
		return rapid.SampledFrom([]any{json.Number(string(x) + ".0"), string(x), json.Number(string(x) + "1")}).Draw(t, "near number")
	case bool:
		return !x
	case nil:
		return rapid.SampledFrom([]any{false, "", json.Number("0"), "null"}).Draw(t, "near null")
	case []any:
		return append(slices.Clone(x), nil)
	case map[string]any:
		m := maps.Clone(x)
		m["zz"] = true
		return m
	}
	return "different"
}
