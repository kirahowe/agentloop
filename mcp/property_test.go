package mcp

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
	"testing"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop/trifecta"
)

// jsonValue is a generated JSON value.
type jsonValue struct {
	kind   int // 0 string, 1 number, 2 bool, 3 null, 4 array, 5 object
	str    string
	num    string
	b      bool
	items  []jsonValue
	keys   []string
	values []jsonValue
}

func genValue(depth int) *rapid.Generator[jsonValue] {
	return rapid.Custom(func(t *rapid.T) jsonValue {
		max := 5
		if depth <= 0 {
			max = 3
		}
		v := jsonValue{kind: rapid.IntRange(0, max).Draw(t, "kind")}
		switch v.kind {
		case 0:
			v.str = rapid.StringOf(rapid.RuneFrom([]rune("ab<>&\"\\é\n \u2028"))).Draw(t, "str")
		case 1:
			v.num = rapid.SampledFrom([]string{"0", "-1", "1.50", "1e3", "12345678901234567890", "0.1"}).Draw(t, "num")
		case 2:
			v.b = rapid.Bool().Draw(t, "b")
		case 4:
			v.items = rapid.SliceOfN(genValue(depth-1), 0, 3).Draw(t, "items")
		case 5:
			v.keys = rapid.SliceOfNDistinct(rapid.StringMatching(`[a-z<&]{1,4}`), 0, 4, rapid.ID[string]).Draw(t, "keys")
			for i := range v.keys {
				v.values = append(v.values, genValue(depth-1).Draw(t, fmt.Sprint("value", i)))
			}
		}
		return v
	})
}

// write serialises v with keys in an order perm chooses and whitespace
// ws chooses.
func write(t *rapid.T, v jsonValue, b *strings.Builder) {
	ws := func() { b.WriteString(rapid.SampledFrom([]string{"", " ", "\n", "\t  "}).Draw(t, "ws")) }
	ws()
	switch v.kind {
	case 0:
		s, _ := json.Marshal(v.str)
		b.Write(s)
	case 1:
		b.WriteString(v.num)
	case 2:
		fmt.Fprint(b, v.b)
	case 3:
		b.WriteString("null")
	case 4:
		b.WriteString("[")
		for i, it := range v.items {
			if i > 0 {
				b.WriteString(",")
			}
			write(t, it, b)
		}
		b.WriteString("]")
	case 5:
		order := rapid.Permutation(indices(len(v.keys))).Draw(t, "order")
		b.WriteString("{")
		for i, k := range order {
			if i > 0 {
				b.WriteString(",")
			}
			ws()
			s, _ := json.Marshal(v.keys[k])
			b.Write(s)
			ws()
			b.WriteString(":")
			write(t, v.values[k], b)
		}
		b.WriteString("}")
	}
	ws()
}

func indices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// Two spellings of one schema have one digest, and Canonical is a fixed
// point; a different description is a different digest.
func TestPropertyDigestCanonical(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		v := genValue(3).Draw(t, "schema")
		var a, b strings.Builder
		write(t, v, &a)
		write(t, v, &b)
		ca, err := Canonical([]byte(a.String()))
		if err != nil {
			t.Fatalf("%q: %v", a.String(), err)
		}
		cb, err := Canonical([]byte(b.String()))
		if err != nil {
			t.Fatal(err)
		}
		if string(ca) != string(cb) {
			t.Fatalf("%s != %s", ca, cb)
		}
		again, _ := Canonical(ca)
		if string(again) != string(ca) {
			t.Fatalf("not a fixed point: %s, %s", ca, again)
		}
		if strings.ContainsAny(string(ca), "\n\t") {
			t.Fatalf("whitespace kept: %s", ca)
		}
		desc := rapid.String().Draw(t, "desc")
		da, err := DefinitionDigest("t", desc, json.RawMessage(a.String()), nil)
		if err != nil {
			t.Fatal(err)
		}
		db, _ := DefinitionDigest("t", desc, json.RawMessage(b.String()), nil)
		if da != db {
			t.Fatal("one schema, two digests")
		}
		dc, _ := DefinitionDigest("t", desc+"x", json.RawMessage(a.String()), nil)
		if dc == da {
			t.Fatal("a changed description kept its digest")
		}
		dd, _ := DefinitionDigest("t", desc, json.RawMessage(a.String()), json.RawMessage(`{}`))
		if dd == da {
			t.Fatal("adding an outputSchema kept the digest")
		}
	})
}

// A result's labels only add to the configured ones, and any label that
// is not known fails the call.
func TestPropertyLabelMerge(t *testing.T) {
	label := rapid.SampledFrom([]string{"private", "contaminated", "untrusted", "Private", ""})
	rapid.Check(t, func(t *rapid.T) {
		configured := rapid.SliceOfDistinct(rapid.SampledFrom(trifecta.Labels), rapid.ID[string]).Draw(t, "configured")
		added := rapid.SliceOf(label).Draw(t, "added")
		arr := make([]any, len(added))
		for i, l := range added {
			arr[i] = l
		}
		got, why := ResultLabels(map[string]any{LabelsKey: arr, "other": 1})
		unknown := slices.ContainsFunc(added, func(l string) bool { return !trifecta.KnownLabel(l) })
		if unknown != (why != "") {
			t.Fatalf("added %q: why %q", added, why)
		}
		if unknown {
			return
		}
		merged := MergeLabels(configured, got)
		if !sort.StringsAreSorted(merged) || len(slices.Compact(slices.Clone(merged))) != len(merged) {
			t.Fatalf("merged %q", merged)
		}
		for _, l := range append(slices.Clone(configured), added...) {
			if !slices.Contains(merged, l) {
				t.Fatalf("merged %q lost %q", merged, l)
			}
		}
		for _, l := range merged {
			if !slices.Contains(configured, l) && !slices.Contains(added, l) {
				t.Fatalf("merged %q invented %q", merged, l)
			}
		}
		if len(merged) == 0 && merged != nil {
			t.Fatal("empty labels are not nil")
		}
	})
}

// Whatever content a result mixes, the model never sees non-text
// content, and the call fails exactly when a kind is not allowed.
func TestPropertyContent(t *testing.T) {
	kinds := []string{"text", Image, Audio, ResourceLink, Resource}
	rapid.Check(t, func(t *rapid.T) {
		allow := rapid.SliceOfDistinct(rapid.SampledFrom(kinds[1:]), rapid.ID[string]).Draw(t, "allow")
		picked := rapid.SliceOf(rapid.SampledFrom(kinds)).Draw(t, "content")
		var content []sdk.Content
		refused := false
		for _, k := range picked {
			if k != "text" && !slices.Contains(allow, k) {
				refused = true
			}
			switch k {
			case "text":
				content = append(content, &sdk.TextContent{Text: "words"})
			case Image:
				content = append(content, &sdk.ImageContent{Data: []byte("SECRET"), MIMEType: "image/png"})
			case Audio:
				content = append(content, &sdk.AudioContent{Data: []byte("SECRET"), MIMEType: "SECRET"})
			case ResourceLink:
				content = append(content, &sdk.ResourceLink{URI: "SECRET", Name: "SECRET", Description: "SECRET"})
			case Resource:
				content = append(content, &sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "SECRET", Text: "SECRET"}})
			}
		}
		cfg := ToolConfig{Policy: trifecta.Tool{Class: trifecta.Read}, AllowContent: allow}
		r := ToResult("t", cfg, nil, &sdk.CallToolResult{Content: content})
		if strings.Contains(r.Content, "SECRET") {
			t.Fatalf("shown: %q", r.Content)
		}
		if refused != (r.Content == Failed("t") && r.IsError) {
			t.Fatalf("refused %v, got %+v", refused, r)
		}
		if !refused && strings.Count(r.Content, "words") != strings.Count(strings.Join(picked, " "), "text") {
			t.Fatalf("text lost: %q", r.Content)
		}
	})
}

func TestLint(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		class  trifecta.Class
		labels []string
		a      *sdk.ToolAnnotations
		want   int
	}{
		{trifecta.Read, nil, nil, 0},
		{trifecta.Read, nil, &sdk.ToolAnnotations{ReadOnlyHint: true}, 0},
		{trifecta.Read, nil, &sdk.ToolAnnotations{DestructiveHint: &yes}, 1},
		{trifecta.Read, nil, &sdk.ToolAnnotations{DestructiveHint: &no, OpenWorldHint: &no}, 0},
		{trifecta.Read, nil, &sdk.ToolAnnotations{OpenWorldHint: &yes}, 1},
		{trifecta.Read, []string{"contaminated"}, &sdk.ToolAnnotations{OpenWorldHint: &yes}, 0},
		{trifecta.Write, nil, &sdk.ToolAnnotations{OpenWorldHint: &yes}, 1},
		{trifecta.Write, nil, &sdk.ToolAnnotations{ReadOnlyHint: true}, 1},
		{trifecta.Egress, nil, &sdk.ToolAnnotations{OpenWorldHint: &yes, DestructiveHint: &yes}, 0},
		{trifecta.Act, nil, &sdk.ToolAnnotations{ReadOnlyHint: true}, 1},
	}
	for i, c := range cases {
		if got := Lint("x", trifecta.Tool{Class: c.class, Labels: c.labels}, c.a); len(got) != c.want {
			t.Errorf("case %d: %q", i, got)
		}
	}
}
