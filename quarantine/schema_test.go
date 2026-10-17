package quarantine

import (
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestCheckExamples(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		free   bool
		path   string // checked when not empty
	}{
		// Scalars.
		{"boolean", `{"type":"boolean"}`, false, ""},
		{"integer range", `{"type":"integer","minimum":0,"maximum":10}`, false, ""},
		{"number or null", `{"type":["number","null"]}`, false, ""},
		{"string", `{"type":"string"}`, true, ""},
		{"string or null", `{"type":["string","null"]}`, true, ""},
		{"no type at all", `{}`, true, ""},
		{"true", `true`, true, ""},
		{"false admits nothing", `false`, false, ""},
		{"short string is still text", `{"type":"string","maxLength":1}`, true, ""},
		{"pattern does not close", `{"type":"string","pattern":"^[0-9]+$"}`, true, ""},
		{"unknown type", `{"type":"strnig"}`, true, "/type"},
		{"empty type list", `{"type":[]}`, true, "/type"},

		// enum and const close whatever they sit on, since the caller
		// wrote every value.
		{"string enum", `{"type":"string","enum":["yes","no"]}`, false, ""},
		{"enum of objects", `{"enum":[{"note":"anything the caller wrote"}]}`, false, ""},
		{"const", `{"const":"done"}`, false, ""},
		{"enum beside an open sibling", `{"enum":["ab"],"properties":{"x":{"type":"string"}}}`, false, ""},
		{"single-character enum", `{"type":"array","items":{"enum":["a","b","c"]},"maxItems":100}`, true, "/items/enum"},
		{"empty-string const", `{"const":""}`, true, "/const"},
		{"one combined character", "{\"enum\":[\"ab\",\"e\u0301\"]}", true, "/enum"},
		{"two code points that stay two", "{\"enum\":[\"\u00e9x\"]}", false, ""},
		{"short string deep in an enum value", `{"enum":[{"note":["ok","z"]}]}`, true, "/enum"},
		{"short key in an enum value", `{"enum":[{"k":true}]}`, true, "/enum"},
		{"short string in not is fine", `{"type":"boolean","not":{"enum":["a"]}}`, false, ""},
		{"short property name", `{"type":"object","properties":{"h":{"type":"boolean"}},"additionalProperties":false}`, true, "/properties/h"},
		{"short propertyNames", `{"type":"object","propertyNames":{"enum":["h","i"]},"additionalProperties":{"const":true}}`, true, "/propertyNames/enum"},
		{"enum not an array", `{"enum":"a"}`, true, "/enum"},

		// Formats.
		{"date", `{"type":"string","format":"date"}`, false, ""},
		{"date-time", `{"type":"string","format":"date-time"}`, false, ""},
		{"time", `{"type":"string","format":"time"}`, false, ""},
		{"email", `{"type":"string","format":"email"}`, true, ""},
		{"uri", `{"type":"string","format":"uri"}`, true, ""},
		{"format with no type leaves arrays open", `{"format":"date"}`, true, ""},

		// Objects.
		{"open keys", `{"type":"object","properties":{"ab":{"type":"boolean"}}}`, true, ""},
		{"closed keys", `{"type":"object","properties":{"ab":{"type":"boolean"}},"additionalProperties":false}`, false, ""},
		{"typed extra values leave keys open", `{"type":"object","additionalProperties":{"type":"boolean"}}`, true, ""},
		{"propertyNames fixes keys", `{"type":"object","propertyNames":{"enum":["ab","cd"]},"additionalProperties":{"type":"boolean"}}`, false, ""},
		{"propertyNames by format", `{"type":"object","propertyNames":{"type":"string","format":"date"},"additionalProperties":{"type":"integer"}}`, false, ""},
		{"propertyNames without additionalProperties", `{"type":"object","propertyNames":{"enum":["ab"]}}`, true, ""},
		{"propertyNames that admit text", `{"type":"object","propertyNames":{"maxLength":3},"additionalProperties":false}`, false, ""},
		{"propertyNames open, extra open", `{"type":"object","propertyNames":{"maxLength":3},"additionalProperties":{"type":"boolean"}}`, true, "/propertyNames"},
		{"open property value", `{"type":"object","properties":{"ab":{"type":"string"}},"additionalProperties":false}`, true, "/properties/ab"},
		{"patternProperties", `{"type":"object","patternProperties":{"^x":{"type":"boolean"}},"additionalProperties":false}`, true, "/patternProperties"},
		{"unevaluatedProperties", `{"type":"object","unevaluatedProperties":false}`, true, "/unevaluatedProperties"},
		{"dependentSchemas only narrow", `{"type":"object","additionalProperties":false,"dependentSchemas":{"a":{"type":"string"}}}`, false, ""},
		{"required and counts narrow", `{"type":"object","properties":{"ab":{"type":"null"}},"required":["ab"],"minProperties":1,"additionalProperties":false}`, false, ""},
		{"key with a slash", `{"type":"object","properties":{"a/b":{"type":"string"}},"additionalProperties":false}`, true, "/properties/a~1b"},

		// Arrays.
		{"array with no items", `{"type":"array"}`, true, ""},
		{"array of booleans", `{"type":"array","items":{"type":"boolean"},"maxItems":100}`, false, ""},
		{"array of text", `{"type":"array","items":{"type":"string"},"maxItems":5}`, true, "/items"},
		{"prefixItems leave the rest open", `{"type":"array","prefixItems":[{"type":"boolean"}],"maxItems":5}`, true, ""},
		{"tuple", `{"type":"array","prefixItems":[{"type":"boolean"},{"const":1}],"items":false,"maxItems":2}`, false, ""},
		{"open prefix item", `{"type":"array","prefixItems":[{"type":"string"}],"items":false,"maxItems":1}`, true, "/prefixItems/0"},
		{"contains only narrows", `{"type":"array","items":{"type":"integer"},"maxItems":3,"contains":{"type":"string"}}`, false, ""},
		{"contains does not close", `{"type":"array","contains":{"const":1}}`, true, ""},
		{"unevaluatedItems", `{"type":"array","items":false,"unevaluatedItems":false}`, true, "/unevaluatedItems"},
		{"array with no maxItems", `{"type":"array","items":{"type":"boolean"}}`, true, ""},
		{"maxItems over the cap", `{"type":"array","items":{"type":"boolean"},"maxItems":101}`, true, "/maxItems"},
		{"maxItems written as 1e2", `{"type":"array","items":{"type":"boolean"},"maxItems":1e2}`, false, ""},
		{"tuple with no maxItems", `{"type":"array","prefixItems":[{"type":"boolean"}],"items":false}`, true, ""},
		{"tuple then items, no maxItems", `{"type":"array","prefixItems":[{"type":"boolean"}],"items":{"type":"integer"}}`, true, ""},
		{"tuple then items, bounded", `{"type":"array","prefixItems":[{"type":"boolean"}],"items":{"type":"integer"},"maxItems":100}`, false, ""},
		{"maxItems in a sibling allOf does not count", `{"type":"array","items":{"type":"boolean"},"allOf":[{"maxItems":3}]}`, true, ""},
		{"draft-07 tuple items", `{"type":"array","items":[{"type":"boolean"}]}`, true, "/items"},
		{"additionalItems", `{"type":"array","items":{"type":"boolean"},"additionalItems":false}`, true, "/additionalItems"},

		// Combinators.
		{"anyOf all closed", `{"anyOf":[{"type":"boolean"},{"type":"integer"}]}`, false, ""},
		{"anyOf one open", `{"anyOf":[{"type":"boolean"},{"type":"string"}]}`, true, "/anyOf/1"},
		{"anyOf names the member's own reason", `{"anyOf":[{"type":"boolean"},{"type":"array","items":{"type":"boolean"}}]}`, true, "/anyOf/1"},
		{"oneOf all closed", `{"oneOf":[{"type":"boolean"},{"type":"null"}]}`, false, ""},
		{"oneOf one open", `{"oneOf":[{"type":"boolean"},{"type":"string"}]}`, true, "/oneOf/1"},
		{"allOf one closed", `{"allOf":[{"minLength":2},{"enum":["ab","cd"]}]}`, false, ""},
		{"allOf none closed", `{"allOf":[{"minLength":2},{"type":"string"}]}`, true, "/allOf/0"},
		{"empty allOf", `{"allOf":[]}`, true, "/allOf"},
		{"not alone", `{"not":{"type":"string"}}`, true, ""},
		{"not narrows", `{"type":"boolean","not":{"const":true}}`, false, ""},
		{"not with an unknown keyword inside", `{"type":"boolean","not":{"cosnt":true}}`, true, "/not/cosnt"},
		{"if then without else", `{"if":{"type":"integer"},"then":{"type":"integer"}}`, true, ""},
		{"if then else both closed", `{"if":{"type":"integer"},"then":{"type":"integer"},"else":{"type":"boolean"}}`, false, ""},
		{"if then else, else open", `{"if":{"type":"integer"},"then":{"type":"integer"},"else":{"type":"string"}}`, true, "/else"},

		// References.
		{"ref to defs", `{"$defs":{"b":{"type":"boolean"}},"$ref":"#/$defs/b"}`, false, ""},
		{"ref to open def", `{"$defs":{"s":{"type":"string"}},"$ref":"#/$defs/s"}`, true, "/$defs/s"},
		{"ref to a missing def", `{"$ref":"#/$defs/nope"}`, true, "/$ref"},
		{"remote ref", `{"$ref":"https://example.com/s.json"}`, true, "/$ref"},
		{"ref into properties", `{"type":"object","properties":{"ab":{"type":"boolean"}},"additionalProperties":false,"$ref":"#/properties/a"}`, true, "/$ref"},
		{"escaped ref", `{"$defs":{"a/b":{"type":"boolean"}},"$ref":"#/$defs/a~1b"}`, true, "/$ref"},
		{"tree of booleans", `{"$defs":{"t":{"type":"object","properties":{"leaf":{"type":"boolean"},"kids":{"type":"array","items":{"$ref":"#/$defs/t"},"maxItems":4}},"additionalProperties":false}},"$ref":"#/$defs/t"}`, false, ""},
		{"tree from the root", `{"type":"array","items":{"$ref":"#"},"maxItems":4}`, false, ""},
		{"ref to itself", `{"$ref":"#"}`, true, "/$ref"},
		{"def to itself", `{"$defs":{"a":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}`, true, "/$defs/a/$ref"},
		{"defs to each other", `{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"allOf":[{"$ref":"#/$defs/a"}]}},"$ref":"#/$defs/a"}`, true, ""},
		{"nested defs", `{"type":"boolean","not":{"$defs":{}}}`, true, "/not/$defs"},
		{"$id", `{"$id":"https://example.com/s","type":"boolean"}`, true, "/$id"},
		{"$anchor", `{"$anchor":"x","type":"boolean"}`, true, "/$anchor"},
		{"$dynamicRef", `{"$dynamicRef":"#x","type":"boolean"}`, true, "/$dynamicRef"},
		{"$dynamicAnchor", `{"$dynamicAnchor":"x","type":"boolean"}`, true, "/$dynamicAnchor"},
		{"draft-07 definitions", `{"definitions":{},"type":"boolean"}`, true, "/definitions"},

		// Dialects and unknown keywords.
		{"2020-12 declared", `{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"boolean"}`, false, ""},
		{"draft-07 declared", `{"$schema":"http://json-schema.org/draft-07/schema#","type":"boolean"}`, true, "/$schema"},
		{"$schema below the root", `{"type":"array","items":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"boolean"}}`, true, "/items/$schema"},
		{"$vocabulary", `{"$vocabulary":{},"type":"boolean"}`, true, "/$vocabulary"},
		{"unknown keyword", `{"type":"boolean","x-extra":1}`, true, "/x-extra"},
		{"misspelt keyword", `{"type":"object","additionalproperties":false}`, true, "/additionalproperties"},
		{"content keywords", `{"type":"string","contentMediaType":"application/json"}`, true, "/contentMediaType"},
		{"annotations are fine", `{"type":"boolean","title":"t","description":"d","$comment":"c","default":true,"examples":[false],"deprecated":false,"readOnly":true,"writeOnly":false}`, false, ""},

		// JSON that two readers could read differently.
		{"duplicate key", `{"type":"boolean","type":"string"}`, true, ""},
		{"trailing value", `{"type":"boolean"} {"type":"string"}`, true, ""},
		{"not JSON", `{"type":`, true, ""},
		{"a schema that is a number", `1`, true, ""},
		{"a subschema that is a string", `{"type":"array","items":"boolean"}`, true, "/items"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := Check([]byte(c.schema))
			if r.FreeText != c.free {
				t.Fatalf("Check(%s) = %+v, want free text %v", c.schema, r, c.free)
			}
			if r.FreeText && r.Reason == "" {
				t.Fatalf("Check(%s) gives no reason", c.schema)
			}
			if !r.FreeText {
				if _, err := Compile([]byte(c.schema)); err != nil {
					t.Fatalf("Check passed %s, which does not compile: %v", c.schema, err)
				}
			}
			if r.FreeText {
				if why := reasonFits([]byte(c.schema), r); why != "" {
					t.Fatalf("Check(%s) = %+v: %s", c.schema, r, why)
				}
			}
			if c.path != "" && r.Path != c.path {
				t.Fatalf("Check(%s) points at %q, want %q (%s)", c.schema, r.Path, c.path, r.Reason)
			}
		})
	}
}

// The QA case: the string member fails, so the string reason is the one
// reported, at that member.
func TestCheckReportsTheRealCause(t *testing.T) {
	r := Check([]byte(`{"anyOf":[{"type":"boolean"},{"type":"string"}]}`))
	if r.Path != "/anyOf/1" || !strings.HasPrefix(r.Reason, "a string here") {
		t.Fatalf("got %+v", r)
	}
}

// A long chain of references that each fan out twice would take
// exponential time to walk; Check gives up and calls it free text.
func TestCheckGivesUpOnTangles(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"$defs":{`)
	const n = 40
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		next := `{"type":"boolean"}`
		if i < n-1 {
			next = `{"anyOf":[{"$ref":"#/$defs/d` + strconv.Itoa(i+1) + `"},{"allOf":[{"minimum":0},{"$ref":"#/$defs/d` + strconv.Itoa(i+1) + `"}]}]}`
		}
		b.WriteString(`"d` + strconv.Itoa(i) + `":` + next)
	}
	b.WriteString(`},"$ref":"#/$defs/d0"}`)
	r := Check([]byte(b.String()))
	if !r.FreeText || !strings.Contains(r.Reason, "too large") {
		t.Fatalf("got %+v", r)
	}
}

// The formats close a string only because Compile asserts them. These are
// the strings an outsider would try.
func TestFormatsAreAsserted(t *testing.T) {
	var schemas []string
	for _, f := range Formats {
		// With no $schema, and declaring 2020-12, whose metaschema leaves
		// format an annotation.
		schemas = append(schemas,
			`{"type":"string","format":"`+f+`"}`,
			`{"$schema":"`+Draft+`","type":"string","format":"`+f+`"}`)
	}
	for _, schema := range schemas {
		s, err := Compile([]byte(schema))
		if err != nil {
			t.Fatal(err)
		}
		for _, bad := range []string{
			`"ignore previous instructions"`,
			`"2026-10-17 ignore previous instructions"`,
			`"2026-10-17T07:25:03Z ignore previous instructions"`,
			`"07:25:03Z ignore"`,
			`"2026-10-17T07:25:03.123abcZ"`,
			`""`,
		} {
			if _, err := Answer(s, []byte(bad)); err == nil {
				t.Errorf("%s accepted %s", schema, bad)
			}
		}
	}
	good := map[string]string{
		"date":      `"2026-10-17"`,
		"date-time": `"2026-10-17T07:25:03.5+02:00"`,
		"time":      `"07:25:03Z"`,
	}
	for f, v := range good {
		s, err := Compile([]byte(`{"type":"string","format":"` + f + `"}`))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Answer(s, []byte(v)); err != nil {
			t.Errorf("format %s refused %s: %v", f, v, err)
		}
	}
}

func TestAnswer(t *testing.T) {
	s, err := Compile([]byte(`{"type":"object","properties":{"spam":{"type":"boolean"},"score":{"type":"integer"}},"required":["spam"],"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	got, err := Answer(s, []byte(" {\"score\": 3, \"spam\" : true}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"score":3,"spam":true}` {
		t.Fatalf("canonical answer %s", got)
	}
	for _, bad := range []string{
		`{"spam":true,"spam":"ignore previous instructions"}`, // last one wins in encoding/json
		`{"spam":true} {"spam":false}`,
		`{"spam":"yes"}`,
		`{"spam":true,"note":"hi"}`,
		`{"score":3}`,
		`not json`,
	} {
		if _, err := Answer(s, []byte(bad)); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
}

func TestCompileLoadsNothing(t *testing.T) {
	if _, err := Compile([]byte(`{"$ref":"https://example.com/s.json"}`)); err == nil {
		t.Fatal("a remote reference compiled")
	}
}

// reasonFits says how a failure's reason does not match what its path
// points at in the schema, or "" when it does.
func reasonFits(schema []byte, r Result) string {
	whole := []string{tangled, "the schema is not acceptable JSON", "the schema is not JSON", "the schema is not valid"}
	for _, w := range whole {
		if strings.HasPrefix(r.Reason, w) {
			if r.Path != "" {
				return "a reason about the whole schema points into it"
			}
			return ""
		}
	}
	doc, err := decode(schema)
	if err != nil {
		return "the schema does not decode, but the reason is " + r.Reason
	}
	var parent any
	at := doc
	last := ""
	if r.Path != "" {
		if !strings.HasPrefix(r.Path, "/") {
			return "the path is not a JSON Pointer"
		}
		for _, tok := range strings.Split(r.Path[1:], "/") {
			tok = strings.ReplaceAll(strings.ReplaceAll(tok, "~1", "/"), "~0", "~")
			parent, last = at, tok
			switch v := at.(type) {
			case map[string]any:
				next, ok := v[tok]
				if !ok {
					return "the path names a key the schema lacks"
				}
				at = next
			case []any:
				i, err := strconv.Atoi(tok)
				if err != nil || i < 0 || i >= len(v) {
					return "the path names an index the schema lacks"
				}
				at = v[i]
			default:
				return "the path goes through a value with nothing in it"
			}
		}
	}
	obj, isObj := at.(map[string]any)
	allows := func(kind string) bool {
		t, ok := obj["type"]
		if !ok {
			return true
		}
		names, _ := kinds(t)
		return slices.Contains(names, kind)
	}
	// With no type, strings are checked first, so a reason about objects
	// or arrays there means strings were closed by a format.
	plainFirst := func() string {
		if _, ok := obj["type"]; ok {
			return ""
		}
		if f, _ := obj["format"].(string); !slices.Contains(Formats, f) {
			return "a schema with no type and open strings gets a reason about something else"
		}
		return ""
	}
	keyword := func() string {
		if _, ok := parent.(map[string]any); !ok || last == "" {
			return "a keyword reason does not point at a keyword"
		}
		return ""
	}
	why := r.Reason
	switch {
	case why == "a schema must be an object or a boolean":
		if _, ok := at.(bool); ok || isObj {
			return "points at a schema"
		}
	case strings.HasPrefix(why, "the keyword "):
		if !strings.Contains(why, strconv.Quote(last)) {
			return "the unknown keyword is not the one pointed at"
		}
		return keyword()
	case refusedReason(why), strings.Contains(why, " is allowed only at the root"),
		strings.HasPrefix(why, "$schema must be"), strings.HasSuffix(why, " must be an object"),
		strings.HasSuffix(why, " must be a non-empty array"), strings.HasPrefix(why, "type "),
		strings.HasPrefix(why, "enum must"), strings.HasPrefix(why, "format must"), strings.HasPrefix(why, "$ref "):
		return keyword()
	case why == "the schema true admits anything":
		if at != true {
			return "does not point at true"
		}
	case strings.Contains(why, "fixes the string"):
		if last != "enum" && last != "const" {
			return "does not point at enum or const"
		}
		if _, ok := shortString(at); !ok {
			return "no short string there"
		}
	case strings.HasPrefix(why, "the property name"):
		if _, ok := parent.(map[string]any); !ok || !short(last) {
			return "does not point at a short property name"
		}
	case strings.HasPrefix(why, "a string here"):
		if !isObj || !allows("string") {
			return "does not point at a schema that allows strings"
		}
		if f, _ := obj["format"].(string); slices.Contains(Formats, f) {
			return "the string there is closed by its format"
		}
	case strings.HasPrefix(why, "an object here"), strings.HasPrefix(why, "values of keys"):
		if !isObj || !allows("object") {
			return "does not point at a schema that allows objects"
		}
		return plainFirst()
	case strings.HasPrefix(why, "an array here can be any length"):
		if !isObj || !allows("array") {
			return "does not point at a schema that allows arrays"
		}
		if _, ok := obj["maxItems"]; ok {
			return "maxItems is there"
		}
		return plainFirst()
	case strings.HasPrefix(why, "maxItems is"):
		n, ok := at.(json.Number)
		if f, err := n.Float64(); last != "maxItems" || !ok || err != nil || f <= MaxItems {
			return "does not point at a maxItems over the cap"
		}
	case strings.HasPrefix(why, "an array here can hold anything"):
		if !isObj || !allows("array") {
			return "does not point at a schema that allows arrays"
		}
		if _, ok := obj["items"]; ok {
			return "items is there"
		}
		return plainFirst()
	case strings.HasPrefix(why, "the reference "):
		if last != "$ref" {
			return "does not point at a $ref"
		}
	default:
		return "a reason the test does not know: " + why
	}
	return ""
}

func refusedReason(why string) bool {
	for _, r := range refused {
		if r == why {
			return true
		}
	}
	return false
}
