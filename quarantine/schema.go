// Package quarantine holds the pure parts of a quarantined reader: a
// model with no tools that reads content an outsider may have written and
// answers a question about it as JSON matching a schema the caller gave.
//
// Check decides whether a schema admits free text. An answer to a schema
// that admits none can only be made of booleans, numbers, nulls, dates and
// times, strings the caller wrote into the schema itself, and arrays of at
// most MaxItems of these, so it cannot carry an outsider's instructions to
// the caller. A fixed string of one character or none counts as free text,
// since an array of them spells. Anything Check does not fully understand
// counts as free text.
//
// Compile and Answer check an answer against its schema. The security of
// the reader rests on that validation, never on the model having followed
// the schema.
//
// Reader is the read_quarantined tool built on them. Its rules
// (ResultLabels, ReaderPrompt, ParseReply, AnswerContent) are pure
// functions; Reader.CallWith is the shell that calls the inner tool and
// the model. Everything else here is pure: no I/O, no clock, no state.
package quarantine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// Draft is the only dialect a schema may declare in $schema.
const Draft = "https://json-schema.org/draft/2020-12/schema"

// Formats are the string formats that close a string. Each has a fixed
// grammar of digits and a few separators (RFC 3339), so a string that
// matches one cannot spell a word. They close a string only because
// Compile turns on format assertion; JSON Schema 2020-12 treats format as
// an annotation by default.
var Formats = []string{"date", "date-time", "time"}

// MaxItems is the most items an array in an answer may hold. Every array
// a schema admits must say so with maxItems: a bounded array of fixed
// values carries a bounded number of the reader's choices. Numbers are
// not bounded.
const MaxItems = 100

// A Result is Check's answer.
type Result struct {
	// FreeText is true when the schema admits free text, or when Check
	// could not tell.
	FreeText bool `json:"free_text"`
	// Path is a JSON Pointer to the part of the schema that admits it,
	// "" for the root.
	Path string `json:"path,omitempty"`
	// Reason says what admits it.
	Reason string `json:"reason,omitempty"`

	// level is how many schemas deep the walk was when it found this,
	// so that of several failures the most specific can be reported.
	level int
}

// maxSteps bounds the work Check does, since references can make a small
// schema expensive to walk. A schema that needs more is free text.
const maxSteps = 100000

const tangled = "the schema is too large or too tangled to check"

// Check reports whether the JSON Schema in schema admits free text: a
// string, or an object key, that the schema does not fix.
//
// A schema admits no free text when every value it admits is built from
// null, booleans, numbers, strings fixed by enum or const or closed by a
// format in Formats, arrays of at most MaxItems (said by maxItems) whose
// every item is such a value, and objects whose keys are fixed (additionalProperties false, or propertyNames that
// admit no free text) and whose values are such values. A string of one
// code point or none (after NFC), fixed by enum or const anywhere in its
// value, or as a property name, is free text: an array of them spells. A
// schema that Compile refuses is free text too.
func Check(schema []byte) Result {
	if err := strictjson.Input(schema); err != nil {
		return freeText("", "the schema is not acceptable JSON: "+err.Error())
	}
	doc, err := decode(schema)
	if err != nil {
		return freeText("", "the schema is not JSON: "+err.Error())
	}
	c := &checker{root: doc, visiting: map[string]int{}}
	if obj, ok := doc.(map[string]any); ok {
		if defs, ok := obj["$defs"].(map[string]any); ok {
			c.defs = defs
		}
	}
	if r := c.scan(doc, "", true); r != nil {
		return *r
	}
	// A schema the validator cannot compile cannot check an answer, and
	// compiling checks every keyword's value against the 2020-12
	// metaschema, which Check does not do itself.
	if _, err := Compile(schema); err != nil {
		return freeText("", err.Error())
	}
	r := c.closed(doc, "", 0)
	if c.steps > maxSteps {
		return freeText("", tangled)
	}
	if r != nil {
		return *r
	}
	return Result{}
}

func freeText(path, reason string) Result {
	return Result{FreeText: true, Path: path, Reason: reason}
}

func fail(path, reason string) *Result {
	r := freeText(path, reason)
	return &r
}

// decode reads exactly one JSON value, numbers kept as written.
func decode(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("there is more after the value")
	}
	return v, nil
}

// How a keyword is read. Every keyword Check knows is in keywords; any
// other is free text.
type role int

const (
	// value: a keyword whose value is not a schema. Check either reads
	// it (type, enum, const, format, $ref) or it can only narrow what
	// the schema admits (minimum, pattern, required, ...).
	value role = iota
	// sub: one schema.
	sub
	// subMap: an object of schemas.
	subMap
	// subList: a non-empty array of schemas.
	subList
	// rootOnly: a keyword allowed only at the root ($schema, $defs).
	rootOnly
)

var keywords = map[string]role{
	// Annotations: the validator ignores them.
	"title": value, "description": value, "$comment": value, "default": value,
	"examples": value, "deprecated": value, "readOnly": value, "writeOnly": value,

	// Read by Check.
	"type": value, "enum": value, "const": value, "format": value, "$ref": value,

	// Narrow what a value can be, and never widen it.
	"multipleOf": value, "maximum": value, "exclusiveMaximum": value,
	"minimum": value, "exclusiveMinimum": value,
	"maxLength": value, "minLength": value, "pattern": value,
	"maxItems": value, "minItems": value, "uniqueItems": value,
	"maxContains": value, "minContains": value,
	"maxProperties": value, "minProperties": value, "required": value,
	"dependentRequired": value,

	// Applicators.
	"items": sub, "additionalProperties": sub, "propertyNames": sub,
	"contains": sub, "not": sub, "if": sub, "then": sub, "else": sub,
	"properties": subMap, "dependentSchemas": subMap,
	"prefixItems": subList, "allOf": subList, "anyOf": subList, "oneOf": subList,

	"$schema": rootOnly, "$defs": rootOnly,
}

// refused names keywords of JSON Schema (2020-12 and the drafts before
// it) that Check knows and will not take, with the reason.
var refused = map[string]string{
	"patternProperties":     "patternProperties admits every key its patterns match",
	"unevaluatedProperties": "unevaluatedProperties depends on what other keywords evaluated, which Check does not follow",
	"unevaluatedItems":      "unevaluatedItems depends on what other keywords evaluated, which Check does not follow",
	"$id":                   "$id changes how references resolve",
	"$anchor":               "$anchor names a schema that references could reach",
	"$dynamicRef":           "$dynamicRef resolves at validation time",
	"$dynamicAnchor":        "$dynamicAnchor is a target for $dynamicRef",
	"$vocabulary":           "$vocabulary changes which keywords apply",
	"$recursiveRef":         "$recursiveRef is not 2020-12",
	"$recursiveAnchor":      "$recursiveAnchor is not 2020-12",
	"definitions":           "definitions is not 2020-12 (use $defs)",
	"dependencies":          "dependencies is not 2020-12",
	"additionalItems":       "additionalItems is not 2020-12",
	"contentEncoding":       "content keywords describe a string's content without checking it",
	"contentMediaType":      "content keywords describe a string's content without checking it",
	"contentSchema":         "content keywords describe a string's content without checking it",
}

var typeNames = []string{"array", "boolean", "integer", "null", "number", "object", "string"}

type checker struct {
	root any
	defs map[string]any
	// visiting maps each $ref being followed to the instance depth at
	// which it was entered.
	visiting map[string]int
	steps    int
	level    int
}

// fail is a failure found at the walk's current level.
func (c *checker) fail(path, reason string) *Result {
	r := fail(path, reason)
	r.level = c.level
	return r
}

// deepest is the most specific of the failures: the one found deepest in
// the walk, the first of those on a tie.
func deepest(rs ...*Result) *Result {
	var best *Result
	for _, r := range rs {
		if r != nil && (best == nil || r.level > best.level) {
			best = r
		}
	}
	return best
}

// scan checks that every keyword in the schema is one Check knows and
// that the keywords it reads have the shape it reads them in. It looks at
// every subschema, even ones that only narrow, so nothing in the schema
// goes unread.
func (c *checker) scan(node any, path string, root bool) *Result {
	if _, ok := node.(bool); ok {
		return nil
	}
	obj, ok := node.(map[string]any)
	if !ok {
		return fail(path, "a schema must be an object or a boolean")
	}
	for _, k := range sortedKeys(obj) {
		v, p := obj[k], path+"/"+escape(k)
		if why, ok := refused[k]; ok {
			return fail(p, why)
		}
		r, ok := keywords[k]
		if !ok {
			return fail(p, fmt.Sprintf("the keyword %q is not one Check knows", k))
		}
		switch r {
		case rootOnly:
			if !root {
				return fail(p, k+" is allowed only at the root")
			}
			if k == "$schema" && v != Draft {
				return fail(p, "$schema must be "+Draft)
			}
			if k == "$defs" {
				defs, ok := v.(map[string]any)
				if !ok {
					return fail(p, "$defs must be an object")
				}
				for _, name := range sortedKeys(defs) {
					if res := c.scan(defs[name], p+"/"+escape(name), false); res != nil {
						return res
					}
				}
			}
		case sub:
			if res := c.scan(v, p, false); res != nil {
				return res
			}
		case subMap:
			m, ok := v.(map[string]any)
			if !ok {
				return fail(p, k+" must be an object")
			}
			for _, name := range sortedKeys(m) {
				if res := c.scan(m[name], p+"/"+escape(name), false); res != nil {
					return res
				}
			}
		case subList:
			list, ok := v.([]any)
			if !ok || len(list) == 0 {
				return fail(p, k+" must be a non-empty array")
			}
			for i, s := range list {
				if res := c.scan(s, p+"/"+strconv.Itoa(i), false); res != nil {
					return res
				}
			}
		case value:
			if res := c.scanValue(k, v, p); res != nil {
				return res
			}
		}
	}
	return nil
}

func (c *checker) scanValue(k string, v any, p string) *Result {
	switch k {
	case "type":
		if _, err := kinds(v); err != nil {
			return fail(p, err.Error())
		}
	case "enum":
		if _, ok := v.([]any); !ok {
			return fail(p, "enum must be an array")
		}
	case "format":
		if _, ok := v.(string); !ok {
			return fail(p, "format must be a string")
		}
	case "$ref":
		s, ok := v.(string)
		if !ok {
			return fail(p, "$ref must be a string")
		}
		if _, err := c.target(s); err != nil {
			return fail(p, err.Error())
		}
	}
	return nil
}

// target resolves a reference. Only the root and the root's $defs can be
// referred to, by a plain name: nothing remote, no anchors, no pointers
// into other keywords, no escapes.
func (c *checker) target(ref string) (any, error) {
	if ref == "#" {
		return nil, nil
	}
	name, ok := strings.CutPrefix(ref, "#/$defs/")
	if !ok || name == "" || strings.ContainsAny(name, "/~%") {
		return nil, fmt.Errorf("$ref %q is not \"#\" or \"#/$defs/<name>\"", ref)
	}
	t, ok := c.defs[name]
	if !ok {
		return nil, fmt.Errorf("$ref %q names no schema in $defs", ref)
	}
	return t, nil
}

// kinds reads type. Integers are numbers here: both are free of text.
func kinds(v any) ([]string, error) {
	var names []string
	switch t := v.(type) {
	case string:
		names = []string{t}
	case []any:
		if len(t) == 0 {
			return nil, errors.New("type must not be an empty array")
		}
		for _, e := range t {
			s, ok := e.(string)
			if !ok {
				return nil, errors.New("type must name types as strings")
			}
			names = append(names, s)
		}
	default:
		return nil, errors.New("type must be a string or an array of strings")
	}
	for _, n := range names {
		if !slices.Contains(typeNames, n) {
			return nil, fmt.Errorf("type %q is not a JSON Schema type", n)
		}
	}
	return names, nil
}

// closed returns nil when node admits no free text, and otherwise where
// and why it does. depth is how far into the instance the node applies,
// which tells a reference cycle that steps into the instance (a tree of
// booleans, say) from one that never does ({"$ref": "#"}).
//
// The keywords of a schema all apply at once, so a node admits no free
// text if any one of these holds: it has enum or const; it has a $ref to
// a schema that admits none; one member of allOf admits none; every
// member of anyOf, or of oneOf, admits none; it has then and else and
// both admit none; or its own type, format, items, prefixItems,
// properties, propertyNames and additionalProperties leave no free text
// for any type it allows.
func (c *checker) closed(node any, path string, depth int) *Result {
	c.steps++
	if c.steps > maxSteps {
		return c.fail(path, tangled)
	}
	c.level++
	defer func() { c.level-- }()
	if b, ok := node.(bool); ok {
		if b {
			return c.fail(path, "the schema true admits anything")
		}
		return nil
	}
	obj := node.(map[string]any) // scan made sure
	for _, k := range []string{"const", "enum"} {
		if v, ok := obj[k]; ok {
			if s, short := shortString(v); short {
				return c.fail(path+"/"+k, fmt.Sprintf("%s fixes the string %q, and an array of strings that short can spell anything", k, s))
			}
			return nil
		}
	}
	// Each way the node could be closed is tried. If none works, the
	// failure reported is the most specific one found.
	var tried []*Result
	if ref, ok := obj["$ref"].(string); ok {
		r := c.followRef(ref, path, depth)
		if r == nil {
			return nil
		}
		tried = append(tried, r)
	}
	if list, ok := obj["allOf"].([]any); ok {
		for i, s := range list {
			r := c.closed(s, path+"/allOf/"+strconv.Itoa(i), depth)
			if r == nil {
				return nil
			}
			tried = append(tried, r)
		}
	}
	for _, k := range []string{"anyOf", "oneOf"} {
		if list, ok := obj[k].([]any); ok {
			r := c.allClosed(list, path+"/"+k, depth)
			if r == nil {
				return nil
			}
			tried = append(tried, r)
		}
	}
	if th, ok := obj["then"]; ok {
		if el, ok := obj["else"]; ok {
			r := c.closed(th, path+"/then", depth)
			if r == nil {
				r = c.closed(el, path+"/else", depth)
			}
			if r == nil {
				return nil
			}
			tried = append(tried, r)
		}
	}
	r := c.own(obj, path, depth)
	if r == nil {
		return nil
	}
	return deepest(append(tried, r)...)
}

func (c *checker) allClosed(list []any, path string, depth int) *Result {
	for i, s := range list {
		if r := c.closed(s, path+"/"+strconv.Itoa(i), depth); r != nil {
			return r
		}
	}
	return nil
}

func (c *checker) followRef(ref, path string, depth int) *Result {
	if entered, ok := c.visiting[ref]; ok {
		if entered < depth {
			// The cycle stepped into the instance, which is finite, so
			// it bottoms out: assume what is being shown.
			return nil
		}
		return c.fail(path+"/$ref", "the reference "+ref+" refers back to itself without stepping into the instance")
	}
	t, tpath := c.root, ""
	if ref != "#" {
		t, _ = c.target(ref) // scan made sure it resolves
		tpath = "/$defs/" + escape(strings.TrimPrefix(ref, "#/$defs/"))
	}
	c.visiting[ref] = depth
	defer delete(c.visiting, ref)
	return c.closed(t, tpath, depth)
}

// own checks the node's own keywords for each type it allows.
func (c *checker) own(obj map[string]any, path string, depth int) *Result {
	allowed := typeNames
	if t, ok := obj["type"]; ok {
		allowed, _ = kinds(t) // scan made sure
	}
	// Strings first: with no type at all, that is the plainest reason.
	for _, kind := range []string{"string", "object", "array"} {
		if !slices.Contains(allowed, kind) {
			continue
		}
		var r *Result
		switch kind {
		case "string":
			r = c.ownString(obj, path)
		case "array":
			r = c.ownArray(obj, path, depth)
		case "object":
			r = c.ownObject(obj, path, depth)
		}
		if r != nil {
			return r
		}
	}
	return nil
}

func (c *checker) ownString(obj map[string]any, path string) *Result {
	if f, ok := obj["format"].(string); ok && slices.Contains(Formats, f) {
		return nil
	}
	return c.fail(path, "a string here can be any text (fix it with enum or const, or a format in "+strings.Join(Formats, ", ")+")")
}

func (c *checker) ownArray(obj map[string]any, path string, depth int) *Result {
	n, ok := obj["maxItems"].(json.Number)
	if !ok {
		return c.fail(path, fmt.Sprintf("an array here can be any length: maxItems is missing (at most %d)", MaxItems))
	}
	if f, err := n.Float64(); err != nil || f > MaxItems {
		return c.fail(path+"/maxItems", fmt.Sprintf("maxItems is %s, more than %d", n, MaxItems))
	}
	if list, ok := obj["prefixItems"].([]any); ok {
		if r := c.allClosed(list, path+"/prefixItems", depth+1); r != nil {
			return r
		}
	}
	items, ok := obj["items"]
	if !ok {
		return c.fail(path, "an array here can hold anything: items is missing")
	}
	return c.closed(items, path+"/items", depth+1)
}

func (c *checker) ownObject(obj map[string]any, path string, depth int) *Result {
	if props, ok := obj["properties"].(map[string]any); ok {
		for _, name := range sortedKeys(props) {
			if short(name) {
				return c.fail(path+"/properties/"+escape(name), fmt.Sprintf("the property name %q is so short that an array of such objects can spell anything", name))
			}
			if r := c.closed(props[name], path+"/properties/"+escape(name), depth+1); r != nil {
				return r
			}
		}
	}
	extra, hasExtra := obj["additionalProperties"]
	if b, ok := extra.(bool); ok && !b {
		return nil
	}
	pn, ok := obj["propertyNames"]
	if !ok {
		return c.fail(path, "an object here can have any keys: set additionalProperties to false, or fix the keys with propertyNames")
	}
	if r := c.closed(pn, path+"/propertyNames", depth+1); r != nil {
		return r
	}
	if !hasExtra {
		return c.fail(path, "values of keys not in properties can be anything: additionalProperties is missing")
	}
	return c.closed(extra, path+"/additionalProperties", depth+1)
}

// shortString finds a string, or object key, of one code point or none
// anywhere in v.
func shortString(v any) (string, bool) {
	switch v := v.(type) {
	case string:
		return v, short(v)
	case []any:
		for _, e := range v {
			if s, ok := shortString(e); ok {
				return s, true
			}
		}
	case map[string]any:
		for _, k := range sortedKeys(v) {
			if short(k) {
				return k, true
			}
			if s, ok := shortString(v[k]); ok {
				return s, true
			}
		}
	}
	return "", false
}

// short reports whether s is one code point or none, after NFC.
func short(s string) bool {
	return utf8.RuneCountInString(norm.NFC.String(s)) <= 1
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// escape makes a key one token of a JSON Pointer (RFC 6901).
func escape(k string) string {
	return strings.ReplaceAll(strings.ReplaceAll(k, "~", "~0"), "/", "~1")
}
