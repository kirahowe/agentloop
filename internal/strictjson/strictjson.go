// Package strictjson checks JSON that encoding/json would read too
// generously. encoding/json matches object keys to struct fields without
// regard to case, and when an object names a key twice the last one wins,
// so {"labels":["private"],"Labels":[]} decodes to no labels. Check
// refuses both: every object may name a key once, and an object read into
// a struct may name only the struct's fields, spelled exactly.
package strictjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

var rawMessage = reflect.TypeFor[json.RawMessage]()

// Check reads the JSON in data as it would be decoded into a value of type
// t, which may be nil for JSON with no fixed shape. It returns an error
// for a repeated key in any object, and for an object key that is not
// exactly the JSON name of a field where t says a struct goes. It does not
// check anything else (types, required fields, trailing data); decoding
// does that.
func Check(data []byte, t reflect.Type) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	return walk(dec, t, "")
}

// DuplicateKeys reports a key named twice in one object anywhere in data.
func DuplicateKeys(data []byte) error { return Check(data, nil) }

func walk(dec *json.Decoder, t reflect.Type, path string) error {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == rawMessage || (t != nil && t.Kind() == reflect.Interface) {
		t = nil
	}
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	d, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch d {
	case '[':
		var elem reflect.Type
		if t != nil && (t.Kind() == reflect.Slice || t.Kind() == reflect.Array) {
			elem = t.Elem()
		}
		for i := 0; dec.More(); i++ {
			if err := walk(dec, elem, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case '{':
		var fields map[string]reflect.Type
		var elem reflect.Type
		if t != nil {
			switch t.Kind() {
			case reflect.Struct:
				fields = map[string]reflect.Type{}
				collect(t, fields)
			case reflect.Map:
				elem = t.Elem()
			}
		}
		seen := map[string]bool{}
		for dec.More() {
			tok, err := dec.Token()
			if err != nil {
				return err
			}
			key := tok.(string)
			at := path + "." + key
			if seen[key] {
				return fmt.Errorf("%s is given twice", strings.TrimPrefix(at, "."))
			}
			seen[key] = true
			next := elem
			if fields != nil {
				ft, ok := fields[key]
				if !ok {
					return fmt.Errorf("unknown field %q", strings.TrimPrefix(at, "."))
				}
				next = ft
			}
			if err := walk(dec, next, at); err != nil {
				return err
			}
		}
	}
	_, err = dec.Token() // the closing delimiter
	return err
}

// collect maps the JSON name of each field of struct type t to its type,
// with the fields of embedded structs promoted as encoding/json does.
func collect(t reflect.Type, into map[string]reflect.Type) {
	for i := range t.NumField() {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if f.Anonymous && name == "" {
			ft := f.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				collect(ft, into)
				continue
			}
		}
		if !f.IsExported() {
			continue
		}
		if name == "" {
			name = f.Name
		}
		into[name] = f.Type
	}
}
