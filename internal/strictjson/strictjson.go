// Package strictjson is how agentloop reads JSON that another process
// wrote: the ACP client's messages, tool envelopes, MCP traffic, request
// files and the session files it reads back. relay and acp-tui read the
// same dialect (relay#58), so a message means one thing to all three or
// is refused by all three:
//
//   - RFC 8259 JSON in valid UTF-8, with no byte order mark and no
//     unpaired surrogate escape;
//   - exactly one value, with only whitespace after it;
//   - no member name twice in one object, at any depth, compared after
//     unescaping;
//   - member names matched exactly and case-sensitively (unknown members
//     are ignored unless the caller uses UnmarshalKnown);
//   - nesting at most MaxNesting levels, and no number literal longer
//     than MaxNumber characters.
//
// encoding/json/v2 does all but the last rule by default; Valid adds the
// two bounds with jsontext. encoding/json (v1) must not read such input:
// it folds case when it matches names to struct fields (so "Method" and
// "ſ" reach fields named method and s), keeps the last of two repeated
// names, and replaces invalid UTF-8.
package strictjson

import (
	"bytes"
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// The bounds of the dialect. The top value is level 1.
const (
	MaxNesting = 128
	MaxNumber  = 1000
)

// Valid says why data is not one JSON value in the dialect, or returns nil.
func Valid(data []byte) error {
	return limits{depth: MaxNesting, number: MaxNumber}.scan(data)
}

// Object is Valid, and the value must be an object: an ACP or JSON-RPC
// message, a tool envelope, a request.
func Object(data []byte) error {
	if err := Valid(data); err != nil {
		return err
	}
	if jsontext.Value(data).Kind() != '{' {
		return errors.New("not a JSON object")
	}
	return nil
}

// Unmarshal reads data, which must pass Valid, into v with
// encoding/json/v2. A time.Duration reads as a number of nanoseconds, as
// encoding/json writes it.
func Unmarshal(data []byte, v any, opts ...json.Options) error {
	if err := Valid(data); err != nil {
		return err
	}
	return json.Unmarshal(data, v, append([]json.Options{jsonv1.FormatDurationAsNano(true)}, opts...)...)
}

// UnmarshalKnown is Unmarshal that also refuses a member that no field of
// the target struct names, at any depth: an envelope or a request names
// only its own fields.
func UnmarshalKnown(data []byte, v any) error {
	err := Unmarshal(data, v, json.RejectUnknownMembers(true))
	var se *json.SemanticError
	if errors.Is(err, json.ErrUnknownName) && errors.As(err, &se) {
		return fmt.Errorf("unknown field %q", strings.Join(slices.Collect(se.JSONPointer.Tokens()), "."))
	}
	return err
}

// Numbers makes Unmarshal read a JSON number into an interface value as
// an encoding/json Number holding the literal as written, not as a
// float64 that may round it.
var Numbers = json.WithUnmarshalers(json.UnmarshalFromFunc(func(dec *jsontext.Decoder, v *any) error {
	if dec.PeekKind() != '0' {
		return errors.ErrUnsupported
	}
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	*v = jsonv1.Number(tok.String())
	return nil
}))

// Value reads data, which must pass Valid, as a value of no fixed shape:
// objects are map[string]any, arrays []any, and numbers jsonv1.Number
// with the literal as written.
func Value(data []byte) (any, error) {
	var v any
	if err := Unmarshal(data, &v, Numbers); err != nil {
		return nil, err
	}
	return v, nil
}

// The bounds every JSON value agentloop takes from a model must keep. They
// are tighter than the dialect's, so that what agentloop writes reads
// back, and passes on to stricter readers, with room to spare.
const (
	MaxDepth        = 64
	MaxNumberLength = 100
)

// Input checks a tool call's input or a model's structured reply: one
// value in the dialect, nested no deeper than MaxDepth, with no number
// literal longer than MaxNumberLength characters.
func Input(data []byte) error {
	return limits{depth: MaxDepth, number: MaxNumberLength}.scan(data)
}

// Bounded checks what a provider returned for a turn the same way as
// Input: it is written to the session log, and the log must read back by
// the dialect.
func Bounded(data []byte) error { return Input(data) }

type limits struct{ depth, number int }

// scan reads data token by token with jsontext's default options, which
// check the syntax, the UTF-8, the surrogate escapes and repeated names.
// It adds the depth and number bounds, and refuses anything after the
// first value but whitespace.
func (l limits) scan(data []byte) error {
	dec := jsontext.NewDecoder(bytes.NewReader(data))
	for {
		if dec.PeekKind() == '0' {
			v, err := dec.ReadValue()
			if err != nil {
				return err
			}
			if len(v) > l.number {
				return fmt.Errorf("a number longer than %d characters", l.number)
			}
		} else {
			tok, err := dec.ReadToken()
			if err != nil {
				return err
			}
			if k := tok.Kind(); (k == '{' || k == '[') && dec.StackDepth() > l.depth {
				return fmt.Errorf("nesting deeper than %d levels", l.depth)
			}
		}
		if dec.StackDepth() == 0 {
			break
		}
	}
	if _, err := dec.ReadToken(); err != io.EOF {
		if err == nil {
			return errors.New("more than one JSON value")
		}
		return err
	}
	return nil
}
