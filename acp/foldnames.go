package acpagent

import (
	"bytes"
	"encoding/json/jsontext"
	"fmt"
	"reflect"
	"strings"
	"unicode"
	"unicode/utf8"

	acp "github.com/caelis-labs/acp-go-sdk"
)

// The ACP SDK decodes every message with encoding/json (v1), which
// matches a member name to a struct field without regard to case: when
// no field has the exact name, "Method", "METHOD" and "paramſ" reach the
// fields named method and params, and {"cwd":"a","Cwd":"b"} reads as cwd
// "b". The strict dialect matches names exactly, so to relay and acp-tui
// those members are unknown and ignored (relay#58). A line the strict
// reader accepts could therefore still mean something else to the SDK.
//
// sdkNames closes that gap from the other side: it holds every member
// name the SDK decodes into a struct field, by its fold. A line with a
// member name, at any depth, that folds to one of them without being
// spelled exactly as one of them is refused like a line that fails the
// strict reader. The check is by name alone, not by where the member
// sits, so it also refuses such a name as a key of most free-form data
// (a tool's raw input, say). That is the safe direction: the line is not
// passed on, so it cannot mean two things. The exception is the value of
// a member in freeForm, which the SDK never decodes names from.
var sdkNames = foldedNames(sdkRoots...)

// sdkRoots are the types the SDK decodes from what a client sends: the
// params of every method, request and notification, and every result,
// in both directions, so a type the server starts to read later is
// already covered. TestSDKNamesCoverTheSDK checks the set against every
// json tag in the SDK's source.
var sdkRoots = []any{
	acp.AgentNotification{}, acp.AgentRequest{}, acp.AgentResponse{},
	acp.ClientNotification{}, acp.ClientRequest{}, acp.ClientResponse{},
	acp.RequestError{},
	acp.AuthenticateRequest{}, acp.AuthenticateResponse{},
	acp.CancelNotification{}, acp.CancelRequestNotification{},
	acp.CloseSessionRequest{}, acp.CloseSessionResponse{},
	acp.CompleteElicitationNotification{},
	acp.CreateElicitationRequest{}, acp.CreateElicitationResponse{},
	acp.CreateTerminalRequest{}, acp.CreateTerminalResponse{},
	acp.DeleteSessionRequest{}, acp.DeleteSessionResponse{},
	acp.ElicitationFormModeRequest{}, acp.ElicitationUrlModeRequest{},
	acp.ElicitationPropertySchema{}, // decoded on its own, from raw properties
	acp.InitializeRequest{}, acp.InitializeResponse{},
	acp.KillTerminalRequest{}, acp.KillTerminalResponse{},
	acp.ListSessionsRequest{}, acp.ListSessionsResponse{},
	acp.LoadSessionRequest{}, acp.LoadSessionResponse{},
	acp.LogoutRequest{}, acp.LogoutResponse{},
	acp.NewSessionRequest{}, acp.NewSessionResponse{},
	acp.PromptRequest{}, acp.PromptResponse{},
	acp.ReadTextFileRequest{}, acp.ReadTextFileResponse{},
	acp.ReleaseTerminalRequest{}, acp.ReleaseTerminalResponse{},
	acp.RequestPermissionRequest{}, acp.RequestPermissionResponse{},
	acp.ResumeSessionRequest{}, acp.ResumeSessionResponse{},
	acp.SessionNotification{},
	acp.SetSessionConfigOptionRequest{}, acp.SetSessionConfigOptionResponse{},
	acp.SetSessionModeRequest{}, acp.SetSessionModeResponse{},
	acp.TerminalOutputRequest{}, acp.TerminalOutputResponse{},
	acp.WaitForTerminalExitRequest{}, acp.WaitForTerminalExitResponse{},
	acp.WriteTextFileRequest{}, acp.WriteTextFileResponse{},
}

// freeForm are the member names whose value the SDK never decodes into a
// struct, wherever it reads them: _meta, kept as raw JSON per key, and
// data, an error's data or a content block's base64 string, decoded into
// any. Inside such a value v1 matches no name to a field, so a client may
// put any names there, "Title" in _meta included, and foldedName does not
// look. A member spelled otherwise ("_Meta", "Data") is still checked by
// its name like any other. TestFreeFormNamesAreFreeForm checks the SDK.
var freeForm = map[string]bool{"_meta": true, "data": true}

// envelopeNames are the members of the SDK's JSON-RPC envelope, which is
// unexported and so cannot be reflected over.
var envelopeNames = []string{"jsonrpc", "id", "method", "params", "result", "error"}

// foldedNames maps the fold of each struct field name reachable from
// roots, and of each envelope name, to the exact spellings with that fold.
func foldedNames(roots ...any) map[string][]string {
	names := map[string][]string{}
	add := func(name string) {
		k := fold(name)
		for _, n := range names[k] {
			if n == name {
				return
			}
		}
		names[k] = append(names[k], name)
	}
	for _, n := range envelopeNames {
		add(n)
	}
	seen := map[reflect.Type]bool{}
	var walk func(reflect.Type)
	walk = func(t reflect.Type) {
		for t.Kind() == reflect.Pointer || t.Kind() == reflect.Slice || t.Kind() == reflect.Array || t.Kind() == reflect.Map {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct || seen[t] {
			return
		}
		seen[t] = true
		for i := range t.NumField() {
			f := t.Field(i)
			if !f.IsExported() && !f.Anonymous {
				continue
			}
			tag, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			switch {
			case tag == "-" && f.Tag.Get("json") == "-":
				// Not decoded by name, but a custom decoder may fill it.
			case f.Anonymous && tag == "" && indirect(f.Type).Kind() == reflect.Struct:
				// An embedded struct's fields are the outer struct's.
			case tag != "":
				add(tag)
			default:
				add(f.Name)
			}
			walk(f.Type)
		}
	}
	for _, r := range roots {
		walk(reflect.TypeOf(r))
	}
	return names
}

func indirect(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t
}

// fold maps each rune to the smallest rune of its simple case folding
// orbit, so fold(a) == fold(b) exactly when bytes.EqualFold(a, b), the
// rule encoding/json v1 matches names by.
func fold(s string) string {
	var b strings.Builder
	for _, r := range s {
		min := r
		for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
			if f < min {
				min = f
			}
		}
		b.WriteRune(min)
	}
	return b.String()
}

// foldedName returns an error naming the first member of line, at any
// depth, whose name the SDK would match to a field of another spelling.
// line must already have passed the strict reader.
func foldedName(line []byte) error {
	dec := jsontext.NewDecoder(bytes.NewReader(line))
	for {
		tok, err := dec.ReadToken()
		if err != nil {
			return nil // the strict reader has already passed the line
		}
		if tok.Kind() == '"' {
			if k, n := dec.StackIndex(dec.StackDepth()); k == '{' && n%2 == 1 {
				name := tok.String()
				if folds(name) {
					return fmt.Errorf("member %q would be read by the ACP SDK as %q", name, sdkNames[fold(name)][0])
				}
				if freeForm[name] {
					if dec.SkipValue() != nil {
						return nil
					}
				}
			}
		}
		if dec.StackDepth() == 0 {
			return nil
		}
	}
}

// folds reports whether the SDK would match name to a field spelled
// otherwise.
func folds(name string) bool {
	if !utf8.ValidString(name) {
		return false
	}
	exact, ok := sdkNames[fold(name)]
	if !ok {
		return false
	}
	for _, n := range exact {
		if n == name {
			return false
		}
	}
	return true
}
