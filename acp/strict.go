package acpagent

import (
	"bufio"
	"bytes"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// MaxLine is the longest line the server reads from the client, newline
// excluded. It is the SDK's own frame limit.
const MaxLine = 10 << 20

// errLineTooLong ends the client's input: a line longer than MaxLine
// cannot be read whole, so nothing after it can be trusted to start a
// line.
var errLineTooLong = fmt.Errorf("a line from the client is longer than %d bytes", MaxLine)

// strictInput screens each line the client writes before the SDK reads
// it. The SDK decodes with encoding/json (v1), which matches member names
// without regard to case and keeps one of two repeated names, so a line
// relay or relay-tui read one way could mean another here. Every line must
// be one JSON object in the strict dialect of internal/strictjson, the
// one relay and relay-tui read (relay#58). A member whose name the SDK
// would fold onto a field spelled otherwise (see sdkNames) is unknown to
// that dialect, so it is removed before the SDK sees the line, and the
// SDK reads the same members as the strict reader. Any other line that
// passes reaches the SDK byte for byte. A line that fails never does: if it is a request
// whose id can be read, the client gets a JSON-RPC parse error (-32700)
// for that id, and otherwise it is dropped. A blank line is dropped too,
// as the SDK would skip it.
//
// Only a request is answered. A response from the client that fails
// carries the id of a request the agent sent, and an error with that id
// would read, to the client, as the answer to a request of its own.
type strictInput struct {
	r       *bufio.Reader
	out     *serialWriter
	logf    func(string, ...any)
	pending []byte
	err     error
}

func newStrictInput(r io.Reader, out *serialWriter, logf func(string, ...any)) *strictInput {
	return &strictInput{r: bufio.NewReader(r), out: out, logf: logf}
}

func (s *strictInput) Read(p []byte) (int, error) {
	for len(s.pending) == 0 {
		if s.err != nil {
			return 0, s.err
		}
		line, err := s.line()
		if len(line) > 0 {
			s.screen(line)
		}
		if err != nil {
			s.err = err
		}
	}
	n := copy(p, s.pending)
	s.pending = s.pending[n:]
	return n, nil
}

// line reads the next line with its newline, if it has one.
func (s *strictInput) line() ([]byte, error) {
	var line []byte
	for {
		chunk, err := s.r.ReadSlice('\n')
		if len(line)+len(chunk) > MaxLine+1 {
			return nil, errLineTooLong
		}
		line = append(line, chunk...)
		if !errors.Is(err, bufio.ErrBufferFull) {
			return line, err
		}
	}
}

// screen passes a line on, answers it, or drops it.
func (s *strictInput) screen(line []byte) {
	msg := bytes.TrimSuffix(line, []byte("\n"))
	if len(bytes.Trim(msg, " \t\r")) == 0 {
		return
	}
	err := strictjson.Object(msg)
	if err == nil {
		out, removed := withoutFolded(msg)
		if len(removed) == 0 {
			s.pending = line
			return
		}
		s.logf("agentloop: removed members %q from a line from the client, which the ACP SDK would read under another name", removed)
		s.pending = append(out, line[len(msg):]...)
		return
	}
	id, ok := requestID(msg)
	if !ok {
		s.logf("agentloop: dropped a line from the client that is not strict JSON: %v", err)
		return
	}
	s.logf("agentloop: refused request %s from the client, which is not strict JSON: %v", id, err)
	resp := append(append([]byte(`{"jsonrpc":"2.0","id":`), id...), `,"error":{"code":-32700,"message":"Parse error"}}`+"\n"...)
	if _, err := s.out.Write(resp); err != nil {
		s.logf("agentloop: cannot answer the client: %v", err)
	}
}

// requestID reads the id of a line that failed the strict reader, if the
// line is still one JSON object with exactly one member named id, a
// string or a number in the dialect, and a member named method. Repeated
// names other than id and invalid UTF-8 outside the id are let through,
// since the point is to answer the request the client sent.
func requestID(line []byte) (jsontext.Value, bool) {
	dec := jsontext.NewDecoder(bytes.NewReader(line), jsontext.AllowDuplicateNames(true), jsontext.AllowInvalidUTF8(true))
	if tok, err := dec.ReadToken(); err != nil || tok.Kind() != '{' {
		return nil, false
	}
	var id jsontext.Value
	ids, method := 0, false
	for dec.PeekKind() != '}' {
		tok, err := dec.ReadToken()
		if err != nil {
			return nil, false
		}
		name := tok.String() // the token is only good until the next read
		v, err := dec.ReadValue()
		if err != nil {
			return nil, false
		}
		switch name {
		case "id":
			ids++
			id = bytes.Clone(v)
		case "method":
			method = true
		}
	}
	if _, err := dec.ReadToken(); err != nil {
		return nil, false
	}
	if ids != 1 || !method || (id.Kind() != '"' && id.Kind() != '0') || strictjson.Valid(id) != nil {
		return nil, false
	}
	return id, true
}

// serialWriter writes each buffer whole, so a parse error the screen
// writes cannot fall inside a message the SDK writes. The SDK writes
// each message with one Write.
type serialWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *serialWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}
