package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// wire is the stdio connection to the server: newline-delimited JSON-RPC,
// read as one stream of JSON values. It decodes each message with the
// SDK's own decoder, so what it checks is exactly what the SDK acts on,
// and it checks each message before the SDK sees it, so a violation is
// recorded before any message after it is read. These are violations:
//
//   - a request from the server other than ping, and a notification
//     whose method is not notifications/...
//   - a batch (a JSON array)
//   - a frame that is not a JSON-RPC message, that names a key twice in
//     an object, or that shares a line with the message before it
//
// A violation breaks the connection (see conn.violation). A frame longer
// than the cap ends it without being one.
type wire struct {
	r   io.ReadCloser
	w   io.WriteCloser
	cn  *conn
	lim *frameLimit
	dec *json.Decoder
	end int64 // where the last message ended in the stream, or -1

	wmu sync.Mutex
}

var _ sdk.Transport = (*wire)(nil)

func newWire(r io.ReadCloser, w io.WriteCloser, cn *conn, max int) *wire {
	lim := &frameLimit{r: r, max: max}
	return &wire{r: r, w: w, cn: cn, lim: lim, dec: json.NewDecoder(lim), end: -1}
}

// Connect makes the wire its own connection: the SDK calls it once.
func (wr *wire) Connect(context.Context) (sdk.Connection, error) { return wr, nil }

var errFrameTooLong = errors.New("a message from the server is longer than the maximum line length")

// frameLimit fails a read once one frame has taken more than max bytes.
// The decoder reads ahead, so a frame may be charged for the start of
// the next one; the cap is a bound, not an exact size.
type frameLimit struct {
	r     io.Reader
	max   int
	count int
	// read is how many bytes have been read, and newlines where in the
	// stream the newlines among them are that the wire has yet to pass.
	read     int64
	newlines []int64
}

func (f *frameLimit) Read(p []byte) (int, error) {
	if f.count >= f.max {
		return 0, errFrameTooLong
	}
	if len(p) > f.max-f.count {
		p = p[:f.max-f.count]
	}
	n, err := f.r.Read(p)
	for i, b := range p[:n] {
		if b == '\n' {
			f.newlines = append(f.newlines, f.read+int64(i))
		}
	}
	f.count += n
	f.read += int64(n)
	return n, err
}

func (wr *wire) Read(ctx context.Context) (jsonrpc.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// What the decoder already holds counts toward the next frame.
	wr.lim.count = bufferedLen(wr.dec)
	var raw json.RawMessage
	if err := wr.dec.Decode(&raw); err != nil {
		var syn *json.SyntaxError
		if errors.As(err, &syn) {
			wr.cn.violation("the server wrote something that is not JSON: " + err.Error())
		}
		return nil, err
	}
	// Each message must have its own line: a newline between the end of
	// the last one and the start of this one.
	end := wr.dec.InputOffset()
	start := end - int64(len(raw))
	parted := wr.end < 0
	kept := wr.lim.newlines[:0]
	for _, at := range wr.lim.newlines {
		if at >= wr.end && at < start {
			parted = true
		}
		if at >= end {
			kept = append(kept, at)
		}
	}
	wr.lim.newlines, wr.end = kept, end
	if !parted {
		wr.cn.violation("the server wrote more than one JSON value on a line")
		return nil, errors.New("a message from the server shares a line with the one before")
	}
	if strictjson.DuplicateKeys(raw) != nil {
		wr.cn.violation("the server sent a message that names a key twice")
		return nil, errors.New("a message from the server names a key twice")
	}
	msg, why := checkFrame(raw)
	if why != "" {
		wr.cn.violation(why)
	}
	if msg == nil {
		return nil, errors.New(why)
	}
	return msg, nil
}

// checkFrame decodes one frame from the server and says why it breaks the
// protocol, or "". A frame that is not a message returns nil.
func checkFrame(raw []byte) (jsonrpc.Message, string) {
	if t := bytes.TrimLeft(raw, " \t\r\n"); len(t) > 0 && t[0] == '[' {
		return nil, "the server sent a batch, which agentloop does not take"
	}
	msg, err := jsonrpc.DecodeMessage(raw)
	if err != nil {
		return nil, "the server sent something that is not a JSON-RPC message: " + err.Error()
	}
	req, ok := msg.(*jsonrpc.Request)
	if !ok {
		return msg, ""
	}
	switch {
	case req.IsCall() && req.Method == "ping":
	case !req.IsCall() && strings.HasPrefix(req.Method, "notifications/"):
	default:
		return msg, requestViolation(req.Method)
	}
	return msg, ""
}

func bufferedLen(dec *json.Decoder) int {
	if b, ok := dec.Buffered().(*bytes.Reader); ok {
		return b.Len()
	}
	return 0
}

func (wr *wire) Write(ctx context.Context, msg jsonrpc.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := jsonrpc.EncodeMessage(msg)
	if err != nil {
		return err
	}
	wr.wmu.Lock()
	defer wr.wmu.Unlock()
	_, err = wr.w.Write(append(data, '\n'))
	return err
}

func (wr *wire) Close() error {
	err1 := wr.r.Close()
	err2 := wr.w.Close()
	return errors.Join(err1, err2)
}

func (wr *wire) SessionID() string { return "" }
