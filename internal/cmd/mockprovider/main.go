// Command mockprovider is a stand-in for Anthropic's Messages and Models
// APIs, for running agentloop against conformance suites without a key or
// a network.
// It is built only by scripts/tck.sh.
//
//	mockprovider [-addr 127.0.0.1:0]
//
// It prints its base URL on the first line of stdout and serves until
// killed. GET /v1/models/{id} describes any model, with a context window
// of Window tokens. Every other request gets a short streamed text reply,
// except one whose body holds the slow marker: that one streams a first
// chunk and then waits until the client goes away, so a cancel can land
// mid-turn.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// SlowMarker in a prompt makes the reply hang after its first chunk.
const SlowMarker = "MOCK-SLOW"

// Window is the context window, in tokens, of every model described.
const Window = 200000

func main() {
	addr := flag.String("addr", "127.0.0.1:0", "address to listen on")
	flag.Parse()
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mockprovider:", err)
		os.Exit(1)
	}
	fmt.Printf("http://%s\n", ln.Addr())
	os.Stdout.Sync()
	if err := http.Serve(ln, http.HandlerFunc(handle)); err != nil {
		fmt.Fprintln(os.Stderr, "mockprovider:", err)
		os.Exit(1)
	}
}

func handle(w http.ResponseWriter, r *http.Request) {
	if id, ok := strings.CutPrefix(r.URL.Path, "/v1/models/"); ok && r.Method == http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"type":"model","id":%q,"display_name":"Mock model","created_at":"2026-01-01T00:00:00Z","max_input_tokens":%d,"max_tokens":64000,"capabilities":{}}`, id, Window)
		return
	}
	body, _ := io.ReadAll(r.Body)
	w.Header().Set("Content-Type", "text/event-stream")
	send := func(event, data string) {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
		w.(http.Flusher).Flush()
	}
	send("message_start", `{"type":"message_start","message":{"id":"msg_mock","type":"message","role":"assistant","model":"mock","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`)
	send("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	if bytes.Contains(body, []byte(SlowMarker)) {
		send("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Working on it"}}`)
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Minute):
		}
		return
	}
	send("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello from the mock provider."}}`)
	send("content_block_stop", `{"type":"content_block_stop","index":0}`)
	send("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}`)
	send("message_stop", `{"type":"message_stop"}`)
}
