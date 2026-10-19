// Command mcpremote serves fakeremote, a remote MCP server and its OAuth
// authorization server, on 127.0.0.1 at a port of its own, for tests. It
// prints its base URL on a line of its own and serves until its stdin
// closes.
//
//	-tools FILE        the tools, a JSON array, read again for each session
//	-calls FILE        append each tool call to FILE as a JSON line
//	-no-registration   offer no dynamic client registration
//	-client-id ID      the client id it takes without registration
//	-deny              refuse every authorization
//	-ttl SECONDS       how long an access token lasts
//	-rotate            a new refresh token at each refresh
//	-token TOKEN       a token the MCP server also takes
//	-iss               name the issuer in every authorization answer
package main

import (
	"flag"
	"fmt"
	"io"
	"net/http/httptest"
	"os"
	"time"

	"github.com/kirahowe/agentloop/internal/fakeremote"
)

func main() {
	var o fakeremote.Options
	flag.StringVar(&o.ToolsFile, "tools", "", "")
	flag.StringVar(&o.CallsFile, "calls", "", "")
	flag.BoolVar(&o.NoRegistration, "no-registration", false, "")
	flag.StringVar(&o.ClientID, "client-id", "", "")
	flag.BoolVar(&o.Deny, "deny", false, "")
	ttl := flag.Int("ttl", 3600, "")
	flag.BoolVar(&o.Rotate, "rotate", false, "")
	flag.StringVar(&o.StaticToken, "token", "", "")
	flag.BoolVar(&o.Iss, "iss", false, "")
	flag.Parse()
	o.TTL = time.Duration(*ttl) * time.Second
	f := fakeremote.New(o)
	// httptest listens on 127.0.0.1 at a free port.
	srv := httptest.NewUnstartedServer(nil)
	f.Base = "http://" + srv.Listener.Addr().String()
	srv.Config.Handler = f.Handler()
	srv.Start()
	fmt.Println(f.Base)
	io.Copy(io.Discard, os.Stdin)
	srv.Close()
}
