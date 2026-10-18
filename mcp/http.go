package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	mimetype "mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// A remote server (Server.URL) is spoken to over MCP's streamable HTTP
// transport, with the SDK's client: each message is a POST, and the
// answer is JSON or a stream of server-sent events. The SDK keeps the
// session id (Mcp-Session-Id) and sends the negotiated protocol version
// (MCP-Protocol-Version). agentloop adds these rules:
//
//   - The URL is https, or http to a loopback address. It has no user
//     name, password or fragment.
//   - A redirect is followed only to the same origin (scheme, host and
//     port); anything else fails the request. No proxy is used.
//   - The bearer token from Server.Token goes only to the server's
//     origin. After a 401 the token is asked for again once (renew), and
//     the request is sent again with it. A request refused because of its
//     credentials was not acted on, so sending it again is safe.
//   - No standalone event stream is opened (a GET), since agentloop takes
//     no requests from the server, and a broken stream is never resumed.
//   - A response body that is not an event stream is cut at
//     MaxMessageBytes, as is each event of a stream.

// renewEvery is how often a 401 may make the token be asked for again.
const renewEvery = 10 * time.Second

// CheckURL says why a remote server's URL cannot be used, or nil.
func CheckURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("the URL %q does not parse: %w", raw, err)
	}
	if u.User != nil {
		return fmt.Errorf("the URL %q holds a user name or password", raw)
	}
	if u.Fragment != "" || strings.Contains(raw, "#") {
		return fmt.Errorf("the URL %q has a fragment", raw)
	}
	if u.Host == "" || u.Hostname() == "" {
		return fmt.Errorf("the URL %q has no host", raw)
	}
	switch u.Scheme {
	case "https":
		return nil
	case "http":
		if Loopback(u.Hostname()) {
			return nil
		}
		return fmt.Errorf("the URL %q is http to a host that is not this machine; use https", raw)
	default:
		return fmt.Errorf("the URL %q is not http or https", raw)
	}
}

// Loopback reports whether a host name is this machine: localhost or a
// loopback address.
func Loopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// origin is a URL's scheme, host and port, with the default port made
// explicit, so that https://h and https://h:443 are the same.
func origin(u *url.URL) string {
	port := u.Port()
	if port == "" {
		switch u.Scheme {
		case "https":
			port = "443"
		case "http":
			port = "80"
		}
	}
	return strings.ToLower(u.Scheme) + "://" + strings.ToLower(u.Hostname()) + ":" + port
}

// httpTransport is the SDK's streamable client, set up by the rules
// above.
func httpTransport(srv Server, max int) (sdk.Transport, error) {
	if err := CheckURL(srv.URL); err != nil {
		return nil, fmt.Errorf("mcp server %s: %w", srv.Name, err)
	}
	u, _ := url.Parse(srv.URL)
	base := srv.HTTPTransport
	if base == nil {
		t := http.DefaultTransport.(*http.Transport).Clone()
		// Never from the environment.
		t.Proxy = nil
		t.TLSHandshakeTimeout = 15 * time.Second
		base = t
	}
	rt := &authTripper{base: base, origin: origin(u), token: srv.Token, max: int64(max)}
	client := &http.Client{
		Transport: rt,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return errors.New("too many redirects")
			}
			if origin(req.URL) != rt.origin {
				return fmt.Errorf("the server redirected to %s, another origin, which agentloop does not follow", origin(req.URL))
			}
			return nil
		},
	}
	return &sdk.StreamableClientTransport{
		Endpoint:             srv.URL,
		HTTPClient:           client,
		MaxRetries:           -1,
		DisableStandaloneSSE: true,
		MaxEventSize:         max,
	}, nil
}

// authTripper adds the bearer token to requests for the server's origin,
// asks for a new one after a 401, and caps response bodies.
type authTripper struct {
	base   http.RoundTripper
	origin string
	token  func(ctx context.Context, renew bool) (string, error)
	max    int64

	mu      sync.Mutex
	current string
	have    bool
	renewed time.Time
}

func (a *authTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if origin(req.URL) != a.origin {
		return nil, fmt.Errorf("agentloop sends nothing to %s, which is not the server's origin", origin(req.URL))
	}
	if a.token == nil {
		return a.capped(a.base.RoundTrip(req))
	}
	tok, err := a.get(req.Context(), false)
	if err != nil {
		return nil, err
	}
	resp, err := a.base.RoundTrip(withToken(req, tok))
	if err != nil || resp.StatusCode != http.StatusUnauthorized {
		return a.capped(resp, err)
	}
	// Asked for a new token, the credentials command may refresh it.
	if req.Body != nil && req.GetBody == nil {
		return a.capped(resp, nil)
	}
	fresh, err := a.get(req.Context(), true)
	if err != nil || fresh == tok {
		return a.capped(resp, nil)
	}
	again := req.Clone(req.Context())
	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return a.capped(resp, nil)
		}
		again.Body = body
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, a.max))
	resp.Body.Close()
	return a.capped(a.base.RoundTrip(withToken(again, fresh)))
}

// get is the token, asked for again when renew is set and the last
// renewal is long enough ago.
func (a *authTripper) get(ctx context.Context, renew bool) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.have && (!renew || time.Since(a.renewed) < renewEvery) {
		return a.current, nil
	}
	tok, err := a.token(ctx, renew && a.have)
	if err != nil {
		return "", err
	}
	if err := checkToken(tok); err != nil {
		return "", err
	}
	a.current, a.have = tok, true
	if renew {
		a.renewed = time.Now()
	}
	return tok, nil
}

// checkToken refuses a token that cannot go in a header.
func checkToken(tok string) error {
	if tok == "" {
		return errors.New("the bearer token is empty")
	}
	for _, r := range tok {
		if r < 0x21 || r > 0x7e {
			return errors.New("the bearer token holds a space, a control character or a character that is not ASCII")
		}
	}
	return nil
}

func withToken(req *http.Request, tok string) *http.Request {
	r := req.Clone(req.Context())
	r.Header.Set("Authorization", "Bearer "+tok)
	return r
}

// capped cuts a response body that is not an event stream at max bytes:
// reading past it fails.
func (a *authTripper) capped(resp *http.Response, err error) (*http.Response, error) {
	if err != nil || resp == nil {
		return resp, err
	}
	media, _, _ := mimetype.ParseMediaType(resp.Header.Get("Content-Type"))
	if media != "text/event-stream" {
		resp.Body = &capReader{r: resp.Body, left: a.max}
	}
	return resp, nil
}

var errBodyTooLong = errors.New("the server's response is longer than the maximum message size")

type capReader struct {
	r    io.ReadCloser
	left int64
}

func (c *capReader) Read(p []byte) (int, error) {
	if c.left < 0 {
		return 0, errBodyTooLong
	}
	if int64(len(p)) > c.left+1 {
		p = p[:c.left+1]
	}
	n, err := c.r.Read(p)
	c.left -= int64(n)
	if c.left < 0 {
		return 0, errBodyTooLong
	}
	return n, err
}

func (c *capReader) Close() error { return c.r.Close() }
