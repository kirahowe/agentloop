// Package fakeremote is a remote MCP server and the OAuth 2.1
// authorization server in front of it, for tests: of agentloop's
// streamable HTTP client, and of relay connect, which runs the whole
// sign-in against it. It never reaches anything outside the process.
//
// Under one base URL it serves:
//
//	/.well-known/oauth-protected-resource/mcp   protected-resource metadata
//	/.well-known/oauth-authorization-server/as  authorization-server metadata
//	/as/register   dynamic client registration (unless NoRegistration)
//	/as/authorize  approves at once, redirecting with a code (or Deny)
//	/as/token      authorization_code with PKCE S256, and refresh_token
//	/mcp           the MCP server, which takes only a token it issued
//
// The authorize endpoint checks what a real one would: a known client,
// its registered redirect URI, an S256 challenge, a state, and the
// resource being the MCP server. The token endpoint checks the verifier
// against the challenge, the redirect URI, the client and the resource.
package fakeremote

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Options say how the fake behaves.
type Options struct {
	// ToolsFile, when given, holds the tools as a JSON array of MCP tool
	// definitions, read again for each new session: changing it changes
	// what the server lists. Empty means DefaultTools.
	ToolsFile string
	// CallsFile, when given, gets one JSON line per tool call: the tool's
	// name and its arguments.
	CallsFile string
	// NoRegistration leaves out the registration endpoint, so a client
	// id must come from the operator. ClientID is then the one accepted.
	NoRegistration bool
	ClientID       string
	// Deny makes the authorize endpoint refuse.
	Deny bool
	// TTL is how long an access token lasts. Zero means an hour.
	TTL time.Duration
	// Rotate gives a new refresh token at each refresh and retires the
	// old one.
	Rotate bool
	// StaticToken, when given, is also accepted by the MCP server.
	StaticToken string
	// Iss makes the authorization server say it names itself in its
	// answers (RFC 9207), and do so.
	Iss bool
}

// DefaultTools is what the server lists without a ToolsFile: one tool of
// each kind relay connect proposes something different for.
const DefaultTools = `[
 {"name": "list_subscribers", "title": "List subscribers", "description": "List the subscribers of a group.",
  "inputSchema": {"type": "object", "properties": {"group": {"type": "string"}}},
  "annotations": {"readOnlyHint": true}},
 {"name": "send_campaign", "description": "Send a campaign to a group.",
  "inputSchema": {"type": "object", "properties": {"group": {"type": "string"}, "subject": {"type": "string"}, "send_now": {"type": "boolean"}}, "required": ["group"]},
  "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true}},
 {"name": "delete_subscriber", "description": "Delete a subscriber.",
  "inputSchema": {"type": "object", "properties": {"email": {"type": "string"}}},
  "annotations": {"destructiveHint": true}},
 {"name": "add_note", "description": "Add a note to a subscriber.",
  "inputSchema": {"type": "object", "properties": {"email": {"type": "string"}, "note": {"type": "string"}}}}
]`

// Server is the fake. Base is its URL, with no trailing slash, once
// Handler is served.
type Server struct {
	opts Options
	Base string

	mu       sync.Mutex
	clients  map[string][]string // client id -> redirect URIs
	codes    map[string]grant
	access   map[string]time.Time // token -> expiry
	refresh  map[string]string    // refresh token -> client id
	issued   int
	Requests []string // method and path of each request, in order
}

type grant struct {
	client, redirect, challenge, resource string
}

// New makes the fake. Set Base before serving.
func New(opts Options) *Server {
	if opts.TTL == 0 {
		opts.TTL = time.Hour
	}
	s := &Server{opts: opts, clients: map[string][]string{}, codes: map[string]grant{},
		access: map[string]time.Time{}, refresh: map[string]string{}}
	if opts.NoRegistration && opts.ClientID != "" {
		s.clients[opts.ClientID] = nil
	}
	return s
}

func (s *Server) resource() string { return s.Base + "/mcp" }
func (s *Server) issuer() string   { return s.Base + "/as" }

// Handler serves everything.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /.well-known/oauth-protected-resource/mcp", s.prm)
	mux.HandleFunc("GET /.well-known/oauth-authorization-server/as", s.asm)
	if !s.opts.NoRegistration {
		mux.HandleFunc("POST /as/register", s.register)
	}
	mux.HandleFunc("GET /as/authorize", s.authorize)
	mux.HandleFunc("POST /as/token", s.token)
	mcpHandler := sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return s.mcpServer() }, nil)
	mux.Handle("/mcp", s.authed(mcpHandler))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.Requests = append(s.Requests, r.Method+" "+r.URL.Path)
		s.mu.Unlock()
		mux.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func oauthError(w http.ResponseWriter, code int, kind, desc string) {
	writeJSON(w, code, map[string]string{"error": kind, "error_description": desc})
}

func (s *Server) prm(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"resource": s.resource(), "authorization_servers": []string{s.issuer()},
		"scopes_supported": []string{"mcp"}, "bearer_methods_supported": []string{"header"}})
}

func (s *Server) asm(w http.ResponseWriter, _ *http.Request) {
	m := map[string]any{
		"issuer":                                s.issuer(),
		"authorization_endpoint":                s.issuer() + "/authorize",
		"token_endpoint":                        s.issuer() + "/token",
		"response_types_supported":              []string{"code"},
		"grant_types_supported":                 []string{"authorization_code", "refresh_token"},
		"code_challenge_methods_supported":      []string{"S256"},
		"token_endpoint_auth_methods_supported": []string{"none"},
	}
	if !s.opts.NoRegistration {
		m["registration_endpoint"] = s.issuer() + "/register"
	}
	if s.opts.Iss {
		m["authorization_response_iss_parameter_supported"] = true
	}
	writeJSON(w, 200, m)
}

func random() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func loopbackRedirect(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && u.Scheme == "http" && u.Hostname() == "127.0.0.1" && u.Port() != ""
}

func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RedirectURIs            []string `json:"redirect_uris"`
		TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method"`
		ClientName              string   `json:"client_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.RedirectURIs) == 0 {
		oauthError(w, 400, "invalid_client_metadata", "redirect_uris are required")
		return
	}
	for _, u := range req.RedirectURIs {
		if !loopbackRedirect(u) {
			oauthError(w, 400, "invalid_redirect_uri", "a native client redirects to 127.0.0.1")
			return
		}
	}
	if req.TokenEndpointAuthMethod != "none" {
		oauthError(w, 400, "invalid_client_metadata", "only public clients")
		return
	}
	id := "client-" + random()[:8]
	s.mu.Lock()
	s.clients[id] = req.RedirectURIs
	s.mu.Unlock()
	writeJSON(w, 201, map[string]any{"client_id": id, "redirect_uris": req.RedirectURIs,
		"token_endpoint_auth_method": "none", "client_name": req.ClientName})
}

func (s *Server) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	s.mu.Lock()
	uris, known := s.clients[q.Get("client_id")]
	s.mu.Unlock()
	redirect := q.Get("redirect_uri")
	okRedirect := false
	for _, u := range uris {
		okRedirect = okRedirect || u == redirect
	}
	if s.opts.NoRegistration && known && loopbackRedirect(redirect) {
		okRedirect = true
	}
	if !known || !okRedirect {
		http.Error(w, "unknown client or redirect_uri", 400)
		return
	}
	back, _ := url.Parse(redirect)
	v := url.Values{}
	if st := q.Get("state"); st != "" {
		v.Set("state", st)
	}
	if s.opts.Iss {
		v.Set("iss", s.issuer())
	}
	switch {
	case s.opts.Deny:
		v.Set("error", "access_denied")
	case q.Get("response_type") != "code", q.Get("code_challenge_method") != "S256", q.Get("code_challenge") == "",
		q.Get("state") == "", q.Get("resource") != s.resource():
		v.Set("error", "invalid_request")
	default:
		code := random()
		s.mu.Lock()
		s.codes[code] = grant{client: q.Get("client_id"), redirect: redirect, challenge: q.Get("code_challenge"), resource: q.Get("resource")}
		s.mu.Unlock()
		v.Set("code", code)
	}
	back.RawQuery = v.Encode()
	http.Redirect(w, r, back.String(), http.StatusFound)
}

func (s *Server) token(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		oauthError(w, 400, "invalid_request", "bad form")
		return
	}
	f := r.PostForm
	s.mu.Lock()
	defer s.mu.Unlock()
	if f.Get("resource") != s.resource() {
		oauthError(w, 400, "invalid_target", "resource must be the MCP server")
		return
	}
	switch f.Get("grant_type") {
	case "authorization_code":
		g, ok := s.codes[f.Get("code")]
		delete(s.codes, f.Get("code"))
		sum := sha256.Sum256([]byte(f.Get("code_verifier")))
		if !ok || g.client != f.Get("client_id") || g.redirect != f.Get("redirect_uri") ||
			base64.RawURLEncoding.EncodeToString(sum[:]) != g.challenge {
			oauthError(w, 400, "invalid_grant", "the code, verifier, client or redirect does not match")
			return
		}
		s.issue(w, g.client, "")
	case "refresh_token":
		rt := f.Get("refresh_token")
		client, ok := s.refresh[rt]
		if !ok || client != f.Get("client_id") {
			oauthError(w, 400, "invalid_grant", "unknown refresh token")
			return
		}
		if s.opts.Rotate {
			delete(s.refresh, rt)
			rt = ""
		}
		s.issue(w, client, rt)
	default:
		oauthError(w, 400, "unsupported_grant_type", f.Get("grant_type"))
	}
}

// issue writes a token response. The caller holds the lock. A refresh
// token is kept when keep is given, else a new one is made.
func (s *Server) issue(w http.ResponseWriter, client, keep string) {
	s.issued++
	at := "at-" + strconv.Itoa(s.issued) + "-" + random()[:8]
	s.access[at] = time.Now().Add(s.opts.TTL)
	resp := map[string]any{"access_token": at, "token_type": "Bearer", "expires_in": int(s.opts.TTL.Seconds()), "scope": "mcp"}
	rt := keep
	if rt == "" {
		rt = "rt-" + random()
		s.refresh[rt] = client
	}
	resp["refresh_token"] = rt
	writeJSON(w, 200, resp)
}

// Valid reports whether the MCP server takes a token now.
func (s *Server) Valid(tok string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.opts.StaticToken != "" && tok == s.opts.StaticToken {
		return true
	}
	exp, ok := s.access[tok]
	return ok && time.Now().Before(exp)
}

func (s *Server) authed(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || !s.Valid(tok) {
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer resource_metadata="%s/.well-known/oauth-protected-resource/mcp", scope="mcp"`, s.Base))
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) tools() ([]*sdk.Tool, error) {
	raw := []byte(DefaultTools)
	if s.opts.ToolsFile != "" {
		b, err := os.ReadFile(s.opts.ToolsFile)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	var tools []*sdk.Tool
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil, err
	}
	return tools, nil
}

func (s *Server) mcpServer() *sdk.Server {
	srv := sdk.NewServer(&sdk.Implementation{Name: "fakeremote", Version: "1"}, nil)
	tools, err := s.tools()
	if err != nil {
		fmt.Fprintln(os.Stderr, "fakeremote:", err)
		return srv
	}
	for _, t := range tools {
		name := t.Name
		srv.AddTool(t, func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			var args map[string]any
			json.Unmarshal(req.Params.Arguments, &args)
			line, _ := json.Marshal(map[string]any{"tool": name, "args": args})
			if s.opts.CallsFile != "" {
				if f, err := os.OpenFile(s.opts.CallsFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
					f.Write(append(line, '\n'))
					f.Close()
				}
			}
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: string(line)}}}, nil
		})
	}
	return srv
}
