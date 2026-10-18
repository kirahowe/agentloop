package fakeremote

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop/mcp"
	"github.com/kirahowe/agentloop/trifecta"
)

func start(t *testing.T, o Options) (*Server, *http.Client) {
	t.Helper()
	f := New(o)
	srv := httptest.NewUnstartedServer(nil)
	f.Base = "http://" + srv.Listener.Addr().String()
	srv.Config.Handler = f.Handler()
	srv.Start()
	t.Cleanup(srv.Close)
	return f, &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

func getJSON(t *testing.T, c *http.Client, u string, into any) {
	t.Helper()
	resp, err := c.Get(u)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("GET %s: %d", u, resp.StatusCode)
	}
	json.NewDecoder(resp.Body).Decode(into)
}

func postForm(t *testing.T, c *http.Client, u string, v url.Values) (int, map[string]any) {
	t.Helper()
	resp, err := c.PostForm(u, v)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	json.NewDecoder(resp.Body).Decode(&m)
	return resp.StatusCode, m
}

// signIn runs the sign-in as a client would, and returns the client id
// and the token response.
func signIn(t *testing.T, f *Server, c *http.Client, verifier string) (string, map[string]any) {
	t.Helper()
	var prm struct {
		Resource string   `json:"resource"`
		AS       []string `json:"authorization_servers"`
	}
	getJSON(t, c, f.Base+"/.well-known/oauth-protected-resource/mcp", &prm)
	if prm.Resource != f.Base+"/mcp" || len(prm.AS) != 1 {
		t.Fatalf("%+v", prm)
	}
	asURL, _ := url.Parse(prm.AS[0])
	var asm map[string]any
	getJSON(t, c, asURL.Scheme+"://"+asURL.Host+"/.well-known/oauth-authorization-server"+asURL.Path, &asm)
	redirect := "http://127.0.0.1:5555/callback"
	body, _ := json.Marshal(map[string]any{"redirect_uris": []string{redirect}, "token_endpoint_auth_method": "none"})
	resp, err := c.Post(asm["registration_endpoint"].(string), "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	var reg map[string]any
	json.NewDecoder(resp.Body).Decode(&reg)
	resp.Body.Close()
	client := reg["client_id"].(string)
	sum := sha256.Sum256([]byte(verifier))
	q := url.Values{"response_type": {"code"}, "client_id": {client}, "redirect_uri": {redirect},
		"code_challenge": {base64.RawURLEncoding.EncodeToString(sum[:])}, "code_challenge_method": {"S256"},
		"state": {"st"}, "resource": {f.Base + "/mcp"}}
	resp, err = c.Get(asm["authorization_endpoint"].(string) + "?" + q.Encode())
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	back, _ := url.Parse(resp.Header.Get("Location"))
	if back.Query().Get("state") != "st" || back.Query().Get("code") == "" {
		t.Fatalf("redirect %s", back)
	}
	code, tok := postForm(t, c, asm["token_endpoint"].(string), url.Values{"grant_type": {"authorization_code"},
		"code": {back.Query().Get("code")}, "redirect_uri": {redirect}, "client_id": {client},
		"code_verifier": {"right-verifier"}, "resource": {f.Base + "/mcp"}})
	if verifier == "right-verifier" && code != 200 {
		t.Fatalf("token: %d %v", code, tok)
	}
	if verifier != "right-verifier" && code == 200 {
		t.Fatal("a wrong verifier got a token")
	}
	return client, tok
}

func TestSignInRefreshAndCall(t *testing.T) {
	f, c := start(t, Options{Rotate: true})
	client, tok := signIn(t, f, c, "right-verifier")
	at := tok["access_token"].(string)
	rt := tok["refresh_token"].(string)
	code, again := postForm(t, c, f.Base+"/as/token", url.Values{"grant_type": {"refresh_token"}, "refresh_token": {rt},
		"client_id": {client}, "resource": {f.Base + "/mcp"}})
	if code != 200 || again["refresh_token"] == rt {
		t.Fatalf("refresh: %d %v", code, again)
	}
	if code, _ := postForm(t, c, f.Base+"/as/token", url.Values{"grant_type": {"refresh_token"}, "refresh_token": {rt},
		"client_id": {client}, "resource": {f.Base + "/mcp"}}); code == 200 {
		t.Fatal("a rotated refresh token still works")
	}
	listed, err := mcp.List(context.Background(), mcp.Server{Name: "fake", URL: f.Base + "/mcp",
		Token: func(context.Context, bool) (string, error) { return at, nil }})
	if err != nil || len(listed) != 4 {
		t.Fatal(listed, err)
	}
	if _, err := mcp.List(context.Background(), mcp.Server{Name: "fake", URL: f.Base + "/mcp"}); err == nil {
		t.Fatal("the MCP server took no token")
	}
	cl, err := mcp.Connect(context.Background(), mcp.Server{Name: "fake", URL: f.Base + "/mcp",
		Token: func(context.Context, bool) (string, error) { return again["access_token"].(string), nil },
		Tools: map[string]mcp.ToolConfig{"list_subscribers": {Policy: trifecta.Tool{Class: trifecta.Read}}}})
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	if len(cl.Tools()) != 1 {
		t.Fatal(cl.Tools())
	}
}

func TestAWrongVerifierGetsNoToken(t *testing.T) {
	f, c := start(t, Options{})
	signIn(t, f, c, "wrong-verifier")
}

func TestAnExpiredTokenIsRefused(t *testing.T) {
	f, c := start(t, Options{TTL: time.Millisecond})
	_, tok := signIn(t, f, c, "right-verifier")
	time.Sleep(5 * time.Millisecond)
	if f.Valid(tok["access_token"].(string)) {
		t.Fatal("expired token valid")
	}
}
