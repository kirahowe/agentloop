package httpclient_test

import (
	"context"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/anthropic"
	"github.com/kirahowe/agentloop/internal/httpclient"
	"github.com/kirahowe/agentloop/openai"
)

// The test binary runs itself as the child, so the proxy variables are in
// its environment from the start (Go reads them once per process).
var childProvider = flag.String("child.provider", "", "run as the child: call this provider's model once")
var childControl = flag.Bool("child.control", false, "the child uses http.DefaultClient")

// An unreachable host name, so that a proxy is the only way a request
// could get anywhere. Go never proxies loopback addresses.
const modelURL = "http://model.invalid/v1"

func TestChild(t *testing.T) {
	if *childProvider == "" {
		t.Skip("run by TestModelClientsIgnoreProxyVariables")
	}
	var hc *http.Client
	if *childControl {
		hc = http.DefaultClient
	}
	var m agentloop.Model
	switch *childProvider {
	case "anthropic":
		m = anthropic.New(anthropic.Config{APIKey: "k", BaseURL: modelURL, HTTPClient: hc})
	case "openai":
		m = openai.New(openai.Config{APIKey: "k", BaseURL: modelURL, HTTPClient: hc})
	default:
		t.Fatalf("provider %q", *childProvider)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	m.Generate(ctx, &agentloop.Request{Model: "m", MaxOutputTokens: 10, Messages: []agentloop.Message{agentloop.Text(agentloop.User, "hi")}}, nil)
}

// With HTTP_PROXY, HTTPS_PROXY and ALL_PROXY set, the model clients still
// connect directly: nothing reaches the proxy. The control run, with
// http.DefaultClient, does reach it, so the test can tell the difference.
func TestModelClientsIgnoreProxyVariables(t *testing.T) {
	var hits atomic.Int64
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "no", http.StatusBadGateway)
	}))
	defer proxy.Close()
	for _, provider := range []string{"anthropic", "openai"} {
		for _, control := range []bool{false, true} {
			hits.Store(0)
			cmd := exec.Command(os.Args[0], "-test.run=^TestChild$", "-child.provider="+provider, "-child.control="+boolText(control))
			cmd.Env = []string{
				"HTTP_PROXY=" + proxy.URL, "HTTPS_PROXY=" + proxy.URL, "ALL_PROXY=" + proxy.URL,
				"http_proxy=" + proxy.URL, "https_proxy=" + proxy.URL, "all_proxy=" + proxy.URL,
			}
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%s control=%v: child failed: %v\n%s", provider, control, err, out)
			}
			if control && hits.Load() == 0 {
				t.Fatalf("%s: the control run did not reach the proxy, so this test proves nothing", provider)
			}
			if !control && hits.Load() != 0 {
				t.Fatalf("%s: the model client used the proxy %d times", provider, hits.Load())
			}
		}
	}
}

func boolText(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func TestNewHasNoProxy(t *testing.T) {
	tr, ok := httpclient.New().Transport.(*http.Transport)
	if !ok || tr.Proxy != nil {
		t.Fatalf("transport %#v", httpclient.New().Transport)
	}
}
