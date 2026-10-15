package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/kirahowe/agentloop"
)

// modelsAPI serves GET {prefix}/v1/models/{id} with body, and records the
// method, path and key of every request.
type modelsAPI struct {
	*httptest.Server
	mu   sync.Mutex
	seen []string
}

func newModelsAPI(t *testing.T, status int, body string) *modelsAPI {
	t.Helper()
	a := &modelsAPI{}
	a.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		a.seen = append(a.seen, r.Method+" "+r.URL.Path+" key="+r.Header.Get("X-Api-Key")+" version="+r.Header.Get("Anthropic-Version"))
		a.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(a.Close)
	return a
}

func (a *modelsAPI) requests() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string{}, a.seen...)
}

// The Models API's answer, as documented: the context window is
// max_input_tokens, and max_tokens is the output cap, which is not it.
const opusInfo = `{
  "type": "model",
  "id": "claude-opus-5-5",
  "display_name": "Claude Opus 5.5",
  "created_at": "2026-08-01T00:00:00Z",
  "max_input_tokens": 1000000,
  "max_tokens": 128000,
  "capabilities": {}
}`

// Describe asks the Models API under the configured base URL, with the
// same key the Messages API gets, and reads the window from
// max_input_tokens.
func TestDescribe(t *testing.T) {
	for _, prefix := range []string{"", "/proxy"} {
		t.Run("prefix="+prefix, func(t *testing.T) {
			api := newModelsAPI(t, 200, opusInfo)
			m := New(Config{APIKey: "test-key", BaseURL: api.URL + prefix})
			info, err := m.Describe(context.Background(), "claude-opus-5-5")
			if err != nil {
				t.Fatal(err)
			}
			want := agentloop.ModelInfo{ID: "claude-opus-5-5", DisplayName: "Claude Opus 5.5", ContextWindow: 1000000, Prices: info.Prices}
			if info.Prices == nil || *info.Prices != *PricesFor("claude-opus-5-5") {
				t.Fatalf("prices %+v", info.Prices)
			}
			if info != want {
				t.Fatalf("got %+v, want %+v", info, want)
			}
			got := api.requests()
			if len(got) != 1 || got[0] != "GET "+prefix+"/v1/models/claude-opus-5-5 key=test-key version=2023-06-01" {
				t.Fatalf("requests %q", got)
			}
		})
	}
}

// A model the API gives no window for has an unknown window, not a
// guessed one.
func TestDescribeWithoutWindow(t *testing.T) {
	api := newModelsAPI(t, 200, `{"type":"model","id":"claude-x","display_name":"Claude X","created_at":"2026-01-01T00:00:00Z"}`)
	info, err := New(Config{APIKey: "k", BaseURL: api.URL}).Describe(context.Background(), "claude-x")
	if err != nil {
		t.Fatal(err)
	}
	if info != (agentloop.ModelInfo{ID: "claude-x", DisplayName: "Claude X"}) {
		t.Fatalf("%+v", info)
	}
}

// A failed description is a ModelError sorted like a failed message, from
// one request with no retries.
func TestDescribeErrors(t *testing.T) {
	for _, c := range []struct {
		status int
		typ    string
		want   agentloop.ErrorKind
	}{
		{404, "not_found_error", agentloop.NotFound},
		{401, "authentication_error", agentloop.Unauthorized},
		{529, "overloaded_error", agentloop.Overloaded},
	} {
		t.Run(c.typ, func(t *testing.T) {
			api := newModelsAPI(t, c.status, fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"no"}}`, c.typ))
			_, err := New(Config{APIKey: "k", BaseURL: api.URL}).Describe(context.Background(), "claude-x")
			me := agentloop.AsModelError(err)
			if me.Kind != c.want || me.Status != c.status || me.Message != "no" {
				t.Fatalf("%+v", me)
			}
			if n := len(api.requests()); n != 1 {
				t.Fatalf("%d requests", n)
			}
		})
	}
}
