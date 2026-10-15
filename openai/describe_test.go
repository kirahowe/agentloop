package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kirahowe/agentloop"
)

// OpenAI's Models API gives the id and no display name or context window,
// so Describe reports both as unknown rather than guessing.
func TestDescribe(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Method+" "+r.URL.Path+" "+r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"gpt-6","object":"model","created":1790000000,"owned_by":"openai"}`)
	}))
	defer srv.Close()
	info, err := New(Config{APIKey: "sk-test", BaseURL: srv.URL + "/v1"}).Describe(context.Background(), "gpt-6")
	if err != nil {
		t.Fatal(err)
	}
	if info != (agentloop.ModelInfo{ID: "gpt-6"}) {
		t.Fatalf("%+v", info)
	}
	if len(seen) != 1 || seen[0] != "GET /v1/models/gpt-6 Bearer sk-test" {
		t.Fatalf("requests %q", seen)
	}
}

func TestDescribeNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		fmt.Fprint(w, `{"error":{"message":"The model 'gpt-nope' does not exist","type":"invalid_request_error","code":"model_not_found"}}`)
	}))
	defer srv.Close()
	_, err := New(Config{APIKey: "sk-test", BaseURL: srv.URL}).Describe(context.Background(), "gpt-nope")
	if me := agentloop.AsModelError(err); me.Kind != agentloop.NotFound || me.Status != 404 {
		t.Fatalf("%+v", me)
	}
}
