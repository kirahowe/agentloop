// Package httpclient is the HTTP client agentloop's model adapters use
// when the caller gives none.
package httpclient

import "net/http"

// New is a client that never takes a proxy from the environment
// (HTTP_PROXY, HTTPS_PROXY and the rest). The model API is reached
// directly or not at all: a proxy variable set in the process's
// environment by anything else is not a decision about where an agent's
// prompts and API key go.
func New() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.Proxy = nil
	return &http.Client{Transport: t}
}
