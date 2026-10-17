package acpagent

import (
	"fmt"
	"path/filepath"

	acp "github.com/caelis-labs/acp-go-sdk"
)

// The connection's setup, as values: what initialize answers, what is
// remembered about the client, and how a new session is named and kept.

// InitializeResponse is the answer to initialize. agentloop speaks
// protocol version 1 whatever the client asks for; a client that cannot
// speak it closes the connection. A non-empty title is the agent's
// display name in agentInfo. Only what the server implements is
// advertised: text, resource links and embedded text in prompts, and
// loading, resuming, listing and closing sessions. With newOnly the
// server refuses to reopen a session, so loading, resuming and listing
// are left out and only closing is advertised.
func InitializeResponse(version, title string, newOnly bool) acp.InitializeResponse {
	info := &acp.Implementation{Name: "agentloop", Version: version}
	if title != "" {
		info.Title = &title
	}
	r := acp.InitializeResponse{
		ProtocolVersion: acp.ProtocolVersion(acp.WireProtocolVersion),
		AgentInfo:       info,
		AuthMethods:     []acp.AuthMethod{},
		AgentCapabilities: acp.AgentCapabilities{
			LoadSession:        true,
			PromptCapabilities: acp.PromptCapabilities{EmbeddedContext: true},
			SessionCapabilities: acp.SessionCapabilities{
				List:   &acp.SessionListCapabilities{},
				Resume: &acp.SessionResumeCapabilities{},
				Close:  &acp.SessionCloseCapabilities{},
			},
		},
	}
	if newOnly {
		c := &r.AgentCapabilities
		c.LoadSession = false
		c.SessionCapabilities.List = nil
		c.SessionCapabilities.Resume = nil
	}
	return r
}

// errNewOnly answers session/load, session/resume and session/list when
// the server opens only new sessions. They are not advertised then, so
// the method is reported as not found, as for any other.
func errNewOnly() *acp.RequestError {
	return &acp.RequestError{Code: -32601, Message: "Method not found"}
}

// ClientNotices says whether the client can show session notices, which
// carry retry status.
func ClientNotices(req acp.InitializeRequest) bool {
	s := req.ClientCapabilities.Session
	return s != nil && s.Notices != nil
}

// MCPNotes are the lines logged when a session/new, session/load or
// session/resume names n MCP servers: tools come only from the request,
// so the client's are not connected. The cwd gets no line, since every
// client sends one.
func MCPNotes(n int) []string {
	if n > 0 {
		return []string{fmt.Sprintf("agentloop: ignoring %d MCP server(s) from the client: tools come only from the request", n)}
	}
	return nil
}

// NewSessionID makes a session id, a version 4 UUID, from 16 random bytes.
func NewSessionID(b [16]byte) acp.SessionId {
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return acp.SessionId(fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]))
}

// SessionPath is the log file of a session.
func SessionPath(dir string, id acp.SessionId) string {
	return filepath.Join(dir, string(id)+".jsonl")
}

// UnknownSession answers a request for a session this connection does not
// have open.
func UnknownSession(id acp.SessionId) *acp.RequestError {
	return &acp.RequestError{Code: -32002, Message: "Resource not found", Data: map[string]any{"sessionId": string(id)}}
}

// errClosing answers a session/new after the client has gone away or the
// process was told to stop.
func errClosing() *acp.RequestError {
	return internal("agentloop is shutting down", nil)
}
