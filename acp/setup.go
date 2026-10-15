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
// speak it closes the connection. Only what the server implements is
// advertised: text, resource links and embedded text in prompts, and no
// session loading, listing, resuming or closing yet.
func InitializeResponse(version string) acp.InitializeResponse {
	return acp.InitializeResponse{
		ProtocolVersion: acp.ProtocolVersion(acp.WireProtocolVersion),
		AgentInfo:       &acp.Implementation{Name: "agentloop", Version: version},
		AuthMethods:     []acp.AuthMethod{},
		AgentCapabilities: acp.AgentCapabilities{
			PromptCapabilities: acp.PromptCapabilities{EmbeddedContext: true},
		},
	}
}

// ClientNotices says whether the client can show session notices, which
// carry retry status.
func ClientNotices(req acp.InitializeRequest) bool {
	s := req.ClientCapabilities.Session
	return s != nil && s.Notices != nil
}

// NewSessionNotes are the lines logged for a session/new: what the client
// asked for that agentloop does not do. The cwd is not one of them, since
// every client sends one. Tools come only from the request, so the client's
// MCP servers are not connected.
func NewSessionNotes(req acp.NewSessionRequest) []string {
	if n := len(req.McpServers); n > 0 {
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
