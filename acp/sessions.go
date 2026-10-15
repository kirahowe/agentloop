package acpagent

import (
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
)

// Sessions kept on disk, as values: which ids can name a session file,
// when an open session can be loaded again, and what session/list answers.
// The shell reads the directory and the logs; these decide.

// sessionID is the shape of every id NewSessionID makes. Nothing else can
// name a session, so no id can reach a path outside session_dir.
var sessionID = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// SessionFile is the log of the session a client names in session/load
// or session/resume. An id agentloop could not have made names no
// session, and is answered as unknown before any path is built from it.
func SessionFile(dir string, id acp.SessionId) (string, *acp.RequestError) {
	if !sessionID.MatchString(string(id)) {
		return "", UnknownSession(id)
	}
	return SessionPath(dir, id), nil
}

// SessionFileID is the session a file in session_dir holds, if its name
// is that of a session log.
func SessionFileID(name string) (acp.SessionId, bool) {
	id, ok := strings.CutSuffix(name, ".jsonl")
	if !ok || !sessionID.MatchString(id) {
		return "", false
	}
	return acp.SessionId(id), true
}

// SessionInUse refuses a load or resume of a session that something else
// has open: another process, another connection in this one, or a turn
// on this connection.
func SessionInUse(id acp.SessionId) *acp.RequestError {
	return internal("session in use", map[string]any{"sessionId": string(id)})
}

// Reattach decides a load or resume of a session this connection already
// has open. The open session is the one on disk, so an idle one can be
// taken up again as it is; one with a turn in flight cannot, since its
// history is still being written.
func Reattach(id acp.SessionId, t Turn) *acp.RequestError {
	if t.State == Idle {
		return nil
	}
	return SessionInUse(id)
}

// SessionCwd is the working directory every session reports, which ACP
// requires to be absolute: the directory the tools run in. That is the
// request's tool_dir, taken from wd (agentloop's own working directory)
// when it is relative, and wd itself when tool_dir is empty, as for the
// tools. The client's cwd is never used, so it is never reported back.
func SessionCwd(toolDir, wd string) string {
	if toolDir == "" {
		return filepath.Clean(wd)
	}
	if filepath.IsAbs(toolDir) {
		return filepath.Clean(toolDir)
	}
	return filepath.Join(wd, toolDir)
}

// StoredSession is a session log as the shell found it.
type StoredSession struct {
	ID acp.SessionId
	// Updated is the time of the log's last entry; zero when it has none.
	Updated time.Time
}

// ListResponse answers session/list. Every session in session_dir runs
// its tools in cwd, so a cwd filter matches all of them or none. The
// newest come first, and all of them in one page: a cursor, which
// agentloop never hands out, is refused.
func ListResponse(found []StoredSession, cwd string, req acp.ListSessionsRequest) (acp.ListSessionsResponse, *acp.RequestError) {
	if req.Cursor != nil {
		return acp.ListSessionsResponse{}, acp.NewInvalidParams(map[string]any{"error": "unknown cursor"})
	}
	sessions := []acp.SessionInfo{}
	if req.Cwd != nil && filepath.Clean(*req.Cwd) != cwd {
		return acp.ListSessionsResponse{Sessions: sessions}, nil
	}
	found = append([]StoredSession(nil), found...)
	sort.SliceStable(found, func(i, j int) bool {
		a, b := found[i], found[j]
		if !a.Updated.Equal(b.Updated) {
			return a.Updated.After(b.Updated)
		}
		return a.ID < b.ID
	})
	for _, f := range found {
		info := acp.SessionInfo{SessionId: f.ID, Cwd: cwd}
		if !f.Updated.IsZero() {
			info.UpdatedAt = acp.Ptr(f.Updated.UTC().Format(time.RFC3339Nano))
		}
		sessions = append(sessions, info)
	}
	return acp.ListSessionsResponse{Sessions: sessions}, nil
}
