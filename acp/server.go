package acpagent

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/sessionfile"
)

// Server is the imperative shell: it serves one ACP connection, keeps the
// sessions it opened, and carries out what the turn machine (Step) says.
// Every decision is made by the pure parts of this package; the server
// only moves values between them and the connection.
//
// A run's events are translated and sent on the run's own goroutine, each
// update written before the next event, and all of them before the
// session/prompt response, which is sent only once the run has returned.
type Server struct {
	cfg Config

	mu       sync.Mutex
	notices  bool
	lost     bool
	sessions map[acp.SessionId]*session
	nextID   PromptID
	conn     *acp.AgentSideConnection
	in       *heldInput

	runs    sync.WaitGroup // runs in flight
	answers sync.WaitGroup // prompt responses not yet written
	loads   sync.WaitGroup // loads and resumes holding a session not yet registered
}

// Config is what the server needs from the request.
type Config struct {
	// Version is agentloop's, for initialize.
	Version string
	// SessionDir holds a <id>.jsonl log for each session.
	SessionDir string
	// Cwd is the absolute directory the tools run in, which session/list
	// reports for every session (see SessionCwd).
	Cwd string
	// Agent is copied for each run. If it has an Observe, that is called
	// after each event's updates have been sent.
	Agent *agentloop.Agent
	// Options are for the translation. Notices is set from the client's
	// initialize, whatever it says here.
	Options Options
	// Log takes the server's diagnostics. Nothing is ever written to the
	// connection but JSON-RPC.
	Log io.Writer
}

// session is one open session: the log, its lock, where its turn is, and
// what a run in flight needs.
type session struct {
	id      acp.SessionId
	conn    *acp.AgentSideConnection
	sess    *agentloop.Session
	unlock  func()
	turn    Turn
	tr      Translator // used only by the run in flight
	cancel  context.CancelFunc
	waiting map[PromptID]chan answer
	// released is closed when the session is let go of.
	released chan struct{}
}

type answer struct {
	stop acp.StopReason
	err  *acp.RequestError
}

// NewServer makes a server for one connection.
func NewServer(cfg Config) *Server {
	if cfg.Log == nil {
		cfg.Log = io.Discard
	}
	if cfg.Cwd == "" {
		wd, _ := os.Getwd()
		cfg.Cwd = SessionCwd("", wd)
	}
	return &Server{cfg: cfg, sessions: map[acp.SessionId]*session{}}
}

// Connect serves the client that writes to in and reads from out.
//
// The end of in is held back from the connection until Shutdown has
// written the answers it owes. The SDK takes the end of its input as the
// end of the connection and stops writing too, so a client that only
// closed its writing half would otherwise never get the cancelled
// responses of the turns in flight. Ended says when in has ended.
func (s *Server) Connect(out io.Writer, in io.Reader) (*acp.AgentSideConnection, error) {
	held := &heldInput{r: in, ended: make(chan struct{}), release: make(chan struct{})}
	conn, err := acp.NewAgentSideConnectionWithOptions(s, out, held, acp.ConnectionOptions{
		Logger: slog.New(slog.NewTextHandler(s.cfg.Log, nil)),
	})
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.conn, s.in = conn, held
	s.mu.Unlock()
	return conn, nil
}

// Ended is closed when the client's input has ended (or failed). It is
// nil before Connect.
func (s *Server) Ended() <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.in == nil {
		return nil
	}
	return s.in.ended
}

// heldInput reads through to the client's input, but when that ends it
// says so on ended and keeps the end from its reader until released.
type heldInput struct {
	r       io.Reader
	ended   chan struct{}
	release chan struct{}
	once    sync.Once
	free    sync.Once
}

func (h *heldInput) Read(p []byte) (int, error) {
	n, err := h.r.Read(p)
	if err == nil {
		return n, nil
	}
	h.once.Do(func() { close(h.ended) })
	if n > 0 {
		// The data first; the next read meets the end again.
		return n, nil
	}
	<-h.release
	return 0, err
}

func (h *heldInput) let() { h.free.Do(func() { close(h.release) }) }

func (s *Server) logf(format string, args ...any) {
	fmt.Fprintf(s.cfg.Log, format+"\n", args...)
}

// Initialize answers initialize and remembers whether the client shows
// notices.
func (s *Server) Initialize(_ context.Context, req acp.InitializeRequest) (acp.InitializeResponse, error) {
	s.mu.Lock()
	s.notices = ClientNotices(req)
	s.mu.Unlock()
	return InitializeResponse(s.cfg.Version), nil
}

// NewSession opens a new session log under its lock. The file is made at
// once, so the session can be listed and loaded before anything is said
// in it.
func (s *Server) NewSession(ctx context.Context, req acp.NewSessionRequest) (acp.NewSessionResponse, error) {
	s.note(len(req.McpServers))
	conn, _ := acp.AgentSideConnectionFromContext(ctx)
	var b [16]byte
	rand.Read(b[:])
	id := NewSessionID(b)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lost {
		return acp.NewSessionResponse{}, errClosing()
	}
	path := SessionPath(s.cfg.SessionDir, id)
	sess, unlock, err := sessionfile.Open(path)
	if err != nil {
		return acp.NewSessionResponse{}, internal(err.Error(), nil)
	}
	if err := sessionfile.Create(path); err != nil {
		unlock()
		return acp.NewSessionResponse{}, internal(err.Error(), nil)
	}
	s.register(s.newSession(id, conn, sess, unlock))
	return acp.NewSessionResponse{SessionId: id}, nil
}

func (s *Server) note(mcpServers int) {
	for _, line := range MCPNotes(mcpServers) {
		s.logf("%s", line)
	}
}

// newSession is the registry's record of an open session. s.mu is held.
func (s *Server) newSession(id acp.SessionId, conn *acp.AgentSideConnection, sess *agentloop.Session, unlock func()) *session {
	opts := s.cfg.Options
	opts.Notices = s.notices
	return &session{
		id: id, conn: conn, sess: sess, unlock: unlock,
		tr: NewTranslator(opts), waiting: map[PromptID]chan answer{},
		released: make(chan struct{}),
	}
}

func (s *Server) register(ss *session) { s.sessions[ss.id] = ss }

// LoadSession opens a session from its log and sends its history, all of
// it before the response.
func (s *Server) LoadSession(ctx context.Context, req acp.LoadSessionRequest) (acp.LoadSessionResponse, error) {
	s.note(len(req.McpServers))
	if err := s.reopen(ctx, req.SessionId, true); err != nil {
		return acp.LoadSessionResponse{}, err
	}
	return acp.LoadSessionResponse{}, nil
}

// ResumeSession opens a session from its log without sending its history.
func (s *Server) ResumeSession(ctx context.Context, req acp.ResumeSessionRequest) (acp.ResumeSessionResponse, error) {
	s.note(len(req.McpServers))
	if err := s.reopen(ctx, req.SessionId, false); err != nil {
		return acp.ResumeSessionResponse{}, err
	}
	return acp.ResumeSessionResponse{}, nil
}

// reopen makes a stored session open on this connection, and with replay
// sends its history first. A session this connection already has open is
// taken up as it is, if Reattach allows; any other is opened under its
// lock, which fails when another process, or another connection in this
// one, has it open.
//
// While the history is sent the session is out of the registry, so no
// turn can start and mix its updates into the replay: a prompt for it
// then is for a session not yet open.
func (s *Server) reopen(ctx context.Context, id acp.SessionId, replay bool) error {
	path, rpcErr := SessionFile(s.cfg.SessionDir, id)
	if rpcErr != nil {
		return rpcErr
	}
	conn, _ := acp.AgentSideConnectionFromContext(ctx)

	s.mu.Lock()
	if s.lost {
		s.mu.Unlock()
		return errClosing()
	}
	ss, open := s.sessions[id]
	if open {
		if err := Reattach(id, ss.turn); err != nil {
			s.mu.Unlock()
			return err
		}
		if !replay {
			s.mu.Unlock()
			return nil
		}
		delete(s.sessions, id)
	}
	s.loads.Add(1)
	s.mu.Unlock()
	defer s.loads.Done()

	if !open {
		if !sessionfile.Exists(path) {
			return UnknownSession(id)
		}
		sess, unlock, err := sessionfile.Open(path)
		if errors.Is(err, sessionfile.ErrInUse) {
			return SessionInUse(id)
		}
		if err != nil {
			return internal(err.Error(), nil)
		}
		s.mu.Lock()
		ss = s.newSession(id, conn, sess, unlock)
		s.mu.Unlock()
	}
	if replay {
		for _, u := range Replay(ss.tr.opts, ss.sess.Entries()) {
			if err := conn.SessionUpdate(ctx, acp.SessionNotification{SessionId: id, Update: u}); err != nil {
				ss.unlock()
				return err
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lost {
		ss.unlock()
		return errClosing()
	}
	s.register(ss)
	return nil
}

// CloseSession cancels the session's turn, if one is running, and answers
// once the turn has ended and the session is let go of.
func (s *Server) CloseSession(ctx context.Context, req acp.CloseSessionRequest) (acp.CloseSessionResponse, error) {
	s.mu.Lock()
	ss, ok := s.sessions[req.SessionId]
	if !ok {
		s.mu.Unlock()
		return acp.CloseSessionResponse{}, UnknownSession(req.SessionId)
	}
	s.step(ss, Close{})
	s.mu.Unlock()
	select {
	case <-ss.released:
	case <-ctx.Done():
		return acp.CloseSessionResponse{}, ctx.Err()
	}
	return acp.CloseSessionResponse{}, nil
}

// ListSessions lists the session logs in session_dir with the time of
// each one's last entry. A log that cannot be read is listed without one.
func (s *Server) ListSessions(_ context.Context, req acp.ListSessionsRequest) (acp.ListSessionsResponse, error) {
	dirents, err := os.ReadDir(s.cfg.SessionDir)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return acp.ListSessionsResponse{}, internal(err.Error(), nil)
	}
	var found []StoredSession
	for _, d := range dirents {
		id, ok := SessionFileID(d.Name())
		if !ok || !d.Type().IsRegular() {
			continue
		}
		st := StoredSession{ID: id}
		entries, err := agentloop.NewFileStore(filepath.Join(s.cfg.SessionDir, d.Name())).Load()
		if err != nil {
			s.logf("agentloop: session %s: %v", id, err)
		} else if n := len(entries); n > 0 {
			st.Updated = entries[n-1].Time
		}
		found = append(found, st)
	}
	resp, rpcErr := ListResponse(found, s.cfg.Cwd, req)
	if rpcErr != nil {
		return acp.ListSessionsResponse{}, rpcErr
	}
	return resp, nil
}

// Prompt runs a turn and answers when the turn machine says so.
//
// The run does not use the request's context. The SDK cancels it on
// session/cancel, and also when a second prompt arrives for the session,
// which the turn machine refuses without touching the first. Cancelling
// is the machine's decision, so it goes through Step like everything else.
func (s *Server) Prompt(ctx context.Context, req acp.PromptRequest) (acp.PromptResponse, error) {
	s.mu.Lock()
	ss, ok := s.sessions[req.SessionId]
	if !ok {
		s.mu.Unlock()
		return acp.PromptResponse{}, UnknownSession(req.SessionId)
	}
	s.nextID++
	id := s.nextID
	ch := make(chan answer, 1)
	ss.waiting[id] = ch
	// A shutdown waits for the answer to be written. Once it has begun,
	// a prompt is only ever refused, and is not waited for.
	counted := !s.lost
	if counted {
		s.answers.Add(1)
	}
	s.step(ss, Prompt{ID: id, Content: req.Prompt})
	s.mu.Unlock()

	a := <-ch
	written := func(context.Context) error {
		if counted {
			s.answers.Done()
		}
		return nil
	}
	if a.err != nil {
		written(ctx)
		return acp.PromptResponse{}, a.err
	}
	if acp.AfterResponse(ctx, written) != nil {
		written(ctx)
	}
	return acp.PromptResponse{StopReason: a.stop}, nil
}

// Cancel is session/cancel.
func (s *Server) Cancel(_ context.Context, n acp.CancelNotification) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ss, ok := s.sessions[n.SessionId]; ok {
		s.step(ss, Cancel{})
	}
	return nil
}

// Lose tells every session that the client has gone: stdin closed, stdout
// failed, or a signal. Runs are cancelled, sessions are released as their
// runs end, and no new session is opened.
func (s *Server) Lose() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lost = true
	for _, ss := range s.sessions {
		s.step(ss, ConnectionLost{})
	}
}

// Shutdown loses the client, waits for every run to end and its session to
// be released, and for every load in progress to let go of its session,
// then waits until each run's response is written, the connection fails,
// or ctx ends. Last, it lets the connection see the end of its input.
func (s *Server) Shutdown(ctx context.Context) {
	s.Lose()
	s.runs.Wait()
	s.loads.Wait()
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	written := make(chan struct{})
	go func() { s.answers.Wait(); close(written) }()
	var done <-chan struct{}
	if conn != nil {
		done = conn.Done()
	}
	select {
	case <-written:
	case <-done:
	case <-ctx.Done():
	}
	s.mu.Lock()
	in := s.in
	s.mu.Unlock()
	if in != nil {
		in.let()
	}
}

// State is where a session's turn is, and whether the session is open.
func (s *Server) State(id acp.SessionId) (TurnState, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ss, ok := s.sessions[id]
	if !ok {
		return Closed, false
	}
	return ss.turn.State, true
}

// step feeds one input to the session's turn machine and carries out the
// effects, in order. s.mu is held; nothing here blocks.
func (s *Server) step(ss *session, in Input) {
	next, effects := Step(ss.turn, in)
	ss.turn = next
	for _, e := range effects {
		switch e := e.(type) {
		case StartRun:
			s.start(ss, e.Prompt)
		case CancelRun:
			ss.cancel()
		case RespondStop:
			ss.answer(e.ID, answer{stop: e.Stop})
		case RespondError:
			ss.answer(e.ID, answer{err: e.Err})
		case ReleaseSession:
			ss.unlock()
			delete(s.sessions, ss.id)
			close(ss.released)
		}
	}
}

func (ss *session) answer(id PromptID, a answer) {
	if ch, ok := ss.waiting[id]; ok {
		delete(ss.waiting, id)
		ch <- a
	}
}

// start runs the agent on its own goroutine. Each event goes through the
// translator and its updates are sent before the run goes on. When the
// run returns, its end goes back into the machine.
func (s *Server) start(ss *session, prompt string) {
	ctx, cancel := context.WithCancel(context.Background())
	ss.cancel = cancel
	agent := *s.cfg.Agent
	observe := agent.Observe
	agent.Observe = func(ev agentloop.Event) {
		var ups []Update
		ss.tr, ups = ss.tr.Step(ev)
		for _, u := range ups {
			// The connection's own lifetime bounds the write, not the
			// request's context, which session/cancel ends: the updates
			// that report the cancelled calls must still go out.
			if ss.conn != nil {
				ss.conn.SessionUpdate(context.Background(), acp.SessionNotification{SessionId: ss.id, Update: u})
			}
		}
		if observe != nil {
			observe(ev)
		}
	}
	s.runs.Add(1)
	go func() {
		defer s.runs.Done()
		out, err := agent.Run(ctx, ss.sess, prompt)
		cancel()
		if err != nil {
			s.logf("agentloop: session %s: %v", ss.id, err)
		}
		s.mu.Lock()
		s.step(ss, RunEnded{Outcome: out, Err: err})
		s.mu.Unlock()
	}()
}
