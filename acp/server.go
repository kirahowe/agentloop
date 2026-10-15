package acpagent

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"log/slog"
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

	runs    sync.WaitGroup // runs in flight
	answers sync.WaitGroup // prompt responses not yet written
}

// Config is what the server needs from the request.
type Config struct {
	// Version is agentloop's, for initialize.
	Version string
	// SessionDir holds a <id>.jsonl log for each session.
	SessionDir string
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
	return &Server{cfg: cfg, sessions: map[acp.SessionId]*session{}}
}

// Connect serves the client that writes to in and reads from out.
func (s *Server) Connect(out io.Writer, in io.Reader) (*acp.AgentSideConnection, error) {
	conn, err := acp.NewAgentSideConnectionWithOptions(s, out, in, acp.ConnectionOptions{
		Logger: slog.New(slog.NewTextHandler(s.cfg.Log, nil)),
	})
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.conn = conn
	s.mu.Unlock()
	return conn, nil
}

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

// NewSession opens a new session log under its lock.
func (s *Server) NewSession(ctx context.Context, req acp.NewSessionRequest) (acp.NewSessionResponse, error) {
	for _, line := range NewSessionNotes(req) {
		s.logf("%s", line)
	}
	conn, _ := acp.AgentSideConnectionFromContext(ctx)
	var b [16]byte
	rand.Read(b[:])
	id := NewSessionID(b)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lost {
		return acp.NewSessionResponse{}, errClosing()
	}
	sess, unlock, err := sessionfile.Open(SessionPath(s.cfg.SessionDir, id))
	if err != nil {
		return acp.NewSessionResponse{}, internal(err.Error(), nil)
	}
	opts := s.cfg.Options
	opts.Notices = s.notices
	s.sessions[id] = &session{
		id: id, conn: conn, sess: sess, unlock: unlock,
		tr: NewTranslator(opts), waiting: map[PromptID]chan answer{},
	}
	return acp.NewSessionResponse{SessionId: id}, nil
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
// be released, then waits until each run's response is written, the
// connection fails, or ctx ends.
func (s *Server) Shutdown(ctx context.Context) {
	s.Lose()
	s.runs.Wait()
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
