package acpagent

import (
	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
)

// The turn machine is one session's lifecycle as seen from ACP:
//
//	Idle      --Prompt-->          Running   [StartRun]
//	Running   --RunEnded-->        Idle      [Respond or RespondError, from Respond(outcome)]
//	Running   --Cancel-->          Cancelling [CancelRun]
//	Cancelling --RunEnded-->       Idle      [Respond(cancelled)]
//	Running|Cancelling --Close or ConnectionLost--> Closing [CancelRun]
//	Closing   --RunEnded-->        Closed    [Respond(cancelled), ReleaseSession]
//	Idle      --Close or ConnectionLost--> Closed [ReleaseSession]
//
// A Prompt anywhere but Idle is answered with an error at once and
// changes nothing else; queueing prompts is the client's job. Holds are
// not asked about yet: a run that ends held is answered like any other
// outcome (see Respond), and the session goes back to Idle.
//
// Step does no I/O. The shell carries out the effects, in order, and
// feeds back RunEnded when the run it started returns.

// TurnState is where a session's turn is.
type TurnState int

const (
	Idle TurnState = iota
	Running
	Cancelling
	Closing
	Closed
)

func (s TurnState) String() string {
	switch s {
	case Idle:
		return "idle"
	case Running:
		return "running"
	case Cancelling:
		return "cancelling"
	case Closing:
		return "closing"
	case Closed:
		return "closed"
	}
	return "unknown"
}

// PromptID names one session/prompt request, so its answer finds it. The
// shell assigns them.
type PromptID uint64

// A Turn is the state of one session's turn. Its zero value is Idle.
type Turn struct {
	State TurnState
	// Prompt is the request being answered while a run is in flight.
	Prompt PromptID
}

// Inputs.
type (
	// Input is something that happened to the session.
	Input interface{ input() }

	// Prompt is a session/prompt request.
	Prompt struct {
		ID      PromptID
		Content []acp.ContentBlock
	}
	// RunEnded is the run the machine started returning.
	RunEnded struct{ Outcome *agentloop.Outcome }
	// Cancel is session/cancel.
	Cancel struct{}
	// Close is session/close.
	Close struct{}
	// ConnectionLost is the client going away: stdin closed, stdout
	// broken, or a signal.
	ConnectionLost struct{}
)

func (Prompt) input()         {}
func (RunEnded) input()       {}
func (Cancel) input()         {}
func (Close) input()          {}
func (ConnectionLost) input() {}

// Effects.
type (
	// Effect is something the shell must do.
	Effect interface{ effect() }

	// StartRun runs the agent with the prompt text.
	StartRun struct{ Prompt string }
	// CancelRun cancels the run in flight.
	CancelRun struct{}
	// RespondStop answers a prompt with a stop reason.
	RespondStop struct {
		ID   PromptID
		Stop acp.StopReason
	}
	// RespondError answers a prompt with a JSON-RPC error.
	RespondError struct {
		ID  PromptID
		Err *acp.RequestError
	}
	// ReleaseSession lets go of the session: its lock, and its place in
	// the registry.
	ReleaseSession struct{}
)

func (StartRun) effect()       {}
func (CancelRun) effect()      {}
func (RespondStop) effect()    {}
func (RespondError) effect()   {}
func (ReleaseSession) effect() {}

// Step takes one input and returns the next state and what to do.
func Step(t Turn, in Input) (Turn, []Effect) {
	switch in := in.(type) {
	case Prompt:
		return t.prompt(in)
	case RunEnded:
		return t.runEnded(in.Outcome)
	case Cancel:
		if t.State == Running {
			return Turn{State: Cancelling, Prompt: t.Prompt}, []Effect{CancelRun{}}
		}
	case Close, ConnectionLost:
		switch t.State {
		case Idle:
			return Turn{State: Closed}, []Effect{ReleaseSession{}}
		case Running:
			return Turn{State: Closing, Prompt: t.Prompt}, []Effect{CancelRun{}}
		case Cancelling:
			// The run is already being cancelled.
			return Turn{State: Closing, Prompt: t.Prompt}, nil
		}
	}
	return t, nil
}

func (t Turn) prompt(p Prompt) (Turn, []Effect) {
	switch t.State {
	case Idle:
	case Closing, Closed:
		return t, []Effect{RespondError{ID: p.ID, Err: internal("the session is closed", nil)}}
	default:
		return t, []Effect{RespondError{ID: p.ID, Err: internal("a turn is already running", nil)}}
	}
	text, err := PromptText(p.Content)
	if err != nil {
		rpc, ok := err.(*acp.RequestError)
		if !ok {
			rpc = acp.NewInvalidParams(map[string]any{"error": err.Error()})
		}
		return t, []Effect{RespondError{ID: p.ID, Err: rpc}}
	}
	return Turn{State: Running, Prompt: p.ID}, []Effect{StartRun{Prompt: text}}
}

func (t Turn) runEnded(out *agentloop.Outcome) (Turn, []Effect) {
	switch t.State {
	case Running:
		stop, err := Respond(out)
		if err != nil {
			return Turn{}, []Effect{RespondError{ID: t.Prompt, Err: err}}
		}
		return Turn{}, []Effect{RespondStop{ID: t.Prompt, Stop: stop}}
	case Cancelling:
		// ACP requires cancelled after session/cancel, whatever the run
		// managed to finish before it noticed.
		return Turn{}, []Effect{RespondStop{ID: t.Prompt, Stop: acp.StopReasonCancelled}}
	case Closing:
		return Turn{State: Closed}, []Effect{
			RespondStop{ID: t.Prompt, Stop: acp.StopReasonCancelled},
			ReleaseSession{},
		}
	}
	// No run is in flight: the shell never sends this, and it changes
	// nothing.
	return t, nil
}
