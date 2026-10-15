package acpagent

import (
	"fmt"
	"reflect"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// machine drives Step the way the shell does, keeping what it needs to
// check the properties as it goes.
type machine struct {
	t    *rapid.T
	turn Turn

	nextID      PromptID
	outstanding bool     // a StartRun has not been answered by RunEnded
	current     PromptID // the prompt whose run is outstanding
	answers     map[PromptID]int
	cancelled   map[PromptID]bool // a Cancel came while its run was outstanding
	ended       bool              // Close or ConnectionLost was sent
	released    int
}

func (m *machine) step(in Input) []Effect {
	t := m.t
	before := m.turn
	wasReleased := m.released > 0
	next, effs := Step(m.turn, in)
	m.turn = next
	if wasReleased {
		// Once released, a session only refuses prompts.
		for _, e := range effs {
			p, isPrompt := in.(Prompt)
			if re, ok := e.(RespondError); !ok || !isPrompt || re.ID != p.ID {
				t.Fatalf("effect %#v after the session was released", e)
			}
		}
	}

	if p, ok := in.(Prompt); ok && before.State != Idle {
		// P2.6: a prompt while not idle is refused and changes nothing.
		want := []Effect{RespondError{ID: p.ID}}
		if len(effs) != 1 || reflect.TypeOf(effs[0]) != reflect.TypeOf(want[0]) || effs[0].(RespondError).ID != p.ID {
			t.Fatalf("a prompt in %s gave %#v", before.State, effs)
		}
		if next != before {
			t.Fatalf("a prompt in %s changed the turn to %+v", before.State, next)
		}
	}
	if _, ok := in.(RunEnded); ok {
		m.outstanding = false
	}
	switch in.(type) {
	case Close, ConnectionLost:
		m.ended = true
	case Cancel:
		if m.outstanding && m.answers[m.current] == 0 {
			m.cancelled[m.current] = true
		}
	}

	for _, e := range effs {
		switch e := e.(type) {
		case StartRun:
			// P2.2: at most one run at a time.
			if m.outstanding {
				t.Fatalf("StartRun with a run outstanding")
			}
			if m.ended {
				t.Fatalf("StartRun after the session ended")
			}
			p, ok := in.(Prompt)
			if !ok {
				t.Fatalf("StartRun from %T", in)
			}
			m.outstanding, m.current = true, p.ID
		case CancelRun:
			if !m.outstanding {
				t.Fatalf("CancelRun with no run")
			}
		case RespondStop:
			m.answer(e.ID, in)
			// P2.3: a cancel before the answer always gives cancelled.
			if m.cancelled[e.ID] && e.Stop != acp.StopReasonCancelled {
				t.Fatalf("prompt %d was cancelled but stopped %q", e.ID, e.Stop)
			}
			if m.ended && e.Stop != acp.StopReasonCancelled {
				t.Fatalf("after close, prompt %d stopped %q", e.ID, e.Stop)
			}
		case RespondError:
			m.answer(e.ID, in)
			if m.cancelled[e.ID] {
				t.Fatalf("prompt %d was cancelled but got the error %v", e.ID, e.Err)
			}
			if e.Err == nil {
				t.Fatalf("RespondError with no error")
			}
			// P2.5: after close, an error answers only a prompt sent now.
			if p, ok := in.(Prompt); m.ended && (!ok || p.ID != e.ID) {
				t.Fatalf("after close, an error for prompt %d on %T", e.ID, in)
			}
		case ReleaseSession:
			m.released++
			if !m.ended || m.released > 1 || m.outstanding {
				t.Fatalf("ReleaseSession: ended=%v released=%d outstanding=%v", m.ended, m.released, m.outstanding)
			}
		default:
			t.Fatalf("unknown effect %#v", e)
		}
	}
	return effs
}

func (m *machine) answer(id PromptID, in Input) {
	m.answers[id]++
	if m.answers[id] > 1 {
		m.t.Fatalf("prompt %d answered %d times", id, m.answers[id])
	}
	if p, ok := in.(Prompt); ok && p.ID == id {
		return
	}
	if _, ok := in.(RunEnded); !ok || id != m.current {
		m.t.Fatalf("prompt %d answered on %T (current %d)", id, in, m.current)
	}
}

func genPrompt(t *rapid.T, id PromptID) Prompt {
	var content []acp.ContentBlock
	switch rapid.IntRange(0, 5).Draw(t, "prompt") {
	case 0:
		content = []acp.ContentBlock{acp.ImageBlock("aGk=", "image/png")}
	case 1:
		content = []acp.ContentBlock{acp.TextBlock(" ")}
	default:
		content = []acp.ContentBlock{acp.TextBlock(fmt.Sprintf("prompt %d", id))}
	}
	return Prompt{ID: id, Content: content}
}

// TestPropertyTurn holds P2.1, P2.2, P2.3, P2.5 and P2.6 over random
// input sequences. RunEnded comes only for a run the machine started, as
// the shell guarantees, with outcomes of every reason; a stray one, when
// no run is outstanding, must change nothing.
func TestPropertyTurn(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		m := &machine{t: t, answers: map[PromptID]int{}, cancelled: map[PromptID]bool{}}
		var prompts []PromptID
		n := rapid.IntRange(0, 30).Draw(t, "n")
		for i := 0; i < n; i++ {
			switch rapid.IntRange(0, 9).Draw(t, "input") {
			case 0, 1, 2:
				m.nextID++
				prompts = append(prompts, m.nextID)
				m.step(genPrompt(t, m.nextID))
			case 3, 4, 5:
				if m.outstanding {
					m.step(RunEnded{Outcome: genOutcome(t)})
				} else {
					before := m.turn
					if effs := m.step(RunEnded{Outcome: genOutcome(t)}); len(effs) != 0 || m.turn != before {
						t.Fatalf("a stray RunEnded gave %#v", effs)
					}
				}
			case 6, 7:
				m.step(Cancel{})
			case 8:
				m.step(Close{})
			case 9:
				m.step(ConnectionLost{})
			}
		}
		// The run in flight returns, as it always does in the end.
		if m.outstanding {
			m.step(RunEnded{Outcome: genOutcome(t)})
		}
		// P2.1: every prompt was answered exactly once.
		for _, id := range prompts {
			if m.answers[id] != 1 {
				t.Fatalf("prompt %d answered %d times", id, m.answers[id])
			}
		}
		if m.ended && m.released != 1 {
			t.Fatalf("the session ended but was released %d times", m.released)
		}
		if !m.ended && m.released != 0 {
			t.Fatal("released without being closed")
		}
	})
}

// A run that ends held is answered with an error and the session is idle
// again, so the next prompt starts a run.
func TestTurnHeldEndsTheTurn(t *testing.T) {
	turn, effs := Step(Turn{}, Prompt{ID: 1, Content: []acp.ContentBlock{acp.TextBlock("go")}})
	if !reflect.DeepEqual(effs, []Effect{StartRun{Prompt: "go"}}) {
		t.Fatalf("%#v", effs)
	}
	turn, effs = Step(turn, RunEnded{Outcome: &agentloop.Outcome{Reason: agentloop.ReasonHeld,
		Held: []agentloop.ToolCall{{ID: "c1", Name: "rm"}}}})
	if len(effs) != 1 {
		t.Fatalf("%#v", effs)
	}
	re, ok := effs[0].(RespondError)
	if !ok || re.ID != 1 || re.Err.Code != -32603 {
		t.Fatalf("%#v", effs)
	}
	if turn.State != Idle {
		t.Fatalf("state %s", turn.State)
	}
	if _, effs = Step(turn, Prompt{ID: 2, Content: []acp.ContentBlock{acp.TextBlock("again")}}); !reflect.DeepEqual(effs, []Effect{StartRun{Prompt: "again"}}) {
		t.Fatalf("%#v", effs)
	}
}

func TestTurnCloseWhileRunning(t *testing.T) {
	turn, _ := Step(Turn{}, Prompt{ID: 7, Content: []acp.ContentBlock{acp.TextBlock("go")}})
	turn, effs := Step(turn, Close{})
	if !reflect.DeepEqual(effs, []Effect{CancelRun{}}) || turn.State != Closing {
		t.Fatalf("%s %#v", turn.State, effs)
	}
	turn, effs = Step(turn, RunEnded{Outcome: &agentloop.Outcome{Reason: agentloop.ReasonCompleted}})
	want := []Effect{RespondStop{ID: 7, Stop: acp.StopReasonCancelled}, ReleaseSession{}}
	if !reflect.DeepEqual(effs, want) || turn.State != Closed {
		t.Fatalf("%s %#v", turn.State, effs)
	}
}
