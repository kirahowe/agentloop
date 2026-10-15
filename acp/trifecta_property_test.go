package acpagent

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"sync"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/scenario"
)

// TestPropertyTrifectaOverACP drives a trifecta scenario (crashes,
// cancellations and holds left open) into a session file, then loads it
// on a server and carries on over ACP, with the client answering each
// session/request_permission at random: allow once, reject, an option
// it was not offered, or cancelled. The scenario's own checks hold for
// every call the server runs (no egress or act call without a person,
// the labels a tool is given, the labels of everything the model sees),
// and besides:
//   - a held call left in the loaded session is asked about, never run
//     on its own;
//   - a call a person was asked about runs only if the answer was allow
//     once;
//   - each request shows the call's whole input and the session's labels,
//     and offers only allow_once and reject_once.
func TestPropertyTrifectaOverACP(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		sc := scenario.New(t, scenario.GenTrifecta(t), scenario.Faults{
			Crash:  rapid.Bool().Draw(t, "crash"),
			Cancel: rapid.Bool().Draw(t, "cancel"),
		})
		sc.Faulty()
		dir, err := os.MkdirTemp(root, "s")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })
		id := drawID(t)
		path := SessionPath(dir, id)
		sc.HandOver(path)
		sc.PrepareDrain()
		before, err := agentloop.OpenSession(agentloop.NewFileStore(path))
		if err != nil {
			t.Fatal(err)
		}
		leftHeld := before.Held()

		// The client's answers are drawn now: rapid draws only on the
		// test's own goroutine.
		choices := rapid.SliceOfN(rapid.IntRange(0, 3), 32, 32).Draw(t, "answers")
		g := &guard{}
		var mu sync.Mutex
		var problem string
		answers := map[string]acp.RequestPermissionOutcome{}
		var completed bool
		agent := g.agent(sc.Agent())
		agent.Observe = func(ev agentloop.Event) {
			if e := ev.Entry; e != nil && e.RunEnded != nil {
				mu.Lock()
				completed = e.RunEnded.Reason == agentloop.ReasonCompleted
				mu.Unlock()
			}
		}
		h := serveIn(t, dir, agent, Options{}, false)
		h.cl.permit = func(_ context.Context, req acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
			mu.Lock()
			defer mu.Unlock()
			call := string(req.ToolCall.ToolCallId)
			s, err := agentloop.OpenSession(agentloop.NewFileStore(path))
			if err != nil {
				problem = err.Error()
				return acp.RequestPermissionResponse{}, err
			}
			info, ok := s.Call(call)
			raw, _ := json.Marshal(req.ToolCall.RawInput)
			var labels []string
			json.Unmarshal(req.ToolCall.Meta["agentloop/labels"], &labels)
			switch {
			case !ok || !info.Held():
				problem = "asked about a call that is not held: " + call
			case !jsonEqual(raw, info.Call.Input):
				problem = "the request does not show the whole input of " + call
			case !slices.Equal(labels, s.Labels()):
				problem = "the request shows labels other than the session's"
			case len(req.Options) != 2 || req.Options[0].OptionId != OptionAllowOnce || req.Options[1].OptionId != OptionRejectOnce:
				problem = "the request offers other options"
			}
			var out acp.RequestPermissionOutcome
			choice := choices[len(answers)%len(choices)]
			switch choice {
			case 0:
				out = acp.NewRequestPermissionOutcomeSelected(OptionAllowOnce)
			case 1:
				out = acp.NewRequestPermissionOutcomeSelected(OptionRejectOnce)
			case 2:
				out = acp.NewRequestPermissionOutcomeSelected("allow_always")
			default:
				out = acp.NewRequestPermissionOutcomeCancelled()
			}
			answers[call] = out
			return acp.RequestPermissionResponse{Outcome: out}, nil
		}
		if err := h.load(id); err != nil {
			t.Fatalf("load: %v", err)
		}
		for i := 0; i < sc.DrainRuns(); i++ {
			// A scripted model error answers the prompt with an error; the
			// next prompt carries on.
			h.prompt(id, "continue")
			g.check(t)
			mu.Lock()
			done := completed
			mu.Unlock()
			if done {
				break
			}
		}
		mu.Lock()
		defer mu.Unlock()
		if problem != "" {
			t.Fatal(problem)
		}
		if !completed {
			t.Fatal("the conversation never completed")
		}
		after, err := agentloop.OpenSession(agentloop.NewFileStore(path))
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range leftHeld {
			if _, asked := answers[c.ID]; !asked {
				t.Fatalf("held call %s was never asked about", c.ID)
			}
		}
		for call, out := range answers {
			info, _ := after.Call(call)
			allowed := out.Selected != nil && out.Selected.OptionId == OptionAllowOnce
			if info.Started != allowed || !info.Resolved {
				t.Fatalf("call %s answered %+v: started=%v resolved=%v", call, out, info.Started, info.Resolved)
			}
		}
		sc.CheckAtRest(true)
	})
}
