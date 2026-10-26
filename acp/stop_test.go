package acpagent

import (
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

var allReasons = []agentloop.Reason{
	agentloop.ReasonCompleted, agentloop.ReasonMaxTokens, agentloop.ReasonRefused,
	agentloop.ReasonLimit, agentloop.ReasonModelError, agentloop.ReasonCanceled, agentloop.ReasonForbidden, "", "exploded",
}

func genOutcome(t *rapid.T) *agentloop.Outcome {
	if rapid.IntRange(0, 20).Draw(t, "nil") == 0 {
		return nil
	}
	out := &agentloop.Outcome{
		Reason: rapid.SampledFrom(allReasons).Draw(t, "reason"),
		Limit: rapid.SampledFrom([]string{"", agentloop.LimitTurns, agentloop.LimitToolCalls,
			agentloop.LimitTokens, "max_fun"}).Draw(t, "limit"),
		Text: rapid.String().Draw(t, "text"),
	}
	if rapid.Bool().Draw(t, "haserror") {
		out.Error = &agentloop.ModelError{
			Kind: rapid.SampledFrom([]agentloop.ErrorKind{agentloop.RateLimited, agentloop.Overloaded,
				agentloop.Unauthorized, agentloop.InvalidRequest, ""}).Draw(t, "kind"),
			Status:  rapid.SampledFrom([]int{0, 429, 500}).Draw(t, "status"),
			Message: rapid.String().Draw(t, "message"),
		}
		if rapid.Bool().Draw(t, "retryafter") {
			out.Error.RetryAfter = time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
		}
		if rapid.Bool().Draw(t, "resetsat") {
			out.Error.ResetsAt = time.Date(2026, 10, 15, 10, 0, 0, 0, time.FixedZone("x", 3600))
		}
	}
	return out
}

// TestPropertyStopMapping is P1.5: every outcome gives exactly one of a
// stop reason or an error, as the table in Respond says, and the stop
// reason validates as a PromptResponse. model_error never gives a stop
// reason.
func TestPropertyStopMapping(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		out := genOutcome(t)
		stop, err := Respond(out)
		if (stop == "") == (err == nil) {
			t.Fatalf("Respond(%+v) = %q, %v: want exactly one", out, stop, err)
		}
		if err != nil {
			if err.Code != -32603 || err.Message == "" {
				t.Fatalf("error %+v", err)
			}
		} else if verr := validate("PromptResponse", map[string]any{"stopReason": stop}); verr != nil {
			t.Fatal(verr)
		}
		if out == nil {
			if err == nil {
				t.Fatal("a nil outcome gave a stop reason")
			}
			return
		}
		want := map[agentloop.Reason]acp.StopReason{
			agentloop.ReasonCompleted: acp.StopReasonEndTurn,
			agentloop.ReasonMaxTokens: acp.StopReasonMaxTokens,
			agentloop.ReasonRefused:   acp.StopReasonRefusal,
			agentloop.ReasonCanceled:  acp.StopReasonCancelled,
		}
		switch out.Reason {
		case agentloop.ReasonLimit:
			w := acp.StopReasonMaxTurnRequests
			if out.Limit == agentloop.LimitTokens {
				w = acp.StopReasonMaxTokens
			}
			if stop != w {
				t.Fatalf("limit %q gave %q, want %q", out.Limit, stop, w)
			}
		case agentloop.ReasonModelError, agentloop.ReasonForbidden, "", "exploded":
			if err == nil {
				t.Fatalf("%q gave the stop reason %q", out.Reason, stop)
			}
		default:
			if stop != want[out.Reason] {
				t.Fatalf("%q gave %q, want %q", out.Reason, stop, want[out.Reason])
			}
		}
	})
}

func TestRespondModelErrorData(t *testing.T) {
	stop, err := Respond(&agentloop.Outcome{Reason: agentloop.ReasonModelError, Error: &agentloop.ModelError{
		Kind: agentloop.RateLimited, Status: 429, Message: "slow down",
		RetryAfter: time.Date(2026, 10, 15, 9, 0, 12, 0, time.FixedZone("x", 3600)),
	}})
	if stop != "" || err == nil {
		t.Fatalf("got %q, %v", stop, err)
	}
	d := err.Data.(map[string]any)
	if d["kind"] != "rate_limited" || d["status"] != 429 || d["message"] != "slow down" ||
		d["retry_after"] != "2026-10-15T08:00:12Z" || d["resets_at"] != nil {
		t.Fatalf("data %+v", d)
	}
}

// The reason "held" no longer exists: an outcome carrying it is an error
// like any other unknown reason.
func TestRespondHeldIsUnknown(t *testing.T) {
	stop, err := Respond(&agentloop.Outcome{Reason: "held"})
	if stop != "" || err == nil || err.Code != -32603 || err.Data.(map[string]any)["reason"] != "held" {
		t.Fatalf("got %q, %v", stop, err)
	}
}
