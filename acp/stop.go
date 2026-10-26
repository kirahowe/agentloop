package acpagent

import (
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
)

// Respond turns how a run ended into the answer to session/prompt: a stop
// reason, or a JSON-RPC error. Exactly one of the two is set, for every
// outcome, including nil and reasons agentloop does not have.
//
//	completed                 end_turn
//	max_tokens                max_tokens
//	refused                   refusal
//	limit max_tokens          max_tokens
//	limit, any other limit    max_turn_requests
//	canceled                  cancelled
//	model_error               error -32603, data: the error's kind, status, message and times
//	forbidden                 error -32603, data: the labels it may not hold
//
// A turn that was cancelled is answered cancelled whatever the outcome.
func Respond(out *agentloop.Outcome) (acp.StopReason, *acp.RequestError) {
	if out == nil {
		return "", internal("the run ended with no outcome", nil)
	}
	switch out.Reason {
	case agentloop.ReasonCompleted:
		return acp.StopReasonEndTurn, nil
	case agentloop.ReasonMaxTokens:
		return acp.StopReasonMaxTokens, nil
	case agentloop.ReasonRefused:
		return acp.StopReasonRefusal, nil
	case agentloop.ReasonCanceled:
		return acp.StopReasonCancelled, nil
	case agentloop.ReasonLimit:
		if out.Limit == agentloop.LimitTokens {
			return acp.StopReasonMaxTokens, nil
		}
		return acp.StopReasonMaxTurnRequests, nil
	case agentloop.ReasonModelError:
		if out.Error == nil {
			return "", internal("the model call failed", map[string]any{"reason": string(out.Reason)})
		}
		return "", internal(out.Error.Error(), modelErrorData(out.Error))
	case agentloop.ReasonForbidden:
		return "", internal("the session holds a label it may not hold, so it cannot run",
			map[string]any{"reason": string(out.Reason), "labels": out.Forbidden})
	}
	return "", internal("the run ended for an unknown reason", map[string]any{"reason": string(out.Reason)})
}

func internal(msg string, data map[string]any) *acp.RequestError {
	return &acp.RequestError{Code: -32603, Message: msg, Data: data}
}

func modelErrorData(me *agentloop.ModelError) map[string]any {
	d := map[string]any{"kind": string(me.Kind), "message": me.Message}
	if me.Status != 0 {
		d["status"] = me.Status
	}
	if !me.RetryAfter.IsZero() {
		d["retry_after"] = me.RetryAfter.UTC().Format(time.RFC3339Nano)
	}
	if !me.ResetsAt.IsZero() {
		d["resets_at"] = me.ResetsAt.UTC().Format(time.RFC3339Nano)
	}
	return d
}
