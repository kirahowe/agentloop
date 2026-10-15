package acpagent

import (
	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/trifecta"
)

// Asking the client about held calls, as values. The shell sends the
// request and feeds the answer back through PermissionVerdict.

// The options a held call offers. There is no "always": each call is
// approved on its own.
const (
	OptionAllowOnce  acp.PermissionOptionId = "allow_once"
	OptionRejectOnce acp.PermissionOptionId = "reject_once"
)

// PermissionRequest asks the client about one held call: the tool, its
// kind, a title, the whole input as rawInput (never cut short), and the
// session's labels under _meta["agentloop/labels"].
func PermissionRequest(id acp.SessionId, held agentloop.HeldCall, kinds map[string]acp.ToolKind, labels []string) acp.RequestPermissionRequest {
	c := held.ToolCall
	kind := Kind(c.Name, kinds)
	status := acp.ToolCallStatusPending
	return acp.RequestPermissionRequest{
		SessionId: id,
		Options: []acp.PermissionOption{
			{OptionId: OptionAllowOnce, Kind: acp.PermissionOptionKindAllowOnce, Name: "Allow once"},
			{OptionId: OptionRejectOnce, Kind: acp.PermissionOptionKindRejectOnce, Name: "Reject"},
		},
		ToolCall: acp.ToolCallUpdate{
			ToolCallId: acp.ToolCallId(c.ID),
			Title:      acp.Ptr(Title(c)),
			Name:       acp.Ptr(c.Name),
			Kind:       &kind,
			Status:     &status,
			RawInput:   rawJSON(c.Input),
			Meta:       labelsMeta(labels),
		},
	}
}

// PermissionVerdict is the person's answer: allow only when the client
// selected the allow option this agent offered. A rejection, a
// cancellation, an option that was not offered, or no answer at all (an
// error, a closed connection) is a deny.
func PermissionVerdict(resp acp.RequestPermissionResponse, err error) agentloop.Verdict {
	if err != nil || resp.Outcome.Selected == nil || resp.Outcome.Selected.OptionId != OptionAllowOnce {
		return agentloop.Deny
	}
	return agentloop.Allow
}

// NotApproved is what the model is told of a held call a person did not
// allow.
const NotApproved = "A person did not approve this call."

// PromptLabels are the labels a prompt's content brings into the session.
// Text is the person's own; an embedded resource or a resource link is
// content from elsewhere that an outsider could have written, so it
// contaminates the session.
func PromptLabels(blocks []acp.ContentBlock) []string {
	for _, b := range blocks {
		if b.Resource != nil || b.ResourceLink != nil {
			return []string{trifecta.Contaminated}
		}
	}
	return nil
}
