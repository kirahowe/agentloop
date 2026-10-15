package acpagent

import (
	"fmt"
	"strings"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/request"
)

// PromptText turns the content of a session/prompt into the prompt text
// agentloop runs. Each block becomes text and the blocks are joined with a
// newline, in order:
//
//   - text is kept as it is;
//   - a resource_link becomes "<name>: <uri>";
//   - an embedded text resource becomes its uri and a fenced block holding
//     the text.
//
// Image, audio and binary resources are refused with -32602, since
// agentloop does not advertise them. So is a prompt with no text in it:
// Run takes an empty prompt to mean "carry on without one", which is not
// what the person asked for.
func PromptText(blocks []acp.ContentBlock) (string, error) {
	parts := make([]string, 0, len(blocks))
	for i, b := range blocks {
		s, err := blockText(b)
		if err != nil {
			return "", acp.NewInvalidParams(map[string]any{"block": i, "error": err.Error()})
		}
		parts = append(parts, s)
	}
	text := strings.Join(parts, "\n")
	if strings.TrimSpace(text) == "" {
		return "", acp.NewInvalidParams(map[string]any{"error": "the prompt has no text"})
	}
	return text, nil
}

func blockText(b acp.ContentBlock) (string, error) {
	switch {
	case b.Text != nil:
		return b.Text.Text, nil
	case b.ResourceLink != nil:
		return b.ResourceLink.Name + ": " + b.ResourceLink.Uri, nil
	case b.Resource != nil:
		r := b.Resource.Resource
		if r.TextResourceContents == nil {
			return "", fmt.Errorf("binary resources are not supported")
		}
		return r.TextResourceContents.Uri + "\n" + fenced(r.TextResourceContents.Text), nil
	case b.Image != nil:
		return "", fmt.Errorf("images are not supported")
	case b.Audio != nil:
		return "", fmt.Errorf("audio is not supported")
	}
	return "", fmt.Errorf("the content block is empty or of an unknown type")
}

// fenced puts text in a Markdown code fence longer than any run of
// backticks inside it, so the text cannot close the fence early.
func fenced(text string) string {
	longest, run := 0, 0
	for _, r := range text {
		if r == '`' {
			run++
			longest = max(longest, run)
		} else {
			run = 0
		}
	}
	fence := strings.Repeat("`", max(3, longest+1))
	return fence + "\n" + text + "\n" + fence
}

// Title is how a client shows a call: the tool's name and a short form of
// its input.
func Title(call agentloop.ToolCall) string {
	name := call.Name
	if name == "" {
		name = "tool"
	}
	in := strings.TrimSpace(string(call.Input))
	if in == "" || in == "{}" || in == "null" {
		return name
	}
	return name + " " + request.Brief(in)
}

// Kind is the ACP kind of a tool, from the request's tool_kinds. A tool
// that is not listed, or listed with a kind ACP does not have, is "other".
func Kind(name string, kinds map[string]acp.ToolKind) acp.ToolKind {
	switch k := kinds[name]; k {
	case acp.ToolKindRead, acp.ToolKindEdit, acp.ToolKindDelete, acp.ToolKindMove, acp.ToolKindSearch,
		acp.ToolKindExecute, acp.ToolKindThink, acp.ToolKindFetch, acp.ToolKindSwitchMode:
		return k
	}
	return acp.ToolKindOther
}
