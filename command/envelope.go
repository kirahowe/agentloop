package command

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/strictjson"
)

// stdinEnvelope is what an envelope tool reads.
type stdinEnvelope struct {
	Input  json.RawMessage `json:"input"`
	Labels []string        `json:"labels"`
	CallID string          `json:"call_id"`
}

// stdoutEnvelope is what an envelope tool must write. Content and labels
// are required; is_error defaults to false. display is optional: what
// the operator is shown and the model is not (agentloop.Display).
type stdoutEnvelope struct {
	Content *string          `json:"content"`
	IsError bool             `json:"is_error"`
	Labels  *[]string        `json:"labels"`
	Display *displayEnvelope `json:"display"`
}

// displayEnvelope is a display as a tool writes it: title, body and
// labels are required, sources optional. It has no ref; agentloop gives
// it one.
type displayEnvelope struct {
	Title   *string            `json:"title"`
	Body    *string            `json:"body"`
	Labels  *[]string          `json:"labels"`
	Sources []agentloop.Source `json:"sources"`
}

func (d *displayEnvelope) display(known func(string) bool) (*agentloop.Display, string) {
	if d.Title == nil || d.Body == nil || d.Labels == nil {
		return nil, "the display must have title, body and labels"
	}
	out := agentloop.Display{Title: *d.Title, Body: *d.Body, Labels: *d.Labels, Sources: d.Sources}
	if out.Sources == nil {
		out.Sources = []agentloop.Source{}
	}
	if err := agentloop.CheckDisplay(out, known); err != nil {
		return nil, err.Error()
	}
	return &out, ""
}

// Failed is all the model is told of an envelope tool that did not give a
// usable result.
func Failed(name string) string { return fmt.Sprintf("The tool %s failed.", name) }

func (t *Tool) envelope(ctx context.Context, call agentloop.Call) (agentloop.Result, error) {
	failed := func(why string, stderr *capped) agentloop.Result {
		return agentloop.Result{Content: Failed(t.Def.Name), IsError: true, Log: operatorLog(why, stderr)}
	}
	if len(t.Argv) == 0 {
		return failed("the tool has no argv", nil), nil
	}
	if err := ctx.Err(); err != nil {
		return agentloop.Result{}, err
	}
	in := StdinEnvelope(call)
	stdout, stderr, waitErr, err := t.run(ctx, in)
	if err != nil {
		return failed(err.Error(), nil), nil
	}
	if err := ctx.Err(); err != nil {
		// The agent says what a timeout or a cancel means; the operator
		// still gets what the program wrote to stderr.
		return agentloop.Result{Log: operatorLog("stopped: "+err.Error(), stderr)}, err
	}
	if waitErr != nil && !errors.Is(waitErr, exec.ErrWaitDelay) {
		return failed(waitErr.Error(), stderr), nil
	}
	if stdout.cut {
		return failed(fmt.Sprintf("stdout passed the cap of %d bytes", stdout.max), stderr), nil
	}
	res, why := ParseEnvelope(stdout.buf.Bytes(), t.KnownLabel)
	if why != "" {
		return failed(why, stderr), nil
	}
	res.Log = operatorLog("", stderr)
	return res, nil
}

// StdinEnvelope is the JSON an envelope tool reads for the call. Labels
// is always an array, never null.
func StdinEnvelope(call agentloop.Call) []byte {
	input := call.Input
	if len(bytes.TrimSpace(input)) == 0 || !json.Valid(input) {
		input = json.RawMessage(`{}`)
	}
	labels := call.Labels
	if labels == nil {
		labels = []string{}
	}
	b, _ := json.Marshal(stdinEnvelope{Input: input, Labels: labels, CallID: call.ID})
	return b
}

// ParseEnvelope reads an envelope tool's stdout. It returns the result,
// or why stdout is not exactly one valid envelope: unknown, missing,
// repeated or differently spelled fields (so "Labels" or a second
// "labels" cannot replace the labels), any label known does not accept,
// a display that agentloop.CheckDisplay refuses, or anything after the
// object but whitespace.
func ParseEnvelope(stdout []byte, known func(string) bool) (agentloop.Result, string) {
	var env stdoutEnvelope
	if err := strictjson.Object(stdout); err != nil {
		return agentloop.Result{}, "stdout is not an envelope: " + err.Error()
	}
	if err := strictjson.UnmarshalKnown(stdout, &env); err != nil {
		return agentloop.Result{}, "stdout is not an envelope: " + err.Error()
	}
	if env.Content == nil {
		return agentloop.Result{}, "the envelope has no content"
	}
	if env.Labels == nil {
		return agentloop.Result{}, "the envelope has no labels"
	}
	for _, l := range *env.Labels {
		if known == nil || !known(l) {
			return agentloop.Result{}, fmt.Sprintf("the envelope has the unknown label %q", l)
		}
	}
	labels := *env.Labels
	if len(labels) == 0 {
		labels = nil
	}
	res := agentloop.Result{Content: *env.Content, IsError: env.IsError, Labels: labels}
	if env.Display != nil {
		d, why := env.Display.display(known)
		if why != "" {
			return agentloop.Result{}, why
		}
		res.Display = d
	}
	return res, ""
}

// operatorLog is why a call failed, if it did, and what the program wrote
// to stderr.
func operatorLog(why string, stderr *capped) string {
	var parts []string
	if why != "" {
		parts = append(parts, why)
	}
	if stderr != nil {
		if s := text(stderr); s != "" {
			parts = append(parts, "stderr: "+s)
		}
	}
	return strings.Join(parts, "\n")
}
