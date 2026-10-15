package acpagent

import (
	"encoding/json"
	"strings"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

type genBlock struct {
	block acp.ContentBlock
	text  string // the text of a text block
	bad   bool   // image, audio or a binary resource
	other bool   // a resource link or text resource
}

func genContent(t *rapid.T) genBlock {
	switch rapid.IntRange(0, 9).Draw(t, "blocktype") {
	case 0:
		return genBlock{block: acp.ImageBlock("aGk=", "image/png"), bad: true}
	case 1:
		return genBlock{block: acp.AudioBlock("aGk=", "audio/wav"), bad: true}
	case 2:
		return genBlock{block: acp.ResourceLinkBlock(rapid.String().Draw(t, "name"), "file:///x"), other: true}
	case 3:
		return genBlock{block: acp.ResourceBlock(acp.EmbeddedResourceResource{
			BlobResourceContents: &acp.BlobResourceContents{Blob: "aGk=", Uri: "file:///b"},
		}), bad: true}
	case 4:
		return genBlock{block: acp.ResourceBlock(acp.EmbeddedResourceResource{
			TextResourceContents: &acp.TextResourceContents{Text: rapid.String().Draw(t, "res"), Uri: "file:///r"},
		}), other: true}
	}
	s := rapid.String().Draw(t, "text")
	return genBlock{block: acp.TextBlock(s), text: s}
}

// TestPropertyPromptContent is P1.6: a list of text blocks becomes their
// text joined with newlines, in order, and any image, audio or binary
// resource is refused with -32602.
func TestPropertyPromptContent(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		blocks := rapid.SliceOfN(rapid.Custom(genContent), 0, 6).Draw(t, "blocks")
		var content []acp.ContentBlock
		var texts []string
		textOnly, bad := true, false
		for _, b := range blocks {
			content = append(content, b.block)
			texts = append(texts, b.text)
			bad = bad || b.bad
			textOnly = textOnly && !b.bad && !b.other
		}
		for _, b := range content {
			if err := validate("ContentBlock", b); err != nil {
				t.Fatal(err)
			}
		}
		got, err := PromptText(content)
		switch {
		case bad:
			if err == nil {
				t.Fatalf("refused content was accepted: %q", got)
			}
			if rpc, ok := err.(*acp.RequestError); !ok || rpc.Code != -32602 {
				t.Fatalf("error %v", err)
			}
		case textOnly:
			want := strings.Join(texts, "\n")
			if strings.TrimSpace(want) == "" {
				if err == nil {
					t.Fatalf("a blank prompt %q was accepted", got)
				}
				return
			}
			if err != nil || got != want {
				t.Fatalf("got %q, %v; want %q", got, err, want)
			}
		default:
			if err != nil && strings.TrimSpace(got) != "" {
				t.Fatalf("error %v with text %q", err, got)
			}
		}
	})
}

func TestPromptTextShapes(t *testing.T) {
	got, err := PromptText([]acp.ContentBlock{
		acp.TextBlock("Look at"),
		acp.ResourceLinkBlock("main.go", "file:///src/main.go"),
		acp.ResourceBlock(acp.EmbeddedResourceResource{TextResourceContents: &acp.TextResourceContents{
			Uri: "file:///notes.md", Text: "has ``` and ```` in it",
		}}),
	})
	want := "Look at\nmain.go: file:///src/main.go\nfile:///notes.md\n`````\nhas ``` and ```` in it\n`````"
	if err != nil || got != want {
		t.Fatalf("got %q, %v\nwant %q", got, err, want)
	}
}

// An empty prompt would make Run carry on without one, so it is refused.
func TestPromptTextRefusesBlank(t *testing.T) {
	for _, blocks := range [][]acp.ContentBlock{nil, {acp.TextBlock("")}, {acp.TextBlock(" \n")}, {{}}} {
		if got, err := PromptText(blocks); err == nil {
			t.Fatalf("%+v gave %q", blocks, got)
		}
	}
}

func TestTitleAndKind(t *testing.T) {
	cases := []struct {
		call agentloop.ToolCall
		want string
	}{
		{agentloop.ToolCall{Name: "ls", Input: json.RawMessage(`{}`)}, "ls"},
		{agentloop.ToolCall{Name: "ls"}, "ls"},
		{agentloop.ToolCall{Name: "read", Input: json.RawMessage(`{"path":  "a\nb"}`)}, `read {"path": "a\nb"}`},
		{agentloop.ToolCall{Input: json.RawMessage(`null`)}, "tool"},
	}
	for _, c := range cases {
		if got := Title(c.call); got != c.want {
			t.Errorf("Title(%+v) = %q, want %q", c.call, got, c.want)
		}
	}
	long := Title(agentloop.ToolCall{Name: "w", Input: json.RawMessage(`"` + strings.Repeat("x", 500) + `"`)})
	if len([]rune(long)) > 130 {
		t.Errorf("a long title is %d runes", len([]rune(long)))
	}
	kinds := map[string]acp.ToolKind{"read_file": acp.ToolKindRead, "odd": "teleport"}
	if Kind("read_file", kinds) != acp.ToolKindRead || Kind("odd", kinds) != acp.ToolKindOther ||
		Kind("missing", kinds) != acp.ToolKindOther || Kind("x", nil) != acp.ToolKindOther {
		t.Error("Kind")
	}
}
