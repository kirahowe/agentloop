package agentloop

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// A Display is what a tool result shows the operator alone: a title, a
// Markdown body and its sources. The agent keeps it in the tool_result
// entry and sends it to a client, and never puts any of it in the
// model's context. The model is told only that the operator was shown
// something, and under which ref (see DisplayNote).
type Display struct {
	Title string `json:"title"`
	Body  string `json:"body"`
	// Labels are what the display's text carries. The agent adds the
	// result's labels and the session's to them, so a display is never
	// labelled less than the call that made it.
	Labels  []string `json:"labels"`
	Sources []Source `json:"sources"`
	// Ref is where the display is kept, as kind:id, set by the agent
	// when it has a DisplayRefPrefix.
	Ref string `json:"ref,omitempty"`
}

// A Source is one thing a display draws on: a title and a ref, such as
// web:https://example.org/ or doc:notes/venues.md.
type Source struct {
	Title string `json:"title"`
	Ref   string `json:"ref"`
}

// The bounds on a display.
const (
	MaxDisplayTitle   = 200 // characters
	MaxDisplayBody    = 200_000
	MaxDisplaySources = 50
	MaxSourceRef      = 2000
)

// CheckDisplay says what is wrong with a display a tool gave, or returns
// nil: a title of 1 to MaxDisplayTitle characters, a body of at most
// MaxDisplayBody bytes, labels that pass CheckLabels, and at most
// MaxDisplaySources sources, each with a title and a one-line ref.
func CheckDisplay(d Display) error {
	if strings.TrimSpace(d.Title) == "" {
		return errors.New("the display has no title")
	}
	if utf8.RuneCountInString(d.Title) > MaxDisplayTitle {
		return fmt.Errorf("the display's title is longer than %d characters", MaxDisplayTitle)
	}
	if len(d.Body) > MaxDisplayBody {
		return fmt.Errorf("the display's body is longer than %d bytes", MaxDisplayBody)
	}
	if err := CheckLabels(d.Labels); err != nil {
		return fmt.Errorf("the display has a bad label: %w", err)
	}
	if len(d.Sources) > MaxDisplaySources {
		return fmt.Errorf("the display has more than %d sources", MaxDisplaySources)
	}
	for _, s := range d.Sources {
		if strings.TrimSpace(s.Title) == "" || utf8.RuneCountInString(s.Title) > MaxDisplayTitle {
			return fmt.Errorf("a source's title must have 1 to %d characters", MaxDisplayTitle)
		}
		if strings.TrimSpace(s.Ref) == "" || len(s.Ref) > MaxSourceRef || strings.ContainsAny(s.Ref, "\n\r") {
			return fmt.Errorf("a source's ref must be one line of 1 to %d bytes", MaxSourceRef)
		}
	}
	return nil
}

var displayIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,99}$`)

// DisplayID is the name a call's display is kept under: the call id in
// lower case when that is a short run of letters, digits, _ and -, and
// otherwise "call-" and the start of the id's sha256. It is a safe path
// segment either way.
func DisplayID(callID string) string {
	if id := strings.ToLower(callID); displayIDPattern.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(callID))
	return "call-" + hex.EncodeToString(sum[:8])
}

// DisplayNote is what the model is told in place of a display, after the
// result's content. It holds nothing of the display, which may carry
// labels the result does not: only the ref, which the agent made.
func DisplayNote(ref string) string {
	if ref == "" {
		return "The operator was also shown more than this."
	}
	return "The operator was also shown [[" + ref + "]]."
}

// withDisplayNote is content with the note after it.
func withDisplayNote(content, ref string) string {
	if strings.TrimSpace(content) == "" {
		return DisplayNote(ref)
	}
	return content + "\n\n" + DisplayNote(ref)
}

// UnionLabels is the sorted set of the labels in all of ls.
func UnionLabels(ls ...[]string) []string {
	set := map[string]bool{}
	for _, l := range ls {
		for _, x := range l {
			set[x] = true
		}
	}
	out := make([]string, 0, len(set))
	for x := range set {
		out = append(out, x)
	}
	sort.Strings(out)
	return out
}
