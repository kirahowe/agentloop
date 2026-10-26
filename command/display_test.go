package command

import (
	"slices"
	"testing"

	"github.com/kirahowe/agentloop"
)

func TestParseEnvelopeReadsADisplay(t *testing.T) {
	out := `{"content":"closed","labels":["private"],"display":{"title":"Venues","body":"# Venues\nmore",` +
		`"labels":["contaminated"],"sources":[{"title":"List","ref":"web:https://example.org/"}]}}`
	res, why := ParseEnvelope([]byte(out))
	if why != "" {
		t.Fatal(why)
	}
	want := &agentloop.Display{Title: "Venues", Body: "# Venues\nmore", Labels: []string{"contaminated"},
		Sources: []agentloop.Source{{Title: "List", Ref: "web:https://example.org/"}}}
	if res.Display == nil || res.Display.Title != want.Title || res.Display.Body != want.Body ||
		!slices.Equal(res.Display.Labels, want.Labels) || !slices.Equal(res.Display.Sources, want.Sources) {
		t.Errorf("display %+v", res.Display)
	}
	if res.Content != "closed" {
		t.Errorf("content %q", res.Content)
	}
	// No sources is an empty list.
	res, why = ParseEnvelope([]byte(`{"content":"c","labels":[],"display":{"title":"t","body":"b","labels":[]}}`))
	if why != "" || res.Display == nil || res.Display.Sources == nil {
		t.Errorf("display %+v, %s", res.Display, why)
	}
}

// A display that is not exactly one well formed display fails the whole
// call, so a tool cannot slip a label past the check or a ref in.
func TestParseEnvelopeRefusesBadDisplays(t *testing.T) {
	for _, d := range []string{
		`{"title":"t","body":"b"}`,
		`{"title":"t","labels":[]}`,
		`{"body":"b","labels":[]}`,
		`{"title":"","body":"b","labels":[]}`,
		`{"title":"t","body":"b","labels":[""]}`,
		`{"title":"t","body":"b","labels":["contaminated"],"Labels":[]}`,
		`{"title":"t","body":"b","labels":["contaminated"],"labels":[]}`,
		`{"title":"t","body":"b","labels":[],"ref":"shown:x"}`,
		`{"title":"t","body":"b","labels":[],"sources":[{"title":"s"}]}`,
		`{"title":"t","body":"b","labels":[],"sources":[{"title":"s","ref":"a\nb"}]}`,
		`{"title":"t","body":"b","labels":[],"sources":[{"title":"s","ref":"x","extra":1}]}`,
		`"text"`,
	} {
		out := `{"content":"c","labels":[],"display":` + d + `}`
		if res, why := ParseEnvelope([]byte(out)); why == "" {
			t.Errorf("accepted %s as %+v", d, res.Display)
		}
	}
}
