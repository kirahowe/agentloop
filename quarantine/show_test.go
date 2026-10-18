package quarantine

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

func showInput(tool, schema string) json.RawMessage {
	return json.RawMessage(`{"tool":"` + tool + `","input":{"id":"m-1"},"question":"Is it spam?","schema":` + schema + `,"show_operator":true}`)
}

const operatorText = "OPERATOR-TEXT: it is spam, see https://evil.example/x"

func TestShowOperatorGoesOnlyToTheDisplay(t *testing.T) {
	model := fake.Script(reply(`{"answer":{"spam":true,"due":null},"operator":"` + operatorText + `"}`))
	in := &inner{name: "mail_read", content: secretContent, labels: []string{"private", "contaminated"}}
	r := newTestReader(t, model, in)
	res, err := r.CallWith(context.Background(), agentloop.Call{ID: "c1", Input: showInput("mail_read", closedSchema)})
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError || res.Content != `{"answer":{"due":null,"spam":true},"free_text":false}` {
		t.Fatalf("result %+v", res)
	}
	// The caller stays clean: the closed answer drops contaminated.
	if !slices.Equal(res.Labels, []string{"private"}) {
		t.Errorf("labels %v", res.Labels)
	}
	noLeak(t, res, "OPERATOR-TEXT", "evil.example")
	d := res.Display
	if d == nil {
		t.Fatal("no display")
	}
	if d.Body != operatorText || d.Title != "Is it spam?" {
		t.Errorf("display %+v", d)
	}
	// The display keeps every label the reader's input had.
	if !slices.Equal(d.Labels, []string{"contaminated", "private"}) {
		t.Errorf("display labels %v", d.Labels)
	}
	reqs := model.Requests()
	if len(reqs) != 1 || reqs[0].System != ReaderSystemShowing {
		t.Errorf("the reader was not asked for the operator's text")
	}
}

func TestShowOperatorOnAnOpenSchemaKeepsContaminated(t *testing.T) {
	model := fake.Script(reply(`{"answer":{"summary":"s"},"operator":"more"}`))
	in := &inner{name: "mail_read", content: secretContent, labels: []string{"contaminated"}}
	res, err := newTestReader(t, model, in).CallWith(context.Background(), agentloop.Call{Input: showInput("mail_read", openSchema)})
	if err != nil || res.Display == nil {
		t.Fatalf("%+v %v", res, err)
	}
	if !slices.Contains(res.Labels, "contaminated") || !slices.Contains(res.Display.Labels, "contaminated") {
		t.Errorf("labels %v, display %v", res.Labels, res.Display.Labels)
	}
}

func TestShowOperatorWithoutTextHasNoDisplay(t *testing.T) {
	for _, text := range []string{`{"answer":{"spam":false,"due":null}}`, `{"answer":{"spam":false,"due":null},"operator":"  "}`} {
		model := fake.Script(reply(text))
		in := &inner{name: "mail_read", content: "x", labels: []string{"contaminated"}}
		res, err := newTestReader(t, model, in).CallWith(context.Background(), agentloop.Call{Input: showInput("mail_read", closedSchema)})
		if err != nil || res.IsError || res.Display != nil {
			t.Errorf("%s: %+v %v", text, res, err)
		}
	}
}

func TestShowOperatorUnanswerableStillShows(t *testing.T) {
	model := fake.Script(reply(`{"unanswerable":true,"operator":"the mail says nothing about it"}`))
	in := &inner{name: "mail_read", content: "x", labels: []string{"contaminated"}}
	res, _ := newTestReader(t, model, in).CallWith(context.Background(), agentloop.Call{Input: showInput("mail_read", closedSchema)})
	if res.Content != `{"free_text":false,"unanswerable":true}` || res.Display == nil {
		t.Errorf("%+v", res)
	}
}

// Without show_operator, a reply with "operator" is not a valid reply,
// and nothing of it is shown to anyone.
func TestOperatorKeyWithoutShowOperatorIsRefused(t *testing.T) {
	model := fake.Script(reply(`{"answer":{"spam":true,"due":null},"operator":"x"}`), reply(`{"answer":{"spam":true,"due":null},"operator":"x"}`))
	in := &inner{name: "mail_read", content: "x", labels: []string{"contaminated"}}
	res, _ := newTestReader(t, model, in).CallWith(context.Background(), agentloop.Call{Input: readerInput("mail_read", closedSchema)})
	if !res.IsError || res.Content != NoAnswer || res.Display != nil {
		t.Errorf("%+v", res)
	}
}

func TestParseReplyForRefusesRepeatedOperator(t *testing.T) {
	s, _ := Compile(json.RawMessage(closedSchema))
	for _, text := range []string{
		`{"answer":{"spam":true,"due":null},"operator":"a","operator":"b"}`,
		`{"answer":{"spam":true,"due":null},"answer":{"spam":false,"due":null},"operator":"a"}`,
		`{"answer":{"spam":true,"due":null},"operator":5}`,
		`{"answer":{"spam":true,"due":null},"Operator":"a"}`,
		`{"operator":"a"}`,
	} {
		if r, err := ParseReplyFor(s, text, true); err == nil {
			t.Errorf("accepted %s as %+v", text, r)
		}
	}
	r, err := ParseReplyFor(s, "```json\n{\"operator\":\"o\",\"answer\":{\"spam\":true,\"due\":null}}\n```", true)
	if err != nil || r.Operator != "o" || string(r.Answer) != `{"due":null,"spam":true}` {
		t.Errorf("%+v %v", r, err)
	}
}

// The inner tool's own display gives the reader's display its sources
// and its labels.
func TestShowOperatorKeepsTheInnerSources(t *testing.T) {
	model := fake.Script(reply(`{"answer":{"spam":true,"due":null},"operator":"see page 1"}`))
	src := []agentloop.Source{{Title: "Page 1", Ref: "web:https://example.org/1"}}
	in := &displayInner{inner: inner{name: "mail_read", content: "x", labels: []string{"contaminated"}},
		display: &agentloop.Display{Title: "Results", Body: "b", Labels: []string{"private"}, Sources: src}}
	res, _ := newTestReader(t, model, in).CallWith(context.Background(), agentloop.Call{Input: showInput("mail_read", closedSchema)})
	if res.Display == nil || !slices.Equal(res.Display.Sources, src) || !slices.Equal(res.Display.Labels, []string{"contaminated", "private"}) {
		t.Errorf("display %+v", res.Display)
	}
}

type displayInner struct {
	inner
	display *agentloop.Display
}

func (t *displayInner) CallWith(ctx context.Context, c agentloop.Call) (agentloop.Result, error) {
	res, err := t.inner.CallWith(ctx, c)
	res.Display = t.display
	return res, err
}

func TestOperatorDisplayBounds(t *testing.T) {
	d := OperatorDisplay(strings.Repeat("q", 500), strings.Repeat("b", MaxOperatorText+50), []string{"contaminated"}, nil)
	if err := agentloop.CheckDisplay(*d, func(string) bool { return true }); err != nil {
		t.Errorf("an operator display breaks the bounds: %v", err)
	}
}
