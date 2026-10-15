package acpagent

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/kirahowe/agentloop"
)

func delta(pt agentloop.PartType, s string) agentloop.Event {
	return agentloop.Event{Delta: &agentloop.Delta{Type: pt, Text: s}}
}

func entry(seq int, e agentloop.Entry) agentloop.Event {
	e.Seq = seq
	return agentloop.Event{Entry: &e}
}

func assistant(seq int, stop agentloop.Stop, refusal string, parts ...agentloop.Part) agentloop.Event {
	return entry(seq, agentloop.Entry{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{
		Message: agentloop.Message{Role: agentloop.Assistant, Parts: parts}, Stop: stop, Refusal: refusal,
	}})
}

func text(s string) agentloop.Part { return agentloop.Part{Type: agentloop.TextPart, Text: s} }

var runStarted = agentloop.Entry{Type: agentloop.EntryRunStarted, RunStarted: &agentloop.RunStartedEntry{}}

// chunks lists the text of each message chunk as "id:text".
func chunks(ups []Update) []string {
	var out []string
	for _, u := range ups {
		if c := u.AgentMessageChunk; c != nil {
			out = append(out, string(*c.MessageId)+":"+c.Content.Text.Text)
		}
	}
	return out
}

func translate(opts Options, evs ...agentloop.Event) []Update {
	return translateAll(opts, evs)
}

func TestRemainderAfterPartialStream(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		delta(agentloop.TextPart, "Hel"),
		assistant(2, agentloop.StopEndTurn, "", text("Hello"), text(" world")),
	))
	want := []string{"a2:Hel", "a2:lo", "a2: world"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// A stream the message does not start with cannot be taken back, and
// sending the message again would repeat it, so nothing more is sent.
func TestDivergedStreamSendsNothingMore(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		delta(agentloop.TextPart, "Howdy"),
		assistant(2, agentloop.StopEndTurn, "", text("Hello")),
	))
	if want := []string{"a2:Howdy"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// A model call that fails part way through its stream leaves text under
// an id no message gets; the retry streams under a new one, so a client
// does not glue the two together.
func TestRetryStreamsUnderANewID(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		delta(agentloop.TextPart, "Sorry, I"),
		entry(2, agentloop.Entry{Type: agentloop.EntryModelError, ModelError: &agentloop.ModelErrorEntry{
			Error: agentloop.ModelError{Kind: agentloop.Unreachable, Message: "reset"}, Retry: true, WaitMillis: 1500,
		}}),
		delta(agentloop.TextPart, "Hello"),
		assistant(3, agentloop.StopEndTurn, "", text("Hello")),
	))
	if want := []string{"a2:Sorry, I", "a3:Hello"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestEmptyPartsSendNoChunks(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		assistant(2, agentloop.StopEndTurn, "", text("a"), text(""), text("b")),
	))
	if want := []string{"a2:a", "a2:b"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// OpenAI puts the refusal in the text as well; it is not sent twice.
func TestRefusalInTextIsNotRepeated(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		assistant(2, agentloop.StopRefusal, "I can't do that.", text("I can't do that.")),
	))
	if want := []string{"a2:I can't do that."}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Text with invalid UTF-8 reaches the client as JSON gives it, which is
// what the log holds too, so the message adds nothing to the stream.
func TestInvalidUTF8StreamsAsTheLogHoldsIt(t *testing.T) {
	ups := translate(Options{},
		entry(1, runStarted),
		delta(agentloop.TextPart, "bad \xff byte"),
		assistant(2, agentloop.StopEndTurn, "", text("bad � byte")),
	)
	if got, want := chunks(ups), []string{"a2:bad � byte"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	for _, u := range ups {
		if err := validateUpdate(u); err != nil {
			t.Fatal(err)
		}
	}
}

func TestNoNoticeWithoutTheCapability(t *testing.T) {
	ups := translate(Options{},
		entry(1, runStarted),
		entry(2, agentloop.Entry{Type: agentloop.EntryModelError, ModelError: &agentloop.ModelErrorEntry{
			Error: agentloop.ModelError{Kind: agentloop.RateLimited}, Retry: true, WaitMillis: 1,
		}}),
	)
	if len(ups) != 0 {
		t.Fatalf("got %+v", ups)
	}
}

// A retry notice's title rounds the wait up and never says "0 s" for a
// wait that is not zero; its attempt counts failures since the last
// message.
func TestNoticeWaitAndAttempt(t *testing.T) {
	failed := func(seq int, ms int64) agentloop.Event {
		return entry(seq, agentloop.Entry{Type: agentloop.EntryModelError, ModelError: &agentloop.ModelErrorEntry{
			Error: agentloop.ModelError{Kind: agentloop.Overloaded, Message: "busy"}, Retry: true, WaitMillis: ms,
		}})
	}
	ups := translate(Options{Notices: true}, entry(1, runStarted), failed(2, 1), failed(3, 2000))
	if len(ups) != 2 {
		t.Fatalf("got %d updates", len(ups))
	}
	if n := ups[0].Notice; n.Title != "The provider is overloaded, retrying in 1 s" {
		t.Fatalf("title %q", n.Title)
	}
	var meta struct{ Attempt int }
	json.Unmarshal(ups[1].Notice.Meta["agentloop/retry"], &meta)
	if meta.Attempt != 2 {
		t.Fatalf("attempt %d", meta.Attempt)
	}
}

// Step returns a new translator and leaves the one it was called on as it
// was, so the same value stepped twice gives the same result.
func TestStepIsPure(t *testing.T) {
	tr := NewTranslator(Options{})
	tr, _ = tr.Step(entry(1, runStarted))
	tr, _ = tr.Step(delta(agentloop.TextPart, "Hel"))
	msg := assistant(2, agentloop.StopEndTurn, "", text("Hello"))
	_, a := tr.Step(msg)
	_, b := tr.Step(msg)
	if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(chunks(a), []string{"a2:lo"}) {
		t.Fatalf("a %q, b %q", chunks(a), chunks(b))
	}
}

// Entries a translator never sees, such as a decision Session.Resolve
// records outside a run, do not throw its ids off: every run starts with
// run_started, which carries the sequence number.
func TestUnseenEntriesDoNotShiftIDs(t *testing.T) {
	got := chunks(translate(Options{},
		entry(1, runStarted),
		assistant(2, agentloop.StopEndTurn, "", text("one")),
		// seq 3 and 4 happen elsewhere
		entry(5, runStarted),
		delta(agentloop.TextPart, "two"),
		assistant(6, agentloop.StopEndTurn, "", text("two")),
	))
	if want := []string{"a2:one", "a6:two"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}
