package agentloop_test

import (
	"errors"
	"maps"
	"reflect"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// A run records the pins the session lacks, once, and a run whose pin
// differs from the session's does not start and writes nothing, also
// after the session is reopened.
func TestPinsAreKeptAcrossRuns(t *testing.T) {
	s, store := newSession(t)
	a := &agentloop.Agent{Model: fake.Script(fake.Text("hi"), fake.Text("again")), Pins: map[string]string{"mail": "aa"}}
	run(t, a, s, "go")
	var pins []agentloop.Entry
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryPins {
			pins = append(pins, e)
		}
	}
	if len(pins) != 1 || !reflect.DeepEqual(pins[0].Pins.Pins, map[string]string{"mail": "aa"}) {
		t.Fatalf("pins entries %+v", pins)
	}
	// The same pins again write nothing new.
	s = reopen(t, store)
	before := len(s.Entries())
	run(t, a, s, "more")
	for _, e := range s.Entries()[before:] {
		if e.Type == agentloop.EntryPins {
			t.Fatal("the same pin was written twice")
		}
	}
	if !reflect.DeepEqual(s.Pins(), map[string]string{"mail": "aa"}) {
		t.Fatalf("pins %v", s.Pins())
	}

	s = reopen(t, store)
	before = len(s.Entries())
	b := &agentloop.Agent{Model: noModel(t), Pins: map[string]string{"mail": "bb"}}
	_, err := b.Run(t.Context(), s, "go")
	var pe *agentloop.PinError
	if !errors.As(err, &pe) || pe.Tool != "mail" || pe.Pinned != "aa" || pe.Got != "bb" {
		t.Fatalf("err %v", err)
	}
	if len(reopen(t, store).Entries()) != before {
		t.Fatal("a refused run wrote to the session")
	}

	// A new tool's pin is added beside the old ones.
	c := &agentloop.Agent{Model: fake.Script(fake.Text("ok")), Pins: map[string]string{"mail": "aa", "web": "cc"}}
	run(t, c, reopen(t, store), "go")
	if got := reopen(t, store).Pins(); !reflect.DeepEqual(got, map[string]string{"mail": "aa", "web": "cc"}) {
		t.Fatalf("pins %v", got)
	}
}

// A log that pins one tool twice with different digests does not open.
func TestALogWithAChangedPinIsRefused(t *testing.T) {
	e := func(seq int, d string) agentloop.Entry {
		return agentloop.Entry{Seq: seq, Type: agentloop.EntryPins, Pins: &agentloop.PinsEntry{Pins: map[string]string{"t": d}}}
	}
	if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(e(1, "a"), e(2, "a"))); err != nil {
		t.Fatalf("the same pin twice: %v", err)
	}
	if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(e(1, "a"), e(2, "b"))); err == nil {
		t.Fatal("a changed pin opened")
	}
	empty := agentloop.Entry{Seq: 1, Type: agentloop.EntryPins, Pins: &agentloop.PinsEntry{}}
	if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(empty)); err == nil {
		t.Fatal("an empty pins entry opened")
	}
}

// Whatever runs are made, the session's pins are the first digest each
// tool was given, and a run is refused exactly when it names another.
func TestPropertyPinsNeverChange(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		store := agentloop.NewMemoryStore()
		want := map[string]string{}
		for i := range rapid.IntRange(1, 6).Draw(t, "runs") {
			pins := rapid.MapOf(rapid.SampledFrom([]string{"a", "b", "c"}), rapid.SampledFrom([]string{"1", "2"})).Draw(t, "pins")
			s, err := agentloop.OpenSession(store)
			if err != nil {
				t.Fatal(err)
			}
			conflict := false
			for k, v := range pins {
				if old, ok := want[k]; ok && old != v {
					conflict = true
				}
			}
			a := &agentloop.Agent{Model: fake.Script(fake.Text("ok")), Pins: pins}
			_, err = a.Run(t.Context(), s, "go")
			var pe *agentloop.PinError
			if conflict != errors.As(err, &pe) || (!conflict && err != nil) {
				t.Fatalf("run %d: conflict %v, err %v", i, conflict, err)
			}
			if !conflict {
				for k, v := range pins {
					if _, ok := want[k]; !ok {
						want[k] = v
					}
				}
			}
			if got := s.Pins(); !maps.Equal(got, want) {
				t.Fatalf("pins %v, want %v", got, want)
			}
		}
	})
}
