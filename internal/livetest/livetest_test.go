package livetest

import (
	"reflect"
	"strings"
	"testing"
	"testing/quick"
)

func TestSplit(t *testing.T) {
	got, err := Split("  security find-generic-password\t-s relay-anthropic -w ")
	want := []string{"security", "find-generic-password", "-s", "relay-anthropic", "-w"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestSplitEmptyIsAnError(t *testing.T) {
	for _, c := range []string{"", " ", "\t\n"} {
		if _, err := Split(c); err == nil {
			t.Errorf("Split(%q) gave no error", c)
		}
	}
}

// Joining the words of any command back with single spaces and splitting
// again gives the same words.
func TestSplitRoundTrip(t *testing.T) {
	f := func(words []string) bool {
		var clean []string
		for _, w := range words {
			clean = append(clean, strings.Fields(w)...)
		}
		if len(clean) == 0 {
			return true
		}
		got, err := Split(strings.Join(clean, " "))
		return err == nil && reflect.DeepEqual(got, clean)
	}
	if err := quick.Check(f, nil); err != nil {
		t.Fatal(err)
	}
}
