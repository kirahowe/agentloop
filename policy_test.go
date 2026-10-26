package agentloop_test

import (
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

func TestCheckLabel(t *testing.T) {
	for _, ok := range []string{"private", "contaminated", "mail:stranger@example.com", "web", "x", "a_b.c-d/e+f"} {
		if err := agentloop.CheckLabel(ok); err != nil {
			t.Errorf("%q refused: %v", ok, err)
		}
	}
	for name, bad := range map[string]string{
		"empty":       "",
		"too long":    strings.Repeat("x", agentloop.MaxLabelBytes+1),
		"invalid":     "a\xffb",
		"newline":     "a\nb",
		"nul":         "a\x00b",
		"escape code": "\x1b[31mred",
		"capital":     "Private",
		"space":       "private ",
		"inner space": "from a mail",
		"zero width":  "private\u200b",
		"bidi":        "private\u202e",
		"lookalike":   "privatе", // Cyrillic e
		"accent":      "café",
		"leading dot": ".private",
	} {
		if agentloop.CheckLabel(bad) == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// Whatever a label is, CheckLabels agrees with CheckLabel on each one, and
// UnionLabels is sorted, has no repeats, and loses nothing.
func TestPropertyLabelHelpers(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := rapid.SliceOf(rapid.String()).Draw(t, "a")
		b := rapid.SliceOf(rapid.String()).Draw(t, "b")
		wantErr := false
		for _, l := range a {
			wantErr = wantErr || agentloop.CheckLabel(l) != nil
		}
		if got := agentloop.CheckLabels(a) != nil; got != wantErr {
			t.Fatalf("CheckLabels(%q) refused=%v, but per label %v", a, got, wantErr)
		}
		u := agentloop.UnionLabels(a, b)
		seen := map[string]bool{}
		for i, l := range u {
			if seen[l] || i > 0 && u[i-1] >= l {
				t.Fatalf("union not sorted and distinct: %q", u)
			}
			seen[l] = true
		}
		for _, l := range append(append([]string{}, a...), b...) {
			if !seen[l] {
				t.Fatalf("union %q lost %q", u, l)
			}
		}
	})
}

// Any label with the case, spacing or invisible-character variants of an
// accepted label is refused, so a forbid list cannot silently miss.
func TestPropertyLabelVariantsAreRefused(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		base := rapid.SampledFrom([]string{"private", "contaminated"}).Draw(t, "base")
		extra := rapid.SampledFrom([]string{" ", "\t", "\u200b", "\u202e", "\u00a0", "\ufeff", "\u2060"}).Draw(t, "extra")
		pos := rapid.IntRange(0, len(base)).Draw(t, "pos")
		variant := base[:pos] + extra + base[pos:]
		if agentloop.CheckLabel(variant) == nil {
			t.Fatalf("%q accepted", variant)
		}
		if agentloop.CheckLabel(strings.ToUpper(base)) == nil || agentloop.CheckLabel(strings.ToUpper(base[:1])+base[1:]) == nil {
			t.Fatalf("upper-case %q accepted", base)
		}
	})
}
