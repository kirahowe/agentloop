package agentloop_test

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop/internal/scenario"
)

func runProperty(t *testing.T, f scenario.Faults) {
	rapid.Check(t, func(t *rapid.T) {
		cfg := scenario.GenConfig(t)
		scenario.New(t, cfg, f).Execute()
	})
}

// TestPropertyCore covers properties 1 to 8 and 11 over random scripts,
// tools, policies, limits and retry policies, with no faults injected.
func TestPropertyCore(t *testing.T) { runProperty(t, scenario.Faults{}) }

// TestPropertyNativeIsStable runs the same with provider records that carry
// whitespace, HTML characters and escapes, which JSON does not keep
// byte for byte.
func TestPropertyNativeIsStable(t *testing.T) { runProperty(t, scenario.Faults{Wild: true}) }

// TestPropertyCrashResume is property 9: the store dies on the Nth append,
// before or after the line is durable, and a fresh session and agent resume
// from what was persisted, repeatedly.
func TestPropertyCrashResume(t *testing.T) { runProperty(t, scenario.Faults{Crash: true}) }

// TestPropertyCancel is property 10: cancellation at random points.
func TestPropertyCancel(t *testing.T) { runProperty(t, scenario.Faults{Cancel: true}) }

// TestPropertyChaos has every fault at once.
func TestPropertyChaos(t *testing.T) {
	runProperty(t, scenario.Faults{Crash: true, Cancel: true, Wild: true})
}
