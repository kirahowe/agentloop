package agentloop

import (
	"math"
	"testing"

	"pgregory.net/rapid"
)

func TestPricesCost(t *testing.T) {
	p := Prices{Input: 4, Output: 20, CacheWrite: 5, CacheRead: 0.2}
	got := p.Cost(Usage{InputTokens: 1_000_000, OutputTokens: 500_000, CacheWriteTokens: 2_000_000, CacheReadTokens: 10_000_000})
	// 4 + 10 + 10 + 2
	if math.Abs(got-26) > 1e-9 {
		t.Fatalf("got %v, want 26", got)
	}
	if got := p.Cost(Usage{}); got != 0 {
		t.Fatalf("no tokens cost %v", got)
	}
}

func drawUsage(t *rapid.T, label string) Usage {
	n := func(name string) int64 { return rapid.Int64Range(0, 1<<40).Draw(t, label+name) }
	return Usage{InputTokens: n("in"), OutputTokens: n("out"), CacheReadTokens: n("read"), CacheWriteTokens: n("write")}
}

// More tokens of any kind never cost less, no tokens cost nothing, and
// the cost of two usages is the sum of their costs.
func TestPricesCostProperties(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		f := func(name string) float64 { return rapid.Float64Range(0, 1000).Draw(t, name) }
		p := Prices{Input: f("pin"), Output: f("pout"), CacheWrite: f("pw"), CacheRead: f("pr")}
		u := drawUsage(t, "u")
		if p.Cost(Usage{}) != 0 {
			t.Fatal("zero tokens cost something")
		}
		for _, bump := range []func(*Usage){
			func(u *Usage) { u.InputTokens++ }, func(u *Usage) { u.OutputTokens++ },
			func(u *Usage) { u.CacheReadTokens++ }, func(u *Usage) { u.CacheWriteTokens++ },
		} {
			more := u
			bump(&more)
			if p.Cost(more) < p.Cost(u) {
				t.Fatalf("cost fell from %v to %v", p.Cost(u), p.Cost(more))
			}
		}
		v := drawUsage(t, "v")
		sum := u
		sum.add(v)
		a, b := p.Cost(sum), p.Cost(u)+p.Cost(v)
		if math.Abs(a-b) > 1e-9*math.Max(1, math.Abs(a)) {
			t.Fatalf("cost of sum %v, sum of costs %v", a, b)
		}
	})
}
