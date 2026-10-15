package anthropic

import "testing"

func TestPricesFor(t *testing.T) {
	p := PricesFor("claude-opus-5-5")
	if p == nil || p.Input != 4 || p.Output != 20 || p.CacheWrite != 5 || p.CacheRead != 0.2 {
		t.Fatalf("%+v", p)
	}
	if d := PricesFor("claude-haiku-4-5-20251001"); d == nil || *d != *PricesFor("claude-haiku-4-5") {
		t.Fatalf("a dated snapshot should cost what its family costs: %+v", d)
	}
	for _, id := range []string{"", "gpt-5", "claude-unknown-9"} {
		if PricesFor(id) != nil {
			t.Fatalf("%q should be unpriced", id)
		}
	}
}

// Every listed model has a price for every kind of token, and a cache read
// is never dearer than input nor a cache write cheaper.
func TestPriceTableIsSane(t *testing.T) {
	for id, p := range prices {
		if p.Input <= 0 || p.Output < p.Input || p.CacheRead <= 0 || p.CacheRead >= p.Input || p.CacheWrite < p.Input {
			t.Errorf("%s: %+v", id, p)
		}
	}
}
