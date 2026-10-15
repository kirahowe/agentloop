package anthropic

import (
	"regexp"

	"github.com/kirahowe/agentloop"
)

// What each model costs, in US dollars per million tokens, from
// Anthropic's published rates (checked 2026-10-06). Cache writes are the
// 5-minute rate, 1.25 times the input price. Cache reads are 0.1 times
// the input price, except on Opus 5.5 ($0.20, 0.05 times) and on Fable
// 5.1 and Mythos 5.1 ($0.25, 0.025 times). Haiku 5.5 is priced for
// prompts up to 100K tokens; Anthropic charges more beyond that, so the
// cost of such a call comes out low. Models not listed are unpriced, and
// the ACP layer then reports no cost.
var prices = map[string]agentloop.Prices{
	"claude-fable-5-1":  {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 0.25},
	"claude-mythos-5-1": {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 0.25},
	"claude-fable-5":    {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 1},
	"claude-mythos-5":   {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 1},
	"claude-opus-5-5":   {Input: 4, Output: 20, CacheWrite: 5, CacheRead: 0.2},
	"claude-opus-5":     {Input: 5, Output: 25, CacheWrite: 6.25, CacheRead: 0.5},
	"claude-opus-4-8":   {Input: 5, Output: 25, CacheWrite: 6.25, CacheRead: 0.5},
	"claude-opus-4-7":   {Input: 5, Output: 25, CacheWrite: 6.25, CacheRead: 0.5},
	"claude-opus-4-6":   {Input: 5, Output: 25, CacheWrite: 6.25, CacheRead: 0.5},
	"claude-sonnet-5-5": {Input: 2, Output: 10, CacheWrite: 2.5, CacheRead: 0.2},
	"claude-sonnet-5":   {Input: 2, Output: 10, CacheWrite: 2.5, CacheRead: 0.2},
	"claude-sonnet-4-6": {Input: 3, Output: 15, CacheWrite: 3.75, CacheRead: 0.3},
	"claude-haiku-5-5":  {Input: 0.1, Output: 0.5, CacheWrite: 0.125, CacheRead: 0.01},
	"claude-haiku-4-5":  {Input: 1, Output: 5, CacheWrite: 1.25, CacheRead: 0.1},
}

var dateSuffix = regexp.MustCompile(`-\d{8}$`)

// PricesFor is the price of a model by id, or nil when it is not one this
// package knows. A dated snapshot id costs what its family costs.
func PricesFor(model string) *agentloop.Prices {
	p, ok := prices[dateSuffix.ReplaceAllString(model, "")]
	if !ok {
		return nil
	}
	return &p
}
