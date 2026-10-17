package main

import "github.com/kirahowe/agentloop"

// A modelCall is one model call's usage and the model that served it.
type modelCall struct {
	Model string
	Usage agentloop.Usage
}

// callsOf lists the model calls among a run's entries: the assistant
// messages, which are the entries that record a call's usage.
func callsOf(entries []agentloop.Entry) []modelCall {
	var rs []modelCall
	for _, e := range entries {
		if m := e.Message; m != nil && m.Message.Role == agentloop.Assistant {
			rs = append(rs, modelCall{Model: m.Model, Usage: m.Usage})
		}
	}
	return rs
}

// runCost is what a run's requests cost in US dollars, each at the price
// of the model that served it. It is nil, never zero, when no request was
// made or any request's model has no price, because a partial sum would
// pass for the whole cost. priceOf returns nil for an unpriced model.
func runCost(rs []modelCall, priceOf func(model string) *agentloop.Prices) *float64 {
	if len(rs) == 0 {
		return nil
	}
	var total float64
	for _, r := range rs {
		p := priceOf(r.Model)
		if p == nil {
			return nil
		}
		total += p.Cost(r.Usage)
	}
	return &total
}
