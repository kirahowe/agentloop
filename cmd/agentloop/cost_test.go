package main

import (
	"math"
	"testing"

	"github.com/kirahowe/agentloop"
)

func TestRunCost(t *testing.T) {
	prices := map[string]agentloop.Prices{
		"cheap": {Input: 1, Output: 5, CacheWrite: 1.25, CacheRead: 0.1},
		"dear":  {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 1},
	}
	priceOf := func(m string) *agentloop.Prices {
		if p, ok := prices[m]; ok {
			return &p
		}
		return nil
	}
	u := func(in, out, w, r int64) agentloop.Usage {
		return agentloop.Usage{InputTokens: in, OutputTokens: out, CacheWriteTokens: w, CacheReadTokens: r}
	}
	tests := []struct {
		name string
		rs   []modelCall
		want *float64
	}{
		{"no requests", nil, nil},
		{"one request, every kind of token", []modelCall{{"cheap", u(1_000_000, 1_000_000, 1_000_000, 1_000_000)}}, ptr(1 + 5 + 1.25 + 0.1)},
		{"each request at its own model's price", []modelCall{
			{"cheap", u(1_000_000, 0, 0, 0)},
			{"dear", u(0, 1_000_000, 0, 0)},
		}, ptr(1.0 + 50)},
		{"zero usage at a known price is zero", []modelCall{{"cheap", u(0, 0, 0, 0)}}, ptr(0.0)},
		{"one unpriced request leaves the cost out", []modelCall{
			{"cheap", u(100, 100, 0, 0)},
			{"mystery", u(100, 100, 0, 0)},
		}, nil},
		{"a request with no served model is unpriced", []modelCall{{"", u(100, 100, 0, 0)}}, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := runCost(tc.rs, priceOf)
			switch {
			case tc.want == nil && got != nil:
				t.Errorf("cost = %v, want none", *got)
			case tc.want != nil && got == nil:
				t.Errorf("no cost, want %v", *tc.want)
			case tc.want != nil && math.Abs(*got-*tc.want) > 1e-9:
				t.Errorf("cost = %v, want %v", *got, *tc.want)
			}
		})
	}
}

func TestCallsOfCountsAssistantMessagesOnly(t *testing.T) {
	es := []agentloop.Entry{
		{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: agentloop.Text(agentloop.User, "hi")}},
		{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: agentloop.Text(agentloop.Assistant, "yo"), Model: "m", Usage: agentloop.Usage{InputTokens: 3}}},
		{Type: agentloop.EntryRunEnded, RunEnded: &agentloop.Outcome{}},
	}
	rs := callsOf(es)
	if len(rs) != 1 || rs[0].Model != "m" || rs[0].Usage.InputTokens != 3 {
		t.Errorf("requests = %+v", rs)
	}
}
