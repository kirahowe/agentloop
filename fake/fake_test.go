package fake_test

import (
	"context"
	"errors"
	"testing"

	"agentloop"
	"agentloop/fake"
)

func TestScript(t *testing.T) {
	boom := errors.New("boom")
	m := fake.Script(fake.Text("one"), boom, fake.ToolUse("calling", fake.Call("a", "t", "")))
	var deltas []agentloop.Delta
	req := &agentloop.Request{Messages: []agentloop.Message{agentloop.Text(agentloop.User, "hi")}}

	resp, err := m.Generate(context.Background(), req, func(d agentloop.Delta) { deltas = append(deltas, d) })
	if err != nil || resp.Message.Text() != "one" || resp.Stop != agentloop.StopEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	if resp.Message.Native == nil || resp.Message.Native.Provider != "fake" || len(resp.Message.Native.Data) == 0 {
		t.Fatalf("native %+v", resp.Message.Native)
	}
	if len(deltas) != 1 || deltas[0].Text != "one" {
		t.Fatalf("deltas %+v", deltas)
	}
	if _, err := m.Generate(context.Background(), req, nil); err != boom {
		t.Fatalf("err = %v", err)
	}
	resp, err = m.Generate(context.Background(), req, nil)
	if err != nil || resp.Stop != agentloop.StopToolUse || string(resp.Message.ToolCalls()[0].Input) != "{}" {
		t.Fatalf("%+v, %v", resp, err)
	}
	if _, err := m.Generate(context.Background(), req, nil); err == nil {
		t.Fatal("a request past the script succeeded")
	}
	if m.Provider() != "fake" || len(m.Requests()) != 4 {
		t.Fatalf("provider %q, %d requests", m.Provider(), len(m.Requests()))
	}
}

func TestCancelledContext(t *testing.T) {
	m := fake.Script(fake.Text("x"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate(ctx, &agentloop.Request{}, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}
