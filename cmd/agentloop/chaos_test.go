package main

// Process-level chaos tests: real `agentloop run` processes against a fake
// Anthropic server, killed at random moments, then resumed until the script
// completes. The invariants checked afterwards are the ones DESIGN.md
// promises: no call runs twice, every call has exactly one result, every
// call_started follows an allow, the log always loads.
//
//	go test ./cmd/agentloop -run Chaos -agentloop.chaos=300 -agentloop.seed=1234

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"agentloop"
)

var (
	chaosTrials = flag.Int("agentloop.chaos", 0, "number of chaos trials (0 means a default that fits a normal test run)")
	chaosSeed   = flag.Int64("agentloop.seed", 0, "chaos seed (0 means time based; it is logged)")
)

const chaosKey = "sk-chaos-key-0123456789"

func chaosCount(t *testing.T, normal, short int) int {
	if *chaosTrials > 0 {
		return *chaosTrials
	}
	if testing.Short() {
		return short
	}
	return normal
}

func chaosSeedValue(t *testing.T) int64 {
	seed := *chaosSeed
	if seed == 0 {
		seed = time.Now().UnixNano() & 0xffffffff
	}
	t.Logf("seed = %d (rerun with -agentloop.seed=%d)", seed, seed)
	return seed
}

// --- the scripted API ---

// chaosAPI replies to a request according to how many assistant turns the
// request already holds, not to how many requests it has seen. A process
// that is killed mid-request therefore costs the script nothing.
type chaosAPI struct {
	*httptest.Server
	turns [][]string // the call ids of each scripted tool-use turn
	rng   *rand.Rand
	mu    sync.Mutex

	requests   int
	violations []string
}

func newChaosAPI(t testing.TB, rng *rand.Rand, turns [][]string) *chaosAPI {
	a := &chaosAPI{turns: turns, rng: rng}
	a.Server = httptest.NewServer(http.HandlerFunc(a.handle))
	t.Cleanup(a.Close)
	return a
}

type chaosBlock struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
}

type chaosMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

func (a *chaosAPI) violate(format string, args ...any) {
	a.mu.Lock()
	a.violations = append(a.violations, fmt.Sprintf(format, args...))
	a.mu.Unlock()
}

func (a *chaosAPI) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.requests
}

func (a *chaosAPI) jitter() time.Duration {
	a.mu.Lock()
	defer a.mu.Unlock()
	return time.Duration(a.rng.Intn(12)) * time.Millisecond
}

func (a *chaosAPI) handle(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Messages []chaosMsg `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		a.violate("undecodable request: %v", err)
		http.Error(w, "bad", 400)
		return
	}
	a.mu.Lock()
	a.requests++
	a.mu.Unlock()

	// The Messages API rejects a tool_use without a tool_result, and the
	// agent must never send such a conversation.
	assistants := 0
	for i, m := range body.Messages {
		if m.Role != "assistant" {
			continue
		}
		assistants++
		var blocks []chaosBlock
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		var want []string
		for _, b := range blocks {
			if b.Type == "tool_use" {
				want = append(want, b.ID)
			}
		}
		if len(want) == 0 {
			continue
		}
		if i+1 >= len(body.Messages) || body.Messages[i+1].Role != "user" {
			a.violate("assistant message %d has tool_use %v and no following user message", i, want)
			continue
		}
		var got []chaosBlock
		_ = json.Unmarshal(body.Messages[i+1].Content, &got)
		have := map[string]int{}
		for _, b := range got {
			if b.Type == "tool_result" {
				have[b.ToolUseID]++
			}
		}
		for _, id := range want {
			if have[id] != 1 {
				a.violate("call %s has %d tool_result blocks in the next message", id, have[id])
			}
		}
	}
	if n := len(body.Messages); n == 0 || body.Messages[n-1].Role != "user" {
		a.violate("request does not end on a user message")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	fl, _ := w.(http.Flusher)
	emit := func(name, data string) {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, data)
		if fl != nil {
			fl.Flush()
		}
		time.Sleep(a.jitter())
	}
	emit("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`)
	stop := "end_turn"
	if assistants < len(a.turns) {
		stop = "tool_use"
		for i, id := range a.turns[assistants] {
			emit("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":%q,"name":"ledger","input":{}}}`, i, id))
			in, _ := json.Marshal(fmt.Sprintf(`{"n":%q}`, id))
			emit("content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":%s}}`, i, in))
			emit("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i))
		}
	} else {
		emit("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		emit("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}`)
		emit("content_block_stop", `{"type":"content_block_stop","index":0}`)
	}
	emit("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+stop+`","stop_sequence":null,"stop_details":null},"usage":{"output_tokens":7}}`)
	emit("message_stop", `{"type":"message_stop"}`)
}

// --- one trial's world ---

type chaosWorld struct {
	t       *testing.T
	rng     *rand.Rand
	dir     string
	session string
	ledger  string
	pids    string
	api     *chaosAPI
	turns   [][]string
	req     map[string]any
	reqPath string
}

const ledgerScript = `#!/bin/sh
in=$(cat)
echo $$ >> "$PIDS"
printf '%s\n' "$in" >> "$LEDGER"
sleep "0.0$(( $$ % 6 + 1 ))"
echo ok
`

func newChaosWorld(t *testing.T, rng *rand.Rand) *chaosWorld {
	w := &chaosWorld{t: t, rng: rng, dir: t.TempDir()}
	n := 2 + rng.Intn(4)
	id := 0
	for k := 0; k < n; k++ {
		var calls []string
		for j := 0; j <= rng.Intn(3); j++ {
			id++
			calls = append(calls, fmt.Sprintf("toolu_%d_%d", k, id))
		}
		w.turns = append(w.turns, calls)
	}
	w.api = newChaosAPI(t, rand.New(rand.NewSource(rng.Int63())), w.turns)
	w.session = filepath.Join(w.dir, "s", "session.jsonl")
	w.ledger = filepath.Join(w.dir, "ledger")
	w.pids = filepath.Join(w.dir, "pids")
	script := filepath.Join(w.dir, "tool.sh")
	if err := os.WriteFile(script, []byte(ledgerScript), 0o755); err != nil {
		t.Fatal(err)
	}
	w.req = map[string]any{
		"provider":        "anthropic",
		"model":           "claude-test",
		"base_url":        w.api.URL,
		"session":         w.session,
		"api_key_command": []string{"printf", "%s", chaosKey},
		"tools": []map[string]any{{
			"name": "ledger", "description": "append to the ledger",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "string"}}},
		}},
		"tool_command": []string{"/bin/sh", script},
		"tool_env":     map[string]string{"LEDGER": w.ledger, "PIDS": w.pids, "PATH": "/usr/bin:/bin"},
		"tool_dir":     w.dir,
		"retry":        map[string]any{"retries": 0},
	}
	return w
}

func (w *chaosWorld) totalCalls() int {
	n := 0
	for _, c := range w.turns {
		n += len(c)
	}
	return n
}

func (w *chaosWorld) write(prompt string, extra map[string]any) string {
	req := map[string]any{}
	for k, v := range w.req {
		req[k] = v
	}
	if prompt != "" {
		req["prompt"] = prompt
	}
	for k, v := range extra {
		req[k] = v
	}
	b, _ := json.Marshal(req)
	w.reqPath = filepath.Join(w.dir, "request.json")
	if err := os.WriteFile(w.reqPath, b, 0o644); err != nil {
		w.t.Fatal(err)
	}
	return w.reqPath
}

// sessionFile returns the raw lines of the session file, if it exists.
func (w *chaosWorld) lines() [][]byte {
	raw, _ := os.ReadFile(w.session)
	var out [][]byte
	for _, l := range bytes.Split(raw, []byte("\n")) {
		if len(bytes.TrimSpace(l)) > 0 {
			out = append(out, l)
		}
	}
	return out
}

func (w *chaosWorld) hasUserMessage() bool {
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(w.session))
	if err != nil {
		w.t.Fatalf("session does not load: %v", err)
	}
	return len(sess.Messages()) > 0
}

// --- running the binary ---

type chaosRun struct {
	stdout, stderr string
	code           int
	signal         syscall.Signal // set when the process died of a signal
	killed         bool           // the test killed it
	elapsed        time.Duration
}

func (r chaosRun) outputLines() [][]byte {
	var out [][]byte
	for _, l := range bytes.Split([]byte(r.stdout), []byte("\n")) {
		if len(l) > 0 {
			out = append(out, l)
		}
	}
	return out
}

// chaosStart runs the binary. after, if non-nil, is called once the process
// has started and may signal it; wait is how long the process may take.
func chaosExec(t *testing.T, args []string, env []string, after func(*os.Process)) chaosRun {
	t.Helper()
	return chaosExecIn(t, "", args, env, after)
}

func chaosExecIn(t *testing.T, dir string, args []string, env []string, after ...func(*os.Process)) chaosRun {
	t.Helper()
	cmd := exec.Command(binary, args...)
	cmd.Env = env
	cmd.Dir = dir
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	start := time.Now()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	for _, f := range after {
		if f != nil {
			f(cmd.Process)
		}
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var err error
	select {
	case err = <-done:
	case <-time.After(90 * time.Second):
		cmd.Process.Kill()
		<-done
		t.Fatalf("agentloop did not exit in 90s; stdout %q stderr %q", out.String(), errb.String())
	}
	r := chaosRun{stdout: out.String(), stderr: errb.String(), elapsed: time.Since(start)}
	if ee, ok := err.(*exec.ExitError); ok {
		r.code = ee.ExitCode()
		if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
			r.signal = ws.Signal()
		}
	} else if err != nil {
		t.Fatal(err)
	}
	return r
}

func chaosEnv() []string {
	return []string{"PATH=/usr/bin:/bin", "HOME=/nonexistent-home", "LEAK_CANARY=parent-env"}
}

// run starts the binary and optionally kills it after delay with sig.
func (w *chaosWorld) run(prompt string, extra map[string]any, delay time.Duration, sig syscall.Signal) chaosRun {
	path := w.write(prompt, extra)
	var after func(*os.Process)
	if sig != 0 {
		after = func(p *os.Process) {
			time.Sleep(delay)
			p.Signal(sig)
		}
	}
	r := chaosExec(w.t, []string{"run", path}, chaosEnv(), after)
	r.killed = sig != 0 && r.signal == sig
	return r
}

// --- invariants ---

// checkStdout verifies the output contract of one run against the session
// file. before is the number of lines the file had when the run started.
func (w *chaosWorld) checkStdout(r chaosRun, before int) {
	t := w.t
	t.Helper()
	lines := w.lines()
	var seqs []int
	outcomeAt := -1
	out := r.outputLines()
	for i, l := range out {
		var m map[string]any
		if err := json.Unmarshal(l, &m); err != nil {
			t.Fatalf("stdout line %d is not JSON: %q", i, l)
		}
		typ, _ := m["type"].(string)
		if typ == "outcome" {
			if i != len(out)-1 {
				t.Fatalf("outcome line is not last (line %d of %d)", i, len(out))
			}
			outcomeAt = i
			continue
		}
		seq := int(m["seq"].(float64))
		seqs = append(seqs, seq)
		if len(seqs) > 1 && seq != seqs[len(seqs)-2]+1 {
			t.Fatalf("seq jumps from %d to %d", seqs[len(seqs)-2], seq)
		}
		// An entry on stdout was appended first: it is in the file, verbatim.
		if seq < 1 || seq > len(lines) || !bytes.Equal(lines[seq-1], l) {
			t.Fatalf("stdout entry seq %d differs from the session file\nstdout: %s\nfile:   %s", seq, l, lineOrNone(lines, seq))
		}
	}
	if len(seqs) > 0 && seqs[0] != before+1 {
		t.Fatalf("first stdout entry has seq %d, the file had %d lines before the run", seqs[0], before)
	}
	if r.killed {
		return
	}
	// A run that ended on its own wrote every entry it appended, then the outcome.
	if outcomeAt < 0 {
		t.Fatalf("no outcome line; code %d signal %v stdout %q stderr %q", r.code, r.signal, r.stdout, r.stderr)
	}
	if len(seqs) != len(lines)-before {
		t.Fatalf("stdout has %d entries, the run appended %d", len(seqs), len(lines)-before)
	}
}

func lineOrNone(lines [][]byte, seq int) string {
	if seq < 1 || seq > len(lines) {
		return "(none)"
	}
	return string(lines[seq-1])
}

// checkSession verifies the log invariants. final is true once the script
// should be complete.
func (w *chaosWorld) checkSession(final bool) {
	t := w.t
	t.Helper()
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(w.session))
	if err != nil {
		t.Fatalf("session does not load: %v", err)
	}
	entries := sess.Entries()
	decided := map[string]int{} // call id -> seq of the allow decision
	started := map[string]int{}
	results := map[string]int{}
	for i, e := range entries {
		if e.Seq != i+1 {
			t.Fatalf("entry %d has seq %d", i+1, e.Seq)
		}
		switch e.Type {
		case agentloop.EntryDecision:
			if e.Decision.Verdict == agentloop.Allow {
				decided[e.Decision.CallID] = e.Seq
			}
		case agentloop.EntryCallStarted:
			id := e.CallStarted.CallID
			if d, ok := decided[id]; !ok || d > e.Seq {
				t.Fatalf("call %s started at seq %d without an earlier allow", id, e.Seq)
			}
			started[id]++
			if started[id] > 1 {
				t.Fatalf("call %s started twice", id)
			}
		case agentloop.EntryToolResult:
			results[e.ToolResult.Result.CallID]++
		}
	}
	for id, n := range results {
		if n != 1 {
			t.Fatalf("call %s has %d results", id, n)
		}
	}
	// The ledger: every call at most once, and never without call_started.
	seen := map[string]int{}
	if raw, err := os.ReadFile(w.ledger); err == nil {
		for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
			if l == "" {
				continue
			}
			var in struct{ N string }
			if err := json.Unmarshal([]byte(l), &in); err != nil {
				t.Fatalf("ledger line %q: %v", l, err)
			}
			seen[in.N]++
			if seen[in.N] > 1 {
				t.Fatalf("call %s executed %d times", in.N, seen[in.N])
			}
			if started[in.N] == 0 {
				t.Fatalf("call %s executed but the log has no call_started for it", in.N)
			}
		}
	}
	// Every call in the derived conversation has exactly one result.
	callsInTurns := 0
	for _, turn := range sess.Turns() {
		for _, id := range turn.Calls {
			callsInTurns++
			info, _ := sess.Call(id)
			if final && !info.Finished {
				t.Fatalf("call %s has no result in a finished session", id)
			}
		}
	}
	resultParts := 0
	for _, m := range sess.Messages() {
		for _, p := range m.Parts {
			if p.Type == agentloop.ToolResultPart {
				resultParts++
			}
		}
	}
	if len(sess.Pending()) == 0 && resultParts != callsInTurns {
		t.Fatalf("conversation has %d tool results for %d calls", resultParts, callsInTurns)
	}
	if final {
		if len(sess.Pending()) != 0 {
			t.Fatalf("pending after completion: %v", sess.Pending())
		}
		if callsInTurns != w.totalCalls() {
			t.Fatalf("session holds %d calls, the script has %d", callsInTurns, w.totalCalls())
		}
		last := entries[len(entries)-1]
		if last.Type != agentloop.EntryRunEnded || last.RunEnded.Reason != agentloop.ReasonCompleted {
			t.Fatalf("last entry is %s", last.Type)
		}
		// Every line of a finished log is whole: nothing torn survived.
		for i, l := range w.lines() {
			if !json.Valid(l) {
				t.Fatalf("line %d of the final log is not JSON: %q", i+1, l)
			}
		}
	}
	w.api.mu.Lock()
	v := append([]string{}, w.api.violations...)
	w.api.mu.Unlock()
	if len(v) > 0 {
		t.Fatalf("the agent sent an invalid conversation: %v", v)
	}
}

// waitOrphans checks that every tool process the ledger script recorded has
// gone, which a SIGKILL of agentloop itself cannot guarantee at once (the
// tool runs in its own process group) but its short sleep ensures soon.
func (w *chaosWorld) waitOrphans() {
	raw, _ := os.ReadFile(w.pids)
	deadline := time.Now().Add(5 * time.Second)
	for _, f := range strings.Fields(string(raw)) {
		pid, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		for syscall.Kill(pid, 0) == nil && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if syscall.Kill(pid, 0) == nil {
			w.t.Errorf("tool process %d still alive", pid)
		}
	}
}

// drive resumes the session until the script completes, killing some runs.
func (w *chaosWorld) drive(maxKills int, killSig syscall.Signal, stats *chaosStats) {
	t := w.t
	t.Helper()
	kills := w.rng.Intn(maxKills + 1)
	for attempt := 0; attempt < kills+30; attempt++ {
		prompt := ""
		if !w.hasUserMessage() || w.rng.Intn(6) == 0 && attempt == 0 {
			prompt = "go"
		}
		before := len(w.entries())
		var r chaosRun
		if attempt < kills {
			// Skewed towards early moments, where the writes are dense.
			d := time.Duration(w.rng.Float64() * w.rng.Float64() * 400 * float64(time.Millisecond))
			r = w.run(prompt, nil, d, killSig)
			atomic.AddInt64(&stats.runs, 1)
			if r.killed {
				atomic.AddInt64(&stats.kills, 1)
			}
		} else {
			r = w.run(prompt, nil, 0, 0)
			atomic.AddInt64(&stats.runs, 1)
		}
		w.checkStdout(r, before)
		w.checkSession(false)
		if r.killed {
			continue
		}
		if r.code != 0 {
			t.Fatalf("exit %d; stderr: %s\nstdout: %s", r.code, r.stderr, r.stdout)
		}
		lines := r.outputLines()
		var out map[string]any
		json.Unmarshal(lines[len(lines)-1], &out)
		if out["reason"] == "completed" && out["text"] == "done" {
			w.checkSession(true)
			w.waitOrphans()
			return
		}
		t.Fatalf("an uninterrupted run ended with %v", out)
	}
	t.Fatal("never completed")
}

type chaosStats struct{ trials, runs, kills int64 }

// --- the tests ---

func TestChaosSIGKILL(t *testing.T) {
	n := chaosCount(t, 60, 8)
	seed := chaosSeedValue(t)
	var stats chaosStats
	t.Run("trials", func(t *testing.T) {
		for i := 0; i < n; i++ {
			i := i
			t.Run(fmt.Sprintf("trial%03d", i), func(t *testing.T) {
				t.Parallel()
				rng := rand.New(rand.NewSource(seed + int64(i)*7919))
				w := newChaosWorld(t, rng)
				w.drive(6, syscall.SIGKILL, &stats)
				atomic.AddInt64(&stats.trials, 1)
			})
		}
	})
	t.Logf("SIGKILL chaos: %d trials, %d process runs, %d runs actually killed", stats.trials, stats.runs, stats.kills)
}

// TestChaosTornTail leaves a partly written last line after a kill, which is
// what a power cut during an append leaves (a SIGKILL cannot split a single
// write call), and resumes. The log must load, the torn line must go, and
// nothing may run twice.
func TestChaosTornTail(t *testing.T) {
	n := chaosCount(t, 20, 4)
	seed := chaosSeedValue(t)
	for i := 0; i < n; i++ {
		i := i
		t.Run(fmt.Sprintf("trial%03d", i), func(t *testing.T) {
			t.Parallel()
			rng := rand.New(rand.NewSource(seed + int64(i)*104729))
			w := newChaosWorld(t, rng)
			w.run("go", nil, time.Duration(20+rng.Intn(200))*time.Millisecond, syscall.SIGKILL)
			if len(w.lines()) == 0 {
				t.Skip("killed before anything was written")
			}
			full := []byte(`{"seq":99,"time":"2026-10-14T10:00:00Z","type":"tool_result","tool_result":{"result":{"call_id":"x","content":"partial"}}}`)
			torn := full[:rng.Intn(len(full))]
			f, err := os.OpenFile(w.session, os.O_WRONLY|os.O_APPEND, 0)
			if err != nil {
				t.Fatal(err)
			}
			f.Write(torn)
			f.Close()
			w.checkSession(false)
			var stats chaosStats
			w.drive(2, syscall.SIGKILL, &stats)
		})
	}
}

// TestChaosSignals sends SIGTERM, or sets a short timeout, at random moments.
// Either is cancellation: the outcome line is printed, the exit is 0, the
// log stays valid, and the session resumes.
func TestChaosSignals(t *testing.T) {
	n := chaosCount(t, 40, 6)
	seed := chaosSeedValue(t)
	var term, timeouts, canceled, early, completedFirst int64
	t.Run("trials", func(t *testing.T) {
		for i := 0; i < n; i++ {
			i := i
			t.Run(fmt.Sprintf("trial%03d", i), func(t *testing.T) {
				t.Parallel()
				rng := rand.New(rand.NewSource(seed + int64(i)*15485863))
				w := newChaosWorld(t, rng)
				d := time.Duration(rng.Float64() * rng.Float64() * 400 * float64(time.Millisecond))
				var r chaosRun
				if rng.Intn(2) == 0 {
					atomic.AddInt64(&term, 1)
					var sig syscall.Signal = syscall.SIGTERM
					if rng.Intn(3) == 0 {
						sig = syscall.SIGINT
					}
					r = w.run("go", nil, d, sig)
					if r.signal == sig {
						// The signal beat the handler: the process was still
						// starting up. That can only happen in the first
						// moments of the process's life.
						atomic.AddInt64(&early, 1)
						if d > 40*time.Millisecond {
							t.Fatalf("SIGTERM after %v killed the process; stdout %q", d, r.stdout)
						}
						w.checkStdout(r, 0)
						w.checkSession(false)
						var stats chaosStats
						w.drive(0, syscall.SIGKILL, &stats)
						return
					}
				} else {
					atomic.AddInt64(&timeouts, 1)
					r = w.run("go", map[string]any{"timeout_seconds": d.Seconds() + 0.001}, 0, 0)
				}
				if r.code != 0 {
					t.Fatalf("exit %d (signal %v) after %v; stderr %q stdout %q", r.code, r.signal, d, r.stderr, r.stdout)
				}
				w.checkStdout(r, 0)
				w.checkSession(false)
				lines := r.outputLines()
				var out map[string]any
				json.Unmarshal(lines[len(lines)-1], &out)
				switch out["reason"] {
				case "canceled":
					atomic.AddInt64(&canceled, 1)
					ents := w.entries()
					if last := ents[len(ents)-1]; last.Type != agentloop.EntryRunEnded || last.RunEnded.Reason != agentloop.ReasonCanceled {
						t.Fatalf("canceled run's log ends with %s", last.Type)
					}
				case "completed":
					atomic.AddInt64(&completedFirst, 1)
				default:
					t.Fatalf("cancellation produced outcome %v", out)
				}
				// Resume to completion, with nothing interrupting.
				var stats chaosStats
				w.drive(0, syscall.SIGKILL, &stats)
			})
		}
	})
	t.Logf("signals: %d SIGTERM/SIGINT trials, %d timeout trials; %d canceled, %d finished first, %d signals hit the startup window",
		term, timeouts, canceled, completedFirst, early)
}

func (w *chaosWorld) entries() []agentloop.Entry {
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(w.session))
	if err != nil {
		w.t.Fatal(err)
	}
	return sess.Entries()
}

// TestOutputContract checks stdout of an uninterrupted run in detail.
func TestOutputContract(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	w := newChaosWorld(t, rng)
	r := w.run("go", nil, 0, 0)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	w.checkStdout(r, 0)
	w.checkSession(true)
	// A second run on the finished session appends only its own entries.
	before := len(w.entries())
	r = w.run("", nil, 0, 0)
	if r.code != 0 {
		t.Fatalf("resume of a finished session: exit %d: %s", r.code, r.stderr)
	}
	w.checkStdout(r, before)
	// stdout carries only entries and the outcome: stderr is free text.
	for _, l := range r.outputLines() {
		if !strings.HasPrefix(string(l), "{") {
			t.Errorf("stdout line %q", l)
		}
	}
}

// TestResumeWhenFinalMessageWasLogged: a crash between the model's final
// message and run_ended leaves a session whose last entry is a finished
// assistant message. Running it again with no prompt is a no-op that
// completes, not an unusable request.
func TestResumeWhenFinalMessageWasLogged(t *testing.T) {
	w := newChaosWorld(t, rand.New(rand.NewSource(2)))
	if r := w.run("go", nil, 0, 0); r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	lines := w.lines()
	// Drop run_ended, as a kill just before it would.
	if err := os.WriteFile(w.session, append(bytes.Join(lines[:len(lines)-1], []byte("\n")), '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	requestsBefore := w.api.count()
	r := w.run("", nil, 0, 0)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	out := r.outputLines()
	var o map[string]any
	json.Unmarshal(out[len(out)-1], &o)
	if o["reason"] != "completed" || o["text"] != "done" {
		t.Errorf("outcome = %v", o)
	}
	if n := w.api.count(); n != requestsBefore {
		t.Errorf("a finished session made %d model requests", n-requestsBefore)
	}
}

// TestSecondProcessOnSameSession: a second agentloop on a session that a
// live one is working on must be refused. Both would run the same pending
// calls and log over each other's entries.
func TestSecondProcessOnSameSession(t *testing.T) {
	w := newChaosWorld(t, rand.New(rand.NewSource(3)))
	path := w.write("go", nil)
	first := make(chan chaosRun, 1)
	go func() { first <- chaosExec(t, []string{"run", path}, chaosEnv(), nil) }()
	deadline := time.Now().Add(10 * time.Second)
	for len(w.lines()) < 3 && time.Now().Before(deadline) {
		time.Sleep(2 * time.Millisecond)
	}
	second := chaosExec(t, []string{"run", path}, chaosEnv(), nil)
	r1 := <-first
	if r1.code != 0 {
		t.Fatalf("first process: exit %d: %s", r1.code, r1.stderr)
	}
	// The first run lasts well over 100ms; if the second finished before it,
	// it was refused while the first was alive.
	if second.code == 0 {
		t.Logf("second process was not refused (the first may have finished): elapsed %v", second.elapsed)
		if second.elapsed < r1.elapsed/2 {
			t.Errorf("a second process ran to an outcome while the first was working")
		}
	} else if second.code != 1 || !strings.Contains(second.stderr, "in use") || second.stdout != "" {
		t.Errorf("second process: exit %d, stderr %q, stdout %q", second.code, second.stderr, second.stdout)
	}
	w.checkSession(false)
}

// TestTinyToolTimeoutIsATimeout: a tool timeout too small to be a whole
// nanosecond count is still a timeout, not "no timeout".
func TestTinyToolTimeoutIsATimeout(t *testing.T) {
	w := newChaosWorld(t, rand.New(rand.NewSource(4)))
	w.turns = [][]string{{"toolu_t_1"}}
	w.api.turns = w.turns
	r := w.run("go", map[string]any{"tool_timeout_seconds": 1e-10}, 0, 0)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	if !strings.Contains(r.stdout, "timed out") {
		t.Errorf("the call was not timed out: %s", r.stdout)
	}
}
