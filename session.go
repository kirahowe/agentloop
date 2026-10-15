package agentloop

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"sync"
	"time"
)

// A Session is a conversation and everything decided in it. It is the
// entries of a Store and nothing else: the conversation, the labels, the
// usage and the state of every tool call are derived from them, so a
// session reopened from its store is the session that was closed.
type Session struct {
	mu      sync.Mutex
	store   Store
	running bool // a Run is in progress

	entries  []Entry
	messages []Message
	labels   map[string]bool
	usage    Usage
	calls    map[string]*callState
	turns    []Turn
	decided  int
}

// A Turn is one assistant message and how it ended.
type Turn struct {
	// Message is the turn's index in Messages.
	Message int
	Stop    Stop
	// Model is the model that served the turn.
	Model string
	Usage Usage
	// Calls lists the ids of the turn's tool calls, in order.
	Calls []string
}

// CallInfo is what the session knows about one tool call.
type CallInfo struct {
	Call ToolCall
	// Decision is the latest decision on the call, nil until it is checked.
	Decision *Decision
	Started  bool
	Finished bool
	// Result is set once Finished.
	Result *ToolResult
}

// Held reports whether the call waits for a person.
func (c CallInfo) Held() bool {
	return !c.Finished && c.Decision != nil && c.Decision.Verdict == Hold
}

type callState struct {
	info  CallInfo
	turn  int
	index int // position among the turn's calls
}

// OpenSession rebuilds a session from the store. An empty store makes a
// new session.
func OpenSession(store Store) (*Session, error) {
	entries, err := store.Load()
	if err != nil {
		return nil, err
	}
	s := &Session{
		store:  store,
		labels: map[string]bool{},
		calls:  map[string]*callState{},
	}
	for i, e := range entries {
		if e.Seq != i+1 {
			return nil, fmt.Errorf("the log is out of sequence: entry %d has seq %d, want %d", i+1, e.Seq, i+1)
		}
		if err := s.apply(e); err != nil {
			return nil, fmt.Errorf("entry %d: %w", i+1, err)
		}
	}
	return s, nil
}

// Messages is the conversation. Results that follow an assistant turn are
// gathered into one user message, in the order of the calls. While calls
// are held or pending the conversation is not yet one a model can take.
func (s *Session) Messages() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.messages)
}

// Labels lists the session's labels, sorted.
func (s *Session) Labels() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sortedLabels()
}

// Usage is everything the session's model calls cost.
func (s *Session) Usage() Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage
}

// Entries is the log.
func (s *Session) Entries() []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.entries)
}

// View is the session as a policy sees it.
func (s *Session) View() View {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.view()
}

// Turns lists the assistant turns, with the stop of each.
func (s *Session) Turns() []Turn {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.turns)
}

// Call returns the state of one call.
func (s *Session) Call(id string) (CallInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.calls[id]
	if !ok {
		return CallInfo{}, false
	}
	return st.info, true
}

// Held lists the calls waiting for a person, in order.
func (s *Session) Held() []ToolCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	var held []ToolCall
	for _, t := range s.turns {
		for _, id := range t.Calls {
			if info := s.calls[id].info; info.Held() {
				held = append(held, info.Call)
			}
		}
	}
	return held
}

// Pending lists the calls of the last assistant turn that have no result
// yet, in order. A resumed run settles exactly these.
func (s *Session) Pending() []ToolCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pending()
}

// Resolve records a person's decision on a held call. The next Run
// carries it out. The verdict must be allow or deny, and the call must be
// held.
func (s *Session) Resolve(callID string, d Decision) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.calls[callID]
	if !ok {
		return fmt.Errorf("there is no call %q", callID)
	}
	if !st.info.Held() {
		return fmt.Errorf("call %q is not held", callID)
	}
	if d.Verdict != Allow && d.Verdict != Deny {
		return fmt.Errorf("a held call is resolved with allow or deny, not %q", d.Verdict)
	}
	labels := d.Labels
	if d.Verdict == Allow {
		// The labels the policy gave with the hold are the call's own: an
		// allowed call adds them as if it had been allowed at once.
		labels = union(st.info.Decision.Labels, d.Labels)
	}
	_, err := s.appendLocked(Entry{Type: EntryDecision, Decision: &DecisionEntry{
		CallID:   callID,
		Tool:     st.info.Call.Name,
		Verdict:  d.Verdict,
		Reason:   d.Reason,
		Labels:   labels,
		Resolved: true,
	}})
	return err
}

// record stamps the entry, appends it to the store, and applies it.
func (s *Session) record(e Entry) (Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendLocked(e)
}

func (s *Session) appendLocked(e Entry) (Entry, error) {
	e.Seq = len(s.entries) + 1
	e.Time = time.Now().UTC()
	// The entry is kept as the log will give it back: copied, so nothing
	// the caller holds can reach into the session, and passed through
	// JSON once, so a session that has run is the same as one reopened
	// (a store cannot keep invalid UTF-8, or the spacing of raw JSON).
	e, err := normalize(e)
	if err != nil {
		return Entry{}, err
	}
	if err := s.validate(e); err != nil {
		return Entry{}, err
	}
	if err := s.store.Append(e); err != nil {
		return Entry{}, err
	}
	s.mutate(e)
	return e, nil
}

// normalize returns the entry as it reads back from a store.
func normalize(e Entry) (Entry, error) {
	b, err := json.Marshal(e)
	if err != nil {
		return Entry{}, err
	}
	var out Entry
	if err := json.Unmarshal(b, &out); err != nil {
		return Entry{}, err
	}
	return out, nil
}

// begin claims the session for one Run. Two runs at once would interleave
// their entries, so the second is refused.
func (s *Session) begin() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		return fmt.Errorf("another Run is already in progress on this session")
	}
	s.running = true
	return nil
}

func (s *Session) end() {
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
}

func (s *Session) apply(e Entry) error {
	if err := s.validate(e); err != nil {
		return err
	}
	s.mutate(e)
	return nil
}

// validate refuses an entry that does not fit the log so far, before
// anything is written.
func (s *Session) validate(e Entry) error {
	switch e.Type {
	case EntryRunStarted:
		if e.RunStarted == nil {
			return fmt.Errorf("run_started entry has no payload")
		}
	case EntryMessage:
		if e.Message == nil {
			return fmt.Errorf("message entry has no payload")
		}
		if e.Message.Message.Role != Assistant {
			// Results enter the conversation as tool_result entries, which
			// place them after their calls; a message cannot carry them.
			for _, p := range e.Message.Message.Parts {
				if p.Type == ToolResultPart || p.ToolResult != nil {
					return fmt.Errorf("a message entry cannot hold a tool result")
				}
			}
		}
		if e.Message.Message.Role == Assistant {
			seen := map[string]bool{}
			for _, c := range e.Message.Message.ToolCalls() {
				if c.ID == "" {
					return fmt.Errorf("the model made a tool call with no id")
				}
				if _, dup := s.calls[c.ID]; dup || seen[c.ID] {
					return fmt.Errorf("the model reused the tool call id %q", c.ID)
				}
				seen[c.ID] = true
			}
		}
	case EntryDecision:
		if e.Decision == nil {
			return fmt.Errorf("decision entry has no payload")
		}
		if st, err := s.openCall(e.Decision.CallID); err != nil {
			return err
		} else if st.info.Started {
			return fmt.Errorf("call %q was decided after it started", e.Decision.CallID)
		}
	case EntryCallStarted:
		if e.CallStarted == nil {
			return fmt.Errorf("call_started entry has no payload")
		}
		st, err := s.openCall(e.CallStarted.CallID)
		if err != nil {
			return err
		}
		if st.info.Decision == nil || st.info.Decision.Verdict != Allow {
			return fmt.Errorf("call %q started without being allowed", e.CallStarted.CallID)
		}
		if st.info.Started {
			return fmt.Errorf("call %q started twice", e.CallStarted.CallID)
		}
	case EntryToolResult:
		if e.ToolResult == nil {
			return fmt.Errorf("tool_result entry has no payload")
		}
		if _, err := s.openCall(e.ToolResult.Result.CallID); err != nil {
			return err
		}
	case EntryModelError:
		if e.ModelError == nil {
			return fmt.Errorf("model_error entry has no payload")
		}
	case EntryRunEnded:
		if e.RunEnded == nil {
			return fmt.Errorf("run_ended entry has no payload")
		}
	default:
		return fmt.Errorf("unknown entry type %q", e.Type)
	}
	return nil
}

func (s *Session) openCall(id string) (*callState, error) {
	st, ok := s.calls[id]
	if !ok {
		return nil, fmt.Errorf("there is no call %q", id)
	}
	if st.info.Finished {
		return nil, fmt.Errorf("call %q is finished", id)
	}
	return st, nil
}

// mutate folds a validated entry into the derived state.
func (s *Session) mutate(e Entry) {
	s.entries = append(s.entries, e)
	switch e.Type {
	case EntryRunStarted:
		s.addLabels(e.RunStarted.Labels)
	case EntryMessage:
		m := e.Message
		s.messages = append(s.messages, m.Message)
		if m.Message.Role != Assistant {
			return
		}
		s.usage.add(m.Usage)
		turn := Turn{Message: len(s.messages) - 1, Stop: m.Stop, Model: m.Model, Usage: m.Usage}
		for i, c := range m.Message.ToolCalls() {
			turn.Calls = append(turn.Calls, c.ID)
			s.calls[c.ID] = &callState{info: CallInfo{Call: c}, turn: len(s.turns), index: i}
		}
		s.turns = append(s.turns, turn)
	case EntryDecision:
		d := e.Decision
		st := s.calls[d.CallID]
		if st.info.Decision == nil || st.info.Decision.Verdict == Hold {
			// A call counts once, when its first final verdict lands.
			if d.Verdict != Hold {
				s.decided++
			}
		}
		st.info.Decision = &Decision{Verdict: d.Verdict, Reason: d.Reason, Labels: d.Labels}
		if d.Verdict == Allow {
			s.addLabels(d.Labels)
		}
	case EntryCallStarted:
		s.calls[e.CallStarted.CallID].info.Started = true
	case EntryToolResult:
		r := e.ToolResult
		st := s.calls[r.Result.CallID]
		res := r.Result
		st.info.Finished = true
		st.info.Result = &res
		s.addLabels(r.Labels)
		s.placeResult(st, res)
	}
}

// placeResult puts a result in the user message that follows its turn,
// creating the message if it is the turn's first. Parts are kept in call
// order whatever order the results arrive in. The message is replaced,
// not edited, so slices handed out earlier stay as they were.
func (s *Session) placeResult(st *callState, res ToolResult) {
	part := Part{Type: ToolResultPart, ToolResult: &res}
	at := s.turns[st.turn].Message + 1
	if at < len(s.messages) && isResultMessage(s.messages[at]) {
		old := s.messages[at].Parts
		pos := 0
		for _, p := range old {
			if s.calls[p.ToolResult.CallID].index < st.index {
				pos++
			}
		}
		parts := make([]Part, 0, len(old)+1)
		parts = append(parts, old[:pos]...)
		parts = append(parts, part)
		parts = append(parts, old[pos:]...)
		s.messages[at] = Message{Role: User, Parts: parts}
		return
	}
	msg := Message{Role: User, Parts: []Part{part}}
	// Results arrive before anything else joins the conversation, since
	// a run settles a turn before it adds a prompt.
	s.messages = slices.Insert(s.messages, at, msg)
	for i := range s.turns {
		if s.turns[i].Message >= at {
			s.turns[i].Message++
		}
	}
}

func isResultMessage(m Message) bool {
	if m.Role != User || len(m.Parts) == 0 {
		return false
	}
	for _, p := range m.Parts {
		if p.Type != ToolResultPart || p.ToolResult == nil {
			return false
		}
	}
	return true
}

func (s *Session) addLabels(labels []string) {
	for _, l := range labels {
		s.labels[l] = true
	}
}

// union is a and b with no repeats, in order of first appearance.
func union(a, b []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, l := range append(slices.Clone(a), b...) {
		if !seen[l] {
			seen[l] = true
			out = append(out, l)
		}
	}
	return out
}

func (s *Session) sortedLabels() []string {
	out := make([]string, 0, len(s.labels))
	for l := range s.labels {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

func (s *Session) view() View {
	return View{Labels: s.sortedLabels(), ToolCalls: s.decided}
}

func (s *Session) pending() []ToolCall {
	if len(s.turns) == 0 {
		return nil
	}
	var out []ToolCall
	for _, id := range s.turns[len(s.turns)-1].Calls {
		if info := s.calls[id].info; !info.Finished {
			out = append(out, info.Call)
		}
	}
	return out
}

// lastTurn returns a copy of the last assistant turn.
func (s *Session) lastTurn() (Turn, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.turns) == 0 {
		return Turn{}, false
	}
	return s.turns[len(s.turns)-1], true
}

// lastText is the text of the last assistant message.
func (s *Session) lastText() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.messages) - 1; i >= 0; i-- {
		if s.messages[i].Role == Assistant {
			return s.messages[i].Text()
		}
	}
	return ""
}
