package agentloop

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unicode"
)

// A Store keeps a session's entries. It is the only thing a session needs
// to be remembered and resumed.
type Store interface {
	// Load returns every entry, in order.
	Load() ([]Entry, error)
	// Append adds one entry. It is durable when it returns: a crash after
	// that cannot lose it.
	Append(Entry) error
}

// EntryType says which payload of an Entry is set.
type EntryType string

const (
	EntryRunStarted  EntryType = "run_started"
	EntryMessage     EntryType = "message"
	EntryDecision    EntryType = "decision"
	EntryCallStarted EntryType = "call_started"
	EntryToolResult  EntryType = "tool_result"
	EntryModelError  EntryType = "model_error"
	EntryRunEnded    EntryType = "run_ended"
)

// An Entry is one line of a session's log. Exactly one payload field is
// set, the one Type names. Entries are never edited or removed.
type Entry struct {
	Seq  int       `json:"seq"`
	Time time.Time `json:"time"`
	Type EntryType `json:"type"`

	RunStarted  *RunStartedEntry  `json:"run_started,omitempty"`
	Message     *MessageEntry     `json:"message,omitempty"`
	Decision    *DecisionEntry    `json:"decision,omitempty"`
	CallStarted *CallStartedEntry `json:"call_started,omitempty"`
	ToolResult  *ToolResultEntry  `json:"tool_result,omitempty"`
	ModelError  *ModelErrorEntry  `json:"model_error,omitempty"`
	RunEnded    *Outcome          `json:"run_ended,omitempty"`
}

// RunStartedEntry records that Run began.
type RunStartedEntry struct {
	Prompt string `json:"prompt,omitempty"`
	// Labels are the agent's starting labels, added to the session as the
	// run begins.
	Labels []string `json:"labels,omitempty"`
}

// MessageEntry records a message joining the conversation. For an
// assistant turn it also carries why the turn ended and what it cost.
type MessageEntry struct {
	Message Message `json:"message"`
	Stop    Stop    `json:"stop,omitempty"`
	Usage   Usage   `json:"usage"`
	// Model is the model that served the turn.
	Model   string `json:"model,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

// DecisionEntry records the verdict on one call.
type DecisionEntry struct {
	CallID  string   `json:"call_id"`
	Tool    string   `json:"tool"`
	Verdict Verdict  `json:"verdict"`
	Reason  string   `json:"reason,omitempty"`
	Labels  []string `json:"labels,omitempty"`
	// Resolved marks the decision of a person on a held call.
	Resolved bool `json:"resolved,omitempty"`
}

// CallStartedEntry is written immediately before a tool runs. If it is
// the last word on a call, the call may or may not have taken effect.
type CallStartedEntry struct {
	CallID string `json:"call_id"`
}

// ToolResultEntry records how a call ended and what the result added to
// the session's labels.
type ToolResultEntry struct {
	Result ToolResult `json:"result"`
	Labels []string   `json:"labels,omitempty"`
	// Log is what the tool said for the operator alone, such as a
	// program's stderr. It is never part of the conversation.
	Log string `json:"log,omitempty"`
}

// ModelErrorEntry records one failed model call.
type ModelErrorEntry struct {
	Error ModelError `json:"error"`
	// Retry says whether the loop went on to try again, after WaitMillis.
	Retry      bool  `json:"retry"`
	WaitMillis int64 `json:"wait_ms,omitempty"`
}

// MemoryStore keeps entries in memory. Entries pass through JSON on the
// way in and out, so it behaves like a file would, without sharing
// pointers with the caller.
type MemoryStore struct {
	mu   sync.Mutex
	data [][]byte
}

// NewMemoryStore makes a store, optionally holding entries already.
func NewMemoryStore(entries ...Entry) *MemoryStore {
	m := &MemoryStore{}
	for _, e := range entries {
		if err := m.Append(e); err != nil {
			panic(err)
		}
	}
	return m
}

func (m *MemoryStore) Load() ([]Entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	entries := make([]Entry, 0, len(m.data))
	for _, b := range m.data {
		var e Entry
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, err
		}
		entries = append(entries, e)
	}
	return entries, nil
}

func (m *MemoryStore) Append(e Entry) error {
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data = append(m.data, b)
	return nil
}

// FileStore keeps entries as JSON lines in a file, syncing each append.
type FileStore struct {
	path string

	mu       sync.Mutex
	repaired bool
}

// NewFileStore makes a store on path. The file is created by the first
// append; its directory must exist.
func NewFileStore(path string) *FileStore { return &FileStore{path: path} }

// Load reads the file. A final line that is not complete JSON is what a
// write cut short by a crash leaves, and is ignored. A bad line anywhere
// else means the log is damaged, and is an error.
func (f *FileStore) Load() ([]Entry, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	raw, err := os.ReadFile(f.path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	lines := nonBlankLines(raw)
	var entries []Entry
	for i, line := range lines {
		var e Entry
		if err := json.Unmarshal(line, &e); err != nil {
			if i == len(lines)-1 {
				break
			}
			return nil, fmt.Errorf("%s: line %d is not an entry: %w", f.path, i+1, err)
		}
		entries = append(entries, e)
	}
	return entries, nil
}

// Append writes one line and syncs it.
func (f *FileStore) Append(e Entry) error {
	line, err := json.Marshal(e)
	if err != nil {
		return err
	}
	line = append(line, '\n')

	f.mu.Lock()
	defer f.mu.Unlock()
	created := false
	if _, err := os.Stat(f.path); errors.Is(err, fs.ErrNotExist) {
		created = true
	}
	if !f.repaired {
		if err := repairTail(f.path); err != nil {
			return err
		}
		f.repaired = true
	}
	file, err := os.OpenFile(f.path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600)
	if err != nil {
		return err
	}
	if _, err := file.Write(line); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if created {
		// A new file is only durable once its directory entry is.
		if d, err := os.Open(filepath.Dir(f.path)); err == nil {
			d.Sync()
			d.Close()
		}
	}
	return nil
}

// repairTail makes the file end on a line boundary before anything is
// appended to it. A last line that is not a complete entry is a torn
// write, which Load ignores, so it is cut off; otherwise appending would
// glue the next entry onto it. A complete last line missing its newline
// just gets one.
func repairTail(path string) error {
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	// The last line is the last one with anything on it, as in Load.
	end := len(bytes.TrimRightFunc(raw, unicode.IsSpace))
	if end == 0 {
		if len(raw) == 0 {
			return nil
		}
		return truncateTo(path, 0)
	}
	start := bytes.LastIndexByte(raw[:end], '\n') + 1
	if !isEntryLine(raw[start:end]) {
		return truncateTo(path, int64(start))
	}
	if raw[len(raw)-1] == '\n' {
		return nil
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := file.Write([]byte("\n")); err != nil {
		return err
	}
	return file.Sync()
}

// isEntryLine reports whether line is a complete entry. Load and repairTail
// must agree on this, or a line one ignores as torn the other would keep.
func isEntryLine(line []byte) bool {
	var e Entry
	return json.Unmarshal(line, &e) == nil
}

func truncateTo(path string, size int64) error {
	file, err := os.OpenFile(path, os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := file.Truncate(size); err != nil {
		return err
	}
	return file.Sync()
}

func nonBlankLines(raw []byte) [][]byte {
	var lines [][]byte
	for _, l := range bytes.Split(raw, []byte("\n")) {
		if len(bytes.TrimSpace(l)) > 0 {
			lines = append(lines, l)
		}
	}
	return lines
}
