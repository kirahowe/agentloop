package agentloop

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// InputDigest is the SHA-256, in hex, of a call's input in canonical JSON:
// object keys sorted, no insignificant whitespace, no HTML escaping, and
// numbers as written. Input that is not JSON is hashed as it is. A hold
// records it, and a resolution must name it, so a person approves exactly
// the input they were shown.
func InputDigest(input json.RawMessage) string {
	sum := sha256.Sum256(canonicalJSON(input))
	return hex.EncodeToString(sum[:])
}

func canonicalJSON(raw json.RawMessage) []byte {
	if len(bytes.TrimSpace(raw)) == 0 {
		raw = json.RawMessage(`{}`)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return raw
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return raw
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n"))
}

// A Resolution is a person's decision on a held call.
type Resolution struct {
	CallID  string  `json:"call_id"`
	Verdict Verdict `json:"verdict"`
	Reason  string  `json:"reason,omitempty"`
	// InputSHA256 must be the digest the hold recorded (InputDigest of
	// the call's input).
	InputSHA256 string `json:"input_sha256"`
}

// Resolve records a person's decision on a held call, which the next Run
// carries out. An allow takes its labels from the agent's policy now, as
// an allow from the policy would have; a person never supplies labels.
//
// The call must exist and its input must have the digest the resolution
// names. A call a person already resolved with the same verdict is left
// as it is, so resolving can be repeated after a crash. Any other call
// that is not held is an error.
func (a *Agent) Resolve(ctx context.Context, s *Session, r Resolution) error {
	info, ok := s.Call(r.CallID)
	if !ok {
		return fmt.Errorf("there is no call %q", r.CallID)
	}
	if r.InputSHA256 != InputDigest(info.Call.Input) {
		return fmt.Errorf("the input of call %q does not have the digest %q", r.CallID, r.InputSHA256)
	}
	if r.Verdict != Allow && r.Verdict != Deny {
		return fmt.Errorf("a held call is resolved with allow or deny, not %q", r.Verdict)
	}
	if !info.Held() {
		if info.Resolved && info.Decision.Verdict == r.Verdict {
			return nil
		}
		return fmt.Errorf("call %q is not held", r.CallID)
	}
	d := Decision{Verdict: r.Verdict, Reason: r.Reason}
	if r.Verdict == Allow && a.Policy != nil {
		d.Labels = a.Policy.Decide(ctx, info.Call, s.View()).Labels
	}
	return s.Resolve(r.CallID, d)
}
