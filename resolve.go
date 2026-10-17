package agentloop

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
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
// The entry it writes goes to Observe, as a run's entries do.
//
// The call must exist and its input must have the digest the resolution
// names. A call a person already resolved with the same verdict is left
// as it is, so resolving can be repeated after a crash. Any other call
// that is not held is an error. So is an allow by an agent with no policy,
// which could not give the call its labels, or one the policy now denies
// outright: a person can approve what the policy held, not what it
// refuses.
func (a *Agent) Resolve(ctx context.Context, s *Session, r Resolution) error {
	d, noop, err := a.resolution(ctx, s, r)
	if err != nil || noop {
		return err
	}
	if err := s.Resolve(r.CallID, d); err != nil {
		return err
	}
	if a.Observe != nil {
		es := s.Entries()
		a.Observe(Event{Entry: &es[len(es)-1]})
	}
	return nil
}

// ErrPolicyDenies is why Resolve refuses an allow the agent's policy now
// denies outright, such as a call to a tool the agent no longer has.
var ErrPolicyDenies = errors.New("the policy denies the call")

// CheckResolve says whether Resolve would accept r, without writing
// anything, so a list of resolutions can be checked whole first.
func (a *Agent) CheckResolve(ctx context.Context, s *Session, r Resolution) error {
	_, _, err := a.resolution(ctx, s, r)
	return err
}

// resolution is the decision Resolve would record for r, or that it would
// record nothing (noop), or why r is refused.
func (a *Agent) resolution(ctx context.Context, s *Session, r Resolution) (d Decision, noop bool, err error) {
	info, ok := s.Call(r.CallID)
	if !ok {
		return d, false, fmt.Errorf("there is no call %q", r.CallID)
	}
	if r.InputSHA256 != InputDigest(info.Call.Input) {
		return d, false, fmt.Errorf("the input of call %q does not have the digest %q", r.CallID, r.InputSHA256)
	}
	if r.Verdict != Allow && r.Verdict != Deny {
		return d, false, fmt.Errorf("a held call is resolved with allow or deny, not %q", r.Verdict)
	}
	if !info.Held() {
		if info.Resolved && info.Decision.Verdict == r.Verdict {
			return d, true, nil
		}
		return d, false, fmt.Errorf("call %q is not held", r.CallID)
	}
	d = Decision{Verdict: r.Verdict, Reason: r.Reason}
	if r.Verdict == Allow {
		if a.Policy == nil {
			return d, false, fmt.Errorf("call %q cannot be allowed by an agent with no policy", r.CallID)
		}
		pd := a.Policy.Decide(ctx, info.Call, s.View())
		if pd.Verdict == Deny {
			return d, false, fmt.Errorf("%w: call %q: %s", ErrPolicyDenies, r.CallID, pd.Reason)
		}
		if len(Forbidden(append(slices.Clone(pd.Labels), pd.Deferred...), a.Forbid)) > 0 {
			return d, false, fmt.Errorf("%w: call %q: %s", ErrPolicyDenies, r.CallID, ForbiddenDecision)
		}
		d.Labels, d.Deferred = pd.Labels, pd.Deferred
	}
	return d, false, nil
}
