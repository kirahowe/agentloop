package otel_test

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"testing"
)

// checkOTLP asserts the shape of an OTLP/JSON trace export: the field names
// and types of the protobuf JSON mapping of
// opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest, as
// far as this package writes it. OTLP publishes no JSON schema, so the
// rules are written out here. It also checks what the proto cannot say:
// every span ends no earlier than it starts, a parent is a span of the same
// trace, and ids are unique.
func checkOTLP(t testing.TB, raw []byte) {
	t.Helper()
	if err := validateOTLP(raw); err != nil {
		t.Fatalf("not valid OTLP/JSON: %v", err)
	}
}

var (
	hex32 = regexp.MustCompile(`^[0-9a-f]{32}$`)
	hex16 = regexp.MustCompile(`^[0-9a-f]{16}$`)
	dec   = regexp.MustCompile(`^[0-9]+$`)
)

func validateOTLP(raw []byte) error {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return err
	}
	if err := keys("root", root, []string{"resourceSpans"}, nil); err != nil {
		return err
	}
	rss, ok := root["resourceSpans"].([]any)
	if !ok {
		return fmt.Errorf("resourceSpans is not an array")
	}
	type spanRef struct{ trace string }
	ids := map[string]spanRef{} // by spanId
	var parents [][3]string     // trace, parent, name
	for _, rs := range rss {
		rsm, ok := rs.(map[string]any)
		if !ok {
			return fmt.Errorf("resourceSpans entry is not an object")
		}
		if err := keys("resourceSpans", rsm, []string{"resource", "scopeSpans"}, []string{"schemaUrl"}); err != nil {
			return err
		}
		res := rsm["resource"].(map[string]any)
		if err := keys("resource", res, nil, []string{"attributes", "droppedAttributesCount"}); err != nil {
			return err
		}
		if err := checkAttrs("resource", res["attributes"]); err != nil {
			return err
		}
		sss, ok := rsm["scopeSpans"].([]any)
		if !ok {
			return fmt.Errorf("scopeSpans is not an array")
		}
		for _, ss := range sss {
			ssm := ss.(map[string]any)
			if err := keys("scopeSpans", ssm, []string{"scope", "spans"}, []string{"schemaUrl"}); err != nil {
				return err
			}
			if err := keys("scope", ssm["scope"].(map[string]any), []string{"name"}, []string{"version", "attributes"}); err != nil {
				return err
			}
			spans, ok := ssm["spans"].([]any)
			if !ok {
				return fmt.Errorf("spans is not an array")
			}
			for _, sp := range spans {
				m := sp.(map[string]any)
				if err := keys("span", m,
					[]string{"traceId", "spanId", "name", "kind", "startTimeUnixNano", "endTimeUnixNano"},
					[]string{"parentSpanId", "attributes", "events", "status", "traceState", "links"}); err != nil {
					return err
				}
				tid, sid := m["traceId"].(string), m["spanId"].(string)
				if !hex32.MatchString(tid) || tid == "00000000000000000000000000000000" {
					return fmt.Errorf("bad traceId %q", tid)
				}
				if !hex16.MatchString(sid) || sid == "0000000000000000" {
					return fmt.Errorf("bad spanId %q", sid)
				}
				if _, dup := ids[sid]; dup {
					return fmt.Errorf("span id %s is used twice", sid)
				}
				ids[sid] = spanRef{tid}
				name, _ := m["name"].(string)
				if name == "" {
					return fmt.Errorf("span %s has no name", sid)
				}
				if p, has := m["parentSpanId"]; has {
					ps, _ := p.(string)
					if !hex16.MatchString(ps) {
						return fmt.Errorf("bad parentSpanId %q", p)
					}
					parents = append(parents, [3]string{tid, ps, name})
				}
				if k, _ := m["kind"].(float64); k < 1 || k > 5 || k != float64(int(k)) {
					return fmt.Errorf("span %s: bad kind %v", name, m["kind"])
				}
				start, end, err := nanoPair(m["startTimeUnixNano"], m["endTimeUnixNano"])
				if err != nil {
					return fmt.Errorf("span %s: %v", name, err)
				}
				if end < start {
					return fmt.Errorf("span %s ends before it starts", name)
				}
				if err := checkAttrs(name, m["attributes"]); err != nil {
					return err
				}
				if evs, ok := m["events"].([]any); ok {
					for _, ev := range evs {
						em := ev.(map[string]any)
						if err := keys("event", em, []string{"timeUnixNano", "name"}, []string{"attributes", "droppedAttributesCount"}); err != nil {
							return err
						}
						if s, _ := em["timeUnixNano"].(string); !dec.MatchString(s) {
							return fmt.Errorf("event time %v is not a decimal string", em["timeUnixNano"])
						}
						if err := checkAttrs("event", em["attributes"]); err != nil {
							return err
						}
					}
				}
				if st, ok := m["status"].(map[string]any); ok {
					if err := keys("status", st, []string{"code"}, []string{"message"}); err != nil {
						return err
					}
					if c, _ := st["code"].(float64); c < 0 || c > 2 {
						return fmt.Errorf("bad status code %v", st["code"])
					}
				}
			}
		}
	}
	for _, p := range parents {
		ref, ok := ids[p[1]]
		if !ok {
			return fmt.Errorf("span %q has a parent %s that is not in the export", p[2], p[1])
		}
		if ref.trace != p[0] {
			return fmt.Errorf("span %q has a parent in another trace", p[2])
		}
	}
	return nil
}

func nanoPair(a, b any) (uint64, uint64, error) {
	as, _ := a.(string)
	bs, _ := b.(string)
	if !dec.MatchString(as) || !dec.MatchString(bs) {
		return 0, 0, fmt.Errorf("times %v, %v are not decimal strings", a, b)
	}
	x, err1 := strconv.ParseUint(as, 10, 64)
	y, err2 := strconv.ParseUint(bs, 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, fmt.Errorf("times do not fit in uint64")
	}
	return x, y, nil
}

func keys(what string, m map[string]any, required, optional []string) error {
	allowed := map[string]bool{}
	for _, k := range required {
		if _, ok := m[k]; !ok {
			return fmt.Errorf("%s lacks %q", what, k)
		}
		allowed[k] = true
	}
	for _, k := range optional {
		allowed[k] = true
	}
	for k := range m {
		if !allowed[k] {
			return fmt.Errorf("%s has an unknown field %q", what, k)
		}
	}
	return nil
}

func checkAttrs(what string, v any) error {
	if v == nil {
		return nil
	}
	list, ok := v.([]any)
	if !ok {
		return fmt.Errorf("%s: attributes is not an array", what)
	}
	seen := map[string]bool{}
	for _, a := range list {
		kv := a.(map[string]any)
		if err := keys("attribute", kv, []string{"key", "value"}, nil); err != nil {
			return err
		}
		k, _ := kv["key"].(string)
		if k == "" || seen[k] {
			return fmt.Errorf("%s: empty or repeated attribute key %q", what, k)
		}
		seen[k] = true
		if err := checkValue(k, kv["value"]); err != nil {
			return err
		}
	}
	return nil
}

func checkValue(k string, v any) error {
	m, ok := v.(map[string]any)
	if !ok || len(m) != 1 {
		return fmt.Errorf("attribute %s: a value has exactly one field", k)
	}
	for field, x := range m {
		switch field {
		case "stringValue":
			if _, ok := x.(string); !ok {
				return fmt.Errorf("attribute %s: stringValue is not a string", k)
			}
		case "boolValue":
			if _, ok := x.(bool); !ok {
				return fmt.Errorf("attribute %s: boolValue is not a bool", k)
			}
		case "intValue":
			s, ok := x.(string)
			if _, err := strconv.ParseInt(s, 10, 64); !ok || err != nil {
				return fmt.Errorf("attribute %s: intValue %v is not an int64 string", k, x)
			}
		case "arrayValue":
			am, ok := x.(map[string]any)
			if !ok {
				return fmt.Errorf("attribute %s: arrayValue is not an object", k)
			}
			if err := keys("arrayValue", am, []string{"values"}, nil); err != nil {
				return err
			}
			for _, e := range am["values"].([]any) {
				if err := checkValue(k, e); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("attribute %s: unknown value field %q", k, field)
		}
	}
	return nil
}
