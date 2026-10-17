package otel

import "strconv"

// The types below are the part of the OTLP/JSON trace encoding that this
// package writes: opentelemetry/proto/collector/trace/v1 ExportTraceServiceRequest
// with the protobuf JSON mapping. Ids are hex strings, 64-bit integers and
// times are decimal strings, and enums are numbers.

// TracesData is the body of an OTLP/HTTP export request.
type TracesData struct {
	ResourceSpans []ResourceSpans `json:"resourceSpans"`
}

type ResourceSpans struct {
	Resource   Resource     `json:"resource"`
	ScopeSpans []ScopeSpans `json:"scopeSpans"`
}

type Resource struct {
	Attributes []KeyValue `json:"attributes"`
}

type ScopeSpans struct {
	Scope     Scope  `json:"scope"`
	Spans     []Span `json:"spans"`
	SchemaURL string `json:"schemaUrl,omitempty"`
}

type Scope struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}

// Span kinds, as the proto enum numbers.
const (
	KindInternal = 1
	KindClient   = 3
)

// Status codes, as the proto enum numbers.
const (
	StatusUnset = 0
	StatusOK    = 1
	StatusError = 2
)

type Span struct {
	TraceID           string     `json:"traceId"`
	SpanID            string     `json:"spanId"`
	ParentSpanID      string     `json:"parentSpanId,omitempty"`
	Name              string     `json:"name"`
	Kind              int        `json:"kind"`
	StartTimeUnixNano string     `json:"startTimeUnixNano"`
	EndTimeUnixNano   string     `json:"endTimeUnixNano"`
	Attributes        []KeyValue `json:"attributes,omitempty"`
	Events            []Event    `json:"events,omitempty"`
	Status            *Status    `json:"status,omitempty"`
}

type Event struct {
	TimeUnixNano string     `json:"timeUnixNano"`
	Name         string     `json:"name"`
	Attributes   []KeyValue `json:"attributes,omitempty"`
}

type Status struct {
	Message string `json:"message,omitempty"`
	Code    int    `json:"code"`
}

type KeyValue struct {
	Key   string   `json:"key"`
	Value AnyValue `json:"value"`
}

// AnyValue has exactly one field set.
type AnyValue struct {
	StringValue *string     `json:"stringValue,omitempty"`
	BoolValue   *bool       `json:"boolValue,omitempty"`
	IntValue    *string     `json:"intValue,omitempty"`
	ArrayValue  *ArrayValue `json:"arrayValue,omitempty"`
}

type ArrayValue struct {
	Values []AnyValue `json:"values"`
}

func str(k, v string) KeyValue { return KeyValue{k, AnyValue{StringValue: &v}} }

func boolean(k string, v bool) KeyValue { return KeyValue{k, AnyValue{BoolValue: &v}} }

func integer(k string, v int64) KeyValue {
	s := strconv.FormatInt(v, 10)
	return KeyValue{k, AnyValue{IntValue: &s}}
}

func strArray(k string, vs []string) KeyValue {
	vals := make([]AnyValue, len(vs))
	for i := range vs {
		v := vs[i]
		vals[i] = AnyValue{StringValue: &v}
	}
	return KeyValue{k, AnyValue{ArrayValue: &ArrayValue{Values: vals}}}
}
