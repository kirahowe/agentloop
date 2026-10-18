package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// A Tool is something the model may call. An Agent has no tools but the
// ones it is given, and runs nothing else.
type Tool interface {
	Spec() ToolSpec
	// Call runs the tool on input that has already been checked against
	// the spec's schema and allowed by the policy. An error means the
	// tool could not run; the model is told so.
	Call(ctx context.Context, input json.RawMessage) (Result, error)
}

// A Result is what a tool call produced.
type Result struct {
	Content string
	// IsError marks a result the model should read as a failure.
	IsError bool
	// Labels are added to the session for good, describing what the
	// result brought into it, such as "private" or "untrusted".
	Labels []string
	// Log is for the operator: it is kept in the session's log and never
	// shown to the model.
	Log string
	// Usage is the model calls the tool made for the run, such as a
	// quarantined reader's. They count toward the run's usage and its
	// MaxTokens limit, and are kept in the tool_result entry even when
	// the result is refused.
	Usage []ModelUsage
	// Quarantine is set by a quarantined reader, to say how its answer
	// was labelled. A client can show it.
	Quarantine *QuarantineReport
	// Display is what the result shows the operator and never the
	// model (see Display).
	Display *Display
}

// A ModelUsage is what one model call cost, and the model that served it.
type ModelUsage struct {
	Model string `json:"model"`
	Usage Usage  `json:"usage"`
}

// A QuarantineReport says how a quarantined reader's answer was labelled:
// the inner tool it read, whether the answer's schema admits free text,
// and the labels the result left out because it does not.
type QuarantineReport struct {
	Tool     string   `json:"tool"`
	FreeText bool     `json:"free_text"`
	Dropped  []string `json:"dropped"`
}

// A Call is one call as a CallTool sees it.
type Call struct {
	ID    string
	Input json.RawMessage
	// Labels are the session's labels as the call runs, which include
	// the labels its decision added.
	Labels []string
	// Deferred are the labels its decision deferred to the result (see
	// Decision.Deferred).
	Deferred []string
}

// An InputChecker is a Tool with a check of its own on input that
// already matches its schema. The agent runs it before the policy, and a
// reason it returns denies the call, as a schema mismatch does.
type InputChecker interface {
	Tool
	CheckInput(input json.RawMessage) string
}

// A CallTool is a Tool that is told which call it serves and what the
// session holds. The agent uses CallWith instead of Call.
type CallTool interface {
	Tool
	CallWith(ctx context.Context, call Call) (Result, error)
}

// Func makes a Tool from a function.
func Func(spec ToolSpec, call func(ctx context.Context, input json.RawMessage) (Result, error)) Tool {
	return funcTool{spec, call}
}

type funcTool struct {
	spec ToolSpec
	call func(context.Context, json.RawMessage) (Result, error)
}

func (t funcTool) Spec() ToolSpec { return t.spec }
func (t funcTool) Call(ctx context.Context, in json.RawMessage) (Result, error) {
	return t.call(ctx, in)
}

// toolbox is an agent's tools with their schemas compiled.
type toolbox struct {
	order   []string
	tools   map[string]Tool
	schemas map[string]*InputSchema
}

// An InputSchema is a tool's parameters compiled, to check input as the
// agent checks a call's input.
type InputSchema struct {
	name   string
	schema *jsonschema.Schema
}

// CompileInput compiles a tool's parameters as the agent does: no
// reference is loaded from anywhere.
func CompileInput(spec ToolSpec) (*InputSchema, error) {
	s, err := compileSchema(spec)
	if err != nil {
		return nil, err
	}
	return &InputSchema{name: spec.Name, schema: s}, nil
}

func newToolbox(tools []Tool) (*toolbox, error) {
	tb := &toolbox{tools: map[string]Tool{}, schemas: map[string]*InputSchema{}}
	for _, t := range tools {
		spec := t.Spec()
		if spec.Name == "" {
			return nil, fmt.Errorf("a tool has no name")
		}
		if _, dup := tb.tools[spec.Name]; dup {
			return nil, fmt.Errorf("two tools are named %q", spec.Name)
		}
		schema, err := CompileInput(spec)
		if err != nil {
			return nil, err
		}
		tb.order = append(tb.order, spec.Name)
		tb.tools[spec.Name] = t
		tb.schemas[spec.Name] = schema
	}
	return tb, nil
}

func compileSchema(spec ToolSpec) (*jsonschema.Schema, error) {
	raw := spec.Parameters
	if len(raw) == 0 {
		raw = json.RawMessage(`{"type":"object"}`)
	}
	doc, err := jsonschema.UnmarshalJSON(strings.NewReader(string(raw)))
	if err != nil {
		return nil, fmt.Errorf("tool %q: parameters are not JSON: %w", spec.Name, err)
	}
	c := jsonschema.NewCompiler()
	// A schema may not fetch anything to resolve a reference.
	c.UseLoader(noLoader{})
	url := "tool:" + spec.Name
	if err := c.AddResource(url, doc); err != nil {
		return nil, fmt.Errorf("tool %q: %w", spec.Name, err)
	}
	s, err := c.Compile(url)
	if err != nil {
		return nil, fmt.Errorf("tool %q: parameters are not a valid schema: %w", spec.Name, err)
	}
	return s, nil
}

type noLoader struct{}

func (noLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("schemas may not load %s", url)
}

func (tb *toolbox) specs() []ToolSpec {
	specs := make([]ToolSpec, 0, len(tb.order))
	for _, name := range tb.order {
		specs = append(specs, tb.tools[name].Spec())
	}
	return specs
}

// check returns why input is not acceptable for the named tool, or "".
func (tb *toolbox) check(name string, input json.RawMessage) string {
	schema, ok := tb.schemas[name]
	if !ok {
		return fmt.Sprintf("There is no tool called %s.", name)
	}
	if why := schema.Check(input); why != "" {
		return why
	}
	if ic, ok := tb.tools[name].(InputChecker); ok {
		return ic.CheckInput(input)
	}
	return ""
}

// Check returns why input is not acceptable, or "".
func (s *InputSchema) Check(input json.RawMessage) string {
	name, schema := s.name, s.schema
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	v, err := jsonschema.UnmarshalJSON(strings.NewReader(string(input)))
	if err != nil {
		return fmt.Sprintf("The input to %s is not valid JSON: %v", name, err)
	}
	// A key given twice means different things to different readers (the
	// schema, the digest a person approves, the tool), so it is refused;
	// so is input nested too deep or with a number too long for a strict
	// reader of the log.
	if err := strictjson.Input(input); err != nil {
		return fmt.Sprintf("The input to %s is not acceptable: %v.", name, err)
	}
	if err := schema.Validate(v); err != nil {
		return fmt.Sprintf("The input to %s does not match its schema: %v", name, err)
	}
	return ""
}
