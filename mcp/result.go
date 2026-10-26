package mcp

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kirahowe/agentloop"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// ToResult turns a tools/call result into what the agent records. It is
// pure. output is the tool's compiled outputSchema, or nil.
func ToResult(name string, cfg ToolConfig, output *jsonschema.Schema, r *sdk.CallToolResult) agentloop.Result {
	failed := func(why string) agentloop.Result {
		return agentloop.Result{Content: Failed(name), IsError: true, Log: why}
	}
	if r == nil {
		return failed("the server sent no result")
	}
	if r.NeedsInput() || r.InputRequests != nil {
		return failed("the server asked for input (input_required), which is a protocol violation: agentloop declares no capability to give it")
	}
	added, why := ResultLabels(r.Meta)
	if why != "" {
		return failed(why)
	}
	labels := MergeLabels(cfg.Labels, added)
	var texts []string
	for i, c := range r.Content {
		kind, placeholder := describe(c)
		if kind == "text" {
			texts = append(texts, c.(*sdk.TextContent).Text)
			continue
		}
		if !slices.Contains(cfg.AllowContent, kind) {
			return failed(fmt.Sprintf("content %d is %s, which the tool does not allow", i, kind))
		}
		if t, ok := resourceText(c); ok {
			texts = append(texts, t)
			continue
		}
		texts = append(texts, placeholder)
	}
	text := strings.Join(texts, "\n")
	if output == nil {
		return agentloop.Result{Content: text, IsError: r.IsError, Labels: labels}
	}
	// With an outputSchema the model sees only validated structured
	// content, never the server's text, even for an error.
	if r.IsError {
		return agentloop.Result{
			Content: fmt.Sprintf("The tool %s returned an error.", name),
			IsError: true, Labels: labels, Log: "error text: " + text,
		}
	}
	if r.StructuredContent == nil {
		return failed("the result has no structuredContent, though the tool has an outputSchema")
	}
	raw, err := canonicalValue(r.StructuredContent)
	if err != nil {
		return failed("structuredContent: " + err.Error())
	}
	v, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return failed("structuredContent: " + err.Error())
	}
	if err := output.Validate(v); err != nil {
		return failed("structuredContent does not match the outputSchema: " + err.Error())
	}
	return agentloop.Result{Content: string(raw), Labels: labels}
}

// ResultLabels reads the labels a result adds from its _meta. No key is
// no labels. Anything but an array of valid labels is why the call fails.
func ResultLabels(meta map[string]any) ([]string, string) {
	v, ok := meta[LabelsKey]
	if !ok {
		return nil, ""
	}
	arr, ok := v.([]any)
	if !ok {
		return nil, fmt.Sprintf("_meta %s is not an array", LabelsKey)
	}
	var out []string
	for _, x := range arr {
		l, ok := x.(string)
		if !ok {
			return nil, fmt.Sprintf("_meta %s holds a non-string", LabelsKey)
		}
		if err := agentloop.CheckLabel(l); err != nil {
			return nil, fmt.Sprintf("_meta %s has a bad label: %v", LabelsKey, err)
		}
		out = append(out, l)
	}
	return out, ""
}

// MergeLabels is the configured labels with those a result adds: sorted,
// no repeats, nil when there are none. A result can only add.
func MergeLabels(configured, added []string) []string {
	u := agentloop.UnionLabels(configured, added)
	if len(u) == 0 {
		return nil
	}
	return u
}

// resourceText is the text of an embedded text resource. Its URI and
// MIME type are the server's free text too, and are not shown.
func resourceText(c sdk.Content) (string, bool) {
	r, ok := c.(*sdk.EmbeddedResource)
	if !ok || r.Resource == nil || r.Resource.Blob != nil || r.Resource.Text == "" {
		return "", false
	}
	return r.Resource.Text, true
}

// mimeTops are the top-level media types a placeholder may name. Nothing
// else of a server's MIME type reaches the model: a subtype is free text.
var mimeTops = []string{"application", "audio", "font", "image", "message", "model", "multipart", "text", "video"}

func mime(s string) string {
	top, _, _ := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "/")
	if slices.Contains(mimeTops, top) {
		return top + "/*"
	}
	return "an unknown type"
}

// describe returns a content block's kind and, for a non-text kind, the
// placeholder the model sees instead of it.
func describe(c sdk.Content) (kind, placeholder string) {
	switch c := c.(type) {
	case *sdk.TextContent:
		return "text", ""
	case *sdk.ImageContent:
		return Image, fmt.Sprintf("[image not shown: %s, %d bytes]", mime(c.MIMEType), len(c.Data))
	case *sdk.AudioContent:
		return Audio, fmt.Sprintf("[audio not shown: %s, %d bytes]", mime(c.MIMEType), len(c.Data))
	case *sdk.ResourceLink:
		return ResourceLink, fmt.Sprintf("[resource link not shown: %s]", mime(c.MIMEType))
	case *sdk.EmbeddedResource:
		n := 0
		mt := ""
		if c.Resource != nil {
			n = len(c.Resource.Text) + len(c.Resource.Blob)
			mt = c.Resource.MIMEType
		}
		return Resource, fmt.Sprintf("[embedded resource not shown: %s, %d bytes]", mime(mt), n)
	}
	return fmt.Sprintf("%T", c), ""
}

// Digest is the SHA-256, in hex, of a tool's definition: its name,
// description, inputSchema and outputSchema as canonical JSON. The
// annotations are left out: they decide nothing, and the SDK
// would have to encode them again, so the digest would follow the SDK's
// version.
func Digest(t *sdk.Tool) (string, error) {
	in, err := canonicalValue(t.InputSchema)
	if err != nil {
		return "", err
	}
	var out json.RawMessage
	if t.OutputSchema != nil {
		if out, err = canonicalValue(t.OutputSchema); err != nil {
			return "", err
		}
	}
	return DefinitionDigest(t.Name, t.Description, in, out)
}

// DefinitionDigest is Digest from the parts. An empty output means the
// tool has no outputSchema.
func DefinitionDigest(name, description string, input, output json.RawMessage) (string, error) {
	def := struct {
		Name         string          `json:"name"`
		Description  string          `json:"description"`
		InputSchema  json.RawMessage `json:"inputSchema"`
		OutputSchema json.RawMessage `json:"outputSchema,omitempty"`
	}{name, description, input, output}
	raw, err := json.Marshal(def)
	if err != nil {
		return "", err
	}
	c, err := Canonical(raw)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(c)
	return hex.EncodeToString(sum[:]), nil
}

// Canonical is JSON with object keys sorted, no insignificant whitespace,
// no HTML escaping and numbers as written, as agentloop.InputDigest uses.
func Canonical(raw []byte) ([]byte, error) {
	v, err := strictjson.Value(raw)
	if err != nil {
		return nil, err
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n")), nil
}

func canonicalValue(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return Canonical(raw)
}

// compileSchema compiles a schema that may load nothing to resolve a
// reference.
func compileSchema(url string, raw []byte) (*jsonschema.Schema, error) {
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	c := jsonschema.NewCompiler()
	c.UseLoader(noLoader{})
	if err := c.AddResource(url, doc); err != nil {
		return nil, err
	}
	return c.Compile(url)
}

type noLoader struct{}

func (noLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("schemas may not load %s", url)
}
