package main

import (
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/usage"
	"github.com/kirahowe/agentloop/otel"
)

const otelUsage = "usage: agentloop otel [--conversation-id ID] [--parent-trace ID --parent-span ID] [--out FILE] SESSION.jsonl"

var (
	traceIDRe = regexp.MustCompile(`^[0-9a-f]{32}$`)
	spanIDRe  = regexp.MustCompile(`^[0-9a-f]{16}$`)
)

// otelCmd converts one session file (or a run's transcript) to OTLP/JSON traces, offline. The
// output is one ExportTraceServiceRequest on one line, with no content:
// no messages, tool inputs or results.
func otelCmd(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("otel", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	conv := fs.String("conversation-id", "", "")
	out := fs.String("out", "", "")
	parentTrace := fs.String("parent-trace", "", "")
	parentSpan := fs.String("parent-span", "", "")
	var paths []string
	// Flags may come before or after the path.
	for len(args) > 0 {
		if err := fs.Parse(args); err != nil {
			return usage.Errorf("%v\n%s", err, otelUsage)
		}
		args = fs.Args()
		if len(args) > 0 {
			paths = append(paths, args[0])
			args = args[1:]
		}
	}
	if len(paths) != 1 {
		return usage.Errorf("%s", otelUsage)
	}
	if (*parentTrace == "") != (*parentSpan == "") {
		return usage.Errorf("--parent-trace and --parent-span go together")
	}
	if *parentTrace != "" && (!traceIDRe.MatchString(*parentTrace) || !spanIDRe.MatchString(*parentSpan)) {
		return usage.Errorf("--parent-trace is 32 lower-case hex characters and --parent-span is 16")
	}
	path := paths[0]
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return usage.Errorf("no such session file: %s", path)
	}
	entries, err := agentloop.NewFileStore(path).Load()
	if err != nil {
		return usage.Errorf("%v", err)
	}
	// A run's stdout (a transcript) is a session log plus a last outcome
	// line, which is not an entry.
	entries = slices.DeleteFunc(entries, func(e agentloop.Entry) bool { return e.Type == "outcome" })
	id := *conv
	if id == "" {
		base := filepath.Base(path)
		id = strings.TrimSuffix(base, filepath.Ext(base))
	}
	b, err := otel.Marshal(entries, otel.Options{
		ConversationID: id, AgentName: "agentloop",
		ParentTraceID: *parentTrace, ParentSpanID: *parentSpan,
	})
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if *out != "" {
		return os.WriteFile(*out, b, 0o644)
	}
	_, err = stdout.Write(b)
	return err
}
