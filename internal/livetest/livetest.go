// Package livetest gates the smoke tests that call the real provider APIs.
// Importing it registers test flags; nothing is read from the environment.
//
//	go test ./anthropic ./openai -run Live -args -live \
//	    -anthropic-key-command 'security find-generic-password -s relay-anthropic -w'
//
// A key command is split on whitespace and run with no shell, the same as
// the api_key_command of a request. Only the packages that import this one
// know the flags, so name them on the command line instead of using ./...
package livetest

import (
	"context"
	"errors"
	"flag"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop/internal/request"
)

var errNoCommand = errors.New("no key command given")

var (
	live            = flag.Bool("live", false, "run the smoke tests against the real provider APIs (spends money)")
	anthropicKeyCmd = flag.String("anthropic-key-command", "", "command that prints the Anthropic API key")
	anthropicModel  = flag.String("anthropic-model", "claude-haiku-5-5", "model for the Anthropic live test")
	openaiKeyCmd    = flag.String("openai-key-command", "", "command that prints the OpenAI API key")
	openaiModel     = flag.String("openai-model", "gpt-5", "reasoning model for the OpenAI live test")
	keyCommandWait  = 30 * time.Second
)

// Anthropic skips the test unless -live and -anthropic-key-command are
// given. It returns the key and the model to use.
func Anthropic(t *testing.T) (key, model string) {
	t.Helper()
	return gate(t, "anthropic", *anthropicKeyCmd), *anthropicModel
}

// OpenAI skips the test unless -live and -openai-key-command are given. It
// returns the key and the model to use.
func OpenAI(t *testing.T) (key, model string) {
	t.Helper()
	return gate(t, "openai", *openaiKeyCmd), *openaiModel
}

func gate(t *testing.T, provider, keyCommand string) string {
	t.Helper()
	if !*live {
		t.Skip("live test: pass -live to run it (it spends money)")
	}
	argv, err := Split(keyCommand)
	if err != nil {
		t.Skipf("live test: no %s key command: pass -%s-key-command '<command that prints the key>'", provider, provider)
	}
	key, err := request.FetchKey(context.Background(), argv, keyCommandWait)
	if err != nil {
		t.Fatalf("%s key command: %v", provider, err)
	}
	return key
}

// Split turns a key command into an argv. Arguments are separated by
// whitespace; there is no quoting and no shell. An empty command is an error.
func Split(command string) ([]string, error) {
	argv := strings.Fields(command)
	if len(argv) == 0 {
		return nil, errNoCommand
	}
	return argv, nil
}
