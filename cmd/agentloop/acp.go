package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	acpagent "github.com/kirahowe/agentloop/acp"
	"github.com/kirahowe/agentloop/internal/request"
	"github.com/kirahowe/agentloop/internal/usage"
)

// answerGrace bounds the wait, at shutdown, for the responses of the turns
// that were cancelled to be written.
const answerGrace = 5 * time.Second

// acpCmd serves the agent to one ACP client over stdin and stdout until
// the client closes stdin, stdout fails, or a signal comes. Each of those
// cancels every turn, waits for the runs to end and release their
// sessions, and exits 0. A second signal exits at once.
func acpCmd(req *request.Request, stdin io.Reader, stdout, stderr io.Writer) error {
	sigs := make(chan os.Signal, 2)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigs)

	key, err := request.FetchKey(context.Background(), req.APIKeyCommand, request.KeyTimeout)
	if err != nil {
		return usage.Errorf("%v", err)
	}
	kinds := map[string]acp.ToolKind{}
	for name, kind := range req.ToolKinds {
		kinds[name] = acp.ToolKind(kind)
	}
	srv := acpagent.NewServer(acpagent.Config{
		Version:    version,
		SessionDir: req.SessionDir,
		Agent:      req.Agent(key),
		Options:    acpagent.Options{Kinds: kinds, ContextWindow: req.ContextWindow},
		Log:        stderr,
	})
	conn, err := srv.Connect(stdout, stdin)
	if err != nil {
		return err
	}
	select {
	case <-conn.Done():
	case <-sigs:
		go func() {
			<-sigs
			fmt.Fprintln(stderr)
			os.Exit(130)
		}()
	}
	ctx, cancel := context.WithTimeout(context.Background(), answerGrace)
	defer cancel()
	srv.Shutdown(ctx)
	conn.Close()
	return nil
}
