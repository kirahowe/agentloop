package command

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

// escapeArg makes the test binary a program that leaves its process
// group and ignores SIGTERM.
const escapeArg = "command-test-escape"

func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == escapeArg {
		syscall.Setpgid(0, 0)
		signal.Ignore(syscall.SIGTERM)
		os.Stdout.WriteString("escaped\n")
		time.Sleep(time.Hour)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// Regression: a program that left its process group and ignored SIGTERM
// outlived the group's sweep, and Call waited on it for ever after its
// timeout. The program is now killed by its pid as well.
func TestTimeoutKillsAProgramThatLeftItsGroup(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	tool := &Tool{Argv: []string{exe, escapeArg}}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	start := time.Now()
	go func() {
		_, err := tool.Call(ctx, json.RawMessage(`{}`))
		done <- err
	}()
	select {
	case err := <-done:
		if err != context.DeadlineExceeded {
			t.Fatalf("err %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("Call hung")
	}
	if took := time.Since(start); took > killGrace+3*time.Second {
		t.Fatalf("Call took %s", took)
	}
}
