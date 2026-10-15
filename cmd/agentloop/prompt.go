package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

// lineReader reads the chat's input one line at a time, showing prompt.
// At io.EOF the returned line holds whatever was read before it.
type lineReader interface {
	readLine(prompt string) (string, error)
}

// newLineReader edits lines on a terminal (arrow keys, history, the usual
// control keys) and reads plain lines from anything else, such as a pipe.
func newLineReader(stdin io.Reader, stderr io.Writer) lineReader {
	in, ok := stdin.(*os.File)
	out, ok2 := stderr.(*os.File)
	if ok && ok2 && term.IsTerminal(int(in.Fd())) && term.IsTerminal(int(out.Fd())) {
		return &termReader{fd: int(in.Fd()), t: term.NewTerminal(struct {
			io.Reader
			io.Writer
		}{in, out}, "")}
	}
	return &plainReader{in: bufio.NewReader(stdin), prompt: stderr}
}

type plainReader struct {
	in     *bufio.Reader
	prompt io.Writer
}

func (r *plainReader) readLine(prompt string) (string, error) {
	fmt.Fprint(r.prompt, prompt)
	line, err := r.in.ReadString('\n')
	return strings.TrimSuffix(line, "\n"), err
}

// termReader holds the terminal in raw mode only while a line is being
// typed, so a run's output and Ctrl-C during a run behave as on a plain
// terminal. Ctrl-D on an empty line, or Ctrl-C, ends the input.
type termReader struct {
	fd int
	t  *term.Terminal
}

func (r *termReader) readLine(prompt string) (string, error) {
	old, err := term.MakeRaw(r.fd)
	if err != nil {
		return "", err
	}
	defer term.Restore(r.fd, old)
	r.t.SetPrompt(prompt)
	line, err := r.t.ReadLine()
	if err == term.ErrPasteIndicator {
		err = nil
	}
	return line, err
}
