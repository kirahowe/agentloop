// Package sessionfile opens a session file for one process at a time.
package sessionfile

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/usage"
)

// ErrInUse is the error of Open when another open of the session holds
// its lock: another process, or another open in this one, since the lock
// belongs to the open file and not to the process.
var ErrInUse = errors.New("in use by another agentloop")

// Exists reports whether a session file is already there. It creates
// nothing.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return !errors.Is(err, fs.ErrNotExist)
}

// Open opens the session and takes an exclusive lock on it for as
// long as the process lives (or until the returned function is called).
// Two processes on one session would each run its pending calls and log
// over the other's entries; the lock is released by the OS if the process
// is killed.
func Open(path string) (*agentloop.Session, func(), error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, nil, usage.Errorf("cannot create the session directory: %v", err)
	}
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return nil, nil, usage.Errorf("session %s is a directory", path)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, nil, usage.Errorf("cannot lock the session: %v", err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		return nil, nil, fmt.Errorf("session %s is %w", path, ErrInUse)
	}
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(path))
	if err != nil {
		lock.Close()
		return nil, nil, err
	}
	return sess, func() { lock.Close() }, nil
}

// Create makes an empty session file if there is none, and makes it
// durable, so a session exists from the moment it has an id and not only
// once something has been logged in it.
func Create(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	d, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
