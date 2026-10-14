// Command envdump is a tool for the isolation tests. It records everything
// it can see of its own process in dump-<pid>.json in its working
// directory, and prints "ok".
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"syscall"
)

func main() {
	// Probe descriptors first, before the runtime opens any of its own.
	var fds []string
	for fd := 0; fd < 256; fd++ {
		if _, err := fcntl(fd); err == nil {
			fds = append(fds, fmt.Sprint(fd))
		}
	}
	// Read fd 0 directly: os.Stdin would open the runtime's poller, which is
	// a descriptor of this process and not one it was given.
	var stdin []byte
	buf := make([]byte, 4096)
	for {
		n, err := syscall.Read(0, buf)
		if n > 0 {
			stdin = append(stdin, buf[:n]...)
		}
		if n <= 0 || err != nil {
			break
		}
	}
	cwd, _ := os.Getwd()
	b, _ := json.Marshal(map[string]any{
		"env":   os.Environ(),
		"args":  os.Args,
		"stdin": string(stdin),
		"fds":   fds,
		"cwd":   cwd,
		"pgid":  syscall.Getpgrp(),
		"pid":   os.Getpid(),
	})
	os.WriteFile(fmt.Sprintf("dump-%d.json", os.Getpid()), b, 0o644)
	fmt.Println("ok")
}

func fcntl(fd int) (int, error) {
	r, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_GETFD, 0)
	if errno != 0 {
		return 0, errno
	}
	return int(r), nil
}
