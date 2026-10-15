// Command keycmd is an api_key_command for the isolation tests. It prints
// its first argument as the key and records what its process was given in
// the file named by its second argument: environment, argv and stdin.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

func main() {
	stdin, _ := io.ReadAll(os.Stdin)
	b, _ := json.Marshal(map[string]any{
		"env":   os.Environ(),
		"args":  os.Args,
		"stdin": string(stdin),
	})
	if len(os.Args) > 2 {
		os.WriteFile(os.Args[2], b, 0o644)
	}
	fmt.Println(os.Args[1])
}
