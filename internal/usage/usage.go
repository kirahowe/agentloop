// Package usage is the error for input the operator can fix: an unusable
// request, command line or session path. The command exits 2 for it.
package usage

import "fmt"

// Error is an unusable request or command line.
type Error struct{ msg string }

func (e *Error) Error() string { return e.msg }

// Errorf formats an Error.
func Errorf(format string, args ...any) error {
	return &Error{fmt.Sprintf(format, args...)}
}
