// Package workbench is a tool that hands a task to a child model session
// whose only tool, sh, runs each command under macOS sandbox-exec. The
// sandbox has no network of any kind, reads the disk except a deny list,
// and writes only to a scratch directory of its own. The caller gets
// only the child's answer, checked against a JSON Schema the caller
// gives, and labelled by the quarantined reader's rule.
//
// The profile is the pure function Profile. The shell around it (running
// a command, killing what it left behind) is in run_darwin.go.
package workbench

import (
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// sysctlRules are the sysctl values a command may read: what the
// hardware is, what the system is, and the few kernel numbers the C
// library asks for. Reading sysctl at large would hand over
// kern.procargs2, which is every other process's command line, where a
// password or a token can sit.
//
// The tests found that without these, sort, shasum, perl, rg and
// python3 all fail; with them, nothing a workbench command plausibly
// runs does.
var sysctlRules = []string{
	`(sysctl-name-prefix "hw.")`,
	`(sysctl-name-prefix "kern.os")`,
	`(sysctl-name-prefix "machdep.cpu.")`,
	`(sysctl-name "kern.argmax")`,
	`(sysctl-name "kern.boottime")`,
	`(sysctl-name "kern.hostname")`,
	`(sysctl-name "kern.maxfiles")`,
	`(sysctl-name "kern.maxproc")`,
	`(sysctl-name "kern.ostype")`,
	`(sysctl-name "kern.usrstack64")`,
	`(sysctl-name "kern.version")`,
	`(sysctl-name "vm.overcommit")`,
}

// A Sandbox is what one session's profile allows.
type Sandbox struct {
	// Scratch is the only directory the commands may write, and they may
	// read it whatever DenyRead says. Absolute, symlinks resolved.
	Scratch string
	// DenyRead are paths the commands may not read, below or at each.
	// An entry that ends in "*" denies every path that starts with what
	// comes before it, so "/Users/k/.codex*" covers .codex and
	// .codex-work. Absolute, symlinks resolved.
	DenyRead []string
}

// Profile is the sandbox-exec profile for s. It denies everything, then
// allows only:
//
//   - forking and running programs, which stay in the sandbox;
//   - signals to processes in the same sandbox, and nothing outside it;
//   - reading the sysctl values in sysctlRules;
//   - reading any file, except under DenyRead;
//   - reading and writing Scratch;
//   - writing /dev/null and the process's own standard output and error.
//
// There is no network rule, so every network operation (IP, unix
// sockets, binding) is denied, and no mach-lookup rule, so no system
// service can be reached: not LaunchServices (open), AppleEvents
// (osascript), the pasteboard, DNS (mDNSResponder), Spotlight or launchd.
// The tests found that sh, coreutils, find, grep, rg, python3, sqlite3,
// jq and pdftotext need none.
//
// Later rules win in a profile, which is why the order matters: the deny
// list comes after the broad read rule, and Scratch after the deny list.
func Profile(s Sandbox) (string, error) {
	if err := checkPath(s.Scratch, false); err != nil {
		return "", fmt.Errorf("scratch: %w", err)
	}
	deny := append([]string(nil), s.DenyRead...)
	sort.Strings(deny)
	var b strings.Builder
	b.WriteString("(version 1)\n")
	b.WriteString("(deny default)\n")
	// Said again so that a reader of the profile need not know that
	// deny default covers it.
	b.WriteString("(deny network*)\n")
	b.WriteString("(allow process-fork)\n")
	b.WriteString("(allow process-exec)\n")
	b.WriteString("(allow signal (target same-sandbox))\n")
	b.WriteString("(allow process-info* (target same-sandbox))\n")
	for _, rule := range sysctlRules {
		fmt.Fprintf(&b, "(allow sysctl-read %s)\n", rule)
	}
	b.WriteString("(allow file-read*)\n")
	seen := map[string]bool{}
	for _, p := range deny {
		if seen[p] {
			continue
		}
		seen[p] = true
		if err := checkPath(p, true); err != nil {
			return "", fmt.Errorf("deny_read %q: %w", p, err)
		}
		if prefix, ok := strings.CutSuffix(p, "*"); ok {
			fmt.Fprintf(&b, "(deny file-read* (regex #\"^%s\"))\n", regexQuote(prefix))
		} else {
			fmt.Fprintf(&b, "(deny file-read* (subpath %s))\n", stringQuote(p))
		}
	}
	// The scratch directory's ancestors may be on the deny list (it lives
	// in a session directory that is). A command can still stat them, or
	// it could not cd into its own scratch.
	for _, a := range ancestors(s.Scratch) {
		fmt.Fprintf(&b, "(allow file-read-metadata (literal %s))\n", stringQuote(a))
	}
	fmt.Fprintf(&b, "(allow file-read* file-write* (subpath %s))\n", stringQuote(s.Scratch))
	b.WriteString("(allow file-write-data (literal \"/dev/null\") (literal \"/dev/stdout\") (literal \"/dev/stderr\") (regex #\"^/dev/fd/[0-9]+$\"))\n")
	return b.String(), nil
}

// checkPath refuses a path the profile cannot hold exactly: relative, not
// clean, or with a character that would need quoting in the profile's
// strings. prefix allows one trailing "*".
func checkPath(p string, prefix bool) error {
	body := p
	if prefix {
		body = strings.TrimSuffix(p, "*")
	}
	switch {
	case body == "":
		return errors.New("empty")
	case !filepath.IsAbs(body):
		return errors.New("not absolute")
	case body == "/":
		return errors.New("the whole disk")
	case strings.HasSuffix(p, "*") && !prefix:
		return errors.New("a wildcard is not allowed here")
	// A prefix may end in "/" ("/x/*" is everything in /x); a path may not.
	case filepath.Clean(body) != strings.TrimSuffix(body, "/"),
		body == p && strings.HasSuffix(body, "/"):
		return errors.New("not a clean path")
	}
	for _, r := range body {
		if r < 0x20 || r == 0x7f || r == '"' || r == '\\' || r == '*' {
			return fmt.Errorf("holds %q, which a deny path may not", r)
		}
	}
	return nil
}

// ancestors are the directories above p, nearest last, without "/".
func ancestors(p string) []string {
	var out []string
	for d := filepath.Dir(p); d != "/" && d != "."; d = filepath.Dir(d) {
		out = append([]string{d}, out...)
	}
	return out
}

// stringQuote is p as a profile string. checkPath has refused anything
// that would need escaping.
func stringQuote(p string) string { return `"` + p + `"` }

// regexQuote escapes every character a profile regex treats specially.
func regexQuote(s string) string {
	var b strings.Builder
	for _, r := range s {
		if strings.ContainsRune(`\.^$|?*+()[]{}`, r) {
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

// DefaultDenyRead are the paths no session may read, whatever the
// caller adds: where this user keeps keys, passwords, tokens and browser
// profiles, coding agents' settings and sessions, other workbench
// sessions, and agentloop's own credential files. home is the user's home
// directory and tmp the user's temporary directory, both with symlinks
// resolved.
func DefaultDenyRead(home, tmp string) []string {
	h := func(rel string) string { return filepath.Join(home, rel) }
	out := []string{
		h("Library/Keychains"),
		h(".ssh"),
		h(".gnupg"),
		h(".aws"),
		h(".netrc"),
		h(".git-credentials"),
		h(".docker"),
		h(".kube"),
		h(".npmrc"),
		h(".pypirc"),
		h(".password-store"),
		h(".config/gh"),
		h(".config/op"),
		h(".config/gcloud"),
		h(".azure"),
		h(".gem/credentials"),
		h(".cargo/credentials"),
		h(".cargo/credentials.toml"),
		h(".m2/settings.xml"),
		h(".terraform.d"),
		h(".kaggle"),
		h(".huggingface"),
		h(".cache/huggingface"),
		h(".codex") + "*",
		h(".claude") + "*",
		h("Library/Cookies"),
		h("Library/Safari"),
		h("Library/Containers/com.apple.Safari"),
		h("Library/Application Support/Google/Chrome"),
		h("Library/Application Support/Chromium"),
		h("Library/Application Support/BraveSoftware"),
		h("Library/Application Support/Microsoft Edge"),
		h("Library/Application Support/Arc"),
		h("Library/Application Support/Vivaldi"),
		h("Library/Application Support/com.operasoftware.Opera"),
		h("Library/Application Support/Firefox"),
		h("Library/Application Support/1Password"),
		h("Library/Application Support/Bitwarden"),
		h("Library/Group Containers/2BUA8C4S2C.com.1password"),
		"/Volumes/.timemachine",
	}
	if tmp != "" {
		out = append(out, filepath.Join(tmp, "agentloop-")+"*")
	}
	return out
}
