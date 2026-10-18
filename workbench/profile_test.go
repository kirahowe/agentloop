package workbench

import (
	"regexp"
	"strings"
	"testing"

	"pgregory.net/rapid"
)

func TestProfileShape(t *testing.T) {
	p, err := Profile(Sandbox{Scratch: "/w/s/scratch", DenyRead: []string{"/Users/k/.ssh", "/Users/k/.codex*", "/w/s"}})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(p), "\n")
	if lines[0] != "(version 1)" || lines[1] != "(deny default)" {
		t.Fatalf("profile does not start deny-default:\n%s", p)
	}
	for _, bad := range []string{"mach-lookup", "allow network", "allow default", "iokit", "ipc-posix", "mach-register", "process-info* )", "user-preference"} {
		if strings.Contains(p, bad) {
			t.Errorf("profile holds %q:\n%s", bad, p)
		}
	}
	idx := func(s string) int {
		for i, l := range lines {
			if strings.Contains(l, s) {
				return i
			}
		}
		t.Fatalf("no line with %q in\n%s", s, p)
		return -1
	}
	read := idx("(allow file-read*)")
	for _, d := range []string{`(deny file-read* (subpath "/Users/k/.ssh"))`, `(deny file-read* (regex #"^/Users/k/\.codex"))`, `(deny file-read* (subpath "/w/s"))`} {
		if i := idx(d); i < read {
			t.Errorf("%s comes before the broad read rule, so it would not win", d)
		}
	}
	scratch := idx(`(allow file-read* file-write* (subpath "/w/s/scratch"))`)
	if scratch < idx(`(subpath "/w/s"))`) {
		t.Error("scratch comes before the deny list, so a deny on its parent would win")
	}
	if idx(`(allow file-read-metadata (literal "/w/s"))`) < 0 {
		t.Error("no metadata rule for the scratch directory's parent")
	}
	// The only writes are scratch and the standard streams.
	for _, l := range lines {
		if strings.Contains(l, "file-write") && !strings.Contains(l, "/w/s/scratch") && !strings.HasPrefix(l, "(allow file-write-data (literal \"/dev/null\")") {
			t.Errorf("unexpected write rule %s", l)
		}
	}
}

func TestProfileRefusesPathsItCannotHold(t *testing.T) {
	for _, p := range []string{"", "relative", "/", "/a/../b", "/a/", `/a"b`, `/a\b`, "/a\nb", "/a*b", "/a**", "*"} {
		if _, err := Profile(Sandbox{Scratch: "/s", DenyRead: []string{p}}); err == nil {
			t.Errorf("deny path %q was accepted", p)
		}
	}
	for _, s := range []string{"", "rel", "/", "/s*", `/s"`, "/s/"} {
		if _, err := Profile(Sandbox{Scratch: s}); err == nil {
			t.Errorf("scratch %q was accepted", s)
		}
	}
	for _, p := range []string{"/a", "/a b/c", "/a*", "/a/*", "/Library/Application Support/X"} {
		if _, err := Profile(Sandbox{Scratch: "/s", DenyRead: []string{p}}); err != nil {
			t.Errorf("deny path %q was refused: %v", p, err)
		}
	}
}

// Every path Profile accepts appears in it exactly as a deny rule, and
// nothing in a path can end its string or regex early.
func TestProfileProperty(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		paths := rapid.SliceOfN(rapid.StringMatching(`/[ -~]{1,20}`), 0, 6).Draw(t, "paths")
		p, err := Profile(Sandbox{Scratch: "/scratch/s", DenyRead: paths})
		if err != nil {
			return
		}
		for _, d := range paths {
			var want string
			if prefix, ok := strings.CutSuffix(d, "*"); ok {
				want = `(deny file-read* (regex #"^` + regexQuote(prefix) + `"))`
				// The quoted prefix matches only itself at the start.
				re := regexp.MustCompile("^" + regexQuote(prefix))
				if !re.MatchString(prefix+"x") || re.MatchString("a"+prefix) {
					t.Fatalf("regex for %q is wrong", prefix)
				}
			} else {
				want = `(deny file-read* (subpath "` + d + `"))`
			}
			if !strings.Contains(p, want+"\n") {
				t.Fatalf("no rule %s in\n%s", want, p)
			}
		}
		// Every line is one balanced form.
		for _, l := range strings.Split(strings.TrimSpace(p), "\n") {
			if strings.Count(l, `"`)%2 != 0 || !strings.HasPrefix(l, "(") || !strings.HasSuffix(l, ")") {
				t.Fatalf("line %q is not one form", l)
			}
		}
	})
}

func TestRegexQuoteIsLiteral(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := rapid.StringMatching(`[ -~]{0,20}`).Draw(t, "s")
		re := regexp.MustCompile("^" + regexQuote(s) + "$")
		if !re.MatchString(s) {
			t.Fatalf("%q does not match itself", s)
		}
	})
}

func TestDefaultDenyRead(t *testing.T) {
	d := DefaultDenyRead("/Users/k", "/private/var/folders/x/T")
	for _, want := range []string{"/Users/k/.ssh", "/Users/k/Library/Keychains", "/Users/k/.gnupg", "/Users/k/.codex*", "/Users/k/.claude*",
		"/Users/k/Library/Application Support/Google/Chrome", "/Users/k/Library/Safari", "/private/var/folders/x/T/agentloop-*"} {
		found := false
		for _, p := range d {
			found = found || p == want
		}
		if !found {
			t.Errorf("default deny list lacks %s", want)
		}
	}
	if _, err := Profile(Sandbox{Scratch: "/s", DenyRead: d}); err != nil {
		t.Fatal(err)
	}
}
