package acpagent

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// Inert is text an outsider may have written, made safe for a client that
// renders markdown and knows nothing of relay's _meta. Characters that
// draw as nothing show as \u{XXXX}, every link is defanged (http becomes
// hxxp, any other scheme's :// becomes [:]//, a dot inside a word becomes
// [.]), and the whole goes inside a code fence longer than any run of
// backticks in it, so no image is fetched and no link can be clicked
// before anyone approves anything.
func Inert(s string) string {
	t := defang(visible(s))
	n := 3
	for _, run := range backticks.FindAllString(t, -1) {
		n = max(n, len(run)+1)
	}
	f := strings.Repeat("`", n)
	return f + "text\n" + t + "\n" + f
}

var backticks = regexp.MustCompile("`+")

// hidden reports whether r draws as nothing, or could change how the
// text around it reads: format and control characters (line breaks and
// tabs apart), variation selectors and the blank fillers.
func hidden(r rune) bool {
	switch r {
	case '\n', '\t':
		return false
	case 'ᅟ', 'ᅠ', 'ㅤ', 'ﾠ', '⠀', '᠎':
		return true
	}
	return unicode.Is(unicode.Cf, r) || unicode.IsControl(r) || unicode.Is(unicode.Variation_Selector, r) ||
		unicode.Is(unicode.Co, r) || r == unicode.ReplacementChar
}

func visible(s string) string {
	var b strings.Builder
	for _, r := range s {
		if hidden(r) {
			fmt.Fprintf(&b, "\\u{%04x}", r)
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

var (
	scheme = regexp.MustCompile(`(?i)([a-z][a-z0-9+.\-]*)://`)
	dot    = regexp.MustCompile(`(\S)[.\x{3002}\x{FF0E}\x{FF61}](\S)`)
)

func defang(s string) string {
	s = scheme.ReplaceAllStringFunc(s, func(m string) string {
		name := strings.ToLower(strings.TrimSuffix(m, "://"))
		switch name {
		case "http":
			return "hxxp://"
		case "https":
			return "hxxps://"
		case "ftp":
			return "fxp://"
		case "hxxp", "hxxps", "fxp":
			return m
		}
		return strings.TrimSuffix(m, "://") + "[:]//"
	})
	// Go's regexp has no lookaround, so a dot between two dots ("a..b")
	// needs a second pass.
	for range 2 {
		s = dot.ReplaceAllString(s, "$1[.]$2")
	}
	return s
}
