#!/bin/sh
# Run the ACP Test Compatibility Kit (agentclientprotocol/acp-tck) against
# `agentloop acp`, with a mock Anthropic API standing in for the model.
# The binary is built into out/agentloop.
#
#   scripts/tck.sh            (or: make tck)
#
# Needs Go and uv. acp-tck is not on PyPI; uvx fetches it from GitHub at
# the commit pinned below, once, and caches it. The JSON report goes to
# out/tck.json. The script fails if any requirement FAILs, or if anything
# is SKIPPED other than the requirements listed in ALLOWED_SKIPS, which
# cover capabilities agentloop does not advertise.
set -eu

# acp-tck has no releases; this is main as of 2026-10-02.
TCK_REF=fa00ccdc13dfaa4b288a91877c1f3e468af4aa7d

# Requirements that may be SKIPPED, each for a capability agentloop does
# not advertise in initialize (or an opt-in flag this script does not pass).
# ACP-CONFIG-001 runs, since sessions carry the model as a config option.
# ACP-CONFIG-002 skips: it needs a second value to switch to, and the
# model option has only one.
ALLOWED_SKIPS="
ACP-DELETE-001 ACP-DELETE-002
ACP-ADDDIRS-001
ACP-MODES-001 ACP-MODES-002
ACP-CONFIG-002
ACP-PROMPTCAP-001 ACP-PROMPTCAP-002
ACP-AUTH-003 ACP-AUTH-004
"

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p out
work=$(mktemp -d)
mock_pid=
cleanup() {
	if [ -n "$mock_pid" ]; then kill "$mock_pid" 2>/dev/null || true; fi
	rm -rf "$work"
}
trap cleanup EXIT INT TERM

# The binary under test goes in out/, not bin/, so a run never replaces a
# bin/agentloop that something else is using.
go build -o out/agentloop ./cmd/agentloop
go build -o "$work/mockprovider" ./internal/cmd/mockprovider

"$work/mockprovider" >"$work/mock.url" 2>"$work/mock.err" &
mock_pid=$!
i=0
until [ -s "$work/mock.url" ]; do
	i=$((i + 1))
	if [ "$i" -gt 100 ]; then
		echo "tck: the mock provider did not start" >&2
		cat "$work/mock.err" >&2
		exit 1
	fi
	sleep 0.1
done
url=$(head -n 1 "$work/mock.url")

cat >"$work/request.json" <<JSON
{
  "provider": "anthropic",
  "model": "mock",
  "base_url": "$url",
  "api_key_command": ["printf", "mock-key"],
  "system": "You are under test.",
  "session_dir": "$work/sessions",
  "retry": {"retries": 0}
}
JSON

mkdir -p "$work/agent-cwd"
status=0
uvx --from "git+https://github.com/agentclientprotocol/acp-tck@$TCK_REF" acp-tck \
	--agent-cwd "$work/agent-cwd" \
	--cancel-prompt "MOCK-SLOW: keep busy until cancelled" \
	--report-json out/tck.json \
	-- "$root/out/agentloop" acp "$work/request.json" || status=$?

python3 -I - out/tck.json "$ALLOWED_SKIPS" "$status" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
allowed = set(sys.argv[2].split())
counts, bad = {}, []
for r in report["requirements"]:
    st = r["status"]
    counts[st] = counts.get(st, 0) + 1
    if st == "FAIL" or st == "NOT_TESTED" and r["tier"] == "MANDATORY":
        bad.append(f"{r['id']} {st}")
    elif st == "SKIPPED":
        if r["id"] in allowed:
            print(f"tck: skipped (not advertised) {r['id']}")
        else:
            reasons = "; ".join(t.get("message") or "" for t in r.get("tests", []))
            bad.append(f"{r['id']} SKIPPED unexpectedly: {reasons}")
print("tck:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
print("tck: verdict", "CONFORMANT" if report["verdict"]["conformant"] else "NOT CONFORMANT")
for b in bad:
    print("tck: problem:", b)
sys.exit(1 if bad or sys.argv[3] != "0" else 0)
PY
