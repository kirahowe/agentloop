.PHONY: build test live update tck

build:
	go build -o bin/agentloop ./cmd/agentloop

test:
	@test -z "$$(gofmt -l .)" || { echo "gofmt needed:"; gofmt -l .; exit 1; }
	go vet ./...
	go test -race ./...

# Smoke tests against the real APIs. They spend a few cents of the key in the
# macOS keychain. The OpenAI test skips because no OpenAI key command is given;
# add -openai-key-command '<command>' after -args to run it too.
live:
	go test -count=1 -v -run Live ./anthropic ./openai -args -live \
		-anthropic-key-command 'security find-generic-password -s relay-anthropic -w' \
		-anthropic-model claude-haiku-5-5

# Keep dependencies current.
update:
	go get -u ./...
	go mod tidy
	$(MAKE) test

tck:
	scripts/tck.sh
