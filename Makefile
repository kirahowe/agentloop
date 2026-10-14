.PHONY: build test update

build:
	go build -o bin/agentloop ./cmd/agentloop

test:
	@test -z "$$(gofmt -l .)" || { echo "gofmt needed:"; gofmt -l .; exit 1; }
	go vet ./...
	go test -race ./...

# Keep dependencies current.
update:
	go get -u ./...
	go mod tidy
	$(MAKE) test
