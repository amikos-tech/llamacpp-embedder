
APP_NAME=llama-embedder-server

.PHONY: build
build:
	go build -x -a -o ./bin/$(APP_NAME) ./cmd/server/main.go

.PHONY: test
test:
	go test -v ./...