
APP_NAME=llama-embedder-server

.PHONY: build
build:
	go build -x -a -o ./bin/$(APP_NAME) ./cmd/server/main.go

.PHONY: test
test:
	go test -v ./...

.PHONY: run
run:
	LLAMA_CACHED_MODELS="ChristianAzinn/snowflake-arctic-embed-s-gguf/snowflake-arctic-embed-s-f16.GGUF;leliuga/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2.Q4_0.gguf" go run ./cmd/server/main.go

.PHONY: lint
lint:
	golangci-lint run

.PHONY: lint-fix
lint-fix:
	golangci-lint run --fix ./...

.PHONYT: clean
clean:
	rm -rf ./bin