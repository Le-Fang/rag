.PHONY: build test test-integration run up down

build:
	go build -o bin/poc-rag ./cmd/app

test:
	go test ./...

# -p 1: the integration-test packages (qdrant_test, app_test) share one Qdrant
# instance and drop shared collections — parallel test binaries would collide.
# WARNING: several tests DROP the shared "documents" and "variants"
# collections to get a clean slate before running. Point this at a dev/local
# Qdrant only — running it against an instance with real data is data loss.
test-integration: up
	go test -tags integration -p 1 ./...

run: build
	./bin/poc-rag server --config .apprc.yaml

up:
	docker compose up -d --wait

down:
	docker compose down
