.PHONY: build up down logs shell test lint clean

# Default target
all: build up

# Build all Docker images
build:
	docker compose build

# Start services in the background
up:
	docker compose up -d

# Stop services and keep volumes
down:
	docker compose down

# Follow logs from all services
logs:
	docker compose logs -f

# Open an interactive shell inside the backend container
shell:
	docker compose exec backend sh

# Run python tests inside the backend container
test:
	docker compose exec backend pytest

# Run linter inside the backend container
lint:
	docker compose exec backend ruff check .

# Clean up containers, volumes, networks, and local cache artifacts
clean:
	docker compose down -v --remove-orphans
