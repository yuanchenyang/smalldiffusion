.PHONY: build test test-slow publish install-local

build:
	uv build
	
test:
	uv run pytest

test-slow:
	uv run pytest --run_slow

publish:
	uv publish

install-local:
	uv sync --all-extras
