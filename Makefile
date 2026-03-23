.PHONY: build test upload install-local

build:
	uv run python -m build

test:
	uv run pytest

test-slow:
	uv run pytest --run_slow

upload:
	uv run twine upload --repository pypi dist/*

install-local:
	uv sync --extra dev --extra test --extra examples
