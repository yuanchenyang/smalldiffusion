.PHONY: build test upload install-local

build:
	python -m build

test:
	python -m pytest

upload:
	python -m twine upload --repository pypi dist/*

install-local:
	python -m pip install -e .[dev,test]
