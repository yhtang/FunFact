.PHONY: lint test test-coverage docs cleandocs

VENV ?= venv

default: lint


lint:
	flake8 \
		--max-line-length=80 \
		--per-file-ignores "test_*.py":F401 \
		funfact/ example/

test:
	tox -e py38

test-coverage:
	tox -e coverage

docs:
	mkdocs build

cleandocs:
	rm -rf site/

