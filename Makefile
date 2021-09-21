.PHONY: setup dev docs test test-coverage

VENV ?= venv

default: lint


lint:
	flake8 --max-line-length=80 funfact/ example/

test:
	tox -e py38

test-coverage:
	tox -e coverage

docs:
	m2r2 CHANGELOG.md
	cd docs && make html
