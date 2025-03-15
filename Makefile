.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: mypy
mypy:
	uv run mypy .

.PHONY: tests
tests:
	uv run pytest

.PHONY: build-docs
build-docs:
	uv run mkdocs build

.PHONY: serve-docs
serve-docs:
	uv run mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	uv run mkdocs gh-deploy --force --verbose
