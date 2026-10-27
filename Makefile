.PHONY: gate test lint format typecheck

gate:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src scripts app.py
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src scripts app.py

test:
	uv run pytest -q
