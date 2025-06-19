.PHONY: init setup run

init:
	uv tool run pre-commit install

lint:
	uv tool run ruff check --fix src
	uv tool run ruff format src

sync-requirements:
	uv export --no-hashes --format requirements-txt > requirements.txt

run:
	uv run app.py
