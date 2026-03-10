.PHONY: lint lint-fix format lint-all

lint:
	@echo "checking lint with ruff"
	poetry run ruff check .

lint-fix:
	@echo "auto fix lint with ruff"
	poetry run ruff check . --fix

format:
	@echo "checking format" 
	poetry run ruff format .

lint-all: lint lint-fix format