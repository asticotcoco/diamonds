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

.PHONY: start-mlflow stop-mlflow
MLFLOW_PORT=5000
MLFLOW_PID_FILE=.mlflow.pid

start-mlflow:
	mlflow ui --port $(MLFLOW_PORT) & echo $$! > $(MLFLOW_PID_FILE)

stop-mlflow:
	@if [ -f $(MLFLOW_PID_FILE) ]; then \
		kill `cat $(MLFLOW_PID_FILE)` && rm $(MLFLOW_PID_FILE); \
		echo "MLflow server stopped"; \
	else \
		echo "No MLflow server running"; \
	fi
