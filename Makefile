.PHONY: help install test test-cov lint format demo up down clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint: ## Run ruff linter
	ruff check src/ tests/ scripts/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/ scripts/

format: ## Format code with ruff
	ruff format src/ tests/ scripts/

format-check: ## Check formatting without modifying
	ruff format --check src/ tests/ scripts/

demo: ## Run the full lifecycle demo
	python scripts/demo.py

up: ## Start Docker Compose stack (MLflow + API + Prometheus + Grafana)
	docker-compose up --build -d

down: ## Stop Docker Compose stack
	docker-compose down

clean: ## Remove generated data and caches
	rm -rf data/mlflow.db data/ab_tests data/alerts data/reports data/artifacts mlruns .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true