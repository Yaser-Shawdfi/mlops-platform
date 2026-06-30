# Contributing to MLOps Platform

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Yaser-Shawdfi/mlops-platform.git
cd mlops-platform

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov ruff

# Copy environment file
cp .env.example .env
```

## Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run a specific test suite
python -m pytest tests/test_drift.py -v
```

## Code Quality

```bash
# Lint
make lint

# Auto-fix lint issues
make lint-fix

# Format
make format

# Check formatting (CI uses this)
make format-check
```

## Running the Demo

```bash
make demo
```

This runs the full ML lifecycle: train -> register -> drift detect -> alert -> retrain -> A/B test.

## Docker Stack

```bash
# Start all services
make up

# Services:
#   API:         http://localhost:8000/docs
#   MLflow:      http://localhost:5000
#   Prometheus:  http://localhost:9090
#   Grafana:     http://localhost:3000 (admin/admin)

# Stop all services
make down
```

## Adding a New Model

1. Add a data generator function in `src/models/model_wrappers.py`
2. Add a training function returning `(model, metrics, params)`
3. Register it in the `MODEL_REGISTRY` dict
4. Add tests in `tests/test_models.py`
5. Run `make test` to verify

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] Format passes (`make format-check`)
- [ ] Demo runs without error (`make demo`)
- [ ] No emojis in source code, configs, or documentation
- [ ] New features include tests