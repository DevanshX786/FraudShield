.PHONY: help setup up down train evaluate drift retrain test lint format api logs clean

PYTHON ?= python

help:
	@echo "FraudShield commands"
	@echo "  make setup     Install Python dependencies"
	@echo "  make up        Start local Docker stack"
	@echo "  make down      Stop local Docker stack"
	@echo "  make train     Run training pipeline"
	@echo "  make evaluate  Run model evaluation"
	@echo "  make drift     Generate and check drift"
	@echo "  make retrain   Run retraining pipeline"
	@echo "  make test      Run tests"
	@echo "  make lint      Run lint checks"
	@echo "  make format    Format Python code"
	@echo "  make api       Run FastAPI locally"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

up:
	docker compose up --build

down:
	docker compose down

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

drift:
	$(PYTHON) scripts/generate_inference_batch.py
	$(PYTHON) scripts/check_drift.py

retrain:
	$(PYTHON) -m src.retraining_pipeline

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m flake8 src scripts tests
	$(PYTHON) -m black --check src scripts tests

format:
	$(PYTHON) -m black src scripts tests

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

logs:
	docker compose logs -f api

clean:
	$(PYTHON) scripts/clean_artifacts.py
