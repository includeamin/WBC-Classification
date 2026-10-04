.DEFAULT_GOAL := help

RUN    := poetry run
CONFIG ?= configs/resnet18.yaml
CKPT   ?=
IMAGES ?=

.PHONY: help install install-docs lint format typecheck test check \
        docs-cli docs docs-serve build data train evaluate predict clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the project with dev tools and the Kaggle extra
	poetry install --with dev --extras kaggle

install-docs: ## Also install the docs tooling
	poetry install --with dev,docs --extras kaggle

lint: ## Run ruff (lint + format check)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Auto-fix lint issues and format the code
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## Run mypy
	$(RUN) mypy src

test: ## Run the test suite
	$(RUN) pytest -q

check: lint typecheck test ## Run everything CI runs (lint, mypy, tests)

docs-cli: ## Generate the CLI reference page
	$(RUN) typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md

docs: docs-cli ## Build the docs site (strict) into site/
	$(RUN) mkdocs build --strict

docs-serve: docs-cli ## Serve the docs locally with live reload
	$(RUN) mkdocs serve

build: ## Build the wheel and sdist into dist/
	poetry build

data: ## Download the Kaggle dataset into data/
	$(RUN) wbc download-data --dest data

train: ## Train a model: make train [CONFIG=configs/resnet18.yaml]
	$(RUN) wbc -v train -c $(CONFIG)

evaluate: ## Evaluate on TEST: make evaluate CKPT=runs/<run>/best.pt
	@test -n "$(CKPT)" || { echo "Usage: make evaluate CKPT=runs/<run>/best.pt"; exit 1; }
	$(RUN) wbc evaluate $(CKPT) --output $(dir $(CKPT))test_metrics.json

predict: ## Predict images: make predict CKPT=runs/<run>/best.pt IMAGES=path/to/cell.jpeg
	@test -n "$(CKPT)" -a -n "$(IMAGES)" || { echo "Usage: make predict CKPT=runs/<run>/best.pt IMAGES=<file-or-dir>"; exit 1; }
	$(RUN) wbc predict $(IMAGES) -c $(CKPT)

clean: ## Remove build outputs and caches
	rm -rf dist site .pytest_cache .mypy_cache .ruff_cache docs/reference/cli.md
	find . -name __pycache__ -type d -not -path './.git/*' -prune -exec rm -rf {} +
