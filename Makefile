
SHELL = /bin/bash
.SHELLFLAGS = -o pipefail -c

.PHONY: help
help: ## Print info about all commands
	@echo "Commands:"
	@echo
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[01;32m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: deps
deps: ## Install dependencies (eg, create virtualenv using pipenv)
	pipenv install --dev --deploy

.PHONY: test
test: ## Run all tests and lints
	pipenv run python -m pytest --ignore extra/
	pipenv run pylint -j 0 -E pdf_trio tests/*.py
	#pipenv run flake8 tests/ pdf_trio/ --count --select=E9,F63,F7,F82 --show-source --statistics

.PHONY: coverage
coverage: ## Run tests, collecting code coverage
	pipenv run pytest --cov --cov-report html --ignore extra/

.PHONY: serve
serve: ## Run web service locally
	pipenv run flask run -h localhost

.PHONY: fetch-models
fetch-models: ## Download model files from archive.org
	./fetch_models.sh
