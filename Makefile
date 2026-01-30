.PHONY: all install test lint clean

all: install test

install:
	poetry install

test:
	poetry run pytest || test $$? -eq 5

lint:
	poetry run python -m py_compile tfidf_zones/tokenizer.py
	poetry run python -m py_compile tfidf_zones/tfidf_engine.py
	poetry run python -m py_compile tfidf_zones/scikit_engine.py
	poetry run python -m py_compile tfidf_zones/zones.py
	poetry run python -m py_compile tfidf_zones/formatter.py
	poetry run python -m py_compile tfidf_zones/runner.py
	poetry run python -m py_compile tfidf_zones/cli.py

clean:
	rm -rf dist/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
