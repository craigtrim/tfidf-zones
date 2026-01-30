# tfidf-zones

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

CLI tool that classifies terms in text documents into three zones based on TF-IDF and document frequency:

- **Too Common** - top 10% by document frequency
- **Goldilocks** - 45th to 55th percentile by document frequency
- **Too Rare** - bottom 10% by document frequency

Useful for stylometric analysis, authorship attribution, and understanding term importance.

## Install

```bash
poetry install
```

## Usage

```bash
# Analyze a single file
poetry run tfidf-zones --file novel.txt

# Use scikit-learn engine with bigrams
poetry run tfidf-zones --file novel.txt --engine scikit --ngram 2

# Analyze a directory of .txt files
poetry run tfidf-zones --dir ./texts/

# Show top 25 terms per zone with custom chunk size
poetry run tfidf-zones --file novel.txt --top-k 25 --chunk-size 500
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | | Path to a single text file |
| `--dir` | | Path to a directory of `.txt` files |
| `--engine` | `pure` | TF-IDF engine: `pure` or `scikit` |
| `--top-k` | `10` | Number of terms per zone |
| `--ngram` | `1` | N-gram level (1-5, or 6 for skipgrams) |
| `--chunk-size` | `2000` | Tokens per chunk (min 100) |

Either `--file` or `--dir` is required (not both).

## How It Works

Text is tokenized, split into chunks, and scored with TF-IDF. Chunking a single document into sub-documents prevents IDF from collapsing to a constant. Terms are then bucketed into zones by their document-frequency percentile.

Two engines are available: a pure-Python implementation and a scikit-learn backed implementation. Both use smooth IDF (`log((1+N)/(1+DF)) + 1`) and produce comparable results.
