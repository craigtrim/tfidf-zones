# tfidf-zones

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Downloads](https://img.shields.io/pypi/dd/tfidf-zones)](https://pypi.org/project/tfidf-zones/)
[![Downloads/Month](https://img.shields.io/pypi/dm/tfidf-zones)](https://pypi.org/project/tfidf-zones/)

CLI tool that classifies terms in text documents into three zones based on TF-IDF and document frequency:

- **Too Common** — high document frequency (df > 0.2N)
- **Goldilocks** — high TF-IDF score within a moderate DF band (3 ≤ df ≤ 0.2N, tfidf ≥ Q95)
- **Too Rare** — low document frequency (df < 3)

Useful for stylometric analysis, authorship attribution, and understanding term importance.

## Install

```bash
poetry install
```

## Usage

```bash
# Analyze a single file
poetry run tfidf-zones --file novel.txt --output results.csv

# Analyze with bigrams
poetry run tfidf-zones --file novel.txt --ngram 2 --output results.csv

# Analyze a directory of .txt files
poetry run tfidf-zones --dir ./texts/ --output results.csv

# Show top 25 terms per zone with custom chunk size
poetry run tfidf-zones --file novel.txt --top-k 25 --chunk-size 500 --output results.csv
```

## Recipes

**Find content-word bigrams across a corpus:**

```bash
poetry run tfidf-zones \
  --dir ./texts/ --limit 100 --output results.csv \
  --no-chunk --wordnet --ngram 2 --no-ngram-stopwords
```

Combines `--wordnet` (only real English words), `--ngram 2` (bigrams), and `--no-ngram-stopwords` (discard bigrams containing stop/function words like "of_the") to surface meaningful two-word terms.

**Find content phrases (trigrams and above):**

```bash
poetry run tfidf-zones \
  --dir ./texts/ --output results.csv \
  --no-chunk --wordnet --ngram 3 --no-ngram-stopwords
```

Increase `--ngram` to 3, 4, or 5 to find longer phrases. The stopword filter removes any n-gram where at least one token is a stop word or function word, so only content-rich phrases survive.

**Corpus analysis with post-processing filters:**

```bash
poetry run tfidf-zones \
  --dir ./texts/ --output results.csv \
  --no-chunk --wordnet --min-df 2 --min-tf 2
```

Use `--min-df` and `--min-tf` to remove terms that appear in too few documents or have too few total occurrences, reducing noise from hapax legomena.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | | Path to a single text file |
| `--dir` | | Path to a directory of `.txt` files |
| `--scikit` | off | Use scikit-learn TF-IDF engine (optional; supports all the same flags) |
| `--top-k` | `10` | Number of terms per zone |
| `--ngram` | `1` | N-gram level (1–5, or 6 for skipgrams) |
| `--chunk-size` | `2000` | Tokens per chunk (min 100) |
| `--limit` | all | Randomly select N files from directory (requires `--dir`) |
| `--no-chunk` | off | Each file = one document, no chunking (requires `--dir`) |
| `--wordnet` | off | Only recognized English words participate in TF-IDF |
| `--no-ngram-stopwords` | off | Discard n-grams containing stop/function words (requires `--ngram` ≥ 2) |
| `--min-df` | | Remove terms with document frequency below this value |
| `--min-tf` | | Remove terms with term frequency below this value |
| `--output` | | Output CSV file path (required) |

Either `--file` or `--dir` is required (not both).

## How It Works

Text is tokenized, split into chunks, and scored with TF-IDF. Chunking a single document into sub-documents prevents IDF from collapsing to a constant. Terms are then bucketed into zones by their document-frequency percentile.

## Engines

The default engine is a **pure-Python implementation** that requires no heavy dependencies and generally produces better results for zone analysis. It uses smooth IDF (`log((1+N)/(1+DF)) + 1`) with full control over tokenization, n-gram generation, and scoring.

A **scikit-learn engine** (`--scikit`) is also available for users familiar with the scikit-learn ecosystem or who want to compare results against its `TfidfVectorizer`. Both engines use the same IDF formula and produce comparable output. scikit-learn is an optional dependency — install it with:

```bash
pip install tfidf-zones[scikit]
```

or if using Poetry:

```bash
poetry install -E scikit
```
