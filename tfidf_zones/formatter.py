# =============================================================================
# CLI FORMATTER
# =============================================================================
#
# Colored CLI output for TF-IDF Zone Analysis results.
#
# =============================================================================

from __future__ import annotations

import sys

# ── ANSI Colors ──────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"


def _no_color() -> bool:
    """Check if color output should be disabled."""
    return not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Apply ANSI color code to text, respecting NO_COLOR."""
    if _no_color():
        return text
    return f"{code}{text}{RESET}"


def _kv(key: str, value: str) -> None:
    """Print a key-value pair with consistent formatting."""
    print(f"  {_c(CYAN, f'{key:<13}')} {_c(WHITE, value)}")


# ── Output Functions ─────────────────────────────────────────────────────────


def print_header() -> None:
    """Print the TF-IDF Zone Analysis header."""
    print()
    print(f"  {_c(BOLD + CYAN, 'TF-IDF Zone Analysis')}")
    print(f"  {_c(DIM, '═' * 40)}")
    print()


def print_summary(
    filename: str,
    engine: str,
    ngram_type: str,
    text_length: int,
    tokens: int,
    chunks: int,
    chunk_size: int,
    elapsed: float,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
    min_df: int | None = None,
    min_tf: int | None = None,
) -> None:
    """Print the summary block."""
    _kv("file", filename)
    _kv("engine", engine)
    _kv("ngram_type", ngram_type)
    _kv("text_length", f"{text_length:,} chars")
    _kv("tokens", f"{tokens:,}")
    _kv("chunks", str(chunks))
    _kv("chunk_size", str(chunk_size))
    if wordnet:
        _kv("wordnet", "on")
    if no_ngram_stopwords:
        _kv("stopwords", "on")
    if min_df is not None:
        _kv("min_df", str(min_df))
    if min_tf is not None:
        _kv("min_tf", str(min_tf))
    print()


def print_df_stats(df_stats: dict) -> None:
    """Print the DF distribution statistics."""
    print(f"  {_c(BOLD + CYAN, 'DF Distribution')}")
    print(f"  {_c(DIM, '═' * 40)}")
    _kv("mean", str(df_stats["mean"]))
    _kv("median", str(df_stats["median"]))
    _kv("mode", str(df_stats["mode"]))
    print()


def print_zone(label: str, color: str, terms: list[dict]) -> None:
    """Print a single zone table."""
    print(f"  {_c(color + BOLD, label)}")
    print(f"  {_c(DIM, '─' * 68)}")
    print(f"  {_c(DIM, 'term                     tf     df     idf      tfidf')}")
    for t in terms:
        term = t["term"]
        tf = t.get("tf", 0)
        score = t["score"]
        df = t["df"]
        idf = t["idf"]
        print(f"  {term:<24} tf={tf:<5} df={df:<4} idf={idf:.4f}  tfidf={score:.4f}")
    print()


def print_zones(zones: dict) -> None:
    """Print all three zone tables."""
    print_zone("TOO COMMON  (df > 0.2N)", YELLOW, zones["too_common"])
    print_zone("GOLDILOCKS  (tfidf >= Q95, 3 <= df <= 0.2N)", GREEN, zones["goldilocks"])
    print_zone("TOO RARE    (df < 3)", MAGENTA, zones["too_rare"])


def print_footer(elapsed: float) -> None:
    """Print the footer with elapsed time."""
    secs = int(round(elapsed))
    print(f"  {_c(DIM, '─' * 40)}")
    print(f"  {_c(GREEN, f'Completed in {secs}s')}")
    print()


def print_corpus_summary(
    dirname: str,
    engine: str,
    ngram_type: str,
    file_count: int,
    total_text_length: int,
    tokens: int,
    chunks: int,
    chunk_size: int,
    elapsed: float,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
    min_df: int | None = None,
    min_tf: int | None = None,
) -> None:
    """Print the summary block for corpus (directory) mode."""
    _kv("directory", dirname)
    _kv("engine", engine)
    _kv("ngram_type", ngram_type)
    _kv("files", str(file_count))
    _kv("text_length", f"{total_text_length:,} chars")
    _kv("tokens", f"{tokens:,}")
    if chunk_size == 0:
        _kv("documents", str(chunks))
        _kv("chunking", "off (1 file = 1 doc)")
    else:
        _kv("chunks", str(chunks))
        _kv("chunk_size", str(chunk_size))
    if wordnet:
        _kv("wordnet", "on")
    if no_ngram_stopwords:
        _kv("stopwords", "on")
    if min_df is not None:
        _kv("min_df", str(min_df))
    if min_tf is not None:
        _kv("min_tf", str(min_tf))
    print()


def print_progress(current: int, total: int, filename: str) -> None:
    """Print file reading progress for directory mode (single updating line)."""
    end = "\n" if current == total else ""
    print(f"\r  {_c(DIM, f'[{current}/{total}]')} {_c(WHITE, 'Files Processed')}", end=end, flush=True)


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  {_c(RED + BOLD, 'ERROR')}  {message}", file=sys.stderr)
