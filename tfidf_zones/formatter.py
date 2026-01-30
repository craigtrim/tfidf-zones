# =============================================================================
# CLI FORMATTER
# =============================================================================
#
# Colored CLI table output replicating the analyze-document.sh shell script.
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
) -> None:
    """Print the summary block."""
    print(f"  {_c(DIM, 'file')}         {_c(WHITE, filename)}")
    print(f"  {_c(DIM, 'engine')}       {_c(WHITE, engine)}")
    print(f"  {_c(DIM, 'ngram_type')}   {_c(WHITE, ngram_type)}")
    print(f"  {_c(DIM, 'text_length')}  {_c(WHITE, f'{text_length:,} chars')}")
    print(f"  {_c(DIM, 'tokens')}       {_c(WHITE, f'{tokens:,}')}")
    print(f"  {_c(DIM, 'chunks')}       {_c(WHITE, str(chunks))}")
    print(f"  {_c(DIM, 'chunk_size')}   {_c(WHITE, str(chunk_size))}")
    print(f"  {_c(DIM, 'elapsed')}      {_c(GREEN, f'{elapsed:.3f}s')}")
    print()


def print_df_stats(df_stats: dict) -> None:
    """Print the DF distribution statistics."""
    print(f"  {_c(BOLD + CYAN, 'DF Distribution')}")
    print(f"  {_c(DIM, '═' * 40)}")
    print(f"  {_c(DIM, 'mean')}         {_c(WHITE, str(df_stats['mean']))}")
    print(f"  {_c(DIM, 'median')}       {_c(WHITE, str(df_stats['median']))}")
    print(f"  {_c(DIM, 'mode')}         {_c(WHITE, str(df_stats['mode']))}")
    print()


def print_zone(label: str, color: str, terms: list[dict]) -> None:
    """Print a single zone table."""
    print(f"  {_c(color + BOLD, label)}")
    print(f"  {_c(DIM, '─' * 60)}")
    print(f"  {_c(DIM, 'term                     score      df     idf')}")
    for t in terms:
        term = t["term"]
        score = t["score"]
        df = t["df"]
        idf = t["idf"]
        print(f"  {term:<24} {score:.6f}   df={df}  idf={idf:.4f}")
    print()


def print_zones(zones: dict) -> None:
    """Print all three zone tables."""
    print_zone("TOO COMMON  (top 10%)", YELLOW, zones["too_common"])
    print_zone("GOLDILOCKS  (45th\u201355th pct)", GREEN, zones["goldilocks"])
    print_zone("TOO RARE    (bottom 10%)", MAGENTA, zones["too_rare"])


def print_footer() -> None:
    """Print the footer."""
    print(f"  {_c(DIM, 'Done.')}")
    print()


def print_file_separator(filename: str) -> None:
    """Print a separator between files in directory mode."""
    print()
    print(f"  {_c(DIM, '━' * 60)}")
    print(f"  {_c(BOLD + WHITE, filename)}")
    print(f"  {_c(DIM, '━' * 60)}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  {_c(RED + BOLD, 'ERROR')}  {message}", file=sys.stderr)
