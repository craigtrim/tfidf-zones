# =============================================================================
# CLI FORMATTER
# =============================================================================
#
# Rich CLI output for TF-IDF Zone Analysis results.
#
# =============================================================================

from __future__ import annotations

from rich.console import Console

console = Console()
_err_console = Console(stderr=True)


def _cfg(label: str, value: str) -> str:
    """Format a config line with dim label and bold value."""
    return f"  [dim]{label:<13}[/dim] [bold]{value}[/bold]"


# ── Output Functions ─────────────────────────────────────────────────────────


def print_header() -> None:
    """Print the TF-IDF Zone Analysis header."""
    console.print()
    console.rule("[bold cyan]TF-IDF Zone Analysis[/bold cyan]", style="cyan")
    console.print()


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
    console.print(_cfg("file", filename))
    console.print(_cfg("engine", engine))
    console.print(_cfg("ngram_type", ngram_type))
    console.print(_cfg("text_length", f"{text_length:,} chars"))
    console.print(_cfg("tokens", f"{tokens:,}"))
    console.print(_cfg("chunks", str(chunks)))
    console.print(_cfg("chunk_size", str(chunk_size)))
    if wordnet:
        console.print(_cfg("wordnet", "on"))
    if no_ngram_stopwords:
        console.print(_cfg("stopwords", "on"))
    if min_df is not None:
        console.print(_cfg("min_df", str(min_df)))
    if min_tf is not None:
        console.print(_cfg("min_tf", str(min_tf)))
    console.print()


def print_df_stats(df_stats: dict) -> None:
    """Print the DF distribution statistics."""
    console.print(f"  [bold cyan]DF Distribution[/bold cyan]")
    console.print(f"  [dim]{'═' * 40}[/dim]")
    console.print(_cfg("mean", str(df_stats["mean"])))
    console.print(_cfg("median", str(df_stats["median"])))
    console.print(_cfg("mode", str(df_stats["mode"])))
    console.print()


def print_zone(label: str, color: str, terms: list[dict]) -> None:
    """Print a single zone table."""
    console.print(f"  [{color} bold]{label}[/{color} bold]")
    console.print(f"  [dim]{'─' * 68}[/dim]")
    console.print(f"  [dim]term                     tf     df     idf      tfidf[/dim]")
    for t in terms:
        term = t["term"]
        tf = t.get("tf", 0)
        score = t["score"]
        df = t["df"]
        idf = t["idf"]
        console.print(f"  {term:<24} tf={tf:<5} df={df:<4} idf={idf:.4f}  tfidf={score:.4f}")
    console.print()


def print_zones(zones: dict) -> None:
    """Print all three zone tables."""
    print_zone("TOO COMMON  (df > 0.2N)", "yellow", zones["too_common"])
    print_zone("GOLDILOCKS  (tfidf >= Q95, 3 <= df <= 0.2N)", "green", zones["goldilocks"])
    print_zone("TOO RARE    (df < 3)", "magenta", zones["too_rare"])


def print_output(path: str, rows: int, size_bytes: int) -> None:
    """Print output file info."""
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    console.print(f"  [bold cyan]Output[/bold cyan]")
    console.print(f"  [dim]{'═' * 40}[/dim]")
    console.print(_cfg("file", path))
    console.print(_cfg("rows", f"{rows:,}"))
    console.print(_cfg("size", size_str))
    console.print()


def print_footer(elapsed: float) -> None:
    """Print the footer with elapsed time."""
    secs = int(round(elapsed))
    console.rule(style="dim")
    console.print(f"  [green]Completed in {secs}s[/green]")
    console.print()


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
    console.print(_cfg("directory", dirname))
    console.print(_cfg("engine", engine))
    console.print(_cfg("ngram_type", ngram_type))
    console.print(_cfg("files", str(file_count)))
    console.print(_cfg("text_length", f"{total_text_length:,} chars"))
    console.print(_cfg("tokens", f"{tokens:,}"))
    if chunk_size == 0:
        console.print(_cfg("documents", str(chunks)))
        console.print(_cfg("chunking", "off (1 file = 1 doc)"))
    else:
        console.print(_cfg("chunks", str(chunks)))
        console.print(_cfg("chunk_size", str(chunk_size)))
    if wordnet:
        console.print(_cfg("wordnet", "on"))
    if no_ngram_stopwords:
        console.print(_cfg("stopwords", "on"))
    if min_df is not None:
        console.print(_cfg("min_df", str(min_df)))
    if min_tf is not None:
        console.print(_cfg("min_tf", str(min_tf)))
    console.print()


def print_progress(current: int, total: int, filename: str) -> None:
    """Print file reading progress for directory mode (single updating line).

    Uses raw print with ANSI for \\r carriage-return overwrite, which
    rich.console.print does not support.
    """
    dim = "\033[2m"
    bold = "\033[1m"
    reset = "\033[0m"
    end = "\n" if current == total else ""
    print(f"\r  {dim}[{current}/{total}]{reset} {bold}Files Processed{reset}", end=end, flush=True)


def print_error(message: str) -> None:
    """Print an error message."""
    _err_console.print(f"  [bold red]ERROR[/bold red]  {message}")
