# =============================================================================
# RUNNER
# =============================================================================
#
# Orchestrates file reading, engine dispatch, zone classification, and timing.
#
# =============================================================================

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tfidf_zones.tfidf_engine import EngineResult
from tfidf_zones.zones import classify_zones


@dataclass
class AnalysisResult:
    """Result of analyzing a file or corpus."""

    filename: str
    text_length: int
    engine_name: str
    engine_result: EngineResult
    zones: dict
    elapsed: float
    chunk_size: int
    file_count: int | None = None


def analyze_file(
    file_path: Path,
    engine: str = "pure",
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 10,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
) -> AnalysisResult:
    """Read a file and run TF-IDF zone analysis.

    Args:
        file_path: Path to the input text file.
        engine: Engine to use: "pure" or "scikit".
        ngram: N-gram level (1-6).
        chunk_size: Tokens per chunk.
        top_k: Terms per zone.

    Returns:
        AnalysisResult with engine output and zone classification.
    """
    text = file_path.read_text(encoding="utf-8")

    start = time.perf_counter()

    if engine == "scikit":
        from tfidf_zones.scikit_engine import run
    else:
        from tfidf_zones.tfidf_engine import run

    engine_result = run(text, ngram=ngram, chunk_size=chunk_size, top_k=top_k, wordnet=wordnet, no_ngram_stopwords=no_ngram_stopwords)
    zones = classify_zones(engine_result.all_scored, top_k=top_k, chunk_count=engine_result.chunk_count)

    elapsed = time.perf_counter() - start

    return AnalysisResult(
        filename=file_path.name,
        text_length=len(text),
        engine_name=engine,
        engine_result=engine_result,
        zones=zones,
        elapsed=elapsed,
        chunk_size=chunk_size,
    )


def _collect_files(
    dir_path: Path,
    limit: int | None,
    on_progress: Callable[[int, int, str], None] | None,
) -> tuple[list[Path], list[str]]:
    """Collect and read .txt files from a directory.

    Returns:
        Tuple of (file_paths, file_texts).

    Raises:
        FileNotFoundError: If no .txt files found.
    """
    files = sorted(dir_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {dir_path}")

    if limit is not None:
        if limit >= len(files):
            print(
                f"  WARNING: --limit {limit} >= total files {len(files)}, using all files",
                file=sys.stderr,
            )
        else:
            files = sorted(random.sample(files, limit))

    total = len(files)
    texts = []
    for i, f in enumerate(files, 1):
        if on_progress:
            on_progress(i, total, f.name)
        texts.append(f.read_text(encoding="utf-8"))

    return files, texts


def analyze_corpus(
    dir_path: Path,
    engine: str = "pure",
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 10,
    limit: int | None = None,
    no_chunk: bool = False,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> AnalysisResult:
    """Process all .txt files in a directory as a single corpus.

    By default, concatenates all files and chunks them. With no_chunk=True,
    each file is treated as one document (standard TF-IDF, no chunking).

    Args:
        dir_path: Path to directory containing .txt files.
        engine: Engine to use: "pure" or "scikit".
        ngram: N-gram level (1-6).
        chunk_size: Tokens per chunk (ignored when no_chunk=True).
        top_k: Terms per zone.
        limit: Randomly select N files. None means use all files.
        no_chunk: If True, each file = one document (no chunking).
        on_progress: Optional callback(current, total, filename) for progress.

    Returns:
        Single AnalysisResult for the entire corpus.

    Raises:
        FileNotFoundError: If no .txt files found in directory.
    """
    files, texts = _collect_files(dir_path, limit, on_progress)
    combined_length = sum(len(t) for t in texts)

    start = time.perf_counter()

    if no_chunk:
        if engine == "scikit":
            from tfidf_zones.scikit_engine import run_docs
        else:
            from tfidf_zones.tfidf_engine import run_docs

        engine_result = run_docs(texts, ngram=ngram, top_k=top_k, wordnet=wordnet, no_ngram_stopwords=no_ngram_stopwords)
    else:
        combined = "\n\n".join(texts)
        combined_length = len(combined)

        if engine == "scikit":
            from tfidf_zones.scikit_engine import run
        else:
            from tfidf_zones.tfidf_engine import run

        engine_result = run(combined, ngram=ngram, chunk_size=chunk_size, top_k=top_k, wordnet=wordnet, no_ngram_stopwords=no_ngram_stopwords)

    zones = classify_zones(engine_result.all_scored, top_k=top_k, chunk_count=engine_result.chunk_count)

    elapsed = time.perf_counter() - start

    return AnalysisResult(
        filename=dir_path.name,
        text_length=combined_length,
        engine_name=engine,
        engine_result=engine_result,
        zones=zones,
        elapsed=elapsed,
        chunk_size=0 if no_chunk else chunk_size,
        file_count=len(files),
    )
