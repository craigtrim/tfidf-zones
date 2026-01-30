# =============================================================================
# RUNNER
# =============================================================================
#
# Orchestrates file reading, engine dispatch, zone classification, and timing.
#
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from tfidf_zones.tfidf_engine import EngineResult
from tfidf_zones.zones import classify_zones


@dataclass
class AnalysisResult:
    """Result of analyzing a single file."""

    filename: str
    text_length: int
    engine_name: str
    engine_result: EngineResult
    zones: dict
    elapsed: float
    chunk_size: int


def analyze_file(
    file_path: Path,
    engine: str = "pure",
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 10,
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

    engine_result = run(text, ngram=ngram, chunk_size=chunk_size, top_k=top_k)
    zones = classify_zones(engine_result.all_scored, top_k=top_k)

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


def analyze_directory(
    dir_path: Path,
    engine: str = "pure",
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 10,
) -> list[AnalysisResult]:
    """Process all .txt files in a directory.

    Args:
        dir_path: Path to directory containing .txt files.
        engine: Engine to use: "pure" or "scikit".
        ngram: N-gram level (1-6).
        chunk_size: Tokens per chunk.
        top_k: Terms per zone.

    Returns:
        List of AnalysisResult, one per file.

    Raises:
        FileNotFoundError: If no .txt files found in directory.
    """
    files = sorted(dir_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {dir_path}")
    return [
        analyze_file(f, engine=engine, ngram=ngram, chunk_size=chunk_size, top_k=top_k)
        for f in files
    ]
