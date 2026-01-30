# =============================================================================
# CLI ENTRY POINT
# =============================================================================
#
# Usage:
#   poetry run tfidf-zones --file novel.txt --output results.csv
#   poetry run tfidf-zones --file novel.txt --scikit --ngram 2 --output results.csv
#   poetry run tfidf-zones --file novel.txt --wordnet --output results.csv
#   poetry run tfidf-zones --file novel.txt --min-df 2 --min-tf 2 --output results.csv
#   poetry run tfidf-zones --dir ./texts/ --output results.csv
#   poetry run tfidf-zones --dir ./texts/ --limit 50 --output results.csv
#   poetry run tfidf-zones --dir ./texts/ --no-chunk --output results.csv
#
# =============================================================================

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tfidf_zones.formatter import (
    print_corpus_summary,
    print_df_stats,
    print_error,
    print_footer,
    print_header,
    print_progress,
    print_summary,
)
from tfidf_zones.runner import analyze_corpus, analyze_file
from tfidf_zones.zones import classify_zones


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfidf-zones",
        description="TF-IDF Zone Analysis — classify terms into too-common, goldilocks, and too-rare zones.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to input text file",
    )
    input_group.add_argument(
        "--dir",
        type=str,
        help="Path to directory of .txt files",
    )

    parser.add_argument(
        "--scikit",
        action="store_true",
        default=False,
        help="Use scikit-learn TF-IDF engine (default: pure Python)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Terms per zone (default: 10)",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="N-gram level 1-6 (default: 1). 6=skipgrams",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Tokens per chunk (default: 2000, min: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Randomly select N files from directory (requires --dir)",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        default=False,
        help="Each file = one document, no chunking (requires --dir)",
    )
    parser.add_argument(
        "--wordnet",
        action="store_true",
        default=False,
        help="Filter tokens through WordNet — only recognized English words participate in TF-IDF",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=None,
        help="Remove terms with DF below this value (post-processing filter)",
    )
    parser.add_argument(
        "--min-tf",
        type=int,
        default=None,
        help="Remove terms with TF below this value (post-processing filter)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path (required)",
    )

    return parser


def _apply_filters(result, min_df: int | None, min_tf: int | None, top_k: int) -> None:
    """Post-process result to remove terms below min_df or min_tf thresholds.

    Mutates result in place: filters all_scored, top_terms, and re-classifies zones.
    """
    if min_df is None and min_tf is None:
        return

    er = result.engine_result
    filtered = er.all_scored
    if min_df is not None:
        filtered = [t for t in filtered if t["df"] >= min_df]
    if min_tf is not None:
        filtered = [t for t in filtered if t.get("tf", 0) >= min_tf]

    er.all_scored = filtered
    er.top_terms = sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_k]
    result.zones = classify_zones(filtered, top_k=top_k, chunk_count=er.chunk_count)


def _print_result(result, args) -> None:
    """Print a single analysis result."""
    r = result.engine_result
    if result.file_count is not None:
        print_corpus_summary(
            dirname=result.filename,
            engine=result.engine_name,
            ngram_type=r.ngram_type,
            file_count=result.file_count,
            total_text_length=result.text_length,
            tokens=r.total_tokens,
            chunks=r.chunk_count,
            chunk_size=result.chunk_size,
            elapsed=result.elapsed,
            wordnet=args.wordnet,
            min_df=args.min_df,
            min_tf=args.min_tf,
        )
    else:
        print_summary(
            filename=result.filename,
            engine=result.engine_name,
            ngram_type=r.ngram_type,
            text_length=result.text_length,
            tokens=r.total_tokens,
            chunks=r.chunk_count,
            chunk_size=result.chunk_size,
            elapsed=result.elapsed,
            wordnet=args.wordnet,
            min_df=args.min_df,
            min_tf=args.min_tf,
        )
    print_df_stats(r.df_stats)


def _build_zone_lookup(all_scored: list[dict], chunk_count: int) -> dict[str, int]:
    """Classify every term into a zone: 1=too_common, 2=goldilocks, 3=too_rare.

    Uses the same band-pass logic as classify_zones but without top_k limits,
    so every term gets a zone assignment.
    """
    if not all_scored:
        return {}

    n = chunk_count if chunk_count > 0 else max(t["df"] for t in all_scored)
    df_upper = max(3, int(n * 0.2))
    df_lower = 3
    if df_upper <= df_lower:
        df_lower = 2
        df_upper = max(df_lower + 1, df_upper)

    scores = sorted((t["score"] for t in all_scored), reverse=True)
    p95_index = max(0, int(len(scores) * 0.05) - 1)
    tfidf_threshold = scores[p95_index]

    lookup: dict[str, int] = {}
    for t in all_scored:
        df = t["df"]
        if df > df_upper:
            lookup[t["term"]] = 1
        elif df < df_lower:
            lookup[t["term"]] = 3
        elif t["score"] >= tfidf_threshold:
            lookup[t["term"]] = 2
    return lookup


def _write_csv(result, output_path: str) -> None:
    """Write all scored terms to CSV with cumulative TF columns and zone."""
    all_scored = result.engine_result.all_scored
    total_tokens = result.engine_result.total_tokens

    # Build zone lookup: 1=too_common, 2=goldilocks, 3=too_rare
    zone_lookup = _build_zone_lookup(all_scored, result.engine_result.chunk_count)

    # Build cumulative TF lookup: sort by TF descending, compute running sum
    tf_sorted = sorted(all_scored, key=lambda x: x.get("tf", 0), reverse=True)
    cum_lookup: dict[str, tuple[float, int]] = {}
    running = 0.0
    for entry in tf_sorted:
        tf_pct = entry.get("tf", 0) / total_tokens if total_tokens > 0 else 0.0
        running += tf_pct
        tf_cum_norm = max(1, round(running * 100)) if total_tokens > 0 else 0
        cum_lookup[entry["term"]] = (tf_pct, tf_cum_norm)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "tf", "df", "idf", "tfidf", "tf_pct", "tf_cum_norm", "zone"])
        for entry in all_scored:
            tf_pct, tf_cum_norm = cum_lookup[entry["term"]]
            zone = zone_lookup.get(entry["term"], "")
            writer.writerow([
                entry["term"],
                entry.get("tf", 0),
                entry["df"],
                entry["idf"],
                entry["score"],
                tf_pct,
                tf_cum_norm,
                zone,
            ])


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate --limit requires --dir
    if args.limit is not None and not args.dir:
        print_error("--limit requires --dir")
        sys.exit(1)

    # Validate --no-chunk requires --dir
    if args.no_chunk and not args.dir:
        print_error("--no-chunk requires --dir")
        sys.exit(1)

    # Validate limit
    if args.limit is not None and args.limit < 1:
        print_error("--limit must be >= 1")
        sys.exit(1)

    # Validate top-k
    if args.top_k < 1:
        print_error("top-k must be >= 1")
        sys.exit(1)

    # Validate chunk-size
    if args.chunk_size < 100:
        print_error("chunk-size must be >= 100")
        sys.exit(1)

    engine = "scikit" if args.scikit else "pure"

    try:
        if args.file:
            file_path = Path(args.file)
            if not file_path.is_file():
                print_error(f"file not found: {args.file}")
                sys.exit(1)

            print_header()
            result = analyze_file(
                file_path,
                engine=engine,
                ngram=args.ngram,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                wordnet=args.wordnet,
            )
            _apply_filters(result, args.min_df, args.min_tf, args.top_k)
            _print_result(result, args)
            _write_csv(result, args.output)
            print_footer(result.elapsed)

        else:
            dir_path = Path(args.dir)
            if not dir_path.is_dir():
                print_error(f"directory not found: {args.dir}")
                sys.exit(1)

            print_header()
            result = analyze_corpus(
                dir_path,
                engine=engine,
                ngram=args.ngram,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                limit=args.limit,
                no_chunk=args.no_chunk,
                wordnet=args.wordnet,
                on_progress=print_progress,
            )
            _apply_filters(result, args.min_df, args.min_tf, args.top_k)
            _print_result(result, args)
            _write_csv(result, args.output)
            print_footer(result.elapsed)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
