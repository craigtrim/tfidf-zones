# =============================================================================
# CLI ENTRY POINT
# =============================================================================
#
# Usage:
#   poetry run tfidf-zones --file novel.txt --output results.csv
#   poetry run tfidf-zones --file novel.txt --scikit --ngram 2 --output results.csv
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
    print_zones,
)
from tfidf_zones.runner import analyze_corpus, analyze_file


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
        "--output",
        type=str,
        required=True,
        help="Output CSV file path (required)",
    )

    return parser


def _print_result(result) -> None:
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
        )
    print_df_stats(r.df_stats)
    print_zones(result.zones)


def _write_csv(result, output_path: str) -> None:
    """Write all scored terms to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "tf", "df", "idf", "tfidf"])
        for entry in result.engine_result.all_scored:
            writer.writerow([
                entry["term"],
                entry.get("tf", 0),
                entry["df"],
                entry["idf"],
                entry["score"],
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
            _print_result(result)
            _write_csv(result, args.output)
            print_footer()

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
            _print_result(result)
            _write_csv(result, args.output)
            print_footer()

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
