# =============================================================================
# CLI ENTRY POINT
# =============================================================================
#
# Usage:
#   poetry run tfidf-zones --file novel.txt
#   poetry run tfidf-zones --file novel.txt --engine scikit --top-k 25 --ngram 2
#   poetry run tfidf-zones --dir ./texts/ --chunk-size 500
#
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tfidf_zones.formatter import (
    print_df_stats,
    print_error,
    print_file_separator,
    print_footer,
    print_header,
    print_summary,
    print_zones,
)
from tfidf_zones.runner import analyze_directory, analyze_file


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
        "--engine",
        type=str,
        default="pure",
        choices=["pure", "scikit"],
        help="TF-IDF engine: pure (default) or scikit",
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

    return parser


def _print_result(result) -> None:
    """Print a single analysis result."""
    r = result.engine_result
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate chunk-size
    if args.chunk_size < 100:
        print_error("chunk-size must be >= 100")
        sys.exit(1)

    # Validate top-k
    if args.top_k < 1:
        print_error("top-k must be >= 1")
        sys.exit(1)

    try:
        if args.file:
            file_path = Path(args.file)
            if not file_path.is_file():
                print_error(f"file not found: {args.file}")
                sys.exit(1)

            print_header()
            result = analyze_file(
                file_path,
                engine=args.engine,
                ngram=args.ngram,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
            )
            _print_result(result)
            print_footer()

        else:
            dir_path = Path(args.dir)
            if not dir_path.is_dir():
                print_error(f"directory not found: {args.dir}")
                sys.exit(1)

            print_header()
            results = analyze_directory(
                dir_path,
                engine=args.engine,
                ngram=args.ngram,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
            )
            for i, result in enumerate(results):
                if i > 0:
                    print_file_separator(result.filename)
                _print_result(result)
            print_footer()

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
