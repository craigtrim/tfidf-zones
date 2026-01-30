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
import sys
from pathlib import Path

from tfidf_zones.api import _apply_filters as _api_apply_filters
from tfidf_zones.api import to_csv
from tfidf_zones.formatter import (
    print_corpus_summary,
    print_df_stats,
    print_error,
    print_footer,
    print_header,
    print_output,
    print_progress,
    print_summary,
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
        "--no-ngram-stopwords",
        action="store_true",
        default=False,
        help="Discard n-grams containing stop/function words (only applies when --ngram >= 2)",
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
    result.engine_result, result.zones = _api_apply_filters(
        result.engine_result, result.zones, min_df, min_tf, top_k,
    )


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
            no_ngram_stopwords=args.no_ngram_stopwords,
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
            no_ngram_stopwords=args.no_ngram_stopwords,
            min_df=args.min_df,
            min_tf=args.min_tf,
        )
    print_df_stats(r.df_stats)


def _write_csv(result, output_path: str) -> None:
    """Write all scored terms to CSV."""
    csv_content = to_csv(result)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)


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

    # Validate --no-ngram-stopwords requires --ngram >= 2
    if args.no_ngram_stopwords and args.ngram < 2:
        print_error("--no-ngram-stopwords requires --ngram >= 2")
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
                no_ngram_stopwords=args.no_ngram_stopwords,
            )
            _apply_filters(result, args.min_df, args.min_tf, args.top_k)
            _print_result(result, args)
            _write_csv(result, args.output)
            out = Path(args.output)
            print_output(args.output, len(result.engine_result.all_scored), out.stat().st_size)
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
                no_ngram_stopwords=args.no_ngram_stopwords,
                on_progress=print_progress,
            )
            _apply_filters(result, args.min_df, args.min_tf, args.top_k)
            _print_result(result, args)
            _write_csv(result, args.output)
            out = Path(args.output)
            print_output(args.output, len(result.engine_result.all_scored), out.stat().st_size)
            print_footer(result.elapsed)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
