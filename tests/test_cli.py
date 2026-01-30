import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from tfidf_zones.cli import main


def _run_cli(*args):
    """Run the CLI main() with given args, capturing stdout/stderr."""
    stdout = StringIO()
    stderr = StringIO()
    with patch.object(sys, "argv", ["tfidf-zones", *args]), \
         patch.object(sys, "stdout", stdout), \
         patch.object(sys, "stderr", stderr):
        try:
            main()
            returncode = 0
        except SystemExit as e:
            returncode = e.code if e.code is not None else 0
    return returncode, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample text file for testing."""
    text = "The cat sat on the mat. The dog chased the cat. " * 200
    f = tmp_path / "sample.txt"
    f.write_text(text)
    return f


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with multiple text files."""
    for name in ["file1.txt", "file2.txt"]:
        text = f"This is {name} content with some different words and phrases. " * 200
        (tmp_path / name).write_text(text)
    return tmp_path


class TestCliHelp:
    def test_help(self):
        rc, out, err = _run_cli("--help")
        assert rc == 0
        combined = out + err
        assert "tfidf-zones" in combined or "TF-IDF" in combined


class TestCliFile:
    def test_single_file(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file))
        assert rc == 0
        assert "TF-IDF Zone Analysis" in out
        assert "TOO COMMON" in out
        assert "GOLDILOCKS" in out
        assert "TOO RARE" in out
        assert "Done." in out

    def test_file_with_engine_scikit(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--engine", "scikit")
        assert rc == 0
        assert "scikit" in out

    def test_file_with_ngram(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--ngram", "2")
        assert rc == 0
        assert "bigrams" in out

    def test_file_with_top_k(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--top-k", "3")
        assert rc == 0

    def test_file_with_chunk_size(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--chunk-size", "500")
        assert rc == 0

    def test_file_not_found(self):
        rc, out, err = _run_cli("--file", "/nonexistent/file.txt")
        assert rc != 0


class TestCliDir:
    def test_directory(self, sample_dir):
        rc, out, err = _run_cli("--dir", str(sample_dir))
        assert rc == 0
        assert "file1.txt" in out
        assert "file2.txt" in out

    def test_empty_directory(self, tmp_path):
        rc, out, err = _run_cli("--dir", str(tmp_path))
        assert rc != 0

    def test_dir_not_found(self):
        rc, out, err = _run_cli("--dir", "/nonexistent/dir")
        assert rc != 0


class TestCliValidation:
    def test_no_input(self):
        rc, out, err = _run_cli()
        assert rc != 0

    def test_invalid_engine(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--engine", "bad")
        assert rc != 0

    def test_invalid_ngram(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--ngram", "9")
        assert rc != 0

    def test_invalid_chunk_size(self, sample_file):
        rc, out, err = _run_cli("--file", str(sample_file), "--chunk-size", "10")
        assert rc != 0
