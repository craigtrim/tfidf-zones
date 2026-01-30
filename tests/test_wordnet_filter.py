from tfidf_zones.tokenizer import Tokenizer
from wordnet_lookup import is_wordnet_term


def _make_tokenizer(**kwargs):
    defaults = dict(lowercase=True, strip_punctuation=True, strip_numbers=True, min_length=2)
    defaults.update(kwargs)
    return Tokenizer(**defaults)


class TestWordNetTokenizer:
    def test_real_words_pass_through(self):
        t = _make_tokenizer(wordnet_filter=True)
        tokens = t.tokenize("The cat sat on the mat")
        assert "cat" in tokens
        assert "sat" in tokens
        assert "mat" in tokens

    def test_nonsense_filtered_out(self):
        t = _make_tokenizer(wordnet_filter=True)
        tokens = t.tokenize("The xyzzy qwfp cat sat")
        assert "cat" in tokens
        assert "sat" in tokens
        assert "xyzzy" not in tokens
        assert "qwfp" not in tokens

    def test_filter_disabled_keeps_nonsense(self):
        t = _make_tokenizer(wordnet_filter=False)
        tokens = t.tokenize("The xyzzy cat sat")
        assert "xyzzy" in tokens
        assert "cat" in tokens

    def test_wordnet_false_by_default(self):
        t = _make_tokenizer()
        assert t.wordnet_filter is False


class TestWordNetWithEngines:
    def test_pure_engine_with_wordnet(self):
        from tfidf_zones.tfidf_engine import run

        text = "The cat sat on the mat. " * 200
        result = run(text, ngram=1, chunk_size=500, top_k=5, wordnet=True)
        assert result.total_tokens > 0
        for entry in result.top_terms:
            assert is_wordnet_term(entry["term"]), f"'{entry['term']}' not in WordNet"

    def test_scikit_engine_with_wordnet(self):
        from tfidf_zones.scikit_engine import run

        text = "The cat sat on the mat. " * 200
        result = run(text, ngram=1, chunk_size=500, top_k=5, wordnet=True)
        assert result.total_tokens > 0

    def test_wordnet_reduces_token_count(self):
        text = "The xyzzy qwfp cat sat on the zxcvb mat"
        t_plain = _make_tokenizer(wordnet_filter=False)
        t_wordnet = _make_tokenizer(wordnet_filter=True)
        plain_tokens = t_plain.tokenize(text)
        wordnet_tokens = t_wordnet.tokenize(text)
        assert len(wordnet_tokens) < len(plain_tokens)
