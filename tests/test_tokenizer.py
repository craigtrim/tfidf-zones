from tfidf_zones.tokenizer import Tokenizer


def _make_tokenizer(**kwargs):
    defaults = dict(lowercase=True, strip_punctuation=True, strip_numbers=True, min_length=2)
    defaults.update(kwargs)
    return Tokenizer(**defaults)


class TestBasicTokenization:
    def test_simple_sentence(self):
        t = _make_tokenizer()
        tokens = t.tokenize("The cat sat on the mat.")
        assert "the" in tokens
        assert "cat" in tokens
        assert "sat" in tokens
        assert "mat" in tokens

    def test_empty_string(self):
        t = _make_tokenizer()
        assert t.tokenize("") == []

    def test_lowercase(self):
        t = _make_tokenizer(lowercase=True)
        tokens = t.tokenize("Alice Bob Charlie")
        assert all(tok == tok.lower() for tok in tokens)

    def test_no_lowercase(self):
        t = _make_tokenizer(lowercase=False)
        tokens = t.tokenize("Alice Bob")
        assert "Alice" in tokens
        assert "Bob" in tokens

    def test_min_length_filtering(self):
        t = _make_tokenizer(min_length=2)
        tokens = t.tokenize("I am a big dog")
        # "I" and "a" are length 1, should be filtered
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "big" in tokens
        assert "dog" in tokens

    def test_strip_punctuation(self):
        t = _make_tokenizer(strip_punctuation=True)
        tokens = t.tokenize("Hello, world! How are you?")
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens

    def test_strip_numbers(self):
        t = _make_tokenizer(strip_numbers=True)
        tokens = t.tokenize("There are 42 cats and 7 dogs")
        assert "42" not in tokens
        assert "7" not in tokens  # filtered by min_length=2 anyway
        assert "cats" in tokens


class TestContractions:
    def test_contractions_preserved(self):
        t = _make_tokenizer()
        tokens = t.tokenize("I didn't know what she'd say")
        assert "didn't" in tokens or "didn" in tokens

    def test_possessives(self):
        t = _make_tokenizer()
        tokens = t.tokenize("Alice's cat was there")
        # Should have some form of alice's
        combined = " ".join(tokens)
        assert "alice" in combined


class TestHyphenated:
    def test_hyphenated_compounds(self):
        t = _make_tokenizer()
        tokens = t.tokenize("The looking-glass was on the well-known shelf")
        assert "looking-glass" in tokens
        assert "well-known" in tokens


class TestUnicodeNormalization:
    def test_smart_quotes(self):
        t = _make_tokenizer()
        tokens = t.tokenize("\u201cHello\u201d she said")
        assert "hello" in tokens
        assert "she" in tokens
        assert "said" in tokens

    def test_em_dash(self):
        t = _make_tokenizer()
        # Em dash normalizes to "-", creating a hyphenated compound
        tokens = t.tokenize("word\u2014another")
        assert "word-another" in tokens

    def test_ellipsis(self):
        t = _make_tokenizer(strip_punctuation=True)
        tokens = t.tokenize("wait\u2026 what")
        assert "wait" in tokens
        assert "what" in tokens


class TestIterator:
    def test_tokenize_iter(self):
        t = _make_tokenizer()
        tokens_list = t.tokenize("The cat sat on the mat")
        tokens_iter = list(t.tokenize_iter("The cat sat on the mat"))
        assert tokens_list == tokens_iter
