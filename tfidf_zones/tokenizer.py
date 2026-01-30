# =============================================================================
# TOKENIZER
# =============================================================================
#
# Copied from: craigtrim/pystylometry
# Source file:  pystylometry/tokenizer.py
# Repository:   https://github.com/craigtrim/pystylometry
#
# Advanced tokenizer for stylometric analysis with comprehensive unicode
# normalization, contraction handling, and sophisticated token patterns.
#
# No stop word removal — function words carry stylometric signal and TF-IDF
# naturally handles ubiquitous terms via IDF weighting.
#
# =============================================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator

# ===== Unicode Normalization Tables =====

# Single-character replacements (fast lookup with str.maketrans)
_UNICODE_REPLACEMENTS = str.maketrans(
    {
        # Smart quotes
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201a": "'",  # Single low-9 quote
        "\u201b": "'",  # Single high-reversed-9 quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u201e": '"',  # Double low-9 quote
        "\u201f": '"',  # Double high-reversed-9 quote
        # Dashes
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2015": "-",  # Horizontal bar
        "\u2212": "-",  # Minus sign
        # Spaces
        "\u00a0": " ",  # Non-breaking space
        "\u2002": " ",  # En space
        "\u2003": " ",  # Em space
        "\u2009": " ",  # Thin space
        "\u200a": " ",  # Hair space
        # Apostrophes and primes
        "\u02bc": "'",  # Modifier letter apostrophe
        "\u2032": "'",  # Prime
        "\u2033": '"',  # Double prime
        # Ellipsis
        "\u2026": "...",  # Horizontal ellipsis
        # Ligatures (decompose)
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00e6": "ae",  # ae
        "\u00c6": "AE",  # AE
        "\u0153": "oe",  # oe
        "\u0152": "OE",  # OE
        # Mathematical operators
        "\u00d7": "x",  # Multiplication sign
        "\u00f7": "/",  # Division sign
        "\u00b1": "+/-",  # Plus-minus
        # Currency (normalize for analysis)
        "\u00a3": "GBP",  # Pound
        "\u00a5": "JPY",  # Yen
        "\u20ac": "EUR",  # Euro
        # Fractions
        "\u00bc": "1/4",
        "\u00bd": "1/2",
        "\u00be": "3/4",
        "\u2153": "1/3",
        "\u2154": "2/3",
    }
)

# Multi-character patterns (regex-based)
_MULTI_CHAR_PATTERNS = [
    # Multiple dashes to single dash
    (re.compile(r"[-\u2013\u2014]{2,}"), "-"),
    # Multiple dots (not ellipsis)
    (re.compile(r"\.{4,}"), "..."),
    # Zero-width characters
    (re.compile(r"[\u200b-\u200d\ufeff]"), ""),
    # Control characters except newline/tab
    (re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]"), ""),
    # Multiple spaces/tabs to single space
    (re.compile(r"[ \t]+"), " "),
    # HTML entities (common ones)
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&apos;"), "'"),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
]


# ===== Text Cleaning Patterns =====


def _remove_italics_markers(text: str) -> str:
    """Remove markdown/formatting italics markers."""
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    return text


def _remove_brackets(text: str) -> str:
    """Remove bracketed content [like this] and {like this}."""
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = re.sub(r"\{([^\}]+)\}", r"\1", text)
    return text


def _remove_line_break_hyphens(text: str) -> str:
    """Remove hyphens at line breaks (word-\\nbreak -> wordbreak)."""
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


def _remove_page_markers(text: str) -> str:
    """Remove page numbers and headers like [Page 123] or --- Page 45 ---."""
    text = re.sub(r"\[Page\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[-=]{2,}\s*Page\s+\d+\s*[-=]{2,}", "", text, flags=re.IGNORECASE)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===== Token Pattern =====

# Comprehensive token pattern with priority-ordered alternations
_TOKEN_PATTERN = re.compile(
    r"""
    # URLs (highest priority to avoid splitting)
    (?P<url>https?://\S+)|

    # Email addresses
    (?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)|

    # Hashtags and mentions (social media)
    (?P<hashtag>\#\w+)|
    (?P<mention>@\w+)|

    # Time expressions (3:45pm, 10:30:15)
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[ap]m)?)|

    # Dates (ISO format: 2024-01-15)
    (?P<date>\d{4}-\d{2}-\d{2})|

    # Acronyms with periods (U.S.A., Ph.D.)
    (?P<acronym>(?:[A-Z]\.){2,})|

    # Contractions and possessives (complex patterns)
    (?P<contraction_start>
        '(?:tis|twas|twere|twould|twill|em|gainst|cause|bout|til|way)(?![a-z])
    )|
    (?P<internal_elision>
        \w+[''](?:er|re|ve|ll|d|m|s|t|clock)(?![a-z])
    )|
    (?P<hyphen_possessive>
        (?:\w+(?:-\w+)+)['']s?
    )|
    (?P<standard_contraction>
        \w+[''][a-z]{1,3}(?![a-z])
    )|
    (?P<possessive>
        \w+['']s?(?![a-z])
    )|

    # Roman numerals
    (?P<roman>\b[IVXLCDM]+\b)|

    # Ordinals (1st, 2nd, 3rd, 4th, etc.)
    (?P<ordinal>\d+(?:st|nd|rd|th))|

    # Numbers with commas and decimals ($1,234.56)
    (?P<number_currency>\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?)|

    # Abbreviations (Dr., Mr., Mrs., etc.)
    (?P<abbreviation>(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.)|

    # G-dropping (singin', dancin')
    (?P<g_drop>\w+in[''])|

    # Hyphenated compounds (mother-in-law, well-known)
    (?P<hyphenated>(?:\w+-)+\w+)|

    # Regular words (including internal apostrophes like "o'clock")
    (?P<word>\w+(?:[']\w+)*)|

    # Ellipsis
    (?P<ellipsis>\.{3}|…)|

    # Individual punctuation
    (?P<punct>[^\w\s])
    """,
    re.VERBOSE | re.IGNORECASE | re.UNICODE,
)


# ===== Common Abbreviations =====

_COMMON_ABBREV = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "Sr.": "Senior",
    "Jr.": "Junior",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
    "a.m.": "AM",
    "p.m.": "PM",
}

# Contraction expansions
_CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "won't": "will not",
    "shan't": "shall not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'tis": "it is",
    "'twas": "it was",
    "'em": "them",
}


@dataclass
class TokenMetadata:
    """Metadata about a token."""

    token: str
    start: int
    end: int
    token_type: str


class Tokenizer:
    """
    Advanced tokenizer for stylometric analysis.

    Features:
    - Comprehensive unicode normalization
    - Text cleaning (italics, brackets, page markers)
    - Sophisticated token pattern matching
    - Configurable filtering options
    - Memory-efficient iteration
    """

    def __init__(
        self,
        lowercase: bool = True,
        min_length: int = 1,
        max_length: int | None = None,
        strip_numbers: bool = False,
        strip_punctuation: bool = True,
        preserve_urls: bool = False,
        preserve_emails: bool = False,
        preserve_hashtags: bool = False,
        preserve_mentions: bool = False,
        expand_contractions: bool = False,
        expand_abbreviations: bool = False,
        strip_accents: bool = False,
        normalize_unicode: bool = True,
        clean_text: bool = True,
        wordnet_filter: bool = False,
    ):
        self.lowercase = lowercase
        self.min_length = min_length
        self.max_length = max_length
        self.strip_numbers = strip_numbers
        self.strip_punctuation = strip_punctuation
        self.preserve_urls = preserve_urls
        self.preserve_emails = preserve_emails
        self.preserve_hashtags = preserve_hashtags
        self.preserve_mentions = preserve_mentions
        self.expand_contractions = expand_contractions
        self.expand_abbreviations = expand_abbreviations
        self.strip_accents = strip_accents
        self.normalize_unicode = normalize_unicode
        self.clean_text = clean_text
        self.wordnet_filter = wordnet_filter

        if wordnet_filter:
            try:
                from wordnet_lookup import is_wordnet_term

                self._is_wordnet_term = is_wordnet_term
            except ImportError:
                raise ImportError(
                    "wordnet-lookup package is required for --wordnet filtering. "
                    "Install with: poetry install"
                )

    def _preprocess_text(self, text: str) -> str:
        """Apply unicode normalization and text cleaning."""
        if not text:
            return ""

        if self.normalize_unicode:
            text = text.translate(_UNICODE_REPLACEMENTS)
            for pattern, replacement in _MULTI_CHAR_PATTERNS:
                text = pattern.sub(replacement, text)

        if self.clean_text:
            text = _remove_italics_markers(text)
            text = _remove_brackets(text)
            text = _remove_line_break_hyphens(text)
            text = _remove_page_markers(text)
            text = _normalize_whitespace(text)

        if self.strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        return text

    def _expand_token(self, token: str) -> str:
        """Expand contractions and abbreviations if configured."""
        if self.expand_contractions:
            lower_token = token.lower()
            if lower_token in _CONTRACTIONS:
                expanded = _CONTRACTIONS[lower_token]
                if token[0].isupper():
                    expanded = expanded.capitalize()
                return expanded

        if self.expand_abbreviations:
            if token in _COMMON_ABBREV:
                return _COMMON_ABBREV[token]

        return token

    def _should_keep_token(self, token: str, token_type: str) -> bool:
        """Determine if token should be kept based on filters."""
        if len(token) < self.min_length:
            return False
        if self.max_length and len(token) > self.max_length:
            return False

        if token_type == "url" and not self.preserve_urls:
            return False
        if token_type == "email" and not self.preserve_emails:
            return False
        if token_type == "hashtag" and not self.preserve_hashtags:
            return False
        if token_type == "mention" and not self.preserve_mentions:
            return False
        if token_type == "punct" and self.strip_punctuation:
            return False
        if self.strip_numbers and token_type in ("number_currency", "ordinal", "time", "date"):
            return False

        return True

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into a list of tokens."""
        return list(self.tokenize_iter(text))

    def tokenize_iter(self, text: str) -> Iterator[str]:
        """Tokenize text and return an iterator (memory efficient)."""
        text = self._preprocess_text(text)

        for match in _TOKEN_PATTERN.finditer(text):
            token_type = match.lastgroup or "unknown"
            token = match.group(0)

            token = self._expand_token(token)

            if self.lowercase:
                token = token.lower()

            if self._should_keep_token(token, token_type):
                if " " in token:
                    for sub_token in token.split():
                        if self.wordnet_filter and not self._is_wordnet_term(sub_token):
                            continue
                        yield sub_token
                else:
                    if self.wordnet_filter and not self._is_wordnet_term(token):
                        continue
                    yield token
