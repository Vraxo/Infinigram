"""Tests for Tokenizer protocol and ByteTokenizer."""

import pytest
from infinigram.tokenizers import (
    ByteTokenizer,
    Tokenizer,
    tokenizer_config,
    load_tokenizer,
    vocab_size_of,
)


class TestByteTokenizer:
    def test_vocab_size(self):
        assert ByteTokenizer().vocab_size == 256

    def test_encode_ascii(self):
        assert ByteTokenizer().encode("Hello") == [72, 101, 108, 108, 111]

    def test_decode_ascii(self):
        assert ByteTokenizer().decode([72, 101, 108, 108, 111]) == "Hello"

    def test_roundtrip_utf8(self):
        tok = ByteTokenizer()
        for text in ["Hello", "café", "日本語", "🦊", "", "a" * 10000]:
            assert tok.decode(tok.encode(text)) == text

    def test_encode_multibyte_codepoint(self):
        # "€" is E2 82 AC in UTF-8
        assert ByteTokenizer().encode("€") == [0xE2, 0x82, 0xAC]

    def test_decode_invalid_bytes_replaces(self):
        # Lone continuation byte is invalid UTF-8; default errors='replace'
        assert ByteTokenizer().decode([0x80]) == "\ufffd"

    def test_decode_strict_raises(self):
        with pytest.raises(UnicodeDecodeError):
            ByteTokenizer(errors="strict").decode([0x80])

    def test_satisfies_protocol(self):
        assert isinstance(ByteTokenizer(), Tokenizer)

    def test_equality_and_hash(self):
        assert ByteTokenizer() == ByteTokenizer()
        assert ByteTokenizer(encoding="latin-1") != ByteTokenizer(encoding="utf-8")
        assert hash(ByteTokenizer()) == hash(ByteTokenizer())

    def test_repr(self):
        assert "ByteTokenizer" in repr(ByteTokenizer())


class TestTokenizerConfig:
    def test_byte_round_trips(self):
        tok = ByteTokenizer()
        cfg = tokenizer_config(tok)
        assert cfg["kind"] == "byte"
        loaded = load_tokenizer(cfg)
        assert loaded == tok

    def test_byte_preserves_encoding(self):
        tok = ByteTokenizer(encoding="latin-1")
        cfg = tokenizer_config(tok)
        assert cfg["encoding"] == "latin-1"
        assert load_tokenizer(cfg) == tok

    def test_tiktoken_round_trips(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.get_encoding("cl100k_base")
        cfg = tokenizer_config(enc)
        assert cfg == {"kind": "tiktoken", "name": "cl100k_base"}
        loaded = load_tokenizer(cfg)
        assert loaded.encode("hello") == enc.encode("hello")

    def test_custom_tokenizer_serializes_name(self):
        class MyTokenizer:
            vocab_size = 10

            def encode(self, text):
                return [0]

            def decode(self, tokens):
                return ""

        cfg = tokenizer_config(MyTokenizer())
        assert cfg["kind"] == "custom"
        with pytest.raises(ValueError, match="custom tokenizer"):
            load_tokenizer(cfg)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown tokenizer kind"):
            load_tokenizer({"kind": "alien"})


class TestTiktokenCompatibility:
    """Verify tiktoken satisfies our Tokenizer protocol."""

    def test_tiktoken_satisfies_protocol(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.get_encoding("cl100k_base")
        assert isinstance(enc, Tokenizer)

    def test_tiktoken_encode_decode(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.get_encoding("cl100k_base")
        text = "The quick brown fox"
        tokens = enc.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert enc.decode(tokens) == text

    def test_vocab_size_of_tiktoken(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.get_encoding("cl100k_base")
        assert vocab_size_of(enc) == enc.n_vocab


class TestVocabSizeOf:
    def test_byte_tokenizer(self):
        assert vocab_size_of(ByteTokenizer()) == 256

    def test_missing_raises(self):
        class Bad:
            def encode(self, text):
                return []

            def decode(self, tokens):
                return ""

        with pytest.raises(AttributeError, match="vocab_size"):
            vocab_size_of(Bad())
