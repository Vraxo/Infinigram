"""Tests for the suffix array module, both byte- and token-level paths."""

import numpy as np
import pytest
from infinigram.suffix_array import (
    SuffixArray,
    MmapSuffixArray,
    ChunkedMmapSuffixArray,
)


class TestByteLevelSuffixArray:
    def test_build_from_bytes(self):
        sa = SuffixArray(b"the cat sat on the mat")
        assert sa.is_byte_level
        assert sa.n == 22

    def test_search_simple(self):
        sa = SuffixArray(b"the cat sat on the mat")
        assert sorted(sa.search(b"the")) == [0, 15]

    def test_search_missing_returns_empty(self):
        sa = SuffixArray(b"abc")
        assert sa.search(b"xyz") == []

    def test_longest_suffix(self):
        sa = SuffixArray(b"the cat sat on the mat")
        pos, length = sa.find_longest_suffix(b"the cat")
        assert length == 7
        assert pos == 0

    def test_count(self):
        sa = SuffixArray(b"aaabaaaba")
        assert sa.count(b"a") == 7
        assert sa.count(b"aa") == 4


class TestTokenLevelSuffixArray:
    def test_build_from_int_list(self):
        tokens = [1, 100, 2, 500, 1, 100, 3, 1, 100, 2]
        sa = SuffixArray(tokens, vocab_size=1000)
        assert not sa.is_byte_level
        assert sa._corpus.dtype == np.int32
        assert sa.n == 10

    def test_search_token_pattern(self):
        tokens = [1, 100, 2, 500, 1, 100, 3, 1, 100, 2]
        sa = SuffixArray(tokens, vocab_size=1000)
        positions = sorted(sa.search([1, 100]))
        assert positions == [0, 4, 7]

    def test_search_longer_pattern(self):
        tokens = [1, 100, 2, 500, 1, 100, 3, 1, 100, 2]
        sa = SuffixArray(tokens, vocab_size=1000)
        positions = sorted(sa.search([1, 100, 2]))
        assert positions == [0, 7]

    def test_search_missing_token(self):
        sa = SuffixArray([1, 2, 3], vocab_size=1000)
        assert sa.search([999]) == []

    def test_longest_suffix(self):
        tokens = [1, 100, 2, 500, 1, 100, 3, 1, 100, 2]
        sa = SuffixArray(tokens, vocab_size=1000)
        pos, length = sa.find_longest_suffix([99, 1, 100, 2])
        assert length == 3
        assert pos in {0, 7}

    def test_count_token(self):
        sa = SuffixArray([1, 100, 1, 100, 1], vocab_size=200)
        assert sa.count([1]) == 3
        assert sa.count([1, 100]) == 2

    def test_bytes_rejected_for_large_vocab(self):
        with pytest.raises(TypeError):
            SuffixArray(b"hello", vocab_size=1000)

    def test_large_vocab_tokens(self):
        """Verify real-world-ish tokenizer IDs work."""
        np.random.seed(42)
        tokens = np.random.randint(0, 50_000, size=200, dtype=np.int32).tolist()
        sa = SuffixArray(tokens, vocab_size=50_000)
        # First 3-token pattern should occur at least at position 0.
        pattern = tokens[:3]
        assert 0 in sa.search(pattern)

    def test_get_context_token_level(self):
        sa = SuffixArray(list(range(10, 30)), vocab_size=100)
        before, after = sa.get_context(5, window=3)
        assert before == [12, 13, 14]
        assert after == [15, 16, 17]


class TestMmapSuffixArrayTokenLevel:
    def test_roundtrip_byte_level(self, tmp_path):
        sa = MmapSuffixArray.build(b"the cat sat on the mat", str(tmp_path / "idx"))
        assert sa.is_byte_level
        assert sorted(sa.search(b"the")) == [0, 15]
        sa.close()

    def test_roundtrip_token_level(self, tmp_path):
        tokens = [1, 100, 2, 500, 1, 100, 3, 1, 100, 2]
        sa = MmapSuffixArray.build(
            tokens, str(tmp_path / "idx"), vocab_size=1000
        )
        assert not sa.is_byte_level
        assert sorted(sa.search([1, 100])) == [0, 4, 7]
        assert sa.count([1, 100, 2]) == 2
        sa.close()

    def test_reload_preserves_dtype(self, tmp_path):
        MmapSuffixArray.build(
            [1, 100, 2, 500, 1, 100], str(tmp_path / "idx"), vocab_size=1000
        ).close()
        reopened = MmapSuffixArray(str(tmp_path / "idx"))
        assert reopened._corpus.dtype == np.int32
        assert reopened.vocab_size == 1000
        assert sorted(reopened.search([1, 100])) == [0, 4]
        reopened.close()

    def test_get_context_token(self, tmp_path):
        # vocab_size > 256 forces the int32 path, so get_context returns ints.
        sa = MmapSuffixArray.build(
            list(range(20, 40)), str(tmp_path / "idx"), vocab_size=1000
        )
        ctx = sa.get_context(5, window=3)
        assert ctx == [22, 23, 24, 25, 26, 27]
        sa.close()


class TestInfinigramChunkedBuild:
    """Regression: the chunks list in meta.json must survive Infinigram.build()."""

    def test_chunked_build_load_roundtrip(self, tmp_path, monkeypatch):
        from infinigram import Infinigram

        # Force the chunked path by lowering the threshold.
        monkeypatch.setattr(Infinigram, "CHUNK_THRESHOLD", 100)
        model = Infinigram.build(
            b"the cat sat on the mat " * 100,
            str(tmp_path / "idx"),
            chunk_size_gb=1e-6,  # tiny chunks so we actually split
            verbose=False,
        )
        assert model._is_chunked
        assert len(model._sa.chunks) > 1

        # Reload without a tokenizer argument: meta.json must be complete.
        reloaded = Infinigram.load(str(tmp_path / "idx"))
        assert reloaded._is_chunked
        assert len(reloaded._sa.chunks) == len(model._sa.chunks)
        assert reloaded.n == model.n
        ctx = reloaded.tokenizer.encode("the cat")
        assert reloaded.count(ctx) == model.count(ctx)


class TestChunkedMmapSuffixArrayTokenLevel:
    def test_chunked_byte_level(self, tmp_path):
        sa = ChunkedMmapSuffixArray.build(
            b"aaabaaab" * 100,
            str(tmp_path / "chunked"),
            chunk_size=400,
        )
        assert sa.is_byte_level
        # aaab appears 2x per 8-byte repeat, 100 repeats = 200 occurrences
        # (chunk boundary falls between 8-byte repeats, no split patterns).
        assert sa.count(b"aaab") == 200
        sa.close()

    def test_chunked_token_level(self, tmp_path):
        tokens = [1, 2, 3, 4, 500] * 200
        sa = ChunkedMmapSuffixArray.build(
            tokens,
            str(tmp_path / "chunked"),
            chunk_size=300,
            vocab_size=1000,
        )
        assert not sa.is_byte_level
        assert sa.count([1, 2, 3]) == 200
        sa.close()
