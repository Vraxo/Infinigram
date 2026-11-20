"""Integration tests: Infinigram + tiktoken cl100k_base end-to-end.

These exercise the token-level path of the suffix array, validating that a
real LLM-style tokenizer (vocab ~100k) round-trips cleanly through build,
load, predict, generate, and score.
"""

import pytest

from infinigram import Infinigram
from infinigram.tokenizers import ByteTokenizer


tiktoken = pytest.importorskip("tiktoken")


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def corpus_text():
    # Repeated enough that predictions from partial contexts are learnable.
    return "The quick brown fox jumps over the lazy dog. " * 200


@pytest.fixture(scope="module")
def model(enc, corpus_text):
    return Infinigram(corpus_text, tokenizer=enc)


class TestBuildAndQuery:
    def test_vocab_size_matches_tokenizer(self, model, enc):
        assert model.vocab_size == enc.n_vocab
        assert model.vocab_size > 50_000  # sanity: real tokenizer, not byte-level

    def test_model_is_not_byte_level(self, model):
        assert not model._sa.is_byte_level

    def test_search_on_token_ids(self, model, enc, corpus_text):
        pattern = enc.encode(" quick brown")
        assert len(model.search(pattern)) == 200

    def test_count_matches_search(self, model):
        pattern = model.tokenizer.encode(" quick brown")
        assert model.count(pattern) == len(model.search(pattern))


class TestPredictAndGenerate:
    def test_predict_returns_token_distribution(self, model, enc):
        probs = model.predict(enc.encode("The quick brown"))
        assert probs, "predict should return non-empty distribution"
        # All keys are integers in vocab range
        for token_id in probs:
            assert 0 <= token_id < model.vocab_size
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_generate_text_deterministic_with_greedy(self, model):
        text = model.generate_text("The quick brown", max_tokens=10, temperature=0)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should extend the pattern naturally
        assert "fox" in text or "jump" in text

    def test_generate_returns_tokens_for_nonbyte(self, model, enc):
        tokens = model.generate(enc.encode("The quick"), max_tokens=5, temperature=0)
        # Non-byte tokenizer: returns List[int], not bytes
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_generate_with_stop_string(self, model):
        out = model.generate_text(
            "The quick brown fox",
            max_tokens=50,
            temperature=0,
            stop=[" lazy"],
        )
        assert " lazy" not in out


class TestScoring:
    def test_score_prefers_corpus_text(self, model):
        in_corpus = model.score(model.tokenizer.encode("The quick brown fox"))
        out_of_corpus = model.score(model.tokenizer.encode("xylophone zebra marathon"))
        assert in_corpus > out_of_corpus

    def test_score_handles_empty(self, model):
        assert model.score([]) == 0.0


class TestCompressionWin:
    """Verify that token-level uses fewer positions than byte-level for same text."""

    def test_token_level_has_fewer_positions(self, corpus_text, enc):
        byte_model = Infinigram(corpus_text, tokenizer=ByteTokenizer())
        token_model = Infinigram(corpus_text, tokenizer=enc)
        # cl100k averages ~4.5 chars/token, so token-level corpus is ~4-5x smaller.
        assert token_model.n < byte_model.n / 3


class TestRoundTripPersistence:
    def test_build_and_load(self, tmp_path, enc, corpus_text):
        path = tmp_path / "tiktoken_model"
        built = Infinigram.build(
            corpus_text,
            str(path),
            tokenizer=enc,
            verbose=False,
        )
        # Rebuild by loading fresh — no tokenizer argument; it comes from meta.
        loaded = Infinigram.load(str(path))
        assert loaded.vocab_size == enc.n_vocab
        assert loaded.n == built.n
        # Same query, same count.
        pattern = enc.encode(" quick brown")
        assert loaded.count(pattern) == built.count(pattern)
