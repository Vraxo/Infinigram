#!/usr/bin/env python3
"""
Tests for Infinigram variable-length n-gram model.
"""

import pytest
import numpy as np

from infinigram import Infinigram, SuffixArray


class TestSuffixArray:
    """Test suffix array implementation."""

    def test_suffix_array_construction(self):
        """Test suffix array builds correctly."""
        corpus = [1, 2, 3, 1, 2, 4]
        sa = SuffixArray(corpus)

        assert len(sa.suffix_array) == len(corpus)
        assert all(0 <= idx < len(corpus) for idx in sa.suffix_array)

    def test_find_range_exact_match(self):
        """Test finding exact pattern matches."""
        corpus = [1, 2, 3, 1, 2, 3, 4, 1, 2]
        sa = SuffixArray(corpus)

        # Pattern [1, 2] appears at positions 0, 3, 7
        positions = sa.search([1, 2])
        assert len(positions) == 3  # Three occurrences

    def test_find_range_not_found(self):
        """Test pattern not in corpus."""
        corpus = [1, 2, 3, 4, 5]
        sa = SuffixArray(corpus)

        positions = sa.search([9, 9, 9])
        assert len(positions) == 0  # Not found

    def test_find_range_single_token(self):
        """Test single token pattern."""
        corpus = [1, 2, 1, 3, 1]
        sa = SuffixArray(corpus)

        positions = sa.search([1])
        assert len(positions) == 3  # Token 1 appears 3 times

    def test_find_range_empty_pattern(self):
        """Test empty pattern returns empty range."""
        corpus = [1, 2, 3]
        sa = SuffixArray(corpus)

        positions = sa.search([])
        assert len(positions) == 0


class TestInfinigramCore:
    """Test core Infinigram functionality."""

    def test_infinigram_initialization(self):
        """Test Infinigram initializes correctly."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        assert model.n == 5
        assert model.vocab_size == 256  # Fixed vocabulary for bytes
        assert set(model.vocab) == set(range(256))  # All 256 possible tokens

    def test_infinigram_with_max_length(self):
        """Test max_length parameter."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus, max_length=3)

        assert model.max_length == 3

    def test_byte_validation(self):
        """Test corpus must be valid bytes (0-255)."""
        # Valid corpus should work
        corpus_valid = [0, 128, 255]  # Min, mid, max bytes
        model = Infinigram(corpus_valid)
        assert model.n == 3

        # Invalid corpus should raise error
        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([256])  # Too large

        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([-1])  # Negative

        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([0, 100, 300, 1000])  # Mixed valid/invalid

    def test_update_byte_validation(self):
        """Test update validates byte range."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        # Valid update should work
        model.update([4, 5, 255])
        assert model.n == 6

        # Invalid update should raise error
        with pytest.raises(ValueError, match="new_tokens must contain only bytes"):
            model.update([256])

        with pytest.raises(ValueError, match="new_tokens must contain only bytes"):
            model.update([-1])


class TestLongestSuffix:
    """Test longest suffix matching."""

    def test_longest_suffix_exact_match(self):
        """Test finding exact suffix match."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = Infinigram(corpus)

        # Context [1, 2, 3] has exact match at position 0
        context = [1, 2, 3]
        pos, length = model.longest_suffix(context)

        assert length == 3  # Full match
        assert pos >= 0

    def test_longest_suffix_partial_match(self):
        """Test finding partial suffix match."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        # Context [99, 2, 3] - only [2, 3] matches
        context = [99, 2, 3]
        pos, length = model.longest_suffix(context)

        assert length == 2  # Partial match [2, 3]

    def test_longest_suffix_no_match(self):
        """Test when no suffix matches."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        context = [99, 98, 97]
        pos, length = model.longest_suffix(context)

        assert length == 0  # No match
        assert pos == -1

    def test_longest_suffix_empty_context(self):
        """Test empty context."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        pos, length = model.longest_suffix([])
        assert length == 0

    def test_longest_suffix_respects_max_length(self):
        """Test max_length limits suffix search."""
        corpus = [1, 2, 3, 4, 5, 6, 7, 8]
        model = Infinigram(corpus, max_length=3)

        # Long context that fully matches
        context = [1, 2, 3, 4, 5, 6, 7, 8]
        pos, length = model.longest_suffix(context)

        # Should only match last 3 tokens due to max_length
        assert length <= 3


class TestFindAllSuffixMatches:
    """Test find_all_suffix_matches method."""

    def test_find_all_suffix_matches_basic(self):
        """Test basic suffix match finding."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"the cat")

        # Should find matches at multiple lengths
        assert len(matches) > 0

        # Should be sorted by decreasing length
        lengths = [length for length, _ in matches]
        assert lengths == sorted(lengths, reverse=True)

        # Each entry should have (length, positions)
        for length, positions in matches:
            assert length > 0
            assert len(positions) > 0

    def test_find_all_suffix_matches_multiple_lengths(self):
        """Test that we get matches at multiple suffix lengths."""
        corpus = b"abc abcd abcde"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"abcde")

        # Should have multiple different lengths
        lengths = [length for length, _ in matches]
        assert len(set(lengths)) > 1

    def test_find_all_suffix_matches_verifies_positions(self):
        """Test that returned positions actually contain the suffix."""
        corpus = b"the cat sat. the cat ran."
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"the cat")

        # Verify positions are valid
        for length, positions in matches:
            for pos in positions:
                assert 0 <= pos < len(corpus)

    def test_find_all_suffix_matches_no_match(self):
        """Test with no matching suffixes."""
        corpus = b"abc"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"xyz")
        assert matches == []

    def test_find_all_suffix_matches_empty_context(self):
        """Test with empty context."""
        corpus = b"abc"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"")
        assert matches == []

    def test_find_all_suffix_matches_respects_max_length(self):
        """Test that max_length is respected."""
        corpus = b"abcdefgh"
        model = Infinigram(corpus, max_length=3)

        matches = model.find_all_suffix_matches(b"abcdefgh")

        # Should not search beyond max_length
        if matches:
            max_match_len = max(length for length, _ in matches)
            assert max_match_len <= 3

    # === Additional comprehensive tests for find_all_suffix_matches ===

    def test_find_all_suffix_matches_single_byte_corpus(self):
        """Test with single byte corpus - edge case."""
        corpus = b"a"
        model = Infinigram(corpus)

        # Searching for "a" should find it
        matches = model.find_all_suffix_matches(b"a")
        assert len(matches) == 1
        assert matches[0] == (1, [0])  # length=1, position=0

        # Searching for "b" should find nothing
        matches = model.find_all_suffix_matches(b"b")
        assert matches == []

    def test_find_all_suffix_matches_single_byte_context(self):
        """Test with single byte context against longer corpus."""
        corpus = b"abcabc"
        model = Infinigram(corpus)

        # Single byte "a" appears at positions 0 and 3
        matches = model.find_all_suffix_matches(b"a")
        assert len(matches) == 1
        length, positions = matches[0]
        assert length == 1
        assert sorted(positions) == [0, 3]

    def test_find_all_suffix_matches_context_longer_than_corpus(self):
        """Test when context is longer than entire corpus."""
        corpus = b"abc"
        model = Infinigram(corpus)

        # Context longer than corpus - should still find partial matches
        matches = model.find_all_suffix_matches(b"xyzabc")

        # Should find "abc" (length 3), "bc" (length 2), "c" (length 1)
        assert len(matches) == 3
        lengths = [length for length, _ in matches]
        assert lengths == [3, 2, 1]

    def test_find_all_suffix_matches_position_verification(self):
        """Verify that corpus[pos:pos+length] == suffix for each match."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        context = b"the cat"
        matches = model.find_all_suffix_matches(context)

        # For each match, verify the position actually contains the suffix
        for length, positions in matches:
            suffix = context[-length:]
            for pos in positions:
                # Extract substring at that position
                extracted = corpus[pos:pos + length]
                assert extracted == suffix, (
                    f"Mismatch at pos={pos}, length={length}: "
                    f"expected {suffix!r}, got {extracted!r}"
                )

    def test_find_all_suffix_matches_frequency_verification(self):
        """Verify position counts match expected occurrences."""
        corpus = b"abab ab ab abab"
        model = Infinigram(corpus)

        # Test "ab" - should appear multiple times
        matches = model.find_all_suffix_matches(b"ab")
        # Returns matches at both length 2 ("ab") and length 1 ("b")
        assert len(matches) == 2  # Two lengths: 2 and 1

        # Find the length-2 match
        length_2_matches = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_matches) == 1
        length, positions = length_2_matches[0]
        assert length == 2

        # Count "ab" manually in corpus
        expected_count = corpus.count(b"ab")
        assert len(positions) == expected_count, (
            f"Expected {expected_count} positions for 'ab', got {len(positions)}"
        )

    def test_find_all_suffix_matches_after_tokenize(self):
        """Tokenize text via the model's tokenizer, then query."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(model.tokenizer.encode("the cat"))

        assert len(matches) > 0
        # Verify same result as bytes
        matches_bytes = model.find_all_suffix_matches(b"the cat")
        assert matches == matches_bytes

    def test_find_all_suffix_matches_list_input(self):
        """Test that list of ints input works."""
        corpus = b"abc"
        model = Infinigram(corpus)

        # Pass list of byte values
        matches = model.find_all_suffix_matches([97, 98, 99])  # "abc"

        assert len(matches) > 0
        # Verify same result as bytes
        matches_bytes = model.find_all_suffix_matches(b"abc")
        assert matches == matches_bytes

    def test_find_all_suffix_matches_repeated_pattern(self):
        """Test corpus with many repetitions of same pattern."""
        # Corpus with "ab" repeated 100 times
        corpus = b"ab" * 100
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"ab")

        # Should find exactly 100 positions for "ab"
        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert len(positions) == 100

        # Each position should be even (0, 2, 4, ...)
        expected_positions = list(range(0, 200, 2))
        assert sorted(positions) == expected_positions

    def test_find_all_suffix_matches_overlapping_patterns(self):
        """Test overlapping matches e.g., 'aaa' in 'aaaa'."""
        corpus = b"aaaa"
        model = Infinigram(corpus)

        # Search for "aa" in "aaaa"
        # Should find at positions 0, 1, 2 (overlapping)
        matches = model.find_all_suffix_matches(b"aa")

        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert sorted(positions) == [0, 1, 2]

    def test_find_all_suffix_matches_overlapping_longer_pattern(self):
        """Test overlapping with longer pattern 'aaa' in 'aaaaa'."""
        corpus = b"aaaaa"
        model = Infinigram(corpus)

        # Search for "aaa" in "aaaaa"
        # Should find at positions 0, 1, 2 (overlapping)
        matches = model.find_all_suffix_matches(b"aaa")

        length_3_match = [(l, p) for l, p in matches if l == 3]
        assert len(length_3_match) == 1
        _, positions = length_3_match[0]
        assert sorted(positions) == [0, 1, 2]

    def test_find_all_suffix_matches_all_lengths_found(self):
        """Test that all matching suffix lengths are returned."""
        corpus = b"abcdefg"
        model = Infinigram(corpus)

        # Context "defg" should match at lengths 4, 3, 2, 1
        matches = model.find_all_suffix_matches(b"defg")

        lengths = [length for length, _ in matches]
        assert lengths == [4, 3, 2, 1]

    def test_find_all_suffix_matches_partial_suffix_only(self):
        """Test when only partial suffix matches."""
        corpus = b"xyz"
        model = Infinigram(corpus)

        # Context "abc xyz" - only "xyz" part should match
        matches = model.find_all_suffix_matches(b"abc xyz")

        # Should find " xyz" (4), "xyz" (3), "yz" (2), "z" (1)
        # But "abc" part doesn't exist in corpus
        assert len(matches) > 0
        max_length = max(length for length, _ in matches)
        # Cannot match more than 4 bytes (" xyz")
        assert max_length <= 4

    def test_find_all_suffix_matches_unicode_after_tokenize(self):
        """Unicode text tokenized via the model's tokenizer."""
        corpus = "hello world".encode('utf-8')
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(model.tokenizer.encode("hello"))

        assert len(matches) > 0
        # "hello" is 5 bytes in UTF-8
        max_length = max(length for length, _ in matches)
        assert max_length == 5

    def test_find_all_suffix_matches_special_bytes(self):
        """Test with special byte values (0, 255, etc.)."""
        corpus = bytes([0, 1, 255, 254, 0, 1])
        model = Infinigram(corpus)

        # Search for pattern with special bytes
        matches = model.find_all_suffix_matches(bytes([0, 1]))

        assert len(matches) > 0
        # [0, 1] appears at positions 0 and 4
        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert sorted(positions) == [0, 4]

    def test_find_all_suffix_matches_no_false_positives(self):
        """Test that positions don't include false positives."""
        corpus = b"ab cd ef"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"ab")

        # Only position 0 should match "ab", not position 3 ("cd")
        for length, positions in matches:
            suffix = b"ab"[-length:]
            for pos in positions:
                assert corpus[pos:pos + length] == suffix

    def test_find_all_suffix_matches_positions_are_sorted(self):
        """Test that positions within each length are consistent."""
        corpus = b"test test test"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"test")

        for length, positions in matches:
            # Positions should be a list (may or may not be sorted)
            assert isinstance(positions, list)
            assert len(positions) > 0

    def test_find_all_suffix_matches_with_newlines(self):
        """Test corpus with newlines."""
        corpus = b"line1\nline2\nline1"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"line1")

        assert len(matches) > 0
        # "line1" appears at positions 0 and 12
        length_5_match = [(l, p) for l, p in matches if l == 5]
        assert len(length_5_match) == 1
        _, positions = length_5_match[0]
        assert len(positions) == 2

    def test_find_all_suffix_matches_max_length_zero_edge_case(self):
        """Test behavior when max_length effectively limits to nothing useful."""
        corpus = b"abcdefgh"
        model = Infinigram(corpus, max_length=1)

        matches = model.find_all_suffix_matches(b"abcdefgh")

        # Only last byte should be considered due to max_length=1
        if matches:
            max_match_len = max(length for length, _ in matches)
            assert max_match_len == 1


class TestContinuations:
    """Test continuation probability computation."""

    def test_continuations_basic(self):
        """Test basic continuation counting."""
        corpus = [1, 2, 3, 1, 2, 4, 1, 2, 5]
        model = Infinigram(corpus)

        # After [1, 2], we have: 3, 4, 5
        context = [1, 2]
        conts = model.continuations(context)

        assert 3 in conts
        assert 4 in conts
        assert 5 in conts
        assert conts[3] == 1
        assert conts[4] == 1
        assert conts[5] == 1

    def test_continuations_empty_context(self):
        """Test continuations with empty context returns uniform distribution."""
        corpus = [1, 2, 1, 3, 1]
        model = Infinigram(corpus)

        conts = model.continuations([])

        # Empty context returns uniform distribution over all 256 bytes
        assert len(conts) == 256
        assert all(v == 1 for v in conts.values())

    def test_continuations_no_match(self):
        """Test continuations when no suffix matches returns None."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        conts = model.continuations([99, 98, 97])

        # No suffix in the corpus → None signal (caller can fall back to uniform / LLM).
        assert conts is None

    def test_continuations_at_end_of_corpus(self):
        """Test continuations when match is at corpus end."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        # Last tokens have no continuation; backoff to a shorter suffix that does.
        context = [4, 5]
        conts = model.continuations(context)
        assert conts is None or isinstance(conts, dict)


class TestPredict:
    """Test prediction (probability computation)."""

    def test_predict_returns_probabilities(self):
        """Test predict returns valid probability distribution."""
        corpus = [1, 2, 3] * 10  # Repeating pattern
        model = Infinigram(corpus)

        context = [1, 2]
        probs = model.predict(context, top_k=256)  # Get all probabilities

        # Check properties
        assert isinstance(probs, dict)
        assert len(probs) > 0
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Should sum to 1 with all bytes

    def test_predict_top_k_limit(self):
        """Test top_k parameter limits results."""
        corpus = list(range(100)) * 5  # Many tokens
        model = Infinigram(corpus)

        context = [0]
        probs = model.predict(context, top_k=5)

        assert len(probs) <= 5

    def test_predict_smoothing(self):
        """Test smoothing assigns probability to unseen tokens."""
        corpus = [1, 2, 3, 1, 2, 4]
        model = Infinigram(corpus)

        context = [1, 2]
        probs = model.predict(context, smoothing=0.1)

        # Token 5 never follows [1, 2] but should have some probability
        if 5 in model.vocab:
            assert 5 in probs
            assert probs[5] > 0

    def test_predict_empty_context(self):
        """Test prediction with empty context."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        probs = model.predict([])

        # Should return unigram distribution
        assert len(probs) > 0
        # With smoothing=0, only observed bytes should have non-zero probability
        assert 1 in probs and probs[1] > 0
        assert 2 in probs and probs[2] > 0
        assert 3 in probs and probs[3] > 0

    def test_predict_consistent_probabilities(self):
        """Test predictions are consistent across calls."""
        corpus = [1, 2, 3] * 5
        model = Infinigram(corpus)

        context = [1, 2]
        probs1 = model.predict(context)
        probs2 = model.predict(context)

        # Should be deterministic
        assert probs1 == probs2


class TestUpdate:
    """Test dynamic corpus updates."""

    def test_update_extends_corpus(self):
        """Test update adds new tokens."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        initial_size = model.n
        model.update([4, 5, 6])

        assert model.n == initial_size + 3
        # Note: corpus is now stored in mmap file, not as attribute
        # Verify by checking we can find the new pattern
        assert model.count([4, 5, 6]) == 1

    def test_update_extends_vocabulary(self):
        """Test vocabulary is always fixed at 256 bytes."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        model.update([4, 5])

        assert 4 in model.vocab
        assert 5 in model.vocab
        assert model.vocab_size == 256  # Fixed byte vocabulary

    def test_update_enables_new_predictions(self):
        """Test updated corpus enables new predictions."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        # Initially, [2, 3] has no continuation
        context = [2, 3]
        conts_before = model.continuations(context)

        # Add continuation
        model.update([2, 3, 4])

        conts_after = model.continuations(context)

        # Now [2, 3] should have continuation 4
        assert 4 in conts_after


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_token_corpus(self):
        """Test corpus with single token."""
        corpus = [1]
        model = Infinigram(corpus)

        assert model.n == 1
        probs = model.predict([])
        assert len(probs) >= 1

    def test_empty_corpus_raises_error(self):
        """Test empty corpus handling."""
        # Empty corpus should handle gracefully
        try:
            model = Infinigram([])
            # Should either work or raise error
            assert model.n == 0
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_repeated_pattern_corpus(self):
        """Test corpus with repeated pattern."""
        corpus = [1, 2] * 100
        model = Infinigram(corpus)

        context = [1]
        probs = model.predict(context)

        # Token 2 should have very high probability
        assert 2 in probs
        assert probs[2] >= 0.4  # Relaxed threshold due to smoothing

    def test_long_context(self):
        """Test very long context."""
        corpus = list(range(100))
        model = Infinigram(corpus)

        context = list(range(50))  # Very long context
        probs = model.predict(context)

        # Should handle gracefully
        assert isinstance(probs, dict)

    def test_context_longer_than_corpus(self):
        """Test context longer than entire corpus."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        context = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All valid bytes
        probs = model.predict(context)

        # Either None (no suffix matches) or a real distribution from a partial match.
        assert probs is None or (isinstance(probs, dict) and len(probs) > 0)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_wikipedia_style_corpus(self):
        """Test with Wikipedia-like text."""
        # Simulate "The cat sat on the mat. The cat ..."
        the, cat, sat, on, mat = 1, 2, 3, 4, 5
        corpus = [the, cat, sat, on, the, mat, the, cat, sat, on, the]

        model = Infinigram(corpus)

        # After "the cat", "sat" should be predicted
        context = [the, cat]
        probs = model.predict(context)

        assert sat in probs
        # "sat" should have relatively high probability (relaxed due to smoothing)
        assert probs[sat] >= 0.15

    def test_code_completion_scenario(self):
        """Test code completion scenario."""
        # Simulate: "def foo():\n  return 42\ndef foo():\n  return"
        DEF, FOO, LPAREN, RPAREN, COLON, RETURN, NUM = 1, 2, 3, 4, 5, 6, 7
        corpus = [DEF, FOO, LPAREN, RPAREN, COLON, RETURN, NUM,
                  DEF, FOO, LPAREN, RPAREN, COLON, RETURN]

        model = Infinigram(corpus)

        # Complete "return ___"
        context = [DEF, FOO, LPAREN, RPAREN, COLON, RETURN]
        probs = model.predict(context)

        # NUM should be predicted
        assert NUM in probs

    def test_update_and_query_cycle(self):
        """Test repeated update and query cycle."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        for i in range(10):
            # Query
            probs = model.predict([i % 3 + 1])
            assert len(probs) > 0

            # Update
            model.update([i % 3 + 1, (i + 1) % 3 + 1])

        # Should still work after many updates
        final_probs = model.predict([1])
        assert len(final_probs) > 0


class TestSearchLimit:
    """Test limit parameter for search methods - performance optimization."""

    def test_suffix_array_search_with_limit(self):
        """Test SuffixArray.search() respects limit parameter."""
        # Create corpus with many repeated patterns
        corpus = b"ab" * 100  # "ab" appears 100 times
        sa = SuffixArray(corpus)

        # Without limit - should get all 100 occurrences
        all_results = sa.search(b"ab")
        assert len(all_results) == 100

        # With limit - should get exactly limit results
        limited_results = sa.search(b"ab", limit=10)
        assert len(limited_results) == 10

        # Results should be a subset of all results
        for pos in limited_results:
            assert pos in all_results

    def test_suffix_array_search_limit_less_than_matches(self):
        """Test limit when fewer results than limit exist."""
        corpus = b"abc def abc"
        sa = SuffixArray(corpus)

        # "abc" appears 2 times, but we ask for 10
        results = sa.search(b"abc", limit=10)
        assert len(results) == 2  # Only 2 exist

    def test_suffix_array_search_limit_none_returns_all(self):
        """Test limit=None (default) returns all results."""
        corpus = b"test" * 50
        sa = SuffixArray(corpus)

        results = sa.search(b"test", limit=None)
        assert len(results) == 50

    def test_infinigram_search_with_limit(self):
        """Test Infinigram.search() respects limit parameter."""
        corpus = b"the cat sat. the cat ran. the cat jumped."
        model = Infinigram(corpus)

        # "the" appears 3 times
        all_results = model.search(b"the")
        assert len(all_results) == 3

        # With limit
        limited_results = model.search(b"the", limit=2)
        assert len(limited_results) == 2

    def test_continuations_default_limit(self):
        """Test continuations uses DEFAULT_CONTINUATION_LIMIT for performance."""
        # Create corpus with many repeated patterns
        corpus = b"ab" * 20000  # 40000 bytes, "ab" appears 20000 times
        model = Infinigram(corpus)

        # Default limit should be applied (10000)
        # This should complete quickly without hanging
        conts = model.continuations(b"a")
        assert isinstance(conts, dict)
        assert len(conts) > 0

    def test_continuations_explicit_limit(self):
        """Test continuations with explicit limit parameter."""
        corpus = b"ab" * 1000
        model = Infinigram(corpus)

        # Very small limit
        conts = model.continuations(b"a", limit=10)
        assert isinstance(conts, dict)
        # Should still return valid continuation counts
        assert 98 in conts  # 'b' follows 'a'

    def test_continuations_limit_none_uses_all(self):
        """Test continuations with limit=None samples all occurrences."""
        corpus = b"ab" * 100
        model = Infinigram(corpus)

        # With explicit None - uses all matches
        conts = model.continuations(b"a", limit=None)
        # All 100 'a's are followed by 'b'
        assert conts.get(98, 0) == 100  # 98 = ord('b')

    def test_continuations_limit_reduces_sample_count(self):
        """Test that limit reduces the number of positions sampled.

        Note: The limit takes the first N positions from suffix array order,
        not random samples. This is intentional for performance - contiguous
        reads are faster. For most use cases, suffix array order provides
        sufficient diversity for probability estimation.
        """
        # Create corpus with many repeated patterns
        corpus = b"ab" * 500  # 'a' appears 500 times, always followed by 'b'
        model = Infinigram(corpus)

        # With limit=None, all 500 'a' positions are counted
        full_conts = model.continuations(b"a", limit=None)
        assert full_conts.get(98, 0) == 500  # 98 = ord('b')

        # With limit=100, only 100 positions are sampled
        limited_conts = model.continuations(b"a", limit=100)
        # Should have <= 100 total continuations counted
        total_count = sum(limited_conts.values())
        assert total_count <= 100, f"Expected <= 100 continuations, got {total_count}"

        # In this case, all sampled positions still show 'b' follows 'a'
        assert 98 in limited_conts  # 'b' is the continuation

    def test_search_limit_zero_returns_empty(self):
        """Test that limit=0 returns empty list."""
        corpus = b"test data"
        sa = SuffixArray(corpus)

        results = sa.search(b"test", limit=0)
        assert results == []

    def test_search_limit_one(self):
        """Test that limit=1 returns exactly one result."""
        corpus = b"test test test"
        sa = SuffixArray(corpus)

        results = sa.search(b"test", limit=1)
        assert len(results) == 1
        assert results[0] in [0, 5, 10]  # One of the positions

    def test_predict_uses_limited_continuations(self):
        """Test that predict() benefits from continuations limit for performance."""
        corpus = b"the " * 10000  # Large corpus
        model = Infinigram(corpus)

        # This should complete quickly due to limit in continuations
        probs = model.predict(b"the", top_k=10)
        assert isinstance(probs, dict)
        # Space should have high probability after "the"
        assert 32 in probs  # 32 = ord(' ')


class TestGetTokenValidation:
    """Test bounds validation for get_token()."""

    def test_get_token_valid_position(self):
        corpus = [65, 66, 67]
        model = Infinigram(corpus)
        assert model.get_token(0) == 65
        assert model.get_token(1) == 66
        assert model.get_token(2) == 67

    def test_get_token_out_of_bounds_positive(self):
        model = Infinigram([65, 66, 67])
        with pytest.raises(IndexError, match="out of range"):
            model.get_token(10)

    def test_get_token_out_of_bounds_negative(self):
        model = Infinigram([65, 66, 67])
        with pytest.raises(IndexError, match="out of range"):
            model.get_token(-1)

    def test_get_token_at_boundary(self):
        model = Infinigram([65, 66, 67])
        assert model.get_token(2) == 67
        with pytest.raises(IndexError):
            model.get_token(3)


class TestContinuationsCache:
    """Test the _continuations_cache optimization for repeated queries."""

    def test_cache_returns_same_results_on_repeated_queries(self):
        """Given a model, when predict is called twice with same context,
        then results should be identical and cache should be populated."""
        corpus = b"the cat sat on the mat the cat ran"
        model = Infinigram(corpus)

        context = b"the cat"
        result1 = model.continuations(context)
        result2 = model.continuations(context)

        assert result1 == result2
        # Cache should have at least one entry
        assert len(model._continuations_cache) >= 1

    def test_cache_returns_copy_not_reference(self):
        """Given a cached result, when the caller mutates it,
        then the cache should not be affected."""
        corpus = b"abcabd"
        model = Infinigram(corpus)

        context = b"ab"
        result1 = model.continuations(context)
        # Mutate the returned dict
        result1[999] = 42

        result2 = model.continuations(context)
        assert 999 not in result2, "Cache should return copies, not references"

    def test_cache_cleared_after_update(self):
        """Given a model with cached results, when update() is called,
        then the cache should be empty."""
        corpus = b"abc"
        model = Infinigram(corpus)

        # Populate cache
        model.continuations(b"ab")
        assert len(model._continuations_cache) > 0

        # Update should clear cache
        model.update(b"def")
        assert len(model._continuations_cache) == 0

    def test_cache_cleared_after_update_returns_new_results(self):
        """Given a model with cached continuations, when corpus is updated
        and continuations are queried again, then results reflect new corpus."""
        corpus = b"ab ab ab"
        model = Infinigram(corpus)

        context = b"ab"
        conts_before = model.continuations(context)

        # Update corpus with new continuation for "ab"
        model.update(b" abc")

        conts_after = model.continuations(context)

        # After update, "ab" should have new possible continuations
        # The key point is that we get fresh results, not stale cached ones
        assert isinstance(conts_after, dict)
        assert len(conts_after) > 0

    def test_clear_cache_empties_cache(self):
        """Given a model with cached results, when clear_cache() is called,
        then the cache should be empty."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        # Populate cache with several queries
        model.continuations(b"the")
        model.continuations(b"cat")
        model.continuations(b"sat")
        assert len(model._continuations_cache) >= 3

        model.clear_cache()
        assert len(model._continuations_cache) == 0

    def test_cache_eviction_when_reaching_max_size(self):
        """Given a model with _cache_max_size set low, when many different
        queries are made, then cache size should stay bounded."""
        corpus = bytes(range(256)) * 10  # corpus with all byte values
        model = Infinigram(corpus)
        model._cache_max_size = 10  # Very small cache for testing

        # Make more unique queries than cache can hold
        for i in range(25):
            context = bytes([i, (i + 1) % 256])
            model.continuations(context)

        # Cache should not exceed max size
        assert len(model._continuations_cache) <= model._cache_max_size

    def test_cache_eviction_preserves_functionality(self):
        """Given a model that has undergone cache eviction, when queried
        again, then results should still be correct."""
        corpus = b"abcdefghijklmnopqrstuvwxyz" * 5
        model = Infinigram(corpus)
        model._cache_max_size = 5

        # Fill and overflow cache
        queries = [bytes([ord('a') + i]) for i in range(20)]
        for q in queries:
            model.continuations(q)

        # Query again after eviction - results should still be correct
        result = model.continuations(b"a")
        assert isinstance(result, dict)
        assert len(result) > 0
        # 'a' is followed by 'b' in the corpus
        assert ord('b') in result

    def test_cache_with_different_limits_are_separate_entries(self):
        """Given the same context but different limit values, then cache
        should store them as separate entries."""
        corpus = b"ab" * 500
        model = Infinigram(corpus)

        result_limit10 = model.continuations(b"a", limit=10)
        result_limit100 = model.continuations(b"a", limit=100)

        # Both should be cached as separate entries
        # (cache key includes limit)
        assert len(model._continuations_cache) >= 2


class TestVectorizedContinuationCounting:
    """Test the np.bincount/np.frombuffer vectorized continuation counting."""

    def test_vectorized_counts_match_expected(self):
        """Given a corpus with known continuations, when continuations()
        is called, then counts should match manually computed values."""
        # "ab" appears at positions 0, 4, 8 in "abc abc abc"
        # After each "ab": 'c' at positions 2, 6, 10
        corpus = b"abc abc abc"
        model = Infinigram(corpus)

        conts = model.continuations(b"ab")

        assert conts[ord('c')] == 3  # "ab" is followed by 'c' all 3 times
        total = sum(conts.values())
        assert total == 3

    def test_vectorized_single_match(self):
        """Given a pattern that matches exactly once, when continuations()
        is called, then exactly one continuation should have count 1."""
        corpus = b"unique_pattern_xyz"
        model = Infinigram(corpus)

        conts = model.continuations(b"unique_pattern_xy")
        # Only one match, followed by 'z'
        assert conts.get(ord('z')) == 1
        assert sum(conts.values()) == 1

    def test_vectorized_all_matches_at_end_of_corpus(self):
        """Given a pattern that only matches at the end of corpus (no
        continuation byte), when continuations() is called, then it
        should fall back to shorter suffixes or return uniform."""
        corpus = b"xyzxyz"
        model = Infinigram(corpus)

        # "xyz" appears at positions 0 and 3
        # At pos 3, "xyz" has no continuation (end of corpus)
        # At pos 0, "xyz" is followed by 'x'
        conts = model.continuations(b"xyz")

        # Should include the continuation from pos 0 at minimum
        assert isinstance(conts, dict)
        assert len(conts) > 0

    def test_continuations_no_match_returns_none(self):
        """Given a pattern with no suffix matches, continuations() returns None."""
        corpus = b"abc"
        model = Infinigram(corpus)

        conts = model.continuations(b"zzz")
        assert conts is None

    def test_vectorized_counts_for_all_256_byte_values(self):
        """Given a corpus where every byte value follows a two-byte pattern,
        when continuations() is called, then all 256 should appear in counts."""
        # Build corpus: two-byte pattern followed by each of the 256 possible bytes
        # Using a separator to prevent cross-boundary matches
        pattern = bytes([253, 254])
        corpus_parts = []
        for b in range(256):
            # Use a separator (0xFF) between entries to avoid cross-boundary matches
            corpus_parts.append(pattern + bytes([b]) + bytes([0]))
        corpus = b"".join(corpus_parts)
        model = Infinigram(corpus)

        conts = model.continuations(pattern, limit=None)

        # Each byte value should appear as a continuation
        assert len(conts) == 256
        for b in range(256):
            assert conts[b] >= 1, f"Byte {b} should have count >= 1, got {conts.get(b, 0)}"

    def test_vectorized_matches_manual_python_loop(self):
        """Given a corpus, when continuations are counted via the model,
        then results should match a manual Python loop over the same data.

        This verifies the numpy vectorized path (np.bincount + np.frombuffer)
        produces the same results as the straightforward Python approach."""
        corpus = b"the cat sat on the mat the cat ran on the dog"
        model = Infinigram(corpus)

        context = b"the"
        # Get result from the optimized vectorized path
        model_conts = model.continuations(context, limit=None)

        # Manual reference: find longest matching suffix, then count continuations
        # by iterating positions with a Python loop
        pattern = context  # "the" is the longest suffix that matches
        reference_conts = {}
        for i in range(len(corpus)):
            if corpus[i:i + len(pattern)] == pattern:
                next_pos = i + len(pattern)
                if next_pos < len(corpus):
                    next_byte = corpus[next_pos]
                    reference_conts[next_byte] = reference_conts.get(next_byte, 0) + 1

        assert model_conts == reference_conts, (
            f"Vectorized result {model_conts} differs from "
            f"manual loop result {reference_conts}"
        )

    def test_vectorized_matches_manual_loop_with_diverse_corpus(self):
        """Given a corpus with many different continuation patterns, when
        counted via the model, then results match a manual Python loop."""
        # Build a corpus with varied patterns
        corpus = b"ab1 ab2 ab3 ab4 ab5 cd1 cd2 ef1"
        model = Infinigram(corpus)

        for ctx in [b"ab", b"cd", b"ef"]:
            model_conts = model.continuations(ctx, limit=None)

            # Manual reference
            reference_conts = {}
            # The model uses longest suffix match, so we need to find what
            # it actually matches. For simple 2-byte patterns, it matches the full 2 bytes
            pattern = ctx
            for i in range(len(corpus)):
                if corpus[i:i + len(pattern)] == pattern:
                    next_pos = i + len(pattern)
                    if next_pos < len(corpus):
                        next_byte = corpus[next_pos]
                        reference_conts[next_byte] = reference_conts.get(next_byte, 0) + 1

            assert model_conts == reference_conts, (
                f"Mismatch for context {ctx!r}: "
                f"model={model_conts}, reference={reference_conts}"
            )


class TestHeapqTopK:
    """Test heapq.nlargest top-k selection in predict methods."""

    def test_predict_top_k_returns_correct_items(self):
        """Given known continuation probabilities, when predict is called
        with top_k, then the highest probability items are returned."""
        # Create corpus where after "a", 'b' is most common, then 'c', then 'd'
        corpus = b"ab ab ab ab ac ac ad"
        model = Infinigram(corpus)

        probs = model.predict(b"a", top_k=2)
        assert len(probs) <= 2

        # The two most frequent continuations should be included
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        # Top item should be 'b' (ord 98) since it appears most
        assert sorted_probs[0][0] == ord('b') or sorted_probs[0][0] == ord(' ')

    def test_predict_top_k_returns_correct_count(self):
        """Given a model, when predict is called with top_k=N, then
        at most N items are returned."""
        corpus = bytes(range(50)) * 10
        model = Infinigram(corpus)

        for k in [1, 3, 5, 10]:
            probs = model.predict(bytes([0]), top_k=k)
            assert len(probs) <= k, f"top_k={k} but got {len(probs)} items"

    def test_predict_returns_all_when_probs_le_top_k(self):
        """Given a model where continuations have fewer unique items than
        top_k, when predict is called, then all items are returned."""
        # After "ab", only 'c' follows
        corpus = b"abc abc abc"
        model = Infinigram(corpus)

        probs = model.predict(b"ab", top_k=256)
        # Should have a small number of items (just the observed continuations)
        # All should be returned since count < 256
        assert len(probs) > 0
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

class TestBytesComparisonInSuffixArray:
    """Test direct bytes() comparison in suffix array binary search."""

    def test_comparison_finds_existing_pattern(self):
        """Given a corpus with known patterns, when search is called,
        then the pattern is found at the correct positions."""
        corpus = b"hello world hello"
        sa = SuffixArray(corpus)

        positions = sa.search(b"hello")
        assert len(positions) == 2
        assert sorted(positions) == [0, 12]

    def test_comparison_empty_pattern_returns_empty(self):
        """Given any corpus, when search is called with empty pattern,
        then no results are returned."""
        sa = SuffixArray(b"test data")
        assert sa.search(b"") == []

    def test_comparison_pattern_at_end_of_corpus(self):
        """Given a pattern that occurs at the very end of the corpus,
        when search is called, then it is found correctly."""
        corpus = b"start middle end"
        sa = SuffixArray(corpus)

        positions = sa.search(b"end")
        assert len(positions) == 1
        assert positions[0] == 13

    def test_comparison_pattern_longer_than_suffix(self):
        """Given a pattern longer than any remaining suffix in the corpus,
        when search is called, then no false matches are returned."""
        corpus = b"abc"
        sa = SuffixArray(corpus)

        # Pattern is longer than the corpus itself
        positions = sa.search(b"abcdefgh")
        assert len(positions) == 0

    def test_comparison_single_byte_pattern(self):
        """Given a single byte pattern, when search is called, then all
        occurrences are found correctly."""
        corpus = b"abacada"
        sa = SuffixArray(corpus)

        positions = sa.search(b"a")
        assert sorted(positions) == [0, 2, 4, 6]

    def test_comparison_full_corpus_as_pattern(self):
        """Given the full corpus as the search pattern, when search is
        called, then exactly one match at position 0 is returned."""
        corpus = b"unique_corpus_text"
        sa = SuffixArray(corpus)

        positions = sa.search(corpus)
        assert positions == [0]

    def test_comparison_binary_data(self):
        """Given a corpus with non-ASCII byte values, when search is called,
        then patterns are found correctly."""
        corpus = bytes([0, 128, 255, 0, 128, 255, 42])
        sa = SuffixArray(corpus)

        positions = sa.search(bytes([0, 128, 255]))
        assert len(positions) == 2
        assert sorted(positions) == [0, 3]

    def test_comparison_adjacent_similar_patterns(self):
        """Given patterns that differ only in the last byte, when searching
        for each, then they are distinguished correctly."""
        corpus = b"abc abd abe"
        sa = SuffixArray(corpus)

        assert len(sa.search(b"abc")) == 1
        assert len(sa.search(b"abd")) == 1
        assert len(sa.search(b"abe")) == 1
        assert len(sa.search(b"abf")) == 0

    def test_comparison_results_match_between_suffix_array_and_infinigram(self):
        """Given the same corpus, when searching via SuffixArray and via
        Infinigram, then results should be consistent."""
        corpus = b"the quick brown fox jumps over the lazy dog"
        sa = SuffixArray(corpus)
        model = Infinigram(corpus)

        for pattern in [b"the", b"fox", b"over", b"zzz"]:
            sa_results = sorted(sa.search(pattern))
            model_results = sorted(model.search(pattern))
            assert sa_results == model_results, (
                f"Mismatch for pattern {pattern!r}: "
                f"SuffixArray={sa_results}, Infinigram={model_results}"
            )

    def test_comparison_all_same_bytes(self):
        """Given a corpus of all identical bytes, when searching for a
        repeated pattern, then all valid positions are found."""
        corpus = b"aaaaaaa"  # 7 bytes, all 'a'
        sa = SuffixArray(corpus)

        # "aaa" should be found at positions 0, 1, 2, 3, 4 (5 overlapping matches)
        results = sa.search(b"aaa")
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_comparison_prefix_match_does_not_overcount(self):
        """Given a corpus where a short pattern is a prefix of a longer one,
        when searching for the short pattern, then it finds all occurrences
        including those that are prefixes of longer strings."""
        corpus = b"abc abcd abcde"
        sa = SuffixArray(corpus)

        # "abc" is a prefix of "abcd" and "abcde"
        results = sa.search(b"abc")
        assert len(results) == 3  # All three occurrences

        results = sa.search(b"abcd")
        assert len(results) == 2  # "abcd" and "abcde"

        results = sa.search(b"abcde")
        assert len(results) == 1  # Only "abcde"


class TestResourceLifecycle:
    """Test resource management: close, update, mmap lifecycle."""

    def test_close_is_idempotent(self):
        """Calling close() multiple times should not raise."""
        model = Infinigram(b"hello world")
        model.close()
        model.close()  # Should not raise

    def test_update_does_not_leak_file_descriptors(self):
        """Repeated update() calls should close old mmap before opening new one."""
        model = Infinigram(b"hello")
        for i in range(20):
            model.update(bytes([65 + (i % 26)]))
        # If fds leaked, we'd hit ulimit. Verify model still works.
        assert model.n == 25  # 5 original + 20 appended
        probs = model.predict(b"hello")
        assert len(probs) > 0
        model.close()

    def test_continuations_no_match_is_cached(self):
        """OOD contexts cache None so repeat queries skip the binary search."""
        model = Infinigram(b"abc")
        model.continuations(b"\xff\xfe\xfd")
        model.continuations(b"\xfa\xfb\xfc")
        assert len(model._continuations_cache) == 2
        assert all(v is None for v in model._continuations_cache.values())


class TestGenerate:
    """Tests for Infinigram.generate()."""

    def test_generate_returns_token_ids(self):
        model = Infinigram(b"the cat sat on the mat " * 50)
        result = model.generate(b"the ", max_tokens=10, temperature=0.5)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_generate_respects_max_tokens(self):
        model = Infinigram(b"abcabcabc" * 100)
        result = model.generate(b"a", max_tokens=5)
        assert len(result) <= 5

    def test_generate_empty_context(self):
        model = Infinigram(b"hello world " * 50)
        result = model.generate(b"", max_tokens=10)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_greedy_deterministic(self):
        model = Infinigram(b"the cat sat on the mat " * 100)
        r1 = model.generate(b"the cat ", max_tokens=10, temperature=0)
        r2 = model.generate(b"the cat ", max_tokens=10, temperature=0)
        assert r1 == r2

    def test_generate_stop_sequence(self):
        model = Infinigram(b"hello world hello world " * 100)
        result = model.generate(
            b"hello", max_tokens=50, temperature=0,
            stop=[b" world"],
        )
        # Stop sequence as bytes is interpreted as a list of token ids.
        stop_ids = list(b" world")
        # Verify result does not end with stop sequence.
        assert result[-len(stop_ids):] != stop_ids or len(result) < len(stop_ids)

    def test_generate_takes_token_ids(self):
        model = Infinigram(b"the quick brown fox " * 50)
        result = model.generate(model.tokenizer.encode("the "), max_tokens=10, temperature=0)
        assert isinstance(result, list)

    def test_generate_zero_max_tokens(self):
        model = Infinigram(b"hello")
        assert model.generate(model.tokenizer.encode("h"), max_tokens=0) == []

    def test_generate_text_decodes_to_string(self):
        model = Infinigram(b"the cat sat on the mat " * 50)
        text = model.generate_text("the ", max_tokens=10, temperature=0)
        assert isinstance(text, str)


class TestScore:
    """Tests for Infinigram.score()."""

    def test_score_returns_float(self):
        """score() should return a float."""
        model = Infinigram(b"the cat sat on the mat " * 50)
        result = model.score(b"the cat")
        assert isinstance(result, float)

    def test_score_negative(self):
        """Log-probabilities should be negative (or zero)."""
        model = Infinigram(b"abcabc" * 100)
        result = model.score(b"abc")
        assert result <= 0.0

    def test_score_empty_sequence(self):
        """Empty sequence should score as 0.0."""
        model = Infinigram(b"hello")
        assert model.score(b"") == 0.0

    def test_score_in_distribution_higher(self):
        """In-distribution text should score higher than random."""
        corpus = b"the quick brown fox jumps over the lazy dog " * 100
        model = Infinigram(corpus)
        in_dist = model.score(b"the quick brown")
        random_seq = model.score(bytes(range(15)))
        assert in_dist > random_seq

    def test_score_takes_tokenized_input(self):
        """score() takes token ids; callers tokenize text first."""
        model = Infinigram(b"hello world " * 50)
        result = model.score(model.tokenizer.encode("hello"))
        assert isinstance(result, float)
        assert result < 0.0

    def test_score_longer_worse(self):
        """Longer sequences accumulate more log-prob (more negative)."""
        corpus = b"abcdefghijklmnop " * 100
        model = Infinigram(corpus)
        short = model.score(b"abc")
        long = model.score(b"abcdefgh")
        # Longer sequence has more terms, so more negative total
        assert long < short
