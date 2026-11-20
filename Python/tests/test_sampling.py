#!/usr/bin/env python3
"""Tests for the sampling module."""

import pytest
import math
from infinigram.sampling import (
    sample_from_distribution,
    sample_multiple,
    top_k_filter,
    top_p_filter
)


class TestSampleFromDistribution:
    """Tests for sample_from_distribution function."""

    def test_empty_distribution_returns_zero(self):
        """Empty distribution should return 0."""
        result = sample_from_distribution({})
        assert result == 0

    def test_greedy_sampling_returns_max(self):
        """Temperature 0 should return highest probability token."""
        probs = {65: 0.3, 66: 0.5, 67: 0.2}  # B has highest prob
        result = sample_from_distribution(probs, temperature=0)
        assert result == 66

    def test_standard_sampling_returns_valid_token(self):
        """Temperature 1 should return a valid token from distribution."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = sample_from_distribution(probs, temperature=1.0)
        assert result in probs

    def test_high_temperature_sampling(self):
        """High temperature should still return valid tokens."""
        probs = {65: 0.9, 66: 0.1}
        # Run multiple times to verify it works
        for _ in range(100):
            result = sample_from_distribution(probs, temperature=2.0)
            assert result in probs

    def test_low_temperature_sampling(self):
        """Low temperature should favor high probability tokens."""
        probs = {65: 0.9, 66: 0.05, 67: 0.05}
        # With very low temperature, should almost always return 65
        results = [sample_from_distribution(probs, temperature=0.1) for _ in range(100)]
        assert results.count(65) > 90  # Almost all should be 65

    def test_single_token_distribution(self):
        """Single token distribution should always return that token."""
        probs = {42: 1.0}
        result = sample_from_distribution(probs, temperature=1.0)
        assert result == 42


class TestSampleMultiple:
    """Tests for sample_multiple function."""

    def test_empty_distribution_returns_empty(self):
        """Empty distribution should return empty list."""
        result = sample_multiple({}, k=5)
        assert result == []

    def test_zero_samples_returns_empty(self):
        """Requesting zero samples should return empty list."""
        probs = {65: 0.5, 66: 0.5}
        result = sample_multiple(probs, k=0)
        assert result == []

    def test_returns_k_samples_with_replacement(self):
        """Should return exactly k samples with replacement."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = sample_multiple(probs, k=10)
        assert len(result) == 10
        assert all(r in probs for r in result)

    def test_greedy_returns_top_k(self):
        """Temperature 0 should return top-k unique tokens."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = sample_multiple(probs, k=2, temperature=0)
        assert result == [65, 66]  # Top 2 by probability

    def test_without_replacement_returns_unique(self):
        """Without replacement should return distinct tokens."""
        probs = {65: 0.4, 66: 0.3, 67: 0.2, 68: 0.1}
        result = sample_multiple(probs, k=3, with_replacement=False)
        assert len(result) == 3
        assert len(set(result)) == 3
        assert all(r in probs for r in result)

    def test_without_replacement_k_exceeds_vocab(self):
        """Without replacement with k > vocab returns all tokens."""
        probs = {65: 0.5, 66: 0.5}
        result = sample_multiple(probs, k=10, with_replacement=False)
        assert set(result) == {65, 66}

    def test_all_zero_weights_returns_empty(self):
        """All-zero weights should return empty list instead of crashing."""
        probs = {65: 0.0, 66: 0.0}
        assert sample_multiple(probs, k=2, with_replacement=False) == []
        assert sample_multiple(probs, k=2, with_replacement=True) == []

    def test_without_replacement_mixed_zero_stops_early(self):
        """Without replacement: after non-zero tokens are drawn, remaining zero-weight tokens are skipped."""
        probs = {65: 1.0, 66: 0.0, 67: 0.0}
        result = sample_multiple(probs, k=3, with_replacement=False)
        assert result == [65]


class TestTopKFilter:
    """Tests for top_k_filter function."""

    def test_top_k_filters_to_k_tokens(self):
        """Should keep only top k tokens."""
        probs = {65: 0.4, 66: 0.3, 67: 0.2, 68: 0.1}
        result = top_k_filter(probs, k=2)
        assert len(result) == 2
        assert 65 in result
        assert 66 in result

    def test_top_k_normalizes(self):
        """Filtered distribution should sum to 1."""
        probs = {65: 0.4, 66: 0.3, 67: 0.2, 68: 0.1}
        result = top_k_filter(probs, k=2)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_top_k_empty_returns_empty(self):
        """Empty input should return empty."""
        result = top_k_filter({}, k=5)
        assert result == {}

    def test_top_k_zero_returns_empty(self):
        """k=0 should return empty."""
        probs = {65: 0.5, 66: 0.5}
        result = top_k_filter(probs, k=0)
        assert result == {}

    def test_top_k_larger_than_vocab(self):
        """k larger than vocab should return all."""
        probs = {65: 0.5, 66: 0.5}
        result = top_k_filter(probs, k=10)
        assert len(result) == 2

    def test_top_k_all_zero_falls_back_to_uniform(self):
        """All-zero probabilities should fall back to uniform over top-k."""
        probs = {65: 0.0, 66: 0.0, 67: 0.0}
        result = top_k_filter(probs, k=2)
        assert len(result) == 2
        assert all(abs(v - 0.5) < 1e-10 for v in result.values())


class TestTopPFilter:
    """Tests for top_p_filter (nucleus sampling)."""

    def test_top_p_filters_by_cumulative_probability(self):
        """Should keep tokens until cumulative prob exceeds p."""
        probs = {65: 0.5, 66: 0.3, 67: 0.15, 68: 0.05}
        result = top_p_filter(probs, p=0.8)
        # 0.5 + 0.3 = 0.8, so should include 65 and 66
        assert 65 in result
        assert 66 in result

    def test_top_p_normalizes(self):
        """Filtered distribution should sum to 1."""
        probs = {65: 0.5, 66: 0.3, 67: 0.15, 68: 0.05}
        result = top_p_filter(probs, p=0.8)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_top_p_one_returns_all(self):
        """p=1 should return all tokens."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = top_p_filter(probs, p=1.0)
        assert len(result) == 3

    def test_top_p_zero_returns_empty(self):
        """p=0 should return empty."""
        probs = {65: 0.5, 66: 0.5}
        result = top_p_filter(probs, p=0)
        assert result == {}

    def test_top_p_empty_returns_empty(self):
        """Empty input should return empty."""
        result = top_p_filter({}, p=0.5)
        assert result == {}
