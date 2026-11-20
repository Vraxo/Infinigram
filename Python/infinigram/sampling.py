"""Sampling utilities for Infinigram.

Shared helpers for autoregressive generation and nucleus/top-k filtering.
All functions operate on ``Dict[int, float]`` token-id probability distributions.
"""

import math
import random
from typing import Dict, List, Tuple

# Avoids log(0) in temperature scaling and smoothing fallbacks.
LOG_EPSILON = 1e-10


def _apply_temperature(
    probs: Dict[int, float],
    temperature: float,
) -> Tuple[List[int], List[float]]:
    """Return (tokens, weights) after applying temperature scaling.

    At temperature == 1.0 the original weights are returned unchanged.
    Weights are not normalized: ``random.choices`` handles that internally.
    """
    tokens = list(probs.keys())
    weights = list(probs.values())
    if temperature == 1.0:
        return tokens, weights

    # Log-domain with max-subtraction for numerical stability; yields p^(1/T).
    log_probs = [math.log(p + LOG_EPSILON) / temperature for p in weights]
    max_log = max(log_probs)
    weights = [math.exp(lp - max_log) for lp in log_probs]
    return tokens, weights


def sample_from_distribution(
    probs: Dict[int, float],
    temperature: float = 1.0,
) -> int:
    """Sample a single token id from a probability distribution.

    ``temperature`` of 0 is greedy (argmax); values >1 flatten the distribution,
    values <1 sharpen it. Returns 0 for an empty distribution.

    Example:
        >>> probs = {65: 0.5, 66: 0.3, 67: 0.2}
        >>> sample_from_distribution(probs, temperature=1.0) in probs
        True
    """
    if not probs:
        return 0
    if temperature == 0:
        return max(probs.items(), key=lambda x: x[1])[0]

    tokens, weights = _apply_temperature(probs, temperature)
    return random.choices(tokens, weights=weights, k=1)[0]


def sample_multiple(
    probs: Dict[int, float],
    k: int,
    temperature: float = 1.0,
    with_replacement: bool = True,
) -> List[int]:
    """Sample ``k`` token ids from a probability distribution.

    When ``temperature == 0``, returns the top-k tokens by probability (greedy).
    When ``with_replacement=False``, samples without replacement up to
    ``min(k, len(probs))``. Returns an empty list if ``probs`` is empty,
    ``k <= 0``, or all weights are zero.
    """
    if not probs or k <= 0:
        return []
    if temperature == 0:
        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:k]]

    tokens, weights = _apply_temperature(probs, temperature)
    if sum(weights) <= 0:
        return []

    if with_replacement:
        return random.choices(tokens, weights=weights, k=k)

    result: List[int] = []
    for _ in range(min(k, len(tokens))):
        if sum(weights) <= 0:
            break
        idx = random.choices(range(len(tokens)), weights=weights, k=1)[0]
        result.append(tokens[idx])
        # Swap-and-pop: O(1) removal without shifting tail elements.
        tokens[idx], tokens[-1] = tokens[-1], tokens[idx]
        weights[idx], weights[-1] = weights[-1], weights[idx]
        tokens.pop()
        weights.pop()
    return result


def top_k_filter(probs: Dict[int, float], k: int) -> Dict[int, float]:
    """Keep only the k highest-probability tokens, renormalized to sum to 1."""
    if not probs or k <= 0:
        return {}

    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]
    total = sum(p for _, p in sorted_items)
    if total == 0:
        return {t: 1.0 / len(sorted_items) for t, _ in sorted_items}
    return {t: p / total for t, p in sorted_items}


def top_p_filter(probs: Dict[int, float], p: float) -> Dict[int, float]:
    """Nucleus (top-p) sampling filter.

    Keeps the smallest prefix of tokens (sorted by descending probability)
    whose cumulative mass reaches ``p``, then renormalizes to sum to 1.
    """
    if not probs or p <= 0:
        return {}
    if p >= 1.0:
        return dict(probs)

    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    result: Dict[int, float] = {}
    cumsum = 0.0
    for token, prob in sorted_items:
        result[token] = prob
        cumsum += prob
        if cumsum >= p:
            break

    total = sum(result.values())
    if total > 0:
        result = {t: prob / total for t, prob in result.items()}
    return result
