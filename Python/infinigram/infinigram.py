#!/usr/bin/env python3
"""
Infinigram: Variable-length n-gram language model using suffix arrays.

All models use mmap-backed storage for consistent behavior from 1KB to 100GB+.
Supports both:
- In-memory API: Infinigram(corpus) - creates temporary mmap model
- Persistent API: Infinigram.build(corpus, path) / Infinigram.load(path)
"""

import json
import math
import tempfile
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, OrderedDict

from infinigram.suffix_array import (
    MmapSuffixArray,
    ChunkedMmapSuffixArray,
    _to_corpus_array,
    _build_suffix_array,
)
from infinigram.sampling import sample_from_distribution
from infinigram.tokenizers import (
    ByteTokenizer,
    Tokenizer,
    TokenIds,
    vocab_size_of,
    tokenizer_config,
    load_tokenizer,
)

# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".infinigram" / "models"


class Infinigram:
    """
    Variable-length n-gram model using mmap-backed suffix arrays.

    Unlike traditional n-gram models with fixed order, Infinigram finds
    the longest matching suffix in the corpus and uses its continuations
    for prediction.

    All models are disk-backed using memory-mapped files, enabling:
    - Consistent API for any corpus size (1KB to 100GB+)
    - Instant loading (mmap, not RAM)
    - Automatic persistence
    - Minimal memory footprint during queries

    Note:
        Infinigram instances are NOT thread-safe. Each thread should use
        its own model instance, or external synchronization must be used.

    Usage:
        # In-memory style (creates temporary mmap model)
        >>> model = Infinigram([1, 2, 3, 1, 2, 4])
        >>> probs = model.predict([1, 2])

        # Persistent model
        >>> model = Infinigram.build("The cat sat", "my_model")
        >>> model = Infinigram.load("my_model")
    """

    # Threshold for using chunked index (1GB)
    # Suffix array construction needs ~10x chunk size in RAM
    # (corpus + divsufsort working memory + int32 SA)
    # 1GB chunks need ~10GB peak RAM, 2GB chunks need ~20GB
    CHUNK_THRESHOLD = 1 * 1_000_000_000

    def __init__(
        self,
        corpus_or_path: Union[List[int], bytes, str, Path],
        tokenizer: Optional[Tokenizer] = None,
        max_length: Optional[int] = None,
        _from_path: bool = False,
    ):
        """
        Initialize Infinigram model.

        Args:
            corpus_or_path: One of:
                - str: a text corpus (tokenized via ``tokenizer.encode``)
                - bytes: a raw byte corpus (byte tokenizer only)
                - List[int] / ndarray: a pre-tokenized corpus of token ids
                - Path: a directory containing a previously built index (load)
            tokenizer: Tokenizer for text-to-token-id conversion. Defaults to
                ``ByteTokenizer()`` (UTF-8 bytes, vocab=256). Any object with
                ``encode``/``decode`` and ``vocab_size``/``n_vocab`` works.
            max_length: Maximum suffix length to consider (None = unlimited).
            _from_path: Internal flag used by :meth:`load` and :meth:`build`.
        """
        self.max_length = max_length
        self._temp_dir = None

        # Track whether tokenizer was user-supplied so we can detect mismatches
        # against meta.json on load.
        self._user_supplied_tokenizer = tokenizer
        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.vocab_size: int = vocab_size_of(self.tokenizer)
        self.vocab = range(self.vocab_size)

        self._continuations_cache: OrderedDict = OrderedDict()
        self._cache_max_size = 1000

        if _from_path or isinstance(corpus_or_path, Path):
            self._load_from_path(corpus_or_path)
        elif isinstance(corpus_or_path, (list, bytes, np.ndarray)):
            self._build_from_corpus(corpus_or_path)
        elif isinstance(corpus_or_path, str):
            self._build_from_corpus(self.tokenizer.encode(corpus_or_path))
        else:
            raise ValueError(
                f"Invalid argument type: {type(corpus_or_path)}. "
                f"Pass str / bytes / List[int] / ndarray for a corpus, or a Path "
                f"(or call Infinigram.load) for an existing model."
            )

    def _validate_corpus(
        self,
        corpus: Union[List[int], bytes, np.ndarray],
        what: str = "Corpus",
    ) -> np.ndarray:
        """Coerce corpus to a numpy array of the right dtype and validate token ids.

        ``what`` is the noun used in error messages ("Corpus" or "new_tokens").
        Bytes input is only accepted for byte-level tokenizers.
        """
        if isinstance(corpus, bytes):
            if self.vocab_size > 256:
                raise ValueError(
                    "bytes corpora are only valid for byte-level tokenizers; "
                    f"tokenizer has vocab_size={self.vocab_size}. Pass List[int] instead."
                )
            return _to_corpus_array(corpus, self.vocab_size)

        # Validate range on the raw input before narrowing the dtype, so we
        # surface a clean ValueError instead of numpy's OverflowError.
        if isinstance(corpus, np.ndarray):
            if corpus.size == 0:
                return _to_corpus_array(corpus, self.vocab_size)
            max_id, min_id = int(corpus.max()), int(corpus.min())
        else:
            seq = list(corpus)
            if not seq:
                return _to_corpus_array(seq, self.vocab_size)
            max_id, min_id = max(seq), min(seq)
            corpus = seq

        if min_id < 0 or max_id >= self.vocab_size:
            if self.vocab_size == 256:
                message = f"{what} must contain only bytes (0-255), but found {max_id if max_id >= 256 else min_id}"
            else:
                message = (
                    f"{what} contains token ids outside [0, {self.vocab_size}); "
                    f"min={min_id}, max={max_id}"
                )
            raise ValueError(message)
        return _to_corpus_array(corpus, self.vocab_size)

    def _build_from_corpus(self, corpus: Union[List[int], bytes, np.ndarray]):
        """Build a temporary mmap model from in-memory corpus."""
        corpus_arr = self._validate_corpus(corpus)

        self._temp_dir = tempfile.mkdtemp(prefix="infinigram_")
        self.model_path = Path(self._temp_dir)

        np.save(self.model_path / "corpus.npy", corpus_arr)
        np.save(self.model_path / "suffix_array.npy", _build_suffix_array(corpus_arr))

        self.meta = {
            "version": 2,
            "n": int(len(corpus_arr)),
            "dtype": str(corpus_arr.dtype),
            "vocab_size": int(self.vocab_size),
            "tokenizer": tokenizer_config(self.tokenizer),
            "max_length": self.max_length,
        }
        with open(self.model_path / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

        self._sa = MmapSuffixArray(str(self.model_path))
        self._is_chunked = False
        self.n = self._sa.n

    def _load_from_path(self, model_path: Union[str, Path]):
        """Load model from existing directory."""
        self.model_path = Path(model_path)

        meta_file = self.model_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Use Infinigram.build() to create a new model."
            )

        with open(meta_file) as f:
            self.meta = json.load(f)

        # Resolve tokenizer from meta. If the caller also supplied one,
        # raise on mismatch instead of silently using meta's.
        tok_config = self.meta.get("tokenizer")
        if tok_config is not None:
            if self._user_supplied_tokenizer is not None:
                supplied_config = tokenizer_config(self._user_supplied_tokenizer)
                if supplied_config != tok_config:
                    raise ValueError(
                        f"Tokenizer mismatch: index at {self.model_path} was built "
                        f"with {tok_config!r}, but caller supplied {supplied_config!r}. "
                        f"Pass the matching tokenizer or omit the tokenizer= argument."
                    )
            try:
                meta_tokenizer = load_tokenizer(tok_config)
            except ValueError:
                # Custom tokenizer — keep whatever __init__ was given.
                meta_tokenizer = None
            if meta_tokenizer is not None:
                self.tokenizer = meta_tokenizer
                self.vocab_size = vocab_size_of(self.tokenizer)
                self.vocab = range(self.vocab_size)

        self._is_chunked = "num_chunks" in self.meta or "chunks" in self.meta

        if self._is_chunked:
            self._sa = ChunkedMmapSuffixArray(str(self.model_path))
        else:
            self._sa = MmapSuffixArray(str(self.model_path))

        self.n = self._sa.n

    @classmethod
    def build(
        cls,
        corpus: Union[str, bytes, Path, List[int]],
        model_path: Union[str, Path],
        tokenizer: Optional[Tokenizer] = None,
        chunk_size_gb: float = 1.0,
        max_length: Optional[int] = None,
        verbose: bool = True,
    ) -> "Infinigram":
        """Build a persistent Infinigram model.

        Args:
            corpus: Text, bytes, list of token ids, or path to a text/bytes file.
            model_path: Directory to save model files.
            tokenizer: Tokenizer. Defaults to ``ByteTokenizer()``. Must match
                the intended vocabulary of the corpus tokens.
            chunk_size_gb: Per-chunk size for large corpora.
            max_length: Maximum suffix length for predictions.
            verbose: Print progress messages.

        Examples:
            >>> Infinigram.build("Hello world", "hello_model")
            >>> import tiktoken
            >>> Infinigram.build("Hello world", "hello_model", tokenizer=tiktoken.get_encoding("cl100k_base"))
        """
        tok = tokenizer if tokenizer is not None else ByteTokenizer()
        vocab_size = vocab_size_of(tok)

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Resolve input to a numpy array of token ids.
        corpus_arr: np.ndarray
        if isinstance(corpus, Path):
            # Path means "read this file as the corpus source".
            if vocab_size <= 256:
                corpus_arr = np.frombuffer(corpus.read_bytes(), dtype=np.uint8).copy()
            else:
                corpus_arr = _to_corpus_array(tok.encode(corpus.read_text()), vocab_size)
        elif isinstance(corpus, str):
            corpus_arr = _to_corpus_array(tok.encode(corpus), vocab_size)
        elif isinstance(corpus, bytes):
            if vocab_size > 256:
                raise ValueError(
                    "bytes corpora are only supported with byte-level tokenizers; "
                    f"tokenizer has vocab_size={vocab_size}."
                )
            corpus_arr = _to_corpus_array(corpus, vocab_size)
        elif isinstance(corpus, (list, np.ndarray)):
            corpus_arr = _to_corpus_array(corpus, vocab_size)
        else:
            raise ValueError(f"Invalid corpus: {corpus!r}")

        n = int(len(corpus_arr))
        if verbose:
            print(f"Building model at {model_dir}")
            print(f"Corpus: {n:,} tokens (dtype={corpus_arr.dtype}, vocab={vocab_size})")

        start = time.time()
        item_bytes = corpus_arr.dtype.itemsize
        byte_size = n * item_bytes

        if byte_size > cls.CHUNK_THRESHOLD:
            if verbose:
                print(f"Using chunked index (threshold: {cls.CHUNK_THRESHOLD/1e9:.1f} GB)")
            chunk_tokens = int(chunk_size_gb * (1024 ** 3) / max(1, item_bytes))
            ChunkedMmapSuffixArray.build(
                corpus_arr, str(model_dir),
                chunk_size=chunk_tokens, vocab_size=vocab_size,
            )
            is_chunked = True
        else:
            if verbose:
                print("Building suffix array...")
            np.save(model_dir / "corpus.npy", corpus_arr)
            np.save(model_dir / "suffix_array.npy", _build_suffix_array(corpus_arr))
            is_chunked = False

        meta = {
            "version": 2,
            "n": n,
            "dtype": str(corpus_arr.dtype),
            "vocab_size": int(vocab_size),
            "tokenizer": tokenizer_config(tok),
            "max_length": max_length,
        }
        if is_chunked:
            # ChunkedMmapSuffixArray.build wrote the chunk list; preserve it.
            with open(model_dir / "meta.json") as f:
                meta["chunks"] = json.load(f)["chunks"]
        with open(model_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - start
        if verbose and elapsed > 0:
            print(f"Built in {elapsed:.1f}s ({n/elapsed/1e6:.2f} MT/sec)")

        return cls(
            str(model_dir),
            tokenizer=tok,
            max_length=max_length,
            _from_path=True,
        )

    @classmethod
    def load(cls, model_path: Union[str, Path], **kwargs) -> 'Infinigram':
        """
        Load an existing Infinigram model.

        Args:
            model_path: Path to model directory, or model name (looks in ~/.infinigram/models/)
            **kwargs: Override model settings (max_length)

        Returns:
            Infinigram model

        Example:
            >>> model = Infinigram.load("wikipedia")  # loads ~/.infinigram/models/wikipedia
            >>> model = Infinigram.load("/data/my_model")  # absolute path
        """
        path = Path(model_path)

        # If not absolute, check default models directory
        if not path.is_absolute() and not path.exists():
            default_path = DEFAULT_MODELS_DIR / model_path
            if default_path.exists():
                path = default_path

        return cls(str(path), _from_path=True, **kwargs)

    def _normalize_query(self, query: TokenIds) -> List[int]:
        """Normalize a query to ``List[int]`` for cache hashing and slicing.

        Public query methods take ``TokenIds = List[int]`` — callers that have
        text should call ``model.tokenizer.encode(text)`` first. ``np.ndarray``
        and ``bytes`` are also accepted (both iterate as integers), but
        ``str`` is refused: silently calling ``list("abc")`` would produce
        a list of one-character strings that crashes deep in numpy.

        Raises:
            TypeError: if ``query`` is a ``str``.
        """
        if isinstance(query, str):
            raise TypeError(
                "query must be a list of token ids (or bytes/ndarray of token ids); "
                "got str. For text input, call model.tokenizer.encode(text) first."
            )
        if isinstance(query, np.ndarray):
            return query.tolist()
        return list(query)

    # === Query Methods ===

    def search(
        self,
        pattern: TokenIds,
        limit: Optional[int] = None
    ) -> List[int]:
        """
        Search for occurrences of pattern.

        Args:
            pattern: Token ids (use ``model.tokenizer.encode(text)`` for text input)
            limit: Maximum number of results to return (None = all).
                   For large corpora, limiting results significantly improves performance.

        Returns:
            List of positions where pattern occurs
        """
        pattern = self._normalize_query(pattern)
        return self._sa.search(pattern, limit=limit)

    def count(
        self,
        pattern: TokenIds
    ) -> int:
        """
        Count occurrences of pattern.

        Args:
            pattern: Token ids (use ``model.tokenizer.encode(text)`` for text input)

        Returns:
            Number of occurrences
        """
        pattern = self._normalize_query(pattern)
        return self._sa.count(pattern)

    def get_token(self, position: int, chunk_idx: Optional[int] = None) -> int:
        """Return the token id at ``position`` (within ``chunk_idx`` if chunked)."""
        if self._is_chunked:
            if chunk_idx is None:
                raise ValueError("chunk_idx required for chunked models")
            if chunk_idx < 0 or chunk_idx >= len(self._sa.chunks):
                raise IndexError(
                    f"chunk_idx {chunk_idx} out of range [0, {len(self._sa.chunks)})"
                )
            chunk_size = self._sa.chunks[chunk_idx].n
            if position < 0 or position >= chunk_size:
                raise IndexError(
                    f"position {position} out of range [0, {chunk_size}) for chunk {chunk_idx}"
                )
            return int(self._sa.chunks[chunk_idx]._corpus[position])
        if position < 0 or position >= self.n:
            raise IndexError(f"position {position} out of range [0, {self.n})")
        return int(self._sa._corpus[position])

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get size of a specific chunk (for chunked models)."""
        if not self._is_chunked:
            return self.n
        return self._sa.chunks[chunk_idx].n

    def longest_suffix(
        self,
        context: TokenIds
    ) -> Union[Tuple[int, int], Tuple[Tuple[int, int], int]]:
        """
        Find longest matching suffix in corpus.

        Args:
            context: Token ids (use ``model.tokenizer.encode(text)`` for text input)

        Returns:
            For non-chunked models: (position, length) of longest match, or (-1, 0) if no match
            For chunked models: ((chunk_idx, position), length), or (-1, 0) if no match

        Note:
            Use `isinstance(result[0], tuple)` to distinguish chunked from non-chunked results,
            or check `model._is_chunked` before calling.
        """
        context = self._normalize_query(context)
        if not context:
            return (-1, 0)
        if self.max_length:
            context = context[-self.max_length:]

        length, pos = self._longest_match_length(context)
        if length == 0:
            return (-1, 0)
        return (pos, length)

    def _longest_match_length(self, context: TokenIds) -> Tuple[int, int]:
        """Largest k such that context[-k:] occurs as a substring of the corpus,
        plus one example position of that match.

        Uses binary search on k. Monotonicity: if context[-k:] is in the corpus,
        every contiguous substring of context[-k:] is also in the corpus,
        including context[-(k-1):]. So "context[-k:] in corpus" is monotone
        decreasing in k, and the answer is the boundary.

        Total work: O(m log n) (single full-pattern search amortized; the
        log m probes have geometrically shrinking lengths summing to ~2m).

        Returns:
            ``(length, position)`` where length is the longest matching suffix
            length and position is the corpus position of one such match.
            ``(0, -1)`` if no suffix matches at all. For chunked models,
            position is ``(chunk_idx, position_within_chunk)``.
        """
        if len(context) == 0:
            return (0, -1)
        results = self.search(context[-1:], limit=1)
        if not results:
            return (0, -1)
        # Track the best (longest) match's position alongside its length.
        best_length, best_pos = 1, results[0]
        lo, hi = 1, len(context)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            results = self.search(context[-mid:], limit=1)
            if results:
                lo = mid
                best_length, best_pos = mid, results[0]
            else:
                hi = mid - 1
        return (best_length, best_pos)

    def find_all_suffix_matches(
        self,
        context: TokenIds
    ) -> List[Tuple[int, List]]:
        """
        Find all matching suffixes at different lengths.

        For context "abc", searches for "abc", "bc", "c" and returns
        all matches with their corpus positions.

        Args:
            context: Token ids (use ``model.tokenizer.encode(text)`` for text input)

        Returns:
            List of (length, positions) tuples, sorted by decreasing length.
            For chunked models, positions are (chunk_idx, pos) tuples.

        Example:
            >>> model = Infinigram(b"the cat sat on the mat")
            >>> # Returns one entry per length 1..longest, since every length
            >>> # k <= longest_match implies a substring match somewhere by
            >>> # the substring-of-substring property.
            >>> matches = model.find_all_suffix_matches(list(b"the cat"))
            >>> max(L for L, _ in matches)
            7
        """
        context = self._normalize_query(context)
        if not context:
            return []
        if self.max_length:
            context = context[-self.max_length:]

        matches = []
        for length in range(len(context), 0, -1):
            results = self.search(context[-length:])
            if results:
                matches.append((length, list(results)))
        return matches

    # Default limit for continuation sampling - balances accuracy vs performance
    DEFAULT_CONTINUATION_LIMIT = 10000

    def continuations(
        self,
        context: TokenIds,
        limit: Optional[int] = DEFAULT_CONTINUATION_LIMIT
    ) -> Optional[Dict[int, int]]:
        """
        Get continuation counts for the longest matching suffix.

        Args:
            context: Token ids (use ``model.tokenizer.encode(text)`` for text input)
            limit: Maximum occurrences to sample for counting continuations.
                   For common patterns in large corpora, limiting this prevents
                   hanging on millions of matches. Default: 10000.
                   Set to None to count all occurrences (may be slow).

        Returns:
            Dict mapping next_token_id -> count for the longest matching suffix.
            ``None`` if no suffix of context appears in the corpus (the model
            has no information). Empty context returns the uniform-prior dict
            (no context means no information to condition on).
        """
        context = self._normalize_query(context)

        if not context:
            # Fresh copy: callers may mutate the returned dict.
            return dict(self._uniform_counts())

        if self.max_length:
            context = context[-self.max_length:]

        cache_key = (tuple(context), limit)
        if cache_key in self._continuations_cache:
            self._continuations_cache.move_to_end(cache_key)
            cached = self._continuations_cache[cache_key]
            return cached.copy() if cached is not None else None

        # Find the longest suffix length in O(m log n) via binary search.
        length, _ = self._longest_match_length(context)
        if length == 0:
            self._cache_put(cache_key, None)
            return None

        # Walk down lengths to handle the corner case where every match at
        # the chosen length is at the very end of the corpus (no continuation).
        # In practice this loop runs once.
        vocab_size = self.vocab_size
        while length > 0:
            results = self.search(context[-length:], limit=limit)
            if self._is_chunked:
                cont = defaultdict(int)
                for chunk_idx, pos in results:
                    next_pos = pos + length
                    chunk_size = self.get_chunk_size(chunk_idx)
                    if next_pos < chunk_size:
                        cont[self.get_token(next_pos, chunk_idx)] += 1
                if cont:
                    result = dict(cont)
                    self._cache_put(cache_key, result)
                    return result
            else:
                positions = np.asarray(results, dtype=np.int64)
                next_positions = positions + length
                valid_positions = next_positions[next_positions < self.n]
                if len(valid_positions) > 0:
                    next_tokens = self._sa._corpus[valid_positions]
                    counts = np.bincount(next_tokens, minlength=vocab_size)
                    result = {int(t): int(c) for t, c in enumerate(counts) if c > 0}
                    self._cache_put(cache_key, result)
                    return result
            length -= 1

        self._cache_put(cache_key, None)
        return None

    def _uniform_counts(self) -> Dict[int, int]:
        """Lazily-built ``{token_id: 1}`` template for the full vocabulary.

        Returned by reference for internal use; callers that may mutate
        (e.g. ``continuations()``'s empty-context branch) must wrap in
        ``dict(...)`` to get a fresh copy.
        """
        if not hasattr(self, "_uniform_counts_cache"):
            self._uniform_counts_cache = {t: 1 for t in range(self.vocab_size)}
        return self._uniform_counts_cache

    def _cache_put(self, cache_key: tuple, result: Optional[dict]):
        """Store result in LRU cache, evicting oldest entry if full.
        Stores ``None`` for OOD contexts so repeat queries skip the binary search."""
        if len(self._continuations_cache) >= self._cache_max_size:
            self._continuations_cache.popitem(last=False)
        self._continuations_cache[cache_key] = result.copy() if result is not None else None

    def clear_cache(self):
        """Clear prediction caches (call after corpus updates)."""
        self._continuations_cache.clear()

    def predict(
        self,
        context: TokenIds,
        top_k: int = 50,
        smoothing: float = 0.0
    ) -> Optional[Dict[int, float]]:
        """
        Predict next-token probabilities.

        Args:
            context: Token ids (use ``model.tokenizer.encode(text)`` for text input)
            top_k: Return only top k predictions
            smoothing: Laplace smoothing parameter (default: 0.0)

        Returns:
            Dict mapping token id -> probability for the longest matching suffix,
            or ``None`` if no suffix of context appears in the corpus. Callers
            (e.g. mixture-with-LLM grounding) should branch on ``None`` to fall
            back to a baseline distribution.
        """
        counts = self.continuations(context)
        if counts is None:
            return None

        total = sum(counts.values())
        smoothed_total = total + smoothing * self.vocab_size

        probs = {t: (c + smoothing) / smoothed_total for t, c in counts.items()}

        if smoothing > 0:
            unseen_prob = smoothing / smoothed_total
            for t in range(self.vocab_size):
                probs.setdefault(t, unseen_prob)

        if len(probs) > top_k:
            probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return probs

    def predict_logprobs(
        self,
        context: TokenIds,
        top_k: int = 50,
        smoothing: float = 1e-10
    ) -> Optional[Dict[int, float]]:
        """
        Predict next-token log-probabilities.

        Args:
            context: Token ids (use ``model.tokenizer.encode(text)`` for text input)
            top_k: Return only top k predictions
            smoothing: Smoothing to avoid log(0) (default: 1e-10)

        Returns:
            Dict mapping token id -> log probability, or ``None`` if no suffix
            of context appears in the corpus.
        """
        probs = self.predict(context, top_k=top_k, smoothing=smoothing)
        if probs is None:
            return None
        return {token: math.log(p) for token, p in probs.items()}

    def generate(
        self,
        context: Optional[TokenIds] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        smoothing: float = 0.0,
        stop: Optional[List[TokenIds]] = None
    ) -> List[int]:
        """Generate tokens autoregressively.

        Predicts one token at a time, samples from the distribution, and
        extends the context.

        Args:
            context: Starting token ids. ``None`` or ``[]`` for unconditioned
                generation. Convert text via ``model.tokenizer.encode(text)``.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy, 1 = standard).
            top_k: Number of top predictions to consider per step.
                ``None`` (default) uses the full vocabulary; otherwise clamped
                to ``[1, vocab_size]``.
            smoothing: Laplace smoothing parameter.
            stop: Optional stop sequences as lists of token ids. Generation
                halts (and the stop tokens are stripped) when any matches.

        Returns:
            Generated token ids (not including the original context). Call
            ``self.tokenizer.decode(...)`` or :meth:`generate_text` to get
            a string.
        """
        top_k = self.vocab_size if top_k is None else min(top_k, self.vocab_size)

        current_context = self._normalize_query(context if context is not None else [])
        generated: List[int] = []

        stop_lists: Optional[List[List[int]]] = (
            [list(s) for s in stop if s] if stop else None
        )

        for _ in range(max_tokens):
            probs = self.predict(current_context, top_k=top_k, smoothing=smoothing)
            if not probs:
                break

            next_token = sample_from_distribution(probs, temperature)
            generated.append(next_token)
            current_context.append(next_token)

            if stop_lists:
                for stop_seq in stop_lists:
                    L = len(stop_seq)
                    if len(generated) >= L and generated[-L:] == stop_seq:
                        return generated[:-L]

        return generated

    def generate_text(
        self,
        context: str = "",
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        smoothing: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate and decode to text via the model's tokenizer.

        Convenience wrapper that tokenizes ``context`` and ``stop`` strings
        through the model's tokenizer, then calls :meth:`generate` and
        decodes the result.
        """
        seed_tokens = self.tokenizer.encode(context) if context else []
        stop_tokens: Optional[List[List[int]]] = (
            [self.tokenizer.encode(s) for s in stop if s] if stop else None
        )
        tokens = self.generate(
            context=seed_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            smoothing=smoothing,
            stop=stop_tokens,
        )
        return self.tokenizer.decode(tokens)

    def score(
        self,
        sequence: TokenIds,
        smoothing: float = 1e-10
    ) -> float:
        """
        Score a sequence by its total log-probability.

        Computes the sum of log P(token_i | token_0, ..., token_{i-1}) for each
        position, giving the log-likelihood of the sequence under the model.

        Args:
            sequence: Token ids (use ``model.tokenizer.encode(text)`` for text input)
            smoothing: Smoothing to avoid log(0) (default: 1e-10)

        Returns:
            Total log-probability (more negative = less likely)

        Example:
            >>> model = Infinigram(b"the cat sat on the mat " * 100)
            >>> model.score(b"the cat") > model.score(b"xyz abc")
            True
        """
        seq = self._normalize_query(sequence)
        if not seq:
            return 0.0

        vocab_size = self.vocab_size
        # Uniform fallback: used for the first token (no context) and for any
        # later position whose context is OOD (continuations() returns None).
        # Computed once to avoid building a vocab_size-entry dict (matters for
        # tiktoken-scale vocabs).
        uniform_lp = math.log(1.0 / vocab_size)
        total_lp = uniform_lp

        for i in range(1, len(seq)):
            counts = self.continuations(seq[:i])
            if counts is None:
                # No matching suffix: treat as uniform under this model.
                # Callers wanting LLM grounding should mix per-position elsewhere.
                total_lp += uniform_lp
                continue
            total = sum(counts.values())
            count = counts.get(seq[i], 0)
            prob = (count + smoothing) / (total + smoothing * vocab_size)
            if prob <= 0:
                return float('-inf')
            total_lp += math.log(prob)

        return total_lp

    def get_context(
        self,
        position: int,
        window: int = 50,
        chunk_idx: Optional[int] = None
    ) -> Union[bytes, List[int]]:
        """
        Get context around a position.

        Args:
            position: Position in corpus
            window: Number of tokens before and after
            chunk_idx: Chunk index (for chunked models)

        Returns:
            Bytes for byte-level models, list of token ids for token-level models.
        """
        if self._is_chunked:
            if chunk_idx is None:
                raise ValueError("chunk_idx required for chunked models")
            return self._sa.get_context(chunk_idx, position, window)
        else:
            return self._sa.get_context(position, window)

    def update(self, new_tokens: Union[bytes, List[int], str]):
        """
        Update corpus with new tokens and rebuild index.

        Note: This rebuilds the entire suffix array, which is O(n log n).
        For frequent updates, consider batching.

        Args:
            new_tokens: Tokens to append. Strings are tokenized via the model's
                tokenizer; bytes are valid only for byte-level models;
                List[int] is taken as already-tokenized.

        Raises:
            NotImplementedError: If model is chunked (use rebuild instead)
        """
        if self._is_chunked:
            raise NotImplementedError(
                "update() is not supported for chunked models. "
                "Chunked models are designed for large corpora that don't fit in memory. "
                "To add data, rebuild the model with the new corpus using Infinigram.build()."
            )

        if isinstance(new_tokens, str):
            new_arr = _to_corpus_array(self.tokenizer.encode(new_tokens), self.vocab_size)
        else:
            new_arr = self._validate_corpus(new_tokens, what="new_tokens")

        corpus_npy = self.model_path / "corpus.npy"
        existing = np.load(corpus_npy)
        updated = np.concatenate([existing, new_arr])

        # Close old mmaps before overwriting the underlying files.
        self._sa.close()

        np.save(corpus_npy, updated)
        np.save(self.model_path / "suffix_array.npy", _build_suffix_array(updated))

        self.meta["n"] = int(len(updated))
        self.meta["dtype"] = str(updated.dtype)
        with open(self.model_path / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

        self._sa = MmapSuffixArray(str(self.model_path))
        self.n = self._sa.n
        self.clear_cache()

    @property
    def sa(self):
        """Underlying suffix array (``MmapSuffixArray`` or ``ChunkedMmapSuffixArray``)."""
        return self._sa

    def close(self):
        """Close memory-mapped files and clean up temporary directory."""
        if hasattr(self._sa, 'close'):
            self._sa.close()

        # Clean up temporary directory if we created one
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Best effort cleanup

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        chunked_str = f", chunks={len(self._sa.chunks)}" if self._is_chunked else ""
        return f"Infinigram(n={self.n:,}{chunked_str}, path={self.model_path})"

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def list_models(cls, models_dir: Optional[Path] = None) -> List[str]:
        """
        List available models in the models directory.

        Args:
            models_dir: Directory to search (default: ~/.infinigram/models)

        Returns:
            List of model names
        """
        if models_dir is None:
            models_dir = DEFAULT_MODELS_DIR

        if not models_dir.exists():
            return []

        models = []
        for item in models_dir.iterdir():
            if item.is_dir() and (item / "meta.json").exists():
                models.append(item.name)

        return sorted(models)


