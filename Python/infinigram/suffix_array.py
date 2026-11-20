"""Suffix array implementation for efficient n-gram operations.

Supports corpora stored as ``uint8`` (byte-level, up to 256 vocab) or ``int32``
(token-level, up to ~2 billion distinct token ids). Construction uses
``pydivsufsort``, which handles both dtypes natively. Binary-search
comparisons dispatch on dtype: ``uint8`` uses C-level ``memcmp``; wider
dtypes use element-wise numpy comparison.
"""

from typing import Any, Callable, List, Optional, Tuple, Union
import json
from pathlib import Path
import numpy as np
import pydivsufsort

_INT32_MAX = np.iinfo(np.int32).max

CorpusLike = Union[bytes, List[int], np.ndarray]


def _to_corpus_array(tokens: CorpusLike, vocab_size: int = 256) -> np.ndarray:
    """Normalize input to a 1-D numpy array of the narrowest suitable int dtype.

    vocab_size <= 256 selects ``uint8`` (fast byte path); otherwise ``int32``.
    """
    if vocab_size <= 256:
        if isinstance(tokens, bytes):
            return np.frombuffer(tokens, dtype=np.uint8).copy()
        if isinstance(tokens, np.ndarray):
            return tokens.astype(np.uint8, copy=False)
        return np.fromiter(tokens, dtype=np.uint8)
    if isinstance(tokens, bytes):
        raise TypeError(
            f"bytes input is only supported with vocab_size <= 256 "
            f"(got vocab_size={vocab_size}); pass a list of token ids instead"
        )
    if isinstance(tokens, np.ndarray):
        return tokens.astype(np.int32, copy=False)
    return np.fromiter(tokens, dtype=np.int32)


def _build_suffix_array(data: np.ndarray) -> np.ndarray:
    """Build suffix array over a numpy array corpus.

    Returns int32 SA for corpora under 2 GB worth of positions, int64 above.
    Handles empty and single-element corpora directly (pydivsufsort needs n>=2).
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([0], dtype=np.int32)
    sa = pydivsufsort.divsufsort(data)
    if n > _INT32_MAX:
        sa = sa.astype(np.int64)
    return sa


def _normalize_pattern(pattern: CorpusLike, corpus: np.ndarray) -> np.ndarray:
    """Coerce ``pattern`` to a 1-D numpy array matching the corpus dtype."""
    target_dtype = corpus.dtype
    if isinstance(pattern, np.ndarray):
        return pattern.astype(target_dtype, copy=False)
    if isinstance(pattern, bytes):
        if target_dtype == np.uint8:
            return np.frombuffer(pattern, dtype=np.uint8)
        return np.frombuffer(pattern, dtype=np.uint8).astype(target_dtype)
    return np.fromiter(pattern, dtype=target_dtype)


def _compare_bytes(
    pattern: bytes,
    pos: int,
    corpus_bytes: memoryview,
    n: int,
    prefix_only: bool = False,
) -> int:
    """Compare ``pattern`` against the suffix at ``pos`` in a byte-level corpus."""
    if pos < 0 or pos >= n:
        return -1
    pattern_len = len(pattern)
    suffix_len = n - pos
    compare_len = min(pattern_len, suffix_len)
    suffix_slice = bytes(corpus_bytes[pos:pos + compare_len])
    pattern_slice = pattern[:compare_len]
    if pattern_slice < suffix_slice:
        return -1
    if pattern_slice > suffix_slice:
        return 1
    if prefix_only:
        return 0 if pattern_len <= suffix_len else 1
    if pattern_len < suffix_len:
        return -1
    if pattern_len > suffix_len:
        return 1
    return 0


def _compare_array(
    pattern: np.ndarray,
    pos: int,
    corpus: np.ndarray,
    n: int,
    prefix_only: bool = False,
) -> int:
    """Compare ``pattern`` against the suffix at ``pos`` in a token-level corpus."""
    if pos < 0 or pos >= n:
        return -1
    pattern_len = pattern.size
    suffix_len = n - pos
    compare_len = min(pattern_len, suffix_len)
    diff = pattern[:compare_len] != corpus[pos:pos + compare_len]
    if diff.any():
        idx = int(np.argmax(diff))
        return -1 if pattern[idx] < corpus[pos + idx] else 1
    if prefix_only:
        return 0 if pattern_len <= suffix_len else 1
    if pattern_len < suffix_len:
        return -1
    if pattern_len > suffix_len:
        return 1
    return 0


def _binary_search_range(
    pattern_arr: np.ndarray,
    corpus: np.ndarray,
    corpus_bytes: Optional[memoryview],
    suffix_array: np.ndarray,
    n: int,
) -> Tuple[int, int]:
    """Return [left, right) range in ``suffix_array`` matching ``pattern_arr``."""
    if corpus.dtype == np.uint8:
        pattern_bytes = pattern_arr.tobytes()
        left = _bsearch(
            n,
            lambda mid: _compare_bytes(pattern_bytes, int(suffix_array[mid]), corpus_bytes, n) > 0,
        )
        right = _bsearch(
            n,
            lambda mid: _compare_bytes(pattern_bytes, int(suffix_array[mid]), corpus_bytes, n, prefix_only=True) >= 0,
        )
    else:
        left = _bsearch(
            n,
            lambda mid: _compare_array(pattern_arr, int(suffix_array[mid]), corpus, n) > 0,
        )
        right = _bsearch(
            n,
            lambda mid: _compare_array(pattern_arr, int(suffix_array[mid]), corpus, n, prefix_only=True) >= 0,
        )
    return left, right


def _bsearch(n: int, advance: Callable[[int], bool]) -> int:
    """Generic binary search returning the smallest mid for which ``advance(mid)`` is False."""
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if advance(mid):
            left = mid + 1
        else:
            right = mid
    return left


def _search_positions(
    pattern: CorpusLike,
    corpus: np.ndarray,
    corpus_bytes: Optional[memoryview],
    suffix_array: np.ndarray,
    n: int,
    limit: Optional[int],
) -> List[int]:
    """Return positions of ``pattern`` in the corpus, capped at ``limit``."""
    pattern_arr = _normalize_pattern(pattern, corpus)
    if pattern_arr.size == 0:
        return []
    left, right = _binary_search_range(pattern_arr, corpus, corpus_bytes, suffix_array, n)
    if left >= right:
        return []
    if limit is not None and (right - left) > limit:
        right = left + limit
    return [int(suffix_array[i]) for i in range(left, right)]


def _count_pattern(
    pattern: CorpusLike,
    corpus: np.ndarray,
    corpus_bytes: Optional[memoryview],
    suffix_array: np.ndarray,
    n: int,
) -> int:
    """Return the number of occurrences of ``pattern`` in the corpus."""
    pattern_arr = _normalize_pattern(pattern, corpus)
    if pattern_arr.size == 0:
        return 0
    left, right = _binary_search_range(pattern_arr, corpus, corpus_bytes, suffix_array, n)
    return max(0, right - left)


class SuffixArray:
    """In-memory suffix array supporting byte- and token-level corpora."""

    def __init__(self, tokens: CorpusLike, vocab_size: int = 256):
        """Build a suffix array from a token sequence.

        Args:
            tokens: Corpus as bytes (uint8 path only), list of ints, or ndarray.
            vocab_size: Upper bound on token ids. <=256 selects the uint8 path.
        """
        self.vocab_size = vocab_size
        self._corpus = _to_corpus_array(tokens, vocab_size)
        self.n = int(self._corpus.shape[0])
        self.suffix_array = _build_suffix_array(self._corpus)

        # Zero-copy bytes view for the uint8 fast path.
        self._corpus_bytes = memoryview(self._corpus) if self.is_byte_level else None

    @property
    def is_byte_level(self) -> bool:
        return self._corpus.dtype == np.uint8

    def find_longest_suffix(self, query: CorpusLike) -> Tuple[int, int]:
        """Return (position, length) of the longest suffix of query in the corpus.

        Scans longest-to-shortest, returning on the first match. Handles gaps
        correctly (a shorter suffix may be absent while a longer one is present).
        """
        query_arr = _normalize_pattern(query, self._corpus)
        if query_arr.size == 0:
            return (-1, 0)

        for suffix_len in range(query_arr.size, 0, -1):
            positions = _search_positions(
                query_arr[-suffix_len:], self._corpus, self._corpus_bytes,
                self.suffix_array, self.n, limit=1,
            )
            if positions:
                return (positions[0], suffix_len)
        return (-1, 0)

    def search(
        self,
        pattern: CorpusLike,
        limit: Optional[int] = None,
    ) -> List[int]:
        """All positions where ``pattern`` occurs in the corpus."""
        return _search_positions(
            pattern, self._corpus, self._corpus_bytes, self.suffix_array, self.n, limit,
        )

    def count(self, pattern: CorpusLike) -> int:
        """Count occurrences without materializing positions."""
        return _count_pattern(
            pattern, self._corpus, self._corpus_bytes, self.suffix_array, self.n,
        )

    def get_context(self, position: int, window: int = 10) -> Tuple[List[int], List[int]]:
        start = max(0, position - window)
        end = min(self.n, position + window)
        before = self._corpus[start:position].tolist()
        after = self._corpus[position:end].tolist()
        return before, after

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"SuffixArray(n={self.n}, dtype={self._corpus.dtype})"


class MmapSuffixArray:
    """Memory-mapped suffix array for corpora larger than RAM.

    Layout on disk:
        meta.json        {"dtype", "n", "vocab_size", "version"}
        corpus.npy       numpy array of token ids (uint8 or int32)
        suffix_array.npy numpy array of int32/int64 positions
    """

    _FORMAT_VERSION = 2

    def __init__(self, index_path: str):
        index_dir = Path(index_path)
        meta_path = index_dir / "meta.json"
        corpus_path = index_dir / "corpus.npy"
        if not corpus_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                f"Use MmapSuffixArray.build() to create it first."
            )

        with open(meta_path) as f:
            meta = json.load(f)
        self.vocab_size = int(meta.get("vocab_size", 256))

        self._corpus = np.load(corpus_path, mmap_mode="r")
        self.n = int(self._corpus.shape[0])
        self.suffix_array = np.load(index_dir / "suffix_array.npy", mmap_mode="r")

        self._corpus_bytes = memoryview(self._corpus) if self.is_byte_level else None

    @property
    def is_byte_level(self) -> bool:
        return self._corpus.dtype == np.uint8

    @classmethod
    def build(
        cls,
        corpus: Union[bytes, List[int], np.ndarray, Path],
        index_path: str,
        vocab_size: int = 256,
    ) -> "MmapSuffixArray":
        """Build a memory-mapped index.

        ``corpus`` may be a ``Path`` to a raw-bytes file (byte-level only),
        or an in-memory sequence of token ids.
        """
        import time

        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(corpus, Path):
            if vocab_size > 256:
                raise ValueError(
                    "File-path corpora are only supported with vocab_size <= 256"
                )
            print(f"Loading corpus from {corpus}...")
            corpus_arr = np.frombuffer(corpus.read_bytes(), dtype=np.uint8).copy()
        else:
            corpus_arr = _to_corpus_array(corpus, vocab_size)

        n = int(corpus_arr.shape[0])
        print(f"Corpus: {n:,} tokens (dtype={corpus_arr.dtype}, vocab<={vocab_size})")

        np.save(index_dir / "corpus.npy", corpus_arr)

        print("Building suffix array...")
        start = time.time()
        sa = _build_suffix_array(corpus_arr)
        elapsed = time.time() - start
        rate = (n / elapsed / 1_000_000) if elapsed > 0 else float("inf")
        print(f"Built in {elapsed:.2f}s ({rate:.1f} MT/sec)")

        np.save(index_dir / "suffix_array.npy", sa)

        meta = {
            "version": cls._FORMAT_VERSION,
            "dtype": str(corpus_arr.dtype),
            "n": n,
            "vocab_size": int(vocab_size),
        }
        with open(index_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        del corpus_arr, sa
        print(f"Index saved to {index_dir}")
        return cls(str(index_dir))

    def search(
        self,
        pattern: CorpusLike,
        limit: Optional[int] = None,
    ) -> List[int]:
        return _search_positions(
            pattern, self._corpus, self._corpus_bytes, self.suffix_array, self.n, limit,
        )

    def count(self, pattern: CorpusLike) -> int:
        return _count_pattern(
            pattern, self._corpus, self._corpus_bytes, self.suffix_array, self.n,
        )

    def get_context(self, position: int, window: int = 50) -> Any:
        """Bytes for byte-level indexes, list of ints for token-level."""
        start = max(0, position - window)
        end = min(self.n, position + window)
        if self.is_byte_level:
            return bytes(self._corpus_bytes[start:end])
        return self._corpus[start:end].tolist()

    def close(self):
        # Drop numpy views so the underlying mmap can be released.
        self._corpus_bytes = None
        self._corpus = None
        self.suffix_array = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"MmapSuffixArray(n={self.n:,}, dtype={self._corpus.dtype})"


class ChunkedMmapSuffixArray:
    """Chunked suffix array for corpora that don't fit in RAM at build time.

    Splits the corpus into chunks, builds a separate :class:`MmapSuffixArray`
    per chunk, and merges query results. All chunks share dtype and vocab_size.
    """

    _FORMAT_VERSION = 2

    def __init__(self, index_path: str):
        index_dir = Path(index_path)
        meta_file = index_dir / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Chunked index not found at {index_dir}. "
                f"Use ChunkedMmapSuffixArray.build() to create it first."
            )
        with open(meta_file) as f:
            meta = json.load(f)

        self.vocab_size = int(meta.get("vocab_size", 256))
        self.n = int(meta["n"])
        chunk_names = meta.get("chunks", [])
        self.chunks: List[MmapSuffixArray] = [
            MmapSuffixArray(str(index_dir / name)) for name in chunk_names
        ]

    @property
    def is_byte_level(self) -> bool:
        return all(c.is_byte_level for c in self.chunks)

    def get_chunk_size(self, chunk_idx: int) -> int:
        return self.chunks[chunk_idx].n

    @classmethod
    def build(
        cls,
        corpus: Union[bytes, List[int], np.ndarray, Path],
        index_path: str,
        chunk_size: int = 500_000_000,
        vocab_size: int = 256,
    ) -> "ChunkedMmapSuffixArray":
        """Build a chunked index, splitting the corpus into fixed-size segments."""
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        chunk_names: List[str] = []
        if isinstance(corpus, Path):
            if vocab_size > 256:
                raise ValueError(
                    "File-path corpora are only supported with vocab_size <= 256"
                )
            total_n = corpus.stat().st_size
            with open(corpus, "rb") as f:
                idx = 0
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    arr = np.frombuffer(data, dtype=np.uint8).copy()
                    name = f"chunk_{idx:04d}"
                    MmapSuffixArray.build(arr, str(index_dir / name), vocab_size=vocab_size)
                    chunk_names.append(name)
                    idx += 1
        else:
            corpus_arr = _to_corpus_array(corpus, vocab_size)
            total_n = int(corpus_arr.shape[0])
            for i, start in enumerate(range(0, total_n, chunk_size)):
                arr = corpus_arr[start:start + chunk_size]
                name = f"chunk_{i:04d}"
                MmapSuffixArray.build(arr, str(index_dir / name), vocab_size=vocab_size)
                chunk_names.append(name)
            del corpus_arr

        meta = {
            "version": cls._FORMAT_VERSION,
            "n": int(total_n),
            "vocab_size": int(vocab_size),
            "chunks": chunk_names,
        }
        with open(index_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        return cls(str(index_dir))

    def search(
        self,
        pattern: CorpusLike,
        limit: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Return [(chunk_idx, position_within_chunk), ...] across all chunks."""
        results: List[Tuple[int, int]] = []
        remaining = limit
        for chunk_idx, chunk in enumerate(self.chunks):
            positions = chunk.search(pattern, limit=remaining)
            for p in positions:
                results.append((chunk_idx, p))
            if remaining is not None:
                remaining -= len(positions)
                if remaining <= 0:
                    break
        return results

    def count(self, pattern: CorpusLike) -> int:
        return sum(chunk.count(pattern) for chunk in self.chunks)

    def close(self):
        for c in self.chunks:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"ChunkedMmapSuffixArray(n={self.n:,}, chunks={len(self.chunks)})"
