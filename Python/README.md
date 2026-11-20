# Infinigram: Variable-Length N-gram Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-508%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-79%25-yellowgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Efficient variable-length n-gram language models using suffix arrays for O(m log n) pattern matching. Byte-level (UTF-8), works with any text.

## Overview

Infinigram is a corpus-based language model where **the corpus IS the model** — no gradient descent, no training loop. It uses suffix arrays to find variable-length patterns and predict continuations.

**Key properties:**
- **Instant training**: Build a model by pointing at text
- **Variable-length patterns**: Automatically uses longest matching context
- **Exact matching**: Every prediction traces to actual corpus occurrences
- **O(m log n) queries**: Binary search on suffix arrays
- **Byte-level**: Fixed 256-token vocabulary (UTF-8 bytes 0-255)
- **Mmap-backed**: Memory-mapped files for corpora from 1KB to 100GB+

## Quick Start

### Installation

```bash
pip install py-infinigram
```

Or install from source:

```bash
git clone https://github.com/queelius/infinigram.git
cd infinigram
pip install -e .
```

### Basic Usage

```python
from infinigram import Infinigram

# Create model from text (byte-level, automatic UTF-8 encoding)
model = Infinigram(b"the cat sat on the mat the cat ran")

# Predict next byte probabilities
probs = model.predict(b"the cat", top_k=5)
# {32: 1.0}  # space byte (32) has highest probability

# Find longest matching suffix
position, length = model.longest_suffix(b"the cat")
print(f"Matched {length} bytes at position {position}")

# String input works too (auto-encoded to UTF-8)
probs = model.predict("the cat", top_k=5)
```

### Persistent Models

```python
# Build and save a model
model = Infinigram.build("The quick brown fox jumps over the lazy dog.", "my_model")

# Load it later (instant — uses mmap, not RAM)
model = Infinigram.load("my_model")
probs = model.predict("The quick")
```

## Key Features

### Variable-Length Pattern Matching

Infinigram automatically finds the longest matching suffix in the corpus:

```python
model = Infinigram(b"abcdef abcdef abcxyz")

pos, length = model.longest_suffix(b"abcd")   # length=4
pos, length = model.longest_suffix(b"abc")    # length=3
pos, length = model.longest_suffix(b"zzz")    # length=0 (no match)
```

### Multiple Prediction Strategies

```python
# Longest suffix match
probs = model.predict(b"the cat", top_k=10)
```

### Dynamic Updates

```python
model = Infinigram(b"hello")
model.update(b" world")  # Appends and rebuilds suffix array
print(model.n)  # 11
```

## API Reference

### Infinigram Class

```python
Infinigram(
    corpus_or_path,           # bytes, List[int], str, or Path
    max_length: int = None,   # Max suffix length (None = unlimited)
    min_count: int = 1,       # Min frequency threshold
)
```

### Methods

| Method | Description |
|--------|-------------|
| `predict(context, top_k=50, smoothing=0.0)` | Next-token probability distribution |
| `predict_logprobs(context, top_k=50)` | Log-probabilities |
| `continuations(context, limit=10000)` | Raw continuation counts (`limit=None` for all) |
| `longest_suffix(context)` | `(position, length)` of longest match |
| `find_all_suffix_matches(context)` | All matches at all suffix lengths |
| `search(pattern, limit=None)` | All occurrence positions |
| `count(pattern)` | Number of occurrences |
| `update(new_tokens)` | Append tokens and rebuild index |
| `clear_cache()` | Clear prediction caches |
| `get_context(position, window=50)` | Corpus text around a position |

### Class Methods

| Method | Description |
|--------|-------------|
| `Infinigram.build(corpus, path, ...)` | Build persistent model |
| `Infinigram.load(path)` | Load existing model (mmap, instant) |
| `Infinigram.list_models()` | List models in `~/.infinigram/models/` |

## Performance

Byte-level model with mmap-backed suffix arrays:

| Corpus Size | Construction | Query Latency |
|-------------|-------------|---------------|
| 1 KB | <1 ms | <0.1 ms |
| 1 MB | ~100 ms | <1 ms |
| 1 GB | ~30 s | <10 ms |

**Memory**: ~9 bytes per corpus byte (corpus + suffix array on disk, mmap'd).

**Performance optimizations** (v0.5.0):
- Vectorized continuation counting with `np.bincount` (3-10x speedup)
- LRU cache for repeated `continuations()` queries (5-20x for hot paths)
- Direct bytes comparison in binary search (2-5x vs Python loop)
- Batch SQLite inserts for dataset index rebuild

## Architecture

```
Python API              Infinigram class (predict, search, continuations)
Suffix Array Engine     MmapSuffixArray, ChunkedMmapSuffixArray (pydivsufsort)
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Testing

508 tests, 79% coverage:

```bash
pytest tests/                                    # All tests
pytest tests/ -v                                 # Verbose
pytest tests/ --cov=infinigram --cov-report=html # Coverage
pytest tests/test_infinigram.py -k "cache"       # Keyword filter
```

## Thread Safety

Infinigram instances are **not thread-safe**. Each thread should use its own model instance, or external synchronization must be used.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub](https://github.com/queelius/infinigram)
- [Documentation](https://queelius.github.io/infinigram/)
- [PyPI](https://pypi.org/project/py-infinigram/)
- [Issues](https://github.com/queelius/infinigram/issues)
