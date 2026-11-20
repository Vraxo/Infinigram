# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinigram is a **corpus-based language model** using suffix arrays for variable-length n-gram pattern matching.

- **Instant training**: The corpus IS the model (no gradient descent)
- **Variable-length patterns**: Automatically uses longest matching context
- **Exact matching**: Every prediction traces to actual corpus occurrences
- **O(m log n) queries**: Binary search on suffix arrays
- **Tokenizer-agnostic**: Default `ByteTokenizer` (UTF-8 bytes, vocab=256) or any tokenizer satisfying the `Tokenizer` protocol (`tiktoken`, HuggingFace fast tokenizers, SentencePiece, or custom)

`Infinigram.predict(context)` returns a probability distribution; `generate()` samples autoregressively; `score()` returns log-likelihood. Can be used standalone or composed.

## Commands

### Testing
```bash
pytest tests/                                    # All tests
pytest tests/ -v                                 # Verbose
pytest tests/ --cov=infinigram --cov-report=html # Coverage
pytest tests/test_infinigram.py                  # Single file
pytest tests/test_infinigram.py::TestPredict     # Single class
pytest tests/ -k "predict"                       # Keyword filter
```

### Development
```bash
pip install -e .              # Editable install
pip install -e .[dev]         # With dev dependencies
```

### Code Quality
```bash
black infinigram/ tests/      # Format
flake8 infinigram/ tests/     # Lint
mypy infinigram/              # Type check
```

### Documentation
```bash
mkdocs serve                  # Live preview at http://127.0.0.1:8000
mkdocs build                  # Build static site to site/
```

## Architecture

Top-level layout (beyond the package itself):
- `paper/`: LaTeX research paper (`main.tex`, `projections-followup.tex`)
- `docs/`: mkdocs source (`REST_API.md`, `ARCHITECTURE.md`, `index.md`)
- `scripts/build_wikipedia.py`: example large-corpus build helper
- `FUTURE_WORK.md`: research roadmap (augmentations, fuzzy matching, FM-index)

### Core Classes

**`Tokenizer` protocol** (`tokenizers.py`). Any object with `encode(text) -> List[int]`, `decode(tokens) -> str`, and `vocab_size` (or tiktoken's `n_vocab`) is accepted. `ByteTokenizer` is the default identity-over-UTF-8 implementation.

**`SuffixArray` / `MmapSuffixArray` / `ChunkedMmapSuffixArray`** (`suffix_array.py`). O(n log n) construction, O(m log n) queries. Dispatch on dtype: `uint8` uses C-level `memcmp` (byte path); `int32`/`int64` use element-wise numpy comparison (token path).
- `search(pattern)` → all occurrence positions
- `find_longest_suffix(query)` → (position, length) of longest match
- `get_context(position, window)` → token slice around position
- `ngrams(n)` → iterator of (ngram, count) tuples

**`Infinigram`** (`infinigram.py`). Core model; tokenizer-aware.
- `predict(context, top_k)` → probability distribution with Laplace smoothing
- `continuations(context, limit=10000)` → raw continuation counts per token id
- `generate(context)` → token ids (`bytes` for byte-level back compat)
- `generate_text(context)` → decoded string via `self.tokenizer.decode`
- `score(sequence)` → total log-probability
- `longest_suffix(context)` → (position, length)
- `find_all_suffix_matches(context)` → matches at all suffix lengths
- `search(pattern, limit)` / `count(pattern)` → occurrences
- `update(new_tokens)` → append and rebuild
- `clear_cache()` → clear prediction caches (auto-called by `update()`)

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `tokenizers.py` | `Tokenizer` protocol, `ByteTokenizer`, config save/load |
| `sampling.py` | Sampling primitives: `sample_from_distribution`, `top_k_filter`, `top_p_filter` |
| `corpus_utils.py` | Corpus loading and building (text files, JSONL, HuggingFace datasets) |

### Algorithm Flow

**`predict(context)`**:
1. Binary search suffix array for longest matching suffix
2. Collect all tokens following matched pattern
3. Apply Laplace smoothing: `P(token) = (count + α) / (total + α × vocab_size)`
4. Return top_k as probability distribution

## Key Implementation Details

- **Tokenizer is a property of the model**, recorded in `meta.json` (`{"kind": "byte" | "tiktoken" | "custom", ...}`). Load-time mismatch surfaces as a `ValueError`, not silent scrambling.
- **Storage**: v2 on-disk layout is `corpus.npy` (numpy, dtype-tagged) + `suffix_array.npy` + `meta.json`. v1 layout (`corpus.bin` raw bytes) is still readable for backward compatibility.
- **Dtype dispatch**: `vocab_size ≤ 256` → `uint8` corpus (fast byte path with `memcmp`); otherwise → `int32` (token path with element-wise comparison).
- **Suffix array access**: `model._sa` (private; `MmapSuffixArray` or `ChunkedMmapSuffixArray`). The corpus dtype is at `model._sa._corpus.dtype`; `model._sa.is_byte_level` is the convenience predicate.
- **Mmap-backed**: All models use memory-mapped files (even in-memory API creates temp mmap)
- **Thread safety**: Infinigram instances are NOT thread-safe (use one per thread)
- **Chunked models**: Corpora exceeding `CHUNK_THRESHOLD` (1 GB by byte count) use `ChunkedMmapSuffixArray`
- **Cache keys**: `continuations()` LRU uses `(tuple(context), limit)`, which works for any dtype

### Performance Optimizations
- Vectorized continuation counting via `np.bincount` on the corpus array (not `np.frombuffer`)
- LRU cache (`OrderedDict`) for `continuations()` with bounded eviction
- Direct bytes/memcmp comparison for byte-level indexes; numpy element-wise for token-level
- Pattern normalized to array once per `search()`/`count()` call
- Batch SQLite inserts (`executemany`) for dataset index rebuild

### Performance Targets
- Construction: 1M tokens/second
- Query latency: <10ms for 100-token context
- Memory: ~4-5 bytes per corpus position (int32 corpus + int32 SA). Token-level is ~2.5× more memory-efficient per character than byte-level due to ~4× fewer positions.

## Test Structure

Test files with key coverage areas:

| File | Coverage Area |
|------|---------------|
| `test_infinigram.py` | Core model |
| `test_suffix_array.py` | SuffixArray / MmapSuffixArray / ChunkedMmap, byte + token paths |
| `test_tokenizers.py` | `Tokenizer` protocol, `ByteTokenizer`, config round-trip |
| `test_tiktoken_integration.py` | End-to-end with `tiktoken cl100k_base` |
| `test_utf8_and_documents.py` | UTF-8 round-tripping, document separators |
| `test_corpus_utils.py` | Corpus building utilities |
| `test_sampling.py` | Sampling primitives |

## Usage Examples

### Byte-level (default)
```python
from infinigram import Infinigram

model = Infinigram("the cat sat on the mat")  # ByteTokenizer default
probs = model.predict("the cat", top_k=5)       # {32: 1.0, ...}
text = model.generate_text("the", max_tokens=20, temperature=0.5)
```

### Token-level with tiktoken
```python
from infinigram import Infinigram
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
model = Infinigram("The quick brown fox " * 100, tokenizer=enc)

# ~4× fewer SA positions than byte-level for the same text.
assert model.vocab_size == enc.n_vocab
probs = model.predict("The quick brown")          # Dict[token_id, probability]
text = model.generate_text("The quick", max_tokens=10)
```

### Persistent index
```python
import tiktoken
from infinigram import Infinigram

# Build once (tokenizer is recorded in meta.json)
Infinigram.build("large_corpus.txt", "my_index", tokenizer=tiktoken.get_encoding("cl100k_base"))

# Load later. Tokenizer is reconstructed automatically from meta.json.
model = Infinigram.load("my_index")
```

