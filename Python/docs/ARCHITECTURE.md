# Infinigram Architecture & Vision

**Version**: 0.5.0
**Date**: March 2026
**Status**: Production

## Vision

Infinigram is a **high-speed, corpus-based language model** that leverages suffix arrays for variable-length n-gram matching. Unlike traditional neural LMs, Infinigram provides:

- **Instant training**: Models are corpora (no gradient descent)
- **Exact matching**: Finds actual patterns from training data
- **Explainability**: Every prediction traces back to corpus evidence
- **Speed**: Orders of magnitude faster than neural inference
- **LLM grounding**: Weight mixture with neural LM next-token probabilities

## Core Use Cases

### 1. LLM Fine-tuning via Probability Mixing

A future adapter could blend Infinigram's corpus-grounded distribution with
a neural LM's next-token distribution: `P_final = alpha * P_LLM + (1-alpha) * P_corpus`.

**Benefits**:
- Ground LLM outputs in specific corpora (technical docs, legal text, etc.)
- Boost domain-specific vocabulary without expensive fine-tuning
- Reduce hallucinations by anchoring to real text
- Real-time adaptation without retraining

### 2. Hierarchical Matching

Multiple prediction strategies for different use cases:

```python
# Longest suffix match (default)
probs = model.predict(context)
```

## Architectural Layers

```
+-------------------------------------+
|      Python API (Core)               |  Infinigram class
+-------------------------------------+
|    Suffix Array Engine               |  MmapSuffixArray, ChunkedMmapSuffixArray
+-------------------------------------+
```

### Python API

```python
from infinigram import Infinigram

# In-memory (creates temporary mmap model)
model = Infinigram(b"The cat sat on the mat")
probs = model.predict(b"The cat", top_k=10)

# Persistent model
model = Infinigram.build("corpus text", "my_model")
model = Infinigram.load("my_model")
```

## Performance Optimizations

### Vectorized Continuation Counting
`continuations()` uses `np.bincount` on a numpy view of the mmap'd corpus for 3-10x speedup over Python loops.

### LRU Cache
Repeated `continuations()` queries are cached in an `OrderedDict` with `move_to_end()` / `popitem(last=False)` for O(1) LRU operations. Cache is automatically invalidated on `update()`.

### Bytes Comparison in Binary Search
Pattern is converted to `bytes` once in `search()`, then sliced during O(log n) binary search comparisons. Python's bytes comparison uses C-level `memcmp` internally.

### Batch SQLite Inserts
Dataset index rebuilds use `executemany()` with batch size 1000, avoiding per-row INSERT overhead.

## Design Principles

### 1. Speed First
- Pre-computed suffix arrays (no online construction)
- Memory-mapped corpora for large datasets
- Vectorized numpy operations in hot paths
- LRU caching for repeated queries

### 2. Simplicity & Composability
- Unix philosophy: do one thing well (pattern matching + prediction)
- Easy to compose with other models (probability mixing via adapters)
- Clean separation: core logic, server, CLI
- No transforms or projections in core: keep API surface minimal for future C/Go port

### 3. Explainability
Every prediction is traceable:
- `search()` returns corpus positions of matches
- `get_context()` shows actual text around a position
- `find_all_suffix_matches()` shows all matches at all suffix lengths

## Implementation Roadmap

### Phase 1: Core API: Complete
- [x] Unified `Infinigram` class with mmap-backed storage
- [x] `predict()`
- [x] `find_all_suffix_matches()` for introspection

### Phase 2: Performance: Complete
- [x] Vectorized continuation counting (`np.bincount`)
- [x] LRU cache for `continuations()`
- [x] Bytes comparison in binary search
- [x] Batch SQLite inserts
- [x] Chunked suffix arrays for >5GB corpora
- [ ] Compressed suffix arrays
- [ ] Parallel construction

### Phase 5: Ecosystem & Integration
- [ ] Pre-built corpus packages
- [ ] Tokenizer compatibility layer for popular models
- [ ] Evaluation benchmarks
- [ ] C/Go port of core engine

## Performance Targets

- **Construction**: 1M tokens/second
- **Query latency**: <10ms for 100-token context
- **Throughput**: 1000+ queries/second on single CPU
- **Memory**: <10 bytes per corpus token
- **Scaling**: 1B+ token corpora via chunked suffix arrays
