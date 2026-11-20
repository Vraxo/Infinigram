# Future Work: Infinigram Extensions

The core infinigram library focuses on exact byte-sequence matching via suffix arrays.
This document describes research directions for a separate project that could extend
Infinigram with approximate matching and corpus augmentation.

## Corpus Augmentations (infinigram-augmented)

### Case-Insensitive Matching
- **Approach**: Build augmented corpus with lowercase variants
- **Challenge**: Continuation counting - avoid double-counting "The cat" and "the cat"
- **Research needed**: Proper probability normalization across augmented entries

### Multi-Index Architecture
- Separate suffix arrays for original/normalized corpora
- Query-time merging with configurable weights
- Trade-off between memory (multiple indexes) and accuracy

### Query Transforms
- Runtime query normalization (lowercase, strip, normalize_whitespace)
- Removed from core for simplicity; could be reimplemented as wrapper
- Consider: Should transforms affect probability calculation?

### Typo Correction
- Edit-distance based fuzzy matching
- Would require BK-trees or fuzzy suffix arrays for efficiency
- Alternative: Pre-compute common typos and add to augmented corpus

### Semantic Similarity
- Embedding-based soft matching
- Find semantically similar contexts without exact string overlap
- Could use sentence transformers for context embedding

## Performance Optimizations

### C/C++ Implementation
- Core suffix array operations in C for 10-100x speedup
- Python bindings via pybind11 or ctypes
- Focus areas:
  - Suffix array construction (currently via pydivsufsort)
  - Binary search operations
  - Continuation counting loops

### Compressed Indexes
- FM-index for reduced memory footprint
- Grammar-compressed suffix arrays for repetitive corpora
- Trade-off: query speed vs memory usage

### Parallel Query Processing
- Batch multiple contexts for GPU-accelerated lookup
- SIMD vectorization for continuation counting
- Relevant for high-throughput inference

## Integration Features

### LLM Probability Mixing
A `TokenProbabilityAdapter` would let callers mix Infinigram probabilities
with neural LM token probabilities via byte-level chain rule:
```python
P_final = alpha * P_LLM + (1 - alpha) * P_corpus
```

Possible extensions if implemented:
- Learned mixing weights (alpha as function of context)
- Calibration for better probability estimates

### Streaming Corpus Updates
- Incremental suffix array updates without full rebuild
- Sliding window over recent documents
- Useful for real-time corpus tracking

## Evaluation

### OOD Generalization Benchmarks
- Systematic evaluation of case/typo/synonym robustness
- Comparison with neural n-gram models
- Analysis of when exact matching fails

### Computational Efficiency
- Query latency vs corpus size scaling
- Memory usage profiling
- Comparison with trie-based approaches
