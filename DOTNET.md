# Infinigram for .NET (C# recreation)

This is a C#/.NET 11 recreation of the Python reference implementation in
[`Python/`](Python/). It is a **variable-length n-gram language model**: the
corpus *is* the model, and suffix arrays provide O(m log n) pattern matching —
no training loop, no neural weights.

## Project layout

```
Infinigram.slnx                     # Solution (new XML format)
global.json                         # Selects the Microsoft.Testing.Platform runner for `dotnet test`
src/Infinigram/
  Infinigram.csproj                 # net11.0 class library
  Infinigram.cs                     # Model: predict / continuations / generate / score / I/O
  SuffixArray.cs                    # Suffix array
  LruCache.cs                       # Internal LRU cache
  ITokenizer.cs                     # Tokenizer interface
  ByteTokenizer.cs                  # UTF-8 byte tokenizer
  Sampling.cs                       # Temperature sampling, top-k / top-p filters
  CorpusUtils.cs                    # Document/JSONL corpus builders
tests/Infinigram.Tests/
  Infinigram.Tests.csproj           # net11.0 test app (xUnit v3 on Microsoft.Testing.Platform)
  *Tests.cs                         # Port of Python/tests/ (assertions via AwesomeAssertions)
```

## Requirements & build

- .NET 11 SDK (`dotnet --version`). The `.slnx` solution file is supported by
  recent SDKs (and Visual Studio 2026); if your tooling predates it, upgrade
  rather than converting back to `.sln`.

```bash
dotnet build
dotnet test        # runs tests/Infinigram.Tests (xUnit v3 on Microsoft.Testing.Platform)
```

CI (GitHub Actions, `.github/workflows/ci.yml`) builds and tests on every push
to `main` and every PR targeting `main`.

The repo-root `global.json` selects the MTP runner for `dotnet test` (required
on .NET 10+ SDKs; the old VSTest bridge is gone). Without it, run the test app
directly: `dotnet run --project tests/Infinigram.Tests`.

## Quick example

```csharp
using Infinigram;

using var model = new Infinigram("the cat sat on the mat the cat ran");
var context = model.Tokenizer.Encode("the cat");

var probs = model.Predict(context, topK: 5);        // token id -> probability
var (position, length) = model.LongestSuffix(context);
double lp = model.Score(model.Tokenizer.Encode("the cat sat"));
string text = model.GenerateText("the cat ", maxTokens: 20, temperature: 0.8);
```

Persistent models:

```csharp
Infinigram.Build("The quick brown fox jumps over the lazy dog.", "my_model");
using var model = Infinigram.Load("my_model");      // or a bare name under ~/.infinigram/models
```

## Status vs. the Python reference

Ported and available: suffix-array search/count, longest suffix, continuations
(with LRU cache), predict / predict-logprobs, generation (greedy, temperature,
top-k, stop sequences), sequence scoring, model persistence
(`Build`/`BuildFromFile`/`Load`/`ListModels`), `Update` (append + rebuild),
LRU continuations cache, uniform-prior fallbacks.

Deliberate deviations (initial port):

- **Construction** uses prefix doubling + radix sort — O(n log n) — instead of
  `pydivsufsort` (SA-IS, O(n)). Swap in SA-IS later; only
  `SuffixArray.BuildSuffixArray` changes.
- **Persistence format** is simple little-endian `int32` (`corpus.bin`,
  `suffix_array.bin`, `meta.json` with `version: 1`, `engine:
  "infinigram-csharp"`). Python `.npy` indexes are **not** readable yet.
- **No chunked/mmap indexes** for >RAM corpora; models are held in memory.
- **Byte-level corpora** are stored as `int[]` (ids 0–255), not `uint8[]` —
  the `memcmp` fast path is a future optimization.
- Tokenizers beyond `ByteTokenizer`: implement `ITokenizer`; persisted as
  `custom` and must be supplied again at `Load` time (same spirit as Python).

Not yet ported: chunked indexes, token-level dtype paths, tiktoken integration
tests (port behind an optional dependency), the REST/CLI/REPL ideas in
`Python/ARCHITECTURE.md`.

## Verification so far

No .NET SDK was available in the development sandbox (network-restricted), so a
real `dotnet build` / `dotnet test` still needs to happen on a machine with the
.NET 11 SDK. What *has* been verified:

- All sources parse cleanly (tree-sitter-c-sharp).
- The construction and range-search algorithms were re-implemented in Python
  and fuzz-tested against brute force (3,000 random corpora/patterns).
- The longest-match binary search and the continuation walk-down were
  fuzz-tested against brute force (500/300 random corpora).
- The test suite (`tests/Infinigram.Tests/`) was written against the C# sources
  and the AwesomeAssertions 9.6.0 public API (method names verified against the
  release source), but not yet compiled or executed — expect minor nits on the
  first `dotnet test`.

The test suite ports `Python/tests/` (sampling, suffix array, tokenizers,
corpus utils, UTF-8/documents, and the model suites: longest suffix, suffix
matches, continuations, predict, predict-logprobs, update, edge cases,
integration, GetToken, Generate, Score, cache limits/eviction). Assertions use
AwesomeAssertions; xUnit uses the v3 Microsoft.Testing.Platform runner. Tests
that exercise Python-only behaviors (mmap/chunked builds, `Infinigram.build`/
`load` of `.npy` indexes, tiktoken) are intentionally omitted.

## Next steps

1. `dotnet build` and `dotnet test` on a machine with the .NET 11 SDK; fix any
   compile or behavioral nits in the test port.
2. mmap-backed indexes + chunked construction for multi-GB corpora.
3. SA-IS construction; byte-level `uint8` fast path.
4. Converter for Python-built `.npy` indexes (cross-language model sharing).
