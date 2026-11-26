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
  NGramModel.cs                     # Model API and lifecycle
  NGramModel.Persistence.cs         # Public model build and load methods
  ModelStorage.cs                   # Model persistence and metadata
  ContinuationCounter.cs            # Continuation counts and cache
  ContextKey.cs                     # Continuation cache key
  CorpusValidator.cs                # Token ID validation
  CorpusAppender.cs                 # Append tokens and rebuild the index
  SuffixMatcher.cs                  # Context limiting and suffix matching
  TokenPredictor.cs                 # Prediction probabilities
  TextGenerator.cs                  # Token and text generation
  SequenceScorer.cs                 # Sequence scoring
  SuffixArray.cs                    # Suffix-array queries and public API
  SuffixArraySearch.cs              # Suffix-array search and counting
  SuffixArrayBuilder.cs             # Suffix-array construction
  SuffixArrayStorage.cs             # Binary index persistence
  LruCache.cs                       # Internal LRU cache
  ITokenizer.cs                     # Tokenizer interface
  ByteTokenizer.cs                  # UTF-8 byte tokenizer
  BpeTokenizer.cs                   # Trainable byte-pair tokenizer with byte fallback
  Sampling.cs                       # Temperature sampling, top-k / top-p filters
  CorpusUtils.cs                    # Document/JSONL corpus builders
  TextNormalizer.cs                 # Optional text cleanup
  TextNormalizationOptions.cs       # Cleanup options
src/Infinigram.Repl/
  Infinigram.Repl.csproj            # Interactive console application
  ReplSession.cs                    # Interactive generation loop
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

## Interactive REPL

Place a nonempty UTF-8 text file named `corpus.txt` in the current working
directory, then run:

```bash
dotnet run --project src/Infinigram.Repl
```

The REPL loads the file once at startup, showing a spinner while reading and
indexing in an interactive terminal. Redirected output does not include the
spinner. Each line you enter is a prompt; it prints a greedy continuation of up
to 80 UTF-8 bytes. The prompt and status use
Spectre.Console styling when the terminal supports it; redirected output remains
plain text. End input with Ctrl+D (or Ctrl+Z then Enter on Windows) to exit.
Indexing large files can take time and memory.

Normalization is off by default. To enable it for both the file and every prompt,
change the `normalization` variable in `src/Infinigram.Repl/Interactive/Program.cs` to:

```csharp
TextNormalizationOptions? normalization = new() { Lowercase = true };
```

The default options convert text to Unicode NFC and line endings to LF; lowercase
is optional and culture-invariant. `CollapseWhitespace = true` additionally
replaces each run of Unicode whitespace (including newlines) with one space,
without trimming leading or trailing whitespace. Normalized REPL loading reads
text instead of raw bytes, recognizes a byte-order mark, and rejects invalid UTF-8
when UTF-8 is used. Generation returns normalized text.

For library users, normalization is explicit, not applied by the model or saved
in model metadata. Use the same options on corpus text and every query before
tokenizing, including after loading a persisted model:

```csharp
var options = new TextNormalizationOptions { Lowercase = true };
using var model = new NGramModel(TextNormalizer.Normalize(corpus, options));
var query = model.Tokenizer.Encode(TextNormalizer.Normalize(input, options));
var matches = model.Search(query);
```

Normalization can change byte offsets, whitespace, casing, and exact-match results.
Do not apply it to arbitrary binary corpora.

## Quick example

The public model class is `NGramModel` (previously `Infinigram`); existing C#
callers must update references. The namespace and saved model format are unchanged.

```csharp
using Infinigram;

using var model = new NGramModel("the cat sat on the mat the cat ran");
var context = model.Tokenizer.Encode("the cat");

var probs = model.Predict(context, topK: 5);        // token id -> probability
var (position, length) = model.LongestSuffix(context);
double lp = model.Score(model.Tokenizer.Encode("the cat sat"));
string text = model.GenerateText("the cat ", maxTokens: 20, temperature: 0.8);
```

Persistent models:

```csharp
NGramModel.Build("The quick brown fox jumps over the lazy dog.", "my_model");
using var model = NGramModel.Load("my_model");      // or a bare name under ~/.infinigram/models
```

## Tokenizers

The default `ByteTokenizer` maps UTF-8 input to byte IDs (0–255). For a
smaller token sequence on a domain-specific corpus, train a byte-pair tokenizer:

```csharp
string corpus = File.ReadAllText("corpus.txt");
var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 4096, minFrequency: 2);

using var model = NGramModel.Build(corpus, "my_subword_model", tokenizer);
```

The requested BPE vocabulary size includes the 256 byte fallback tokens, so
unseen words and arbitrary UTF-8 text remain encodable. Use
`model.Tokenizer.Encode(...)` for prompts and updates; token IDs from different
BPE vocabularies are not interchangeable. The merge rules are saved with the
model, so `NGramModel.Load("my_subword_model")` restores the matching tokenizer
without a separate tokenizer argument. Custom `ITokenizer` implementations
remain supported, but their configuration is the caller's responsibility.

## Corpus-derived word index

The default model remains byte-level and is still the source of truth. For text
corpora, callers can optionally build a second suffix index over words and
punctuation extracted from the same decoded corpus:

```csharp
using var model = new NGramModel(corpus);
var words = model.CreateWordIndex();

var nextWords = words.Continuations("the cat");
string continuation = words.GenerateText("the cat", maxTokens: 40);
```

This index is not a separately trained language model. Its vocabulary is made
from tokens that occur in the corpus, its sequence is the corpus tokenized at
word/punctuation boundaries, and its continuation counts come directly from suffix-array matches. Generation interpolates counts
from several suffix lengths with Witten–Bell weights computed from those corpus
counts; it stores no learned weights and can be rebuilt from the original model.
It is created only when requested, so ordinary byte-model construction, memory
use, and generation performance are unchanged.

Word-level generation is an experimental alternative, not a replacement for
byte generation. Whitespace is normalized when rendering generated text, and
the most frequent corpus spelling is used for case-insensitive token variants.
The index is intended for text, not arbitrary binary corpora. Unknown trailing
context tokens do not get silently skipped; generation stops when there is no
known matching suffix.

For sentence-shaped generation, use `GenerateStructuredText`:

```csharp
string sentence = words.GenerateStructuredText(
    "the cat",
    maxWords: 40,
    temperature: 0.8);
```

Sentence layouts are extracted from completed corpus sentences. A layout records
the number of words in each clause, the observed clause separators (commas,
semicolons, colons, or dashes), and the sentence-ending punctuation. Generation
samples a layout by its corpus frequency. Within each clause it looks for
two- or three-word continuations after the current context, using exact
suffix-array matches at one-, two-, and three-word context lengths. The phrase
candidate distributions are interpolated from broader to more specific contexts
with Witten–Bell weights computed from their observed corpus counts. When no
phrase chunk is available, it falls back to the word-level corpus counts and
interpolation. The output recombines short observed phrases under a
corpus-observed sentence layout rather than copying a full sentence. Layouts
are bounded to 128 words when indexed; `maxWords` limits the layout selected
for generation. No grammar parser, external rules, or learned weights are
introduced.

## Status vs. the Python reference

Ported and available: suffix-array search/count, longest suffix, continuations
(with LRU cache), predict / predict-logprobs, generation (greedy, temperature,
top-k, stop sequences), sequence scoring, model persistence
(`Build`/`BuildFromFile`/`Load`/`ListModels`), `Update` (append + rebuild),
LRU continuations cache, corpus-frequency unigram fallback.

Deliberate deviations (initial port):

- **Construction** uses prefix doubling + radix sort — O(n log n) — instead of
  `pydivsufsort` (SA-IS, O(n)). Swap in SA-IS later; only
  `SuffixArrayBuilder.Build` changes.
- **Persistence format** is simple little-endian `int32` (`corpus.bin`,
  `suffix_array.bin`, `meta.json` with `version: 1`, `engine:
  "infinigram-csharp"`). Python `.npy` indexes are **not** readable yet.
- **No chunked/mmap indexes** for >RAM corpora; models are held in memory.
- **Byte-level corpora** are stored as `int[]` (ids 0–255), not `uint8[]` —
  the `memcmp` fast path is a future optimization.
- `BpeTokenizer` provides deterministic byte-level BPE with all 256 UTF-8
  bytes retained as fallback tokens. Its learned merge rules are saved in model
  metadata and restored automatically by `NGramModel.Load`.
- Other custom tokenizers can still implement `ITokenizer`; they are persisted
  as `custom` and must be supplied again at `Load` time.

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

## Benchmarks

The BenchmarkDotNet suite measures the byte-tokenized .NET model using seeded,
synthetic ASCII corpora of exactly 32 KiB and 1 MiB. It is not a representation
of every real-world workload. Run it in Release mode on an otherwise idle machine:

```bash
dotnet run -c Release --project benchmarks/Infinigram.Benchmarks -- --list flat
dotnet run -c Release --project benchmarks/Infinigram.Benchmarks -- --filter '*SearchBenchmarks*'
dotnet run -c Release --project benchmarks/Infinigram.Benchmarks -- --filter '*'
```

For a quick setup/compilation check, append `--job Dry` to a filtered command;
its results are not reliable performance measurements. BenchmarkDotNet reports
runtime, OS, CPU, time, and managed allocations in `BenchmarkDotNet.Artifacts/`,
which is ignored by Git. Use the same machine, .NET SDK, settings and corpus
sizes when comparing runs. The suite uses a BenchmarkDotNet preview with .NET 11
support.

- **Build** indexes already-loaded bytes, including model construction and disposal,
  but excludes file reading.
- **Search** counts or returns up to 100 matches for frequent, rare, missing and
  128-byte patterns. Setup verifies the patterns' expected match categories.
- **Prediction** varies context length and `topK`. The cold case includes clearing
  the continuation cache; the warm case primes it in setup. Both include result
  creation. No smoothing is used.
- **Generation** generates exactly 32 or 128 byte tokens greedily; setup verifies
  the output length. Each operation includes clearing the cache. Divide output
  bytes by the reported mean time to estimate bytes per second.
- **Persistence** measures building and saving, loading, and updating and saving.
  Updates append 17 bytes to a freshly built on-disk model outside the timed
  operation. Build/save and update/save overwrite an existing index on later
  iterations, and load may benefit from the OS file cache. Temporary data is
  removed after each benchmark case. Record the disk and filesystem when comparing
  results.

The suite does not yet compare alternative tokenizers, model accuracy, peak
working set, or uncached physical disk reads. It is not run in CI by default.
