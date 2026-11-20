// Infinigram: Variable-length n-gram language model using suffix arrays.
//
// The corpus IS the model — no gradient descent, no training loop. Index a
// corpus once, then ask it for continuations of any context. Unlike traditional
// n-gram models with a fixed order, Infinigram finds the longest matching
// suffix in the corpus and uses its continuations for prediction.
//
// Status relative to the Python reference (Python/):
//   - Ported: search, count, longest suffix, continuations, predict,
//     predict-logprobs, generate, generate_text, score, update, caches.
//   - Persistent models use a simple little-endian int32 format (corpus.bin,
//     suffix_array.bin, meta.json). Python's .npy indexes are NOT readable yet.
//   - Chunked (multi-GB) indexes are not supported yet.
//   - Models are kept in memory rather than mmap-backed; for corpora that fit
//     in RAM this is equivalent in behavior and usually faster.
//
// Note: instances are NOT thread-safe. Use one model per thread or external
// synchronization.

using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Infinigram;

/// <summary>
/// Variable-length n-gram language model over a suffix array.
/// </summary>
/// <example>
/// <code>
/// using var model = new Infinigram("the cat sat on the mat the cat ran");
/// var probs = model.Predict(model.Tokenizer.Encode("the cat"));
/// </code>
/// </example>
public sealed class Infinigram : IDisposable
{
    /// <summary>Default maximum occurrences sampled when counting continuations.</summary>
    public const int DefaultContinuationLimit = 10000;

    private const int DefaultCacheSize = 1000;

    private static string DefaultModelsDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".infinigram", "models");

    private static readonly JsonSerializerOptions JsonWriteOptions = new() { WriteIndented = true };

    // Assigned by every constructing ctor before the instance escapes (CS8618: nulled until then).
    private SuffixArray _sa = null!;
    private readonly LruCache<ContextKey, Dictionary<int, int>?> _cache = new(DefaultCacheSize);
    private Dictionary<int, int>? _uniformCounts;
    private string? _modelDir;
    private bool _disposed;

    /// <summary>The tokenizer associated with this model.</summary>
    public ITokenizer Tokenizer { get; private set; }

    /// <summary>Size of the token vocabulary the model was built with.</summary>
    public int VocabSize { get; private set; }

    /// <summary>Maximum suffix length to consider for predictions (null = unlimited).</summary>
    public int? MaxLength { get; }

    /// <summary>Number of tokens in the corpus.</summary>
    public int N => _sa.Length;

    /// <summary>Directory of the persistent model, or null for in-memory models.</summary>
    public string? ModelPath => _modelDir;

    // === Construction ===

    private Infinigram(ITokenizer tokenizer, int? maxLength)
    {
        Tokenizer = tokenizer;
        VocabSize = tokenizer.VocabSize;
        MaxLength = maxLength;
    }

    /// <summary>
    /// Create a model from a pre-tokenized corpus of token ids.
    /// </summary>
    /// <exception cref="ArgumentException">The corpus is empty or contains ids outside <c>[0, VocabSize)</c>.</exception>
    public Infinigram(IReadOnlyList<int> corpus, ITokenizer? tokenizer = null, int? maxLength = null)
        : this(tokenizer ?? new ByteTokenizer(), maxLength)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (corpus.Count == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(corpus));
        }
        _sa = new SuffixArray(ValidateCorpus(corpus, "Corpus"));
    }

    /// <summary>
    /// Create a model from a raw byte corpus. Only valid for byte-level tokenizers
    /// (vocab ≤ 256); for token-level models pass token ids instead.
    /// </summary>
    public Infinigram(byte[] corpus, ITokenizer? tokenizer = null, int? maxLength = null)
        : this(tokenizer ?? new ByteTokenizer(), maxLength)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (VocabSize > 256)
        {
            throw new ArgumentException(
                $"bytes corpora are only valid for byte-level tokenizers; " +
                $"tokenizer has VocabSize={VocabSize}. Pass IReadOnlyList<int> instead.",
                nameof(corpus));
        }
        if (corpus.Length == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(corpus));
        }
        _sa = new SuffixArray(ValidateCorpus(corpus.Select(b => (int)b).ToList(), "Corpus"));
    }

    /// <summary>
    /// Create a model from a text corpus. The text is encoded with the model's tokenizer
    /// (UTF-8 bytes by default).
    /// </summary>
    public Infinigram(string corpus, ITokenizer? tokenizer = null, int? maxLength = null)
        : this(tokenizer ?? new ByteTokenizer(), maxLength)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (corpus.Length == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(corpus));
        }
        _sa = new SuffixArray(Tokenizer.Encode(corpus));
    }

    // === Persistent models ===

    /// <summary>Build a persistent model from a text corpus.</summary>
    public static Infinigram Build(string corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        var tok = tokenizer ?? new ByteTokenizer();
        return BuildFromTokens(tok.Encode(corpus).ToList(), modelPath, tok, maxLength, verbose);
    }

    /// <summary>Build a persistent model from a byte corpus (byte-level tokenizers only).</summary>
    public static Infinigram Build(byte[] corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        var tok = tokenizer ?? new ByteTokenizer();
        if (tok.VocabSize > 256)
        {
            throw new ArgumentException(
                $"bytes corpora are only supported with byte-level tokenizers; " +
                $"tokenizer has VocabSize={tok.VocabSize}. Pass IReadOnlyList<int> instead.",
                nameof(corpus));
        }
        return BuildFromTokens(corpus.Select(b => (int)b).ToList(), modelPath, tok, maxLength, verbose);
    }

    /// <summary>Build a persistent model from pre-tokenized ids.</summary>
    public static Infinigram Build(IReadOnlyList<int> corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        return BuildFromTokens(corpus.ToList(), modelPath, tokenizer ?? new ByteTokenizer(), maxLength, verbose);
    }

    /// <summary>
    /// Build a persistent model from a corpus file. For byte-level tokenizers the file is
    /// read as raw bytes; otherwise it is read as text and encoded with the tokenizer.
    /// </summary>
    public static Infinigram BuildFromFile(string corpusFilePath, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpusFilePath);
        ArgumentNullException.ThrowIfNull(modelPath);
        if (!File.Exists(corpusFilePath))
        {
            throw new FileNotFoundException($"Corpus file not found: {corpusFilePath}", corpusFilePath);
        }

        var tok = tokenizer ?? new ByteTokenizer();
        List<int> tokens;
        if (tok.VocabSize <= 256)
        {
            if (verbose)
            {
                Console.WriteLine($"Loading corpus from {corpusFilePath}...");
            }
            tokens = File.ReadAllBytes(corpusFilePath).Select(b => (int)b).ToList();
        }
        else
        {
            tokens = tok.Encode(File.ReadAllText(corpusFilePath)).ToList();
        }
        return BuildFromTokens(tokens, modelPath, tok, maxLength, verbose);
    }

    private static Infinigram BuildFromTokens(List<int> tokens, string modelPath, ITokenizer tokenizer,
        int? maxLength, bool verbose)
    {
        if (tokens.Count == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(tokens));
        }

        var stopwatch = Stopwatch.StartNew();
        string fullPath = Path.GetFullPath(modelPath);
        var model = new Infinigram(tokenizer, maxLength);

        var validated = model.ValidateCorpus(tokens, "Corpus");
        int n = validated.Length;
        if (verbose)
        {
            Console.WriteLine($"Building model at {fullPath}");
            Console.WriteLine($"Corpus: {n:N0} tokens (vocab={tokenizer.VocabSize})");
            Console.WriteLine("Building suffix array...");
        }

        var sa = new SuffixArray(validated);
        sa.Save(fullPath);

        model._sa = sa;
        model._modelDir = fullPath;
        model.WriteMeta(fullPath, n);

        if (verbose)
        {
            double elapsed = stopwatch.Elapsed.TotalSeconds;
            double rate = elapsed > 0 ? n / elapsed / 1e6 : double.PositiveInfinity;
            Console.WriteLine($"Built in {elapsed:F1}s ({rate:F2} MT/sec)");
        }
        return model;
    }

    /// <summary>
    /// Load an existing model. Non-rooted paths that do not exist are resolved against
    /// <c>~/.infinigram/models/</c>.
    /// </summary>
    /// <exception cref="FileNotFoundException">No model at the resolved path.</exception>
    /// <exception cref="NotSupportedException">The index is not a C#-built Infinigram model (e.g. a Python .npy index).</exception>
    public static Infinigram Load(string modelPath, ITokenizer? tokenizer = null, int? maxLength = null)
    {
        ArgumentNullException.ThrowIfNull(modelPath);

        string dir = modelPath;
        if (!Path.IsPathRooted(modelPath) && !Directory.Exists(modelPath))
        {
            string candidate = Path.Combine(DefaultModelsDir, modelPath);
            if (Directory.Exists(candidate))
            {
                dir = candidate;
            }
        }

        if (!Directory.Exists(dir))
        {
            throw new FileNotFoundException(
                $"Model not found at {dir}. Use Infinigram.Build(...) to create a new model.");
        }

        string metaPath = Path.Combine(dir, "meta.json");
        if (!File.Exists(metaPath))
        {
            throw new FileNotFoundException(
                $"Model not found at {dir} (missing meta.json). Use Infinigram.Build(...) to create a new model.");
        }

        JsonObject meta;
        try
        {
            meta = JsonNode.Parse(File.ReadAllText(metaPath)) as JsonObject
                   ?? throw new JsonException("meta.json is not a JSON object");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Corrupt meta.json at {metaPath}: {ex.Message}", ex);
        }

        int version = meta.TryGetPropertyValue("version", out var versionNode) && versionNode is not null
            ? versionNode.GetValue<int>()
            : 0;
        if (version != 1)
        {
            throw new NotSupportedException(
                $"Unsupported model format version {version} at {metaPath}. This engine reads C#-built " +
                "indexes (version 1). If this is a Python-built model (.npy files, version 2), it is " +
                "not supported yet — rebuild it with Infinigram.Build.");
        }

        ITokenizer resolved = ResolveTokenizer(meta, metaPath, tokenizer);

        if (meta.TryGetPropertyValue("vocab_size", out var vocabNode) && vocabNode is not null)
        {
            int metaVocab = vocabNode.GetValue<int>();
            if (metaVocab != resolved.VocabSize)
            {
                throw new InvalidDataException(
                    $"Tokenizer mismatch: model at {dir} was built with vocab_size={metaVocab}, " +
                    $"but the resolved tokenizer has VocabSize={resolved.VocabSize}.");
            }
        }

        int? metaMaxLength = null;
        if (meta.TryGetPropertyValue("max_length", out var maxLengthNode) && maxLengthNode is not null)
        {
            metaMaxLength = maxLengthNode.GetValue<int>();
        }

        var model = new Infinigram(resolved, maxLength ?? metaMaxLength);
        string fullDir = Path.GetFullPath(dir);
        model._sa = SuffixArray.Load(fullDir);
        model._modelDir = fullDir;

        if (meta.TryGetPropertyValue("n", out var nNode) && nNode is not null)
        {
            int metaN = nNode.GetValue<int>();
            if (metaN != model.N)
            {
                throw new InvalidDataException(
                    $"Corrupt model at {dir}: meta.json says n={metaN} but the corpus has {model.N} tokens.");
            }
        }
        return model;
    }

    /// <summary>List model names in the models directory (default: ~/.infinigram/models).</summary>
    public static List<string> ListModels(string? modelsDir = null)
    {
        string dir = modelsDir ?? DefaultModelsDir;
        if (!Directory.Exists(dir))
        {
            return new List<string>();
        }
        return Directory
            .EnumerateDirectories(dir)
            .Where(d => File.Exists(Path.Combine(d, "meta.json")))
            .Select(Path.GetFileName)
            .Where(name => name is not null)
            .Select(name => name!)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToList();
    }

    private static ITokenizer ResolveTokenizer(JsonObject meta, string metaPath, ITokenizer? supplied)
    {
        string? kind = null;
        string? customName = null;
        if (meta.TryGetPropertyValue("tokenizer", out var tokenizerNode) && tokenizerNode is JsonObject tokConfig)
        {
            kind = tokConfig.TryGetPropertyValue("kind", out var kindNode) ? kindNode?.GetValue<string>() : null;
            if (tokConfig.TryGetPropertyValue("name", out var nameNode))
            {
                customName = nameNode?.GetValue<string>();
            }
        }

        switch (kind)
        {
            case "byte":
                if (supplied is not null && supplied is not ByteTokenizer)
                {
                    throw new ArgumentException(
                        "Tokenizer mismatch: model was built with a byte tokenizer but a different " +
                        "tokenizer was supplied. Pass the matching tokenizer or omit tokenizer:.");
                }
                return new ByteTokenizer();

            case "custom":
                if (supplied is null)
                {
                    throw new InvalidOperationException(
                        $"Model at {metaPath} was built with a custom tokenizer ({customName}). " +
                        "Pass tokenizer: when loading.");
                }
                return supplied;

            case null:
                // No tokenizer metadata: assume byte-level (forward compatibility).
                return supplied ?? new ByteTokenizer();

            default:
                if (supplied is null)
                {
                    throw new NotSupportedException(
                        $"Unknown tokenizer kind '{kind}' in {metaPath}. Pass tokenizer: explicitly.");
                }
                return supplied;
        }
    }

    private void WriteMeta(string dir, int n)
    {
        var tokenizerConfig = Tokenizer is ByteTokenizer
            ? new Dictionary<string, object?> { ["kind"] = "byte", ["encoding"] = "utf-8", ["errors"] = "replace" }
            : new Dictionary<string, object?> { ["kind"] = "custom", ["name"] = Tokenizer.GetType().FullName };

        var meta = new Dictionary<string, object?>
        {
            ["version"] = 1,
            ["engine"] = "infinigram-csharp",
            ["n"] = n,
            ["vocab_size"] = VocabSize,
            ["max_length"] = MaxLength,
            ["tokenizer"] = tokenizerConfig,
        };
        File.WriteAllText(
            Path.Combine(dir, "meta.json"),
            JsonSerializer.Serialize(meta, JsonWriteOptions));
    }

    private int[] ValidateCorpus(IReadOnlyList<int> tokens, string what)
    {
        if (tokens.Count == 0)
        {
            return Array.Empty<int>();
        }

        var arr = tokens as int[] ?? tokens.ToArray();
        int min = int.MaxValue, max = int.MinValue;
        foreach (int t in arr)
        {
            if (t < min) min = t;
            if (t > max) max = t;
        }

        if (min < 0 || max >= VocabSize)
        {
            string message = VocabSize == 256
                ? $"{what} must contain only bytes (0-255), but found {(max >= 256 ? max : min)}"
                : $"{what} contains token ids outside [0, {VocabSize}); min={min}, max={max}";
            throw new ArgumentOutOfRangeException(what, message);
        }
        return arr;
    }

    private List<int> ApplyMaxLength(IReadOnlyList<int> context)
    {
        // A max_length of 0 (or null) means unlimited, mirroring the Python truthiness check.
        if (MaxLength is int max && max > 0 && context.Count > max)
        {
            return context.Skip(context.Count - max).ToList();
        }
        return context.ToList();
    }

    private List<int> SearchSuffix(List<int> context, int suffixLength, int? limit)
    {
        var pattern = context.GetRange(context.Count - suffixLength, suffixLength);
        return _sa.Search(pattern, limit);
    }

    // Largest k such that context[^k:] occurs in the corpus, plus one example position.
    // Monotonicity (longer suffix in corpus ⇒ every shorter suffix in corpus) makes the
    // answer binary-searchable. Total work is O(m log n).
    private (int Length, int Position) LongestMatchLength(List<int> context)
    {
        if (context.Count == 0)
        {
            return (0, -1);
        }

        var first = SearchSuffix(context, 1, limit: 1);
        if (first.Count == 0)
        {
            return (0, -1);
        }

        int bestLength = 1;
        int bestPosition = first[0];
        int lo = 1, hi = context.Count;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            var results = SearchSuffix(context, mid, limit: 1);
            if (results.Count > 0)
            {
                lo = mid;
                bestLength = mid;
                bestPosition = results[0];
            }
            else
            {
                hi = mid - 1;
            }
        }
        return (bestLength, bestPosition);
    }

    // === Query methods ===

    /// <summary>
    /// All positions where <paramref name="pattern"/> occurs in the corpus.
    /// </summary>
    /// <param name="limit">Maximum number of results (null = all). Limiting significantly improves performance on large corpora.</param>
    public List<int> Search(IReadOnlyList<int> pattern, int? limit = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pattern);
        return _sa.Search(pattern, limit);
    }

    /// <summary>Number of occurrences of <paramref name="pattern"/> in the corpus.</summary>
    public int Count(IReadOnlyList<int> pattern)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pattern);
        return _sa.Count(pattern);
    }

    /// <summary>Return the token id at <paramref name="position"/>, with bounds checking.</summary>
    public int GetToken(int position)
    {
        ThrowIfDisposed();
        return _sa.GetToken(position);
    }

    /// <summary>
    /// Find the longest suffix of <paramref name="context"/> occurring in the corpus.
    /// Returns <c>(-1, 0)</c> when no suffix matches.
    /// </summary>
    public (int Position, int Length) LongestSuffix(IReadOnlyList<int> context)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return (-1, 0);
        }

        var ctx = ApplyMaxLength(context);
        var (length, position) = LongestMatchLength(ctx);
        return length == 0 ? (-1, 0) : (position, length);
    }

    /// <summary>
    /// Find all matching suffixes at different lengths, sorted by decreasing length.
    /// For context "abc", searches "abc", "bc", "c" and returns all non-empty result sets.
    /// </summary>
    public List<(int Length, List<int> Positions)> FindAllSuffixMatches(IReadOnlyList<int> context)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return new List<(int, List<int>)>();
        }

        var ctx = ApplyMaxLength(context);
        var matches = new List<(int Length, List<int> Positions)>();
        for (int length = ctx.Count; length > 0; length--)
        {
            var positions = SearchSuffix(ctx, length, limit: null);
            if (positions.Count > 0)
            {
                matches.Add((length, positions));
            }
        }
        return matches;
    }

    /// <summary>
    /// Get continuation counts for the longest matching suffix of <paramref name="context"/>.
    /// </summary>
    /// <param name="limit">
    /// Maximum occurrences to sample for counting continuations (default: 10000).
    /// Prevents hanging on millions of matches for common patterns. Null counts all occurrences.
    /// </param>
    /// <returns>
    /// Dictionary mapping next-token-id → count, or null when no suffix of the context
    /// appears in the corpus. An empty context returns the uniform-counts dict.
    /// </returns>
    public Dictionary<int, int>? Continuations(IReadOnlyList<int> context, int? limit = DefaultContinuationLimit)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);

        if (context.Count == 0)
        {
            return UniformCounts();
        }

        var ctx = ApplyMaxLength(context);
        var cacheKey = new ContextKey(ctx, limit);
        if (_cache.TryGet(cacheKey, out var cached))
        {
            return cached is null ? null : new Dictionary<int, int>(cached);
        }

        var (matchLength, _) = LongestMatchLength(ctx);
        if (matchLength == 0)
        {
            _cache.Set(cacheKey, null);
            return null;
        }

        // Walk down lengths to handle the corner case where every match at the chosen
        // length sits at the very end of the corpus (no continuation).
        for (int length = matchLength; length > 0; length--)
        {
            var positions = SearchSuffix(ctx, length, limit);
            Dictionary<int, int>? counts = null;
            foreach (int pos in positions)
            {
                int next = pos + length;
                if (next >= N)
                {
                    continue;
                }
                int token = _sa.GetToken(next);
                counts ??= new Dictionary<int, int>();
                counts[token] = counts.TryGetValue(token, out var c) ? c + 1 : 1;
            }

            if (counts is { Count: > 0 })
            {
                _cache.Set(cacheKey, new Dictionary<int, int>(counts));
                return counts;
            }
        }

        _cache.Set(cacheKey, null);
        return null;
    }

    private Dictionary<int, int> UniformCounts()
    {
        if (_uniformCounts is null)
        {
            _uniformCounts = new Dictionary<int, int>(VocabSize);
            for (int t = 0; t < VocabSize; t++)
            {
                _uniformCounts[t] = 1;
            }
        }
        return new Dictionary<int, int>(_uniformCounts);
    }

    /// <summary>Clear prediction caches (call after corpus updates).</summary>
    public void ClearCache()
    {
        _cache.Clear();
    }

    // Test hooks: cache internals visible to the test assembly only.
    internal int ContinuationsCacheCount => _cache.Count;

    internal int ContinuationsCacheCapacity
    {
        get => _cache.Capacity;
        set => _cache.SetCapacity(value);
    }

    /// <summary>
    /// Predict next-token probabilities.
    /// </summary>
    /// <returns>
    /// Dictionary mapping token id → probability, or null when no suffix of the context
    /// appears in the corpus. Callers (e.g. mixture-with-LLM grounding) should branch on
    /// null to fall back to a baseline distribution.
    /// </returns>
    public IReadOnlyDictionary<int, double>? Predict(IReadOnlyList<int> context, int topK = 50, double smoothing = 0.0)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (topK < 1)
        {
            topK = 1;
        }

        var counts = Continuations(context);
        if (counts is null)
        {
            return null;
        }

        double total = 0;
        foreach (var c in counts.Values)
        {
            total += c;
        }
        double smoothedTotal = total + smoothing * VocabSize;

        var probs = new Dictionary<int, double>(counts.Count);
        foreach (var (token, count) in counts)
        {
            probs[token] = (count + smoothing) / smoothedTotal;
        }

        if (smoothing > 0.0)
        {
            double unseen = smoothing / smoothedTotal;
            for (int t = 0; t < VocabSize; t++)
            {
                probs.TryAdd(t, unseen);
            }
        }

        if (probs.Count > topK)
        {
            probs = probs
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .ToDictionary(kv => kv.Key, kv => kv.Value);
        }
        return probs;
    }

    /// <summary>Predict next-token log-probabilities. Returns null when no suffix matches.</summary>
    public IReadOnlyDictionary<int, double>? PredictLogprobs(IReadOnlyList<int> context, int topK = 50,
        double smoothing = 1e-10)
    {
        var probs = Predict(context, topK, smoothing);
        if (probs is null)
        {
            return null;
        }
        var logprobs = new Dictionary<int, double>(probs.Count);
        foreach (var (token, p) in probs)
        {
            logprobs[token] = Math.Log(p);
        }
        return logprobs;
    }

    /// <summary>
    /// Generate tokens autoregressively.
    /// </summary>
    /// <param name="context">Starting token ids; null or empty for unconditioned generation.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <param name="temperature">Sampling temperature (0 = greedy, 1 = standard).</param>
    /// <param name="topK">Number of top predictions to consider per step; null uses the full vocabulary.</param>
    /// <param name="smoothing">Laplace smoothing parameter.</param>
    /// <param name="stop">Optional stop sequences; generation halts and the stop tokens are stripped.</param>
    /// <param name="rng">Optional randomness source (defaults to <see cref="Random.Shared"/>).</param>
    /// <returns>The generated token ids (not including the original context).</returns>
    public List<int> Generate(
        IReadOnlyList<int>? context = null,
        int maxTokens = 100,
        double temperature = 1.0,
        int? topK = null,
        double smoothing = 0.0,
        IEnumerable<IReadOnlyList<int>>? stop = null,
        Random? rng = null)
    {
        ThrowIfDisposed();
        int effectiveTopK = topK is null ? VocabSize : Math.Clamp(topK.Value, 1, VocabSize);

        var currentContext = context?.ToList() ?? new List<int>();
        var generated = new List<int>();

        var stopLists = stop?
            .Where(s => s is { Count: > 0 })
            .Select(s => s.ToList())
            .ToList();

        for (int step = 0; step < maxTokens; step++)
        {
            var probs = Predict(currentContext, topK: effectiveTopK, smoothing: smoothing);
            if (probs is null || probs.Count == 0)
            {
                break;
            }

            int nextToken = Sampling.SampleFromDistribution(probs, temperature, rng);
            generated.Add(nextToken);
            currentContext.Add(nextToken);

            if (stopLists is not null)
            {
                foreach (var stopSeq in stopLists)
                {
                    int stopLength = stopSeq.Count;
                    if (generated.Count >= stopLength &&
                        generated[^stopLength..].SequenceEqual(stopSeq))
                    {
                        generated.RemoveRange(generated.Count - stopLength, stopLength);
                        return generated;
                    }
                }
            }
        }

        return generated;
    }

    /// <summary>
    /// Generate and decode to text via the model's tokenizer.
    /// </summary>
    public string GenerateText(
        string context = "",
        int maxTokens = 100,
        double temperature = 1.0,
        int? topK = 50,
        double smoothing = 0.0,
        IEnumerable<string>? stop = null,
        Random? rng = null)
    {
        ThrowIfDisposed();
        var seedTokens = string.IsNullOrEmpty(context)
            ? new List<int>()
            : Tokenizer.Encode(context).ToList();
        var stopTokens = stop?
            .Where(s => s.Length > 0)
            .Select(s => Tokenizer.Encode(s).ToList())
            .ToList();

        var tokens = Generate(
            context: seedTokens,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            smoothing: smoothing,
            stop: stopTokens,
            rng: rng);
        return Tokenizer.Decode(tokens);
    }

    /// <summary>
    /// Score a sequence by its total log-probability under the model.
    /// Positions with no matching suffix are treated as uniform.
    /// Returns a more-negative value for less likely sequences.
    /// </summary>
    public double Score(IReadOnlyList<int> sequence, double smoothing = 1e-10)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(sequence);
        if (sequence.Count == 0)
        {
            return 0.0;
        }

        double uniformLp = Math.Log(1.0 / VocabSize);
        double totalLp = uniformLp;

        for (int i = 1; i < sequence.Count; i++)
        {
            // Note: no max_length truncation here beyond what Continuations applies.
            var context = sequence.Take(i).ToList();
            var counts = Continuations(context);
            if (counts is null)
            {
                totalLp += uniformLp;
                continue;
            }

            double total = 0;
            foreach (var c in counts.Values)
            {
                total += c;
            }
            counts.TryGetValue(sequence[i], out int count);
            double prob = (count + smoothing) / (total + smoothing * VocabSize);
            if (prob <= 0)
            {
                return double.NegativeInfinity;
            }
            totalLp += Math.Log(prob);
        }

        return totalLp;
    }

    /// <summary>
    /// Append new tokens to the corpus and rebuild the index. O(n log n) — for
    /// frequent updates, consider batching.
    /// </summary>
    public void Update(IReadOnlyList<int> newTokens)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(newTokens);
        UpdateInternal(ValidateCorpus(newTokens, "new_tokens"));
    }

    /// <summary>Append new bytes to the corpus (byte-level models only) and rebuild the index.</summary>
    public void Update(byte[] newTokens)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(newTokens);
        if (VocabSize > 256)
        {
            throw new ArgumentException(
                $"bytes input is only valid for byte-level tokenizers; tokenizer has VocabSize={VocabSize}.",
                nameof(newTokens));
        }
        UpdateInternal(newTokens.Select(b => (int)b).ToArray());
    }

    /// <summary>Append text (encoded with the model's tokenizer) and rebuild the index.</summary>
    public void Update(string newTokens)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(newTokens);
        UpdateInternal(Tokenizer.Encode(newTokens).ToArray());
    }

    private void UpdateInternal(int[] newTokens)
    {
        var existing = _sa.Corpus as int[] ?? _sa.Corpus.ToArray();
        var updated = new int[existing.Length + newTokens.Length];
        Array.Copy(existing, updated, existing.Length);
        Array.Copy(newTokens, 0, updated, existing.Length, newTokens.Length);

        _sa = new SuffixArray(updated);

        if (_modelDir is not null)
        {
            _sa.Save(_modelDir);
            WriteMeta(_modelDir, updated.Length);
        }
        ClearCache();
    }

    /// <summary>
    /// Get the tokens around <paramref name="position"/>: the window before and after,
    /// clamped to the corpus boundaries.
    /// </summary>
    public (List<int> Before, List<int> After) GetContext(int position, int window = 50)
    {
        ThrowIfDisposed();
        return _sa.GetContext(position, window);
    }

    /// <summary>Get the corpus text around <paramref name="position"/>, decoded with the model's tokenizer.</summary>
    public string GetContextText(int position, int window = 50)
    {
        var (before, after) = GetContext(position, window);
        var slice = new List<int>(before.Count + after.Count);
        slice.AddRange(before);
        slice.AddRange(after);
        return Tokenizer.Decode(slice);
    }

    /// <summary>Release resources and clear caches. Model instances are otherwise GC-friendly.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _cache.Clear();
    }

    /// <inheritdoc />
    public override string ToString() =>
        ModelPath is null
            ? $"Infinigram(n={N:N0}, vocab={VocabSize})"
            : $"Infinigram(n={N:N0}, vocab={VocabSize}, path={ModelPath})";

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(Infinigram));
        }
    }

    private readonly struct ContextKey : IEquatable<ContextKey>
    {
        private readonly int[] _items;
        private readonly int _limit;

        public ContextKey(IReadOnlyList<int> items, int? limit)
        {
            _items = items as int[] ?? items.ToArray();
            _limit = limit ?? -1;
        }

        public bool Equals(ContextKey other) =>
            _limit == other._limit && _items.AsSpan().SequenceEqual(other._items);

        public override bool Equals(object? obj) => obj is ContextKey other && Equals(other);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(_limit);
            for (int i = 0; i < _items.Length; i++)
            {
                hash.Add(_items[i]);
            }
            return hash.ToHashCode();
        }
    }
}
