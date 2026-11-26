namespace Infinigram;

public sealed class NGramModel : IDisposable
{
    public const int DefaultContinuationLimit = 10000;

    private SuffixArray _sa = null!;
    private readonly ContinuationCounter _continuations = new();
    private readonly Dictionary<int, int> _unigramCounts = new();
    private string? _modelDir;
    private bool _disposed;

    public ITokenizer Tokenizer { get; private set; }

    public int VocabSize { get; private set; }

    public int? MaxLength { get; }

    public int N => _sa.Length;

    public string? ModelPath => _modelDir;


    private NGramModel(ITokenizer tokenizer, int? maxLength)
    {
        Tokenizer = tokenizer;
        VocabSize = tokenizer.VocabSize;
        MaxLength = maxLength;
    }

    internal NGramModel(ITokenizer tokenizer, int? maxLength, SuffixArray suffixArray, string modelDirectory)
        : this(tokenizer, maxLength)
    {
        _sa = suffixArray;
        _modelDir = modelDirectory;
        CountUnigrams(_sa.Corpus);
    }

    public static NGramModel Build(string corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true) =>
        ModelPersistence.Build(corpus, modelPath, tokenizer, maxLength, verbose);

    public static NGramModel Build(byte[] corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true) =>
        ModelPersistence.Build(corpus, modelPath, tokenizer, maxLength, verbose);

    public static NGramModel Build(IReadOnlyList<int> corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true) =>
        ModelPersistence.Build(corpus, modelPath, tokenizer, maxLength, verbose);

    public static NGramModel BuildFromFile(string corpusFilePath, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true) =>
        ModelPersistence.BuildFromFile(corpusFilePath, modelPath, tokenizer, maxLength, verbose);

    public static NGramModel Load(string modelPath, ITokenizer? tokenizer = null, int? maxLength = null) =>
        ModelPersistence.Load(modelPath, tokenizer, maxLength);

    public static List<string> ListModels(string? modelsDir = null) => ModelPersistence.ListModels(modelsDir);

    public NGramModel(IReadOnlyList<int> corpus, ITokenizer? tokenizer = null, int? maxLength = null)
        : this(tokenizer ?? new ByteTokenizer(), maxLength)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (corpus.Count == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(corpus));
        }
        _sa = new SuffixArray(CorpusValidator.Validate(corpus, VocabSize, "Corpus"));
        CountUnigrams(_sa.Corpus);
    }

    public NGramModel(byte[] corpus, ITokenizer? tokenizer = null, int? maxLength = null)
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
        _sa = new SuffixArray(CorpusValidator.Validate(corpus.Select(b => (int)b).ToList(), VocabSize, "Corpus"));
        CountUnigrams(_sa.Corpus);
    }

    public NGramModel(string corpus, ITokenizer? tokenizer = null, int? maxLength = null)
        : this(tokenizer ?? new ByteTokenizer(), maxLength)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (corpus.Length == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(corpus));
        }
        _sa = new SuffixArray(Tokenizer.Encode(corpus));
        CountUnigrams(_sa.Corpus);
    }


    public List<int> Search(IReadOnlyList<int> pattern, int? limit = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pattern);
        return _sa.Search(pattern, limit);
    }

    public int Count(IReadOnlyList<int> pattern)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pattern);
        return _sa.Count(pattern);
    }

    public int GetToken(int position)
    {
        ThrowIfDisposed();
        return _sa.GetToken(position);
    }

    public (int Position, int Length) LongestSuffix(IReadOnlyList<int> context)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return (-1, 0);
        }

        var ctx = SuffixMatcher.LimitContext(context, MaxLength);
        var (length, position) = _continuations.GetLongestMatch(_sa, ctx);
        return length == 0 ? (-1, 0) : (position, length);
    }

    public List<(int Length, List<int> Positions)> FindAllSuffixMatches(IReadOnlyList<int> context)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return new List<(int, List<int>)>();
        }

        return SuffixMatcher.FindAll(_sa, SuffixMatcher.LimitContext(context, MaxLength));
    }

    public Dictionary<int, int>? Continuations(IReadOnlyList<int> context, int? limit = DefaultContinuationLimit)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return new Dictionary<int, int>(_unigramCounts);
        }
        return _continuations.Get(_sa, SuffixMatcher.LimitContext(context, MaxLength), limit);
    }

    internal Dictionary<int, int>? ContinuationsExact(
        IReadOnlyList<int> context, int? limit = DefaultContinuationLimit)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(context);
        if (context.Count == 0)
        {
            return null;
        }

        return _continuations.GetExact(_sa, SuffixMatcher.LimitContext(context, MaxLength), limit);
    }

    public void ClearCache() => _continuations.Clear();

    internal int ContinuationsCacheCount => _continuations.Count;

    internal int ContinuationsCacheCapacity
    {
        get => _continuations.Capacity;
        set => _continuations.Capacity = value;
    }

    public IReadOnlyDictionary<int, double>? Predict(IReadOnlyList<int> context, int topK = 50, double smoothing = 0.0)
    {
        ThrowIfDisposed();
        return TokenPredictor.Predict(this, context, topK, smoothing);
    }

    public IReadOnlyDictionary<int, double>? PredictLogprobs(IReadOnlyList<int> context, int topK = 50,
        double smoothing = 1e-10) => TokenPredictor.PredictLogprobs(this, context, topK, smoothing);

    public List<int> Generate(IReadOnlyList<int>? context = null, int maxTokens = 100,
        double temperature = 1.0, int? topK = null, double smoothing = 0.0,
        IEnumerable<IReadOnlyList<int>>? stop = null, Random? rng = null)
    {
        ThrowIfDisposed();
        return TextGenerator.Generate(this, context, maxTokens, temperature, topK, smoothing, stop, rng);
    }

    public string GenerateText(string context = "", int maxTokens = 100, double temperature = 1.0,
        int? topK = 50, double smoothing = 0.0, IEnumerable<string>? stop = null, Random? rng = null)
    {
        ThrowIfDisposed();
        return TextGenerator.GenerateText(this, context, maxTokens, temperature, topK, smoothing, stop, rng);
    }

    public double Score(IReadOnlyList<int> sequence, double smoothing = 1e-10)
    {
        ThrowIfDisposed();
        return SequenceScorer.Score(this, sequence, smoothing);
    }

    public void Update(IReadOnlyList<int> newTokens)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(newTokens);
        UpdateInternal(CorpusValidator.Validate(newTokens, VocabSize, "new_tokens"));
    }

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

    public void Update(string newTokens)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(newTokens);
        UpdateInternal(Tokenizer.Encode(newTokens).ToArray());
    }

    private void UpdateInternal(int[] newTokens)
    {
        _sa = CorpusAppender.Append(_sa, newTokens);
        foreach (int token in newTokens)
        {
            _unigramCounts[token] = _unigramCounts.GetValueOrDefault(token) + 1;
        }

        if (_modelDir is not null)
        {
            ModelStorage.Save(_modelDir, _sa, Tokenizer, VocabSize, MaxLength);
        }
        ClearCache();
    }

    public (List<int> Before, List<int> After) GetContext(int position, int window = 50)
    {
        ThrowIfDisposed();
        return _sa.GetContext(position, window);
    }

    public string GetContextText(int position, int window = 50)
    {
        var (before, after) = GetContext(position, window);
        var slice = new List<int>(before.Count + after.Count);
        slice.AddRange(before);
        slice.AddRange(after);
        return Tokenizer.Decode(slice);
    }

    private void CountUnigrams(IReadOnlyList<int> tokens)
    {
        foreach (int token in tokens)
        {
            _unigramCounts[token] = _unigramCounts.GetValueOrDefault(token) + 1;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _continuations.Clear();
    }

    public override string ToString() =>
        ModelPath is null
            ? $"NGramModel(n={N:N0}, vocab={VocabSize})"
            : $"NGramModel(n={N:N0}, vocab={VocabSize}, path={ModelPath})";

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(NGramModel));
        }
    }
}
