namespace Infinigram;

public sealed class SuffixArray
{
    private readonly int[] _corpus;
    private readonly int[] _sa;

    public SuffixArray(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        _corpus = tokens as int[] ?? tokens.ToArray();
        _sa = SuffixArrayBuilder.Build(_corpus);
    }

    private SuffixArray(int[] corpus, int[] suffixArray)
    {
        _corpus = corpus;
        _sa = suffixArray;
    }

    public int Length => _corpus.Length;

    public IReadOnlyList<int> Corpus => _corpus;

    public IReadOnlyList<int> Positions => _sa;

    public int GetToken(int position)
    {
        if (position < 0 || position >= _corpus.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), $"position {position} out of range [0, {_corpus.Length})");
        }
        return _corpus[position];
    }

    public List<int> Search(IReadOnlyList<int> pattern, int? limit = null) =>
        SuffixArraySearch.Search(_corpus, _sa, pattern, limit);

    public int Count(IReadOnlyList<int> pattern) =>
        SuffixArraySearch.Count(_corpus, _sa, pattern);

    internal List<int> SearchSampled(IReadOnlyList<int> pattern, int? limit) =>
        SuffixArraySearch.SearchSampled(_corpus, _sa, pattern, limit);

    public (List<int> Before, List<int> After) GetContext(int position, int window = 10)
    {
        if (position < 0 || position >= _corpus.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), $"position {position} out of range [0, {_corpus.Length})");
        }
        if (window < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(window), "window must be non-negative");
        }

        int start = Math.Max(0, position - window);
        int end = Math.Min(_corpus.Length, position + window);

        var before = new List<int>(position - start);
        for (int i = start; i < position; i++)
        {
            before.Add(_corpus[i]);
        }

        var after = new List<int>(end - position);
        for (int i = position; i < end; i++)
        {
            after.Add(_corpus[i]);
        }
        return (before, after);
    }

    public void Save(string directory) => SuffixArrayStorage.Save(directory, _corpus, _sa);

    public static SuffixArray Load(string directory)
    {
        var (corpus, positions) = SuffixArrayStorage.Load(directory);
        return new SuffixArray(corpus, positions);
    }

}
