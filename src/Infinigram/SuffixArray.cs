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

    public List<int> Search(IReadOnlyList<int> pattern, int? limit = null)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        if (pattern.Count == 0 || limit is <= 0)
        {
            return new List<int>();
        }

        var p = ToArray(pattern);
        var (left, right) = FindRange(p);
        if (left >= right)
        {
            return new List<int>();
        }

        int count = right - left;
        if (limit is int lim && count > lim)
        {
            count = lim;
        }

        var positions = new List<int>(count);
        for (int i = 0; i < count; i++)
        {
            positions.Add(_sa[left + i]);
        }
        return positions;
    }

    public int Count(IReadOnlyList<int> pattern)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        if (pattern.Count == 0)
        {
            return 0;
        }
        var p = ToArray(pattern);
        var (left, right) = FindRange(p);
        return Math.Max(0, right - left);
    }

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

    private (int Left, int Right) FindRange(int[] pattern)
    {
        int n = _sa.Length;

        int lo = 0, hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (CompareAt(pattern, _sa[mid], prefixOnly: false) > 0)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        int left = lo;

        lo = 0;
        hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (CompareAt(pattern, _sa[mid], prefixOnly: true) >= 0)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return (left, lo);
    }

    private int CompareAt(int[] pattern, int pos, bool prefixOnly)
    {
        int n = _corpus.Length;
        if (pos < 0 || pos >= n)
        {
            return -1;
        }

        int patternLen = pattern.Length;
        int suffixLen = n - pos;
        int compareLen = Math.Min(patternLen, suffixLen);

        for (int i = 0; i < compareLen; i++)
        {
            int a = pattern[i];
            int b = _corpus[pos + i];
            if (a != b)
            {
                return a < b ? -1 : 1;
            }
        }

        if (prefixOnly)
        {
            return patternLen <= suffixLen ? 0 : 1;
        }
        if (patternLen < suffixLen)
        {
            return -1;
        }
        if (patternLen > suffixLen)
        {
            return 1;
        }
        return 0;
    }

    private static int[] ToArray(IReadOnlyList<int> values) =>
        values as int[] ?? values.ToArray();
}
