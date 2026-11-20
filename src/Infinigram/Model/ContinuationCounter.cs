namespace Infinigram;

internal sealed class ContinuationCounter
{
    private readonly LruCache<ContextKey, Dictionary<int, int>?> _cache = new(1000);
    private Dictionary<int, int>? _uniformCounts;

    internal int Count => _cache.Count;
    internal int Capacity
    {
        get => _cache.Capacity;
        set => _cache.SetCapacity(value);
    }

    internal void Clear() => _cache.Clear();

    internal Dictionary<int, int>? Get(SuffixArray suffixArray, int vocabSize, List<int> ctx, int? limit)
    {
        if (ctx.Count == 0)
        {
            return UniformCounts(vocabSize);
        }

        var cacheKey = new ContextKey(ctx, limit);
        if (_cache.TryGet(cacheKey, out var cached))
        {
            return cached is null ? null : new Dictionary<int, int>(cached);
        }

        var (matchLength, _) = SuffixMatcher.LongestMatchLength(suffixArray, ctx);
        if (matchLength == 0)
        {
            _cache.Set(cacheKey, null);
            return null;
        }

        for (int length = matchLength; length > 0; length--)
        {
            var positions = SuffixMatcher.SearchSuffix(suffixArray, ctx, length, limit);
            Dictionary<int, int>? counts = null;
            foreach (int pos in positions)
            {
                int next = pos + length;
                if (next >= suffixArray.Length)
                {
                    continue;
                }
                int token = suffixArray.GetToken(next);
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

    private Dictionary<int, int> UniformCounts(int vocabSize)
    {
        if (_uniformCounts is null)
        {
            _uniformCounts = new Dictionary<int, int>(vocabSize);
            for (int t = 0; t < vocabSize; t++)
            {
                _uniformCounts[t] = 1;
            }
        }
        return new Dictionary<int, int>(_uniformCounts);
    }
}
