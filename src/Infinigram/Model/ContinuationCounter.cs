namespace Infinigram;

internal sealed class ContinuationCounter
{
    private readonly LruCache<ContextKey, Dictionary<int, int>?> _cache = new(1000);
    private readonly LruCache<ContextKey, (int Length, int Position)> _longestCache = new(1000);

    internal int Count => _cache.Count;
    internal int Capacity
    {
        get => _cache.Capacity;
        set
        {
            _cache.SetCapacity(value);
            _longestCache.SetCapacity(value);
        }
    }

    internal void Clear()
    {
        _cache.Clear();
        _longestCache.Clear();
    }

    internal (int Length, int Position) GetLongestMatch(
        SuffixArray suffixArray, List<int> context)
    {
        var cacheKey = new ContextKey(context, limit: null);
        if (_longestCache.TryGet(cacheKey, out var cached))
        {
            return cached;
        }

        var match = SuffixMatcher.LongestMatchLength(suffixArray, context);
        _longestCache.Set(cacheKey, match);
        return match;
    }

    internal Dictionary<int, int>? Get(SuffixArray suffixArray, List<int> ctx, int? limit)
    {
        var cacheKey = new ContextKey(ctx, limit);
        if (_cache.TryGet(cacheKey, out var cached))
        {
            return cached is null ? null : new Dictionary<int, int>(cached);
        }

        var (matchLength, _) = GetLongestMatch(suffixArray, ctx);
        if (matchLength == 0)
        {
            _cache.Set(cacheKey, null);
            return null;
        }

        for (int length = matchLength; length > 0; length--)
        {
            var positions = SuffixMatcher.SearchSuffixSampled(suffixArray, ctx, length, limit);
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

    internal IReadOnlyDictionary<int, int>? GetExact(
        SuffixArray suffixArray, List<int> context, int? limit)
    {
        if (context.Count == 0)
        {
            return null;
        }

        var cacheKey = new ContextKey(context, limit, exact: true);
        if (_cache.TryGet(cacheKey, out var cached))
        {
            return cached;
        }

        int length = context.Count;
        var positions = SuffixMatcher.SearchSuffixSampled(suffixArray, context, length, limit);
        Dictionary<int, int>? counts = null;

        foreach (int position in positions)
        {
            int next = position + length;
            if (next >= suffixArray.Length)
            {
                continue;
            }

            int token = suffixArray.GetToken(next);
            counts ??= new Dictionary<int, int>();
            counts[token] = counts.GetValueOrDefault(token) + 1;
        }

        _cache.Set(cacheKey, counts);
        return counts;
    }
}
