namespace Infinigram;

internal static class SuffixMatcher
{
    internal static List<int> LimitContext(IReadOnlyList<int> context, int? maxLength)
    {
        if (maxLength is int max && max > 0 && context.Count > max)
        {
            return context.Skip(context.Count - max).ToList();
        }
        return context.ToList();
    }

    internal static List<(int Length, List<int> Positions)> FindAll(SuffixArray suffixArray, List<int> context)
    {
        var matches = new List<(int Length, List<int> Positions)>();
        for (int length = context.Count; length > 0; length--)
        {
            var positions = SearchSuffix(suffixArray, context, length, limit: null);
            if (positions.Count > 0)
            {
                matches.Add((length, positions));
            }
        }
        return matches;
    }

    internal static List<int> SearchSuffix(SuffixArray suffixArray, List<int> context, int suffixLength, int? limit)
    {
        var pattern = context.GetRange(context.Count - suffixLength, suffixLength);
        return suffixArray.Search(pattern, limit);
    }

    internal static List<int> SearchSuffixSampled(
        SuffixArray suffixArray, List<int> context, int suffixLength, int? limit)
    {
        var pattern = context.GetRange(context.Count - suffixLength, suffixLength);
        return suffixArray.SearchSampled(pattern, limit);
    }

    internal static (int Length, int Position) LongestMatchLength(SuffixArray suffixArray, List<int> context)
    {
        if (context.Count == 0)
        {
            return (0, -1);
        }

        var first = SearchSuffix(suffixArray, context, 1, limit: 1);
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
            var results = SearchSuffix(suffixArray, context, mid, limit: 1);
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
}
