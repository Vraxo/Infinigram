namespace Infinigram;

internal static class InfinigramSuffixMatcher
{
    internal static List<int> SearchSuffix(SuffixArray suffixArray, List<int> context, int suffixLength, int? limit)
    {
        var pattern = context.GetRange(context.Count - suffixLength, suffixLength);
        return suffixArray.Search(pattern, limit);
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
