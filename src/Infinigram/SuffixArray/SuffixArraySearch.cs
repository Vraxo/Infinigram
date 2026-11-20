namespace Infinigram;

internal static class SuffixArraySearch
{
    internal static List<int> Search(int[] corpus, int[] suffixArray, IReadOnlyList<int> pattern, int? limit)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        if (pattern.Count == 0 || limit is <= 0)
        {
            return new List<int>();
        }

        var p = ToArray(pattern);
        var (left, right) = FindRange(corpus, suffixArray, p);
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
            positions.Add(suffixArray[left + i]);
        }
        return positions;
    }

    internal static int Count(int[] corpus, int[] suffixArray, IReadOnlyList<int> pattern)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        if (pattern.Count == 0)
        {
            return 0;
        }
        var p = ToArray(pattern);
        var (left, right) = FindRange(corpus, suffixArray, p);
        return Math.Max(0, right - left);
    }

    private static (int Left, int Right) FindRange(int[] corpus, int[] suffixArray, int[] pattern)
    {
        int n = suffixArray.Length;

        int lo = 0, hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (CompareAt(corpus, pattern, suffixArray[mid], prefixOnly: false) > 0)
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
            if (CompareAt(corpus, pattern, suffixArray[mid], prefixOnly: true) >= 0)
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

    private static int CompareAt(int[] corpus, int[] pattern, int pos, bool prefixOnly)
    {
        int n = corpus.Length;
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
            int b = corpus[pos + i];
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
