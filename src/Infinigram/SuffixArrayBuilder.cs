namespace Infinigram;

internal static class SuffixArrayBuilder
{
    internal static int[] Build(int[] data)
    {
        int n = data.Length;
        if (n == 0)
        {
            return Array.Empty<int>();
        }
        if (n == 1)
        {
            return new[] { 0 };
        }

        var sa = new int[n];
        var rank = new int[n];
        var tmpRank = new int[n];

        var vals = new int[n];
        Array.Copy(data, vals, n);
        Array.Sort(vals);

        var uniq = new int[n];
        int uniqueCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (i == 0 || vals[i] != vals[i - 1])
            {
                uniq[uniqueCount++] = vals[i];
            }
        }
        for (int i = 0; i < n; i++)
        {
            rank[i] = Array.BinarySearch(uniq, 0, uniqueCount, data[i]);
        }

        int nClasses = uniqueCount;
        for (int i = 0; i < n; i++)
        {
            sa[i] = i;
        }

        int k = 1;
        while (k < n)
        {
            sa = CountingSortBy(sa, rank, k, secondKey: true, nClasses: nClasses);
            sa = CountingSortBy(sa, rank, k: 0, secondKey: false, nClasses: nClasses);

            tmpRank[sa[0]] = 0;
            int classes = 1;
            for (int i = 1; i < n; i++)
            {
                int a = sa[i - 1], b = sa[i];
                int ra = rank[a], rb = rank[b];
                int ak = a + k, bk = b + k;
                int rak = ak < n ? rank[ak] : -1;
                int rbk = bk < n ? rank[bk] : -1;
                if (ra != rb || rak != rbk)
                {
                    classes++;
                }
                tmpRank[b] = classes - 1;
            }

            var swap = rank;
            rank = tmpRank;
            tmpRank = swap;
            nClasses = classes;

            if (nClasses == n)
            {
                break;
            }
            k <<= 1;
        }

        return sa;
    }

    private static int[] CountingSortBy(int[] sa, int[] rank, int k, bool secondKey, int nClasses)
    {
        int n = sa.Length;
        int range = secondKey ? nClasses + 1 : nClasses;
        var count = new int[range];
        var output = new int[n];

        for (int i = 0; i < n; i++)
        {
            count[KeyOf(sa[i], rank, k, secondKey)]++;
        }

        int sum = 0;
        for (int i = 0; i < range; i++)
        {
            int c = count[i];
            count[i] = sum;
            sum += c;
        }

        for (int i = 0; i < n; i++)
        {
            int key = KeyOf(sa[i], rank, k, secondKey);
            output[count[key]++] = sa[i];
        }
        return output;
    }

    private static int KeyOf(int i, int[] rank, int k, bool secondKey)
    {
        if (!secondKey)
        {
            return rank[i];
        }
        int idx = i + k;
        return idx < rank.Length ? rank[idx] + 1 : 0;
    }
}
