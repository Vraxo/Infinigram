// Suffix array implementation for efficient n-gram operations.
//
// The Python reference builds the array with `pydivsufsort` (SA-IS, O(n)). No
// equivalent exists for .NET, so construction here uses prefix doubling with
// radix sort: O(n log n) time, O(n) extra memory, and simple enough to verify.
// Swapping in SA-IS later only requires replacing <see cref="BuildSuffixArray"/>.
//
// The corpus is stored as token ids in an <c>int[]</c> — byte-level corpora use
// ids 0..255, token-level corpora use ids up to <c>int.MaxValue</c>. All binary
// searches compare a pattern against the suffix at a suffix-array position;
// ranges are half-open <c>[left, right)</c>.

using System.Text;

namespace Infinigram;

/// <summary>
/// In-memory suffix array supporting byte- and token-level corpora.
/// </summary>
public sealed class SuffixArray
{
    private readonly int[] _corpus;
    private readonly int[] _sa;

    /// <summary>
    /// Build a suffix array from a sequence of token ids.
    /// </summary>
    public SuffixArray(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        _corpus = tokens as int[] ?? tokens.ToArray();
        _sa = BuildSuffixArray(_corpus);
    }

    // Private ctor for Save/Load round-trips that already have both arrays.
    private SuffixArray(int[] corpus, int[] suffixArray)
    {
        _corpus = corpus;
        _sa = suffixArray;
    }

    /// <summary>Number of tokens in the corpus.</summary>
    public int Length => _corpus.Length;

    /// <summary>The corpus as token ids.</summary>
    public IReadOnlyList<int> Corpus => _corpus;

    /// <summary>The suffix array itself: <c>SuffixArray[i]</c> is the start position of the i-th smallest suffix.</summary>
    public IReadOnlyList<int> Positions => _sa;

    /// <summary>Return the token id at <paramref name="position"/>, with bounds checking.</summary>
    public int GetToken(int position)
    {
        if (position < 0 || position >= _corpus.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), $"position {position} out of range [0, {_corpus.Length})");
        }
        return _corpus[position];
    }

    /// <summary>
    /// All positions where <paramref name="pattern"/> occurs in the corpus,
    /// optionally capped at <paramref name="limit"/>.
    /// </summary>
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

    /// <summary>Count occurrences without materializing positions.</summary>
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

    /// <summary>
    /// Return up to <paramref name="window"/> tokens of context on either side of
    /// <paramref name="position"/>.
    /// </summary>
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

    /// <summary>
    /// Persist corpus and suffix array to <paramref name="directory"/> as
    /// <c>corpus.bin</c> and <c>suffix_array.bin</c> (little-endian int32).
    /// </summary>
    public void Save(string directory)
    {
        ArgumentNullException.ThrowIfNull(directory);
        Directory.CreateDirectory(directory);
        WriteIntArray(Path.Combine(directory, "corpus.bin"), _corpus);
        WriteIntArray(Path.Combine(directory, "suffix_array.bin"), _sa);
    }

    /// <summary>Load an index previously written by <see cref="Save"/>.</summary>
    public static SuffixArray Load(string directory)
    {
        ArgumentNullException.ThrowIfNull(directory);
        string corpusPath = Path.Combine(directory, "corpus.bin");
        string saPath = Path.Combine(directory, "suffix_array.bin");
        if (!File.Exists(corpusPath) || !File.Exists(saPath))
        {
            throw new FileNotFoundException(
                $"Index not found at {directory} (expected corpus.bin and suffix_array.bin). " +
                "Note: indexes written by the Python implementation (.npy files) are not " +
                "readable by this engine yet.");
        }

        int[] corpus = ReadIntArray(corpusPath);
        int[] sa = ReadIntArray(saPath);
        if (sa.Length != corpus.Length)
        {
            throw new InvalidDataException(
                $"Corrupt index at {directory}: corpus has {corpus.Length} tokens but the " +
                $"suffix array has {sa.Length} entries.");
        }
        return new SuffixArray(corpus, sa);
    }

    /// <summary>
    /// Half-open range <c>[left, right)</c> in the suffix array whose entries are the
    /// occurrences of <paramref name="pattern"/>. Port of the Python
    /// <c>_binary_search_range</c>.
    /// </summary>
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

    // Compare `pattern` against the suffix of the corpus starting at `pos`.
    // < 0 when pattern < suffix, > 0 when pattern > suffix, 0 when equal over
    // the compared extent; prefixOnly limits comparison to the common prefix
    // and reports pattern-fits-in-suffix as 0.
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

    private static void WriteIntArray(string path, int[] data)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        writer.Write(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            writer.Write(data[i]); // 4 bytes, little-endian
        }
    }

    private static int[] ReadIntArray(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
        int count = reader.ReadInt32();
        if (count < 0 || count != (stream.Length - sizeof(int)) / sizeof(int))
        {
            throw new InvalidDataException($"Corrupt index file {path} (bad element count {count}).");
        }
        var data = new int[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = reader.ReadInt32();
        }
        return data;
    }

    private static int[] BuildSuffixArray(int[] data)
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

        // Initial ranks: order-preserving compression of the token ids.
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
            // Sort by (rank[i], rank[i + k]) via two stable counting sorts.
            sa = CountingSortBy(sa, rank, k, secondKey: true, nClasses: nClasses);
            sa = CountingSortBy(sa, rank, k: 0, secondKey: false, nClasses: nClasses);

            // Re-rank.
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

    // Stable counting sort of `sa` by either the first key (rank[i],
    // secondKey == false) or the second key (rank[i + k] + 1, with 0 meaning
    // "no suffix at i + k"). Ranges derived from nClasses.
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

/// <summary>
/// Small bounded LRU cache used for continuation counts. Values may be
/// <c>null</c> (a cached "no match").
/// </summary>
internal sealed class LruCache<TKey, TValue> where TKey : notnull
{
    private readonly int _capacity;
    private readonly Dictionary<TKey, LinkedListNode<KeyValuePair<TKey, TValue>>> _map;
    private readonly LinkedList<KeyValuePair<TKey, TValue>> _order;

    public LruCache(int capacity)
    {
        if (capacity < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), "capacity must be at least 1");
        }
        _capacity = capacity;
        _map = new Dictionary<TKey, LinkedListNode<KeyValuePair<TKey, TValue>>>(capacity);
        _order = new LinkedList<KeyValuePair<TKey, TValue>>();
    }

    public int Count => _map.Count;

    public bool TryGet(TKey key, out TValue? value)
    {
        if (_map.TryGetValue(key, out var node))
        {
            _order.Remove(node);
            _order.AddLast(node);
            value = node.Value.Value;
            return true;
        }
        value = default;
        return false;
    }

    public void Set(TKey key, TValue value)
    {
        if (_map.TryGetValue(key, out var node))
        {
            node.Value = new KeyValuePair<TKey, TValue>(key, value);
            _order.Remove(node);
            _order.AddLast(node);
            return;
        }
        if (_map.Count >= _capacity)
        {
            var oldest = _order.First!;
            _order.RemoveFirst();
            _map.Remove(oldest.Value.Key);
        }
        var newNode = _order.AddLast(new KeyValuePair<TKey, TValue>(key, value));
        _map[key] = newNode;
    }

    public void Clear()
    {
        _map.Clear();
        _order.Clear();
    }
}
