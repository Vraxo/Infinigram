using System.Text;

namespace Infinigram;

/// <summary>
/// A deterministic byte-level BPE tokenizer. Unseen text remains representable
/// because every UTF-8 byte is part of the base vocabulary.
/// </summary>
public sealed class BpeTokenizer : ITokenizer, IEquatable<BpeTokenizer>
{
    private const int ByteVocabularySize = 256;

    private static readonly Encoding Utf8Replacement =
        new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

    private readonly Merge[] _merges;
    private readonly byte[][] _tokenBytes;
    private readonly Dictionary<long, MergeRule> _rulesByPair;

    public int VocabSize => _tokenBytes.Length;

    public int MergeCount => _merges.Length;

    public string Name => "bpe";

    private BpeTokenizer(Merge[] merges)
    {
        _merges = merges;
        _tokenBytes = new byte[ByteVocabularySize + merges.Length][];
        _rulesByPair = new Dictionary<long, MergeRule>(merges.Length);

        for (int token = 0; token < ByteVocabularySize; token++)
        {
            _tokenBytes[token] = [(byte)token];
        }

        for (int i = 0; i < merges.Length; i++)
        {
            int tokenId = ByteVocabularySize + i;
            Merge merge = merges[i];

            if (merge.Left < 0 || merge.Left >= tokenId ||
                merge.Right < 0 || merge.Right >= tokenId)
            {
                throw new ArgumentException(
                    $"Merge {i} references token ids that must be in [0, {tokenId}).",
                    nameof(merges));
            }

            long key = PairKey(merge.Left, merge.Right);
            if (!_rulesByPair.TryAdd(key, new MergeRule(i, tokenId)))
            {
                throw new ArgumentException(
                    $"Merge {i} duplicates an earlier pair.",
                    nameof(merges));
            }

            byte[] leftBytes = _tokenBytes[merge.Left];
            byte[] rightBytes = _tokenBytes[merge.Right];
            var mergedBytes = new byte[checked(leftBytes.Length + rightBytes.Length)];

            leftBytes.CopyTo(mergedBytes, 0);
            rightBytes.CopyTo(mergedBytes, leftBytes.Length);
            _tokenBytes[tokenId] = mergedBytes;
        }
    }

    /// <summary>
    /// Learns byte-pair merges from a corpus. The requested vocabulary size
    /// includes the 256 base UTF-8 byte tokens.
    /// </summary>
    public static BpeTokenizer Train(string corpus, int vocabSize = 4096, int minFrequency = 2)
    {
        ArgumentNullException.ThrowIfNull(corpus);

        if (vocabSize < ByteVocabularySize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(vocabSize), $"Vocabulary size must be at least {ByteVocabularySize}.");
        }
        if (minFrequency < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minFrequency), "Minimum frequency must be positive.");
        }

        byte[] bytes = Utf8Replacement.GetBytes(corpus);
        if (bytes.Length == 0)
        {
            throw new ArgumentException("Training corpus must not be empty.", nameof(corpus));
        }
        if (vocabSize == ByteVocabularySize || bytes.Length == 1)
        {
            return new BpeTokenizer(Array.Empty<Merge>());
        }

        int length = bytes.Length;
        var tokens = new int[length];
        var previous = new int[length];
        var next = new int[length];
        var alive = new bool[length];

        for (int i = 0; i < length; i++)
        {
            tokens[i] = bytes[i];
            previous[i] = i - 1;
            next[i] = i + 1 < length ? i + 1 : -1;
            alive[i] = true;
        }

        var pairPositions = new Dictionary<long, SortedSet<int>>();
        var rankedPairs = new SortedSet<PairCandidate>(PairCandidateComparer.Instance);

        void AddPair(int left)
        {
            if (left < 0 || !alive[left])
            {
                return;
            }

            int right = next[left];
            if (right < 0 || !alive[right])
            {
                return;
            }

            long key = PairKey(tokens[left], tokens[right]);
            if (!pairPositions.TryGetValue(key, out var positions))
            {
                positions = new SortedSet<int>();
                pairPositions.Add(key, positions);
            }

            rankedPairs.Remove(new PairCandidate(key, positions.Count));
            positions.Add(left);
            rankedPairs.Add(new PairCandidate(key, positions.Count));
        }

        void RemovePair(int left)
        {
            if (left < 0 || !alive[left])
            {
                return;
            }

            int right = next[left];
            if (right < 0 || !alive[right])
            {
                return;
            }

            long key = PairKey(tokens[left], tokens[right]);
            if (!pairPositions.TryGetValue(key, out var positions) || !positions.Contains(left))
            {
                return;
            }

            rankedPairs.Remove(new PairCandidate(key, positions.Count));
            positions.Remove(left);

            if (positions.Count == 0)
            {
                pairPositions.Remove(key);
            }
            else
            {
                rankedPairs.Add(new PairCandidate(key, positions.Count));
            }
        }

        for (int i = 0; i < length - 1; i++)
        {
            AddPair(i);
        }

        var merges = new List<Merge>(Math.Min(vocabSize - ByteVocabularySize, length - 1));

        while (merges.Count < vocabSize - ByteVocabularySize && rankedPairs.Count > 0)
        {
            PairCandidate candidate = rankedPairs.Min;
            if (candidate.Count < minFrequency)
            {
                break;
            }

            int leftToken = LeftFromPairKey(candidate.Key);
            int rightToken = RightFromPairKey(candidate.Key);
            int[] positions = pairPositions[candidate.Key].ToArray();
            int mergedToken = ByteVocabularySize + merges.Count;
            int mergedCount = 0;

            // Sorted positions give deterministic left-to-right, non-overlapping merges.
            foreach (int left in positions)
            {
                int right = next[left];
                if (!alive[left] || right < 0 || !alive[right] ||
                    tokens[left] != leftToken || tokens[right] != rightToken)
                {
                    continue;
                }

                int before = previous[left];
                int after = next[right];

                RemovePair(before);
                RemovePair(left);
                RemovePair(right);

                tokens[left] = mergedToken;
                alive[right] = false;
                previous[right] = -1;
                next[right] = -1;
                next[left] = after;

                if (after >= 0)
                {
                    previous[after] = left;
                }

                AddPair(before);
                AddPair(left);
                mergedCount++;
            }

            if (mergedCount == 0)
            {
                break;
            }

            merges.Add(new Merge(leftToken, rightToken));
        }

        return new BpeTokenizer(merges.ToArray());
    }

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        byte[] bytes = Utf8Replacement.GetBytes(text);
        if (bytes.Length == 0)
        {
            return Array.Empty<int>();
        }

        int length = bytes.Length;
        if (_merges.Length == 0 || length == 1)
        {
            return ConvertBytes(bytes);
        }

        var tokens = new int[length];
        var previous = new int[length];
        var next = new int[length];
        var generations = new int[length];
        var pending = new PendingMergeHeap(length);

        for (int i = 0; i < length; i++)
        {
            tokens[i] = bytes[i];
            previous[i] = i - 1;
            next[i] = i + 1 < length ? i + 1 : -1;
        }

        void QueuePair(int left)
        {
            if (left < 0 || next[left] == PendingMergeHeap.Removed)
            {
                return;
            }

            int generation = ++generations[left];
            int right = next[left];
            if (right < 0)
            {
                return;
            }

            long key = PairKey(tokens[left], tokens[right]);
            if (_rulesByPair.TryGetValue(key, out MergeRule rule))
            {
                pending.Enqueue(new PendingMerge(left, generation, rule.Rank));
            }
        }

        for (int i = 0; i < length - 1; i++)
        {
            QueuePair(i);
        }

        while (pending.TryDequeue(out PendingMerge candidate))
        {
            int left = candidate.Position;
            if (generations[left] != candidate.Generation)
            {
                continue;
            }

            int right = next[left];
            if (right < 0)
            {
                continue;
            }

            int after = next[right];
            tokens[left] = ByteVocabularySize + candidate.Rank;
            next[left] = after;

            if (after >= 0)
            {
                previous[after] = left;
            }

            generations[right]++;
            previous[right] = -1;
            next[right] = PendingMergeHeap.Removed;

            QueuePair(previous[left]);
            QueuePair(left);
        }

        int encodedCount = 0;
        for (int current = 0; current >= 0; current = next[current])
        {
            encodedCount++;
        }

        var encoded = new int[encodedCount];
        int outputIndex = 0;
        for (int current = 0; current >= 0; current = next[current])
        {
            encoded[outputIndex++] = tokens[current];
        }

        return encoded;
    }

    private static int[] ConvertBytes(byte[] bytes)
    {
        var tokens = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            tokens[i] = bytes[i];
        }

        return tokens;
    }

    public string Decode(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);

        long byteCount = 0;
        foreach (int token in tokens)
        {
            if (token < 0 || token >= _tokenBytes.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokens), $"Token id {token} is outside the vocabulary range [0, {VocabSize}).");
            }

            int expansionLength = _tokenBytes[token].Length;
            if (byteCount + expansionLength > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), "Decoded text is too large.");
            }
            byteCount += expansionLength;
        }

        var bytes = new byte[(int)byteCount];
        int offset = 0;

        foreach (int token in tokens)
        {
            byte[] expansion = _tokenBytes[token];
            expansion.CopyTo(bytes, offset);
            offset += expansion.Length;
        }

        return Utf8Replacement.GetString(bytes);
    }

    internal IReadOnlyList<(int Left, int Right)> MergePairs =>
        _merges.Select(static merge => (merge.Left, merge.Right)).ToArray();

    internal static BpeTokenizer FromMerges(IEnumerable<(int Left, int Right)> merges)
    {
        ArgumentNullException.ThrowIfNull(merges);
        return new BpeTokenizer(merges.Select(static pair => new Merge(pair.Left, pair.Right)).ToArray());
    }

    public bool Equals(BpeTokenizer? other) =>
        other is not null && _merges.AsSpan().SequenceEqual(other._merges);

    public override bool Equals(object? obj) => obj is BpeTokenizer other && Equals(other);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (Merge merge in _merges)
        {
            hash.Add(merge.Left);
            hash.Add(merge.Right);
        }
        return hash.ToHashCode();
    }

    public override string ToString() =>
        $"BpeTokenizer(vocab={VocabSize}, merges={MergeCount})";

    private static long PairKey(int left, int right) =>
        ((long)(uint)left << 32) | (uint)right;

    private static int LeftFromPairKey(long key) => (int)(key >> 32);

    private static int RightFromPairKey(long key) => (int)(uint)key;

    private readonly record struct Merge(int Left, int Right);

    private readonly record struct MergeRule(int Rank, int TokenId);

    private readonly record struct PendingMerge(int Position, int Generation, int Rank);

    private struct PendingMergeHeap
    {
        internal const int Removed = -2;

        private PendingMerge[] _items;
        private int _count;

        internal PendingMergeHeap(int capacity)
        {
            _items = new PendingMerge[Math.Max(4, capacity)];
            _count = 0;
        }

        internal void Enqueue(PendingMerge item)
        {
            if (_count == _items.Length)
            {
                Array.Resize(ref _items, checked(_items.Length * 2));
            }

            int index = _count++;
            while (index > 0)
            {
                int parent = (index - 1) >> 1;
                if (!ComesBefore(item, _items[parent]))
                {
                    break;
                }

                _items[index] = _items[parent];
                index = parent;
            }

            _items[index] = item;
        }

        internal bool TryDequeue(out PendingMerge item)
        {
            if (_count == 0)
            {
                item = default;
                return false;
            }

            item = _items[0];
            int remaining = --_count;
            if (remaining == 0)
            {
                return true;
            }

            PendingMerge last = _items[remaining];
            int index = 0;

            while (true)
            {
                int left = (index << 1) + 1;
                if (left >= remaining)
                {
                    break;
                }

                int right = left + 1;
                int child = right < remaining && ComesBefore(_items[right], _items[left])
                    ? right
                    : left;

                if (!ComesBefore(_items[child], last))
                {
                    break;
                }

                _items[index] = _items[child];
                index = child;
            }

            _items[index] = last;
            return true;
        }

        private static bool ComesBefore(PendingMerge left, PendingMerge right) =>
            left.Rank < right.Rank ||
            left.Rank == right.Rank && left.Position < right.Position;
    }

    private readonly record struct PairCandidate(long Key, int Count);

    private sealed class PairCandidateComparer : IComparer<PairCandidate>
    {
        internal static PairCandidateComparer Instance { get; } = new();

        public int Compare(PairCandidate x, PairCandidate y)
        {
            int frequencyOrder = y.Count.CompareTo(x.Count);
            return frequencyOrder != 0 ? frequencyOrder : x.Key.CompareTo(y.Key);
        }
    }
}
