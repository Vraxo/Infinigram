namespace Infinigram;

internal readonly struct ContextKey : IEquatable<ContextKey>
{
    private readonly int[] _items;
    private readonly int _limit;

    public ContextKey(IReadOnlyList<int> items, int? limit)
    {
        _items = items as int[] ?? items.ToArray();
        _limit = limit ?? -1;
    }

    public bool Equals(ContextKey other) =>
        _limit == other._limit && _items.AsSpan().SequenceEqual(other._items);

    public override bool Equals(object? obj) => obj is ContextKey other && Equals(other);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(_limit);
        for (int i = 0; i < _items.Length; i++)
        {
            hash.Add(_items[i]);
        }
        return hash.ToHashCode();
    }
}
