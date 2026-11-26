namespace Infinigram;

internal readonly struct ContextKey : IEquatable<ContextKey>
{
    private readonly int[] _items;
    private readonly int _limit;
    private readonly bool _exact;

    public ContextKey(IReadOnlyList<int> items, int? limit, bool exact = false)
    {
        _items = items as int[] ?? items.ToArray();
        _limit = limit ?? -1;
        _exact = exact;
    }

    public bool Equals(ContextKey other) =>
        _limit == other._limit &&
        _exact == other._exact &&
        _items.AsSpan().SequenceEqual(other._items);

    public override bool Equals(object? obj) => obj is ContextKey other && Equals(other);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(_limit);
        hash.Add(_exact);
        for (int i = 0; i < _items.Length; i++)
        {
            hash.Add(_items[i]);
        }
        return hash.ToHashCode();
    }
}