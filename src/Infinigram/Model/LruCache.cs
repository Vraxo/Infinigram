namespace Infinigram;

internal sealed class LruCache<TKey, TValue> where TKey : notnull
{
    private int _capacity;
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

    internal int Capacity => _capacity;

    internal void SetCapacity(int capacity)
    {
        if (capacity < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), "capacity must be at least 1");
        }
        _capacity = capacity;
        while (_map.Count > _capacity)
        {
            var oldest = _order.First!;
            _order.RemoveFirst();
            _map.Remove(oldest.Value.Key);
        }
    }

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
