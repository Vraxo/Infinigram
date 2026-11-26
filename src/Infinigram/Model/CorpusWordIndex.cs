using System.Text;
using System.Text.RegularExpressions;

namespace Infinigram;

/// <summary>
/// A word-and-punctuation view of the same source corpus.
/// Its vocabulary, token sequence, and continuation counts are derived only from corpus text.
/// </summary>
public sealed class CorpusWordIndex
{
    public const int DefaultContinuationLimit = 10000;

    private const int MaximumContextWords = 12;

    private static readonly Regex TokenPattern = new(
        @"[\p{L}\p{M}\p{N}_]+(?:['’\-‐‑‒–][\p{L}\p{M}\p{N}_]+)*|[\uD800-\uDBFF][\uDC00-\uDFFF]|[^\s]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly HashSet<char> ClosingPunctuation =
        new(",.;:!?%)]}»”’\"".ToCharArray());

    private static readonly HashSet<char> OpeningPunctuation =
        new("([{«“".ToCharArray());

    private readonly Dictionary<string, int> _tokenIds;
    private readonly string[] _vocabulary;
    private readonly SuffixArray _suffixArray;
    private readonly ContinuationCounter _continuations = new();
    private readonly Dictionary<int, int> _unigramCounts = new();

    internal CorpusWordIndex(IReadOnlyList<int> corpus, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(tokenizer);

        string text = tokenizer.Decode(corpus);
        _tokenIds = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var tokenSequence = new List<int>();
        var surfaceCounts = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (Match match in TokenPattern.Matches(text))
        {
            string surface = match.Value;
            if (!_tokenIds.TryGetValue(surface, out int id))
            {
                id = _tokenIds.Count;
                _tokenIds.Add(surface, id);
            }

            tokenSequence.Add(id);
            surfaceCounts[surface] = surfaceCounts.GetValueOrDefault(surface) + 1;
            _unigramCounts[id] = _unigramCounts.GetValueOrDefault(id) + 1;
        }

        _vocabulary = new string[_tokenIds.Count];
        var bestSurfaceCounts = new int[_tokenIds.Count];

        foreach (var (surface, count) in surfaceCounts)
        {
            int id = _tokenIds[surface];
            if (count > bestSurfaceCounts[id] ||
                (count == bestSurfaceCounts[id] &&
                 string.CompareOrdinal(surface, _vocabulary[id]) < 0))
            {
                _vocabulary[id] = surface;
                bestSurfaceCounts[id] = count;
            }
        }

        _suffixArray = new SuffixArray(tokenSequence.ToArray());
        Vocabulary = Array.AsReadOnly(_vocabulary);
    }

    /// <summary>The number of word and punctuation tokens in the corpus.</summary>
    public int WordCount => _suffixArray.Length;

    /// <summary>The number of distinct case-insensitive tokens found in the corpus.</summary>
    public int VocabularySize => _vocabulary.Length;

    /// <summary>Corpus-derived display forms, ordered by their assigned token IDs.</summary>
    public IReadOnlyList<string> Vocabulary { get; }

    /// <summary>Splits text into words and individual punctuation or symbol tokens.</summary>
    public static IReadOnlyList<string> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return TokenPattern.Matches(text).Select(match => match.Value).ToArray();
    }

    /// <summary>
    /// Counts corpus-observed next tokens for the longest matching suffix of the text.
    /// Returns null when a nonempty context has no known trailing token sequence.
    /// </summary>
    public Dictionary<string, int>? Continuations(
        string context, int? limit = DefaultContinuationLimit)
    {
        ArgumentNullException.ThrowIfNull(context);

        var words = Tokenize(context);
        if (words.Count == 0)
        {
            return ToWords(_unigramCounts);
        }

        var ids = EncodeTrailingKnownSuffix(words);
        if (ids.Count == 0)
        {
            return null;
        }

        var counts = _continuations.Get(_suffixArray, ids, limit);
        return counts is null ? null : ToWords(counts);
    }

    /// <summary>
    /// Generates a word-level continuation by sampling only tokens observed in this corpus.
    /// Whitespace is normalized to single spaces; punctuation is attached conventionally.
    /// </summary>
    public string GenerateText(
        string context = "",
        int maxTokens = 40,
        double temperature = 0.8,
        int? topK = 30,
        Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (maxTokens < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTokens));
        }
        if (!double.IsFinite(temperature) || temperature < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature));
        }

        var contextWords = Tokenize(context);
        var current = EncodeTrailingKnownSuffix(contextWords);
        bool hasUnknownTrailingContext = contextWords.Count > 0 && current.Count == 0;
        var output = new StringBuilder(context);
        int contextStart = Math.Max(0, current.Count - MaximumContextWords);
        if (contextStart > 0)
        {
            current.RemoveRange(0, contextStart);
        }

        for (int step = 0; step < maxTokens; step++)
        {
            IReadOnlyDictionary<int, int>? counts;
            if (current.Count == 0)
            {
                if (step > 0 || hasUnknownTrailingContext)
                {
                    break;
                }
                counts = _unigramCounts;
            }
            else
            {
                counts = _continuations.Get(_suffixArray, current, DefaultContinuationLimit);
                if (counts is null || counts.Count == 0)
                {
                    break;
                }
            }

            var probabilities = ToProbabilities(counts, topK);
            if (probabilities.Count == 0)
            {
                break;
            }

            int next = Sampling.SampleFromDistribution(probabilities, temperature, rng);
            AppendToken(output, _vocabulary[next]);
            current.Add(next);
            if (current.Count > MaximumContextWords)
            {
                current.RemoveAt(0);
            }
        }

        return output.ToString();
    }

    private List<int> EncodeTrailingKnownSuffix(IReadOnlyList<string> words)
    {
        int start = words.Count;
        while (start > 0 && _tokenIds.ContainsKey(words[start - 1]))
        {
            start--;
        }

        var ids = new List<int>(words.Count - start);
        for (int i = start; i < words.Count; i++)
        {
            ids.Add(_tokenIds[words[i]]);
        }
        return ids;
    }

    private Dictionary<string, int> ToWords(IReadOnlyDictionary<int, int> counts)
    {
        var result = new Dictionary<string, int>(counts.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var (id, count) in counts)
        {
            result[_vocabulary[id]] = count;
        }
        return result;
    }

    private Dictionary<int, double> ToProbabilities(
        IReadOnlyDictionary<int, int> counts, int? topK)
    {
        if (counts.Count == 0 || _vocabulary.Length == 0)
        {
            return new Dictionary<int, double>();
        }

        IEnumerable<KeyValuePair<int, int>> selected = counts;
        if (topK is int limit)
        {
            selected = selected
                .OrderByDescending(pair => pair.Value)
                .Take(Math.Clamp(limit, 1, _vocabulary.Length));
        }

        var entries = selected.ToArray();
        double total = entries.Sum(pair => (double)pair.Value);
        if (total <= 0.0)
        {
            return new Dictionary<int, double>();
        }

        var probabilities = new Dictionary<int, double>(entries.Length);
        foreach (var (id, count) in entries)
        {
            probabilities[id] = count / total;
        }
        return probabilities;
    }

    private static void AppendToken(StringBuilder output, string token)
    {
        if (output.Length == 0)
        {
            output.Append(token);
            return;
        }

        char first = token[0];
        if (ClosingPunctuation.Contains(first))
        {
            while (output.Length > 0 && char.IsWhiteSpace(output[^1]))
            {
                output.Length--;
            }
            output.Append(token);
            return;
        }

        if (!char.IsWhiteSpace(output[^1]) && !OpeningPunctuation.Contains(output[^1]))
        {
            output.Append(' ');
        }
        output.Append(token);
    }
}
