using System.Text;
using System.Text.RegularExpressions;

namespace Infinigram;

/// <summary>
/// A word-and-punctuation view of the same source corpus.
/// Its vocabulary, token sequence, continuation counts, and sentence shapes are corpus-derived.
/// </summary>
public sealed class CorpusWordIndex
{
    /// <summary>Maximum corpus occurrences sampled for a continuation query by default.</summary>
    public const int DefaultContinuationLimit = 10000;

    private const int MaximumContextWords = 12;
    private const int MaximumBackoffLevels = 8;
    private const int MaximumIndexedSentenceWords = 128;

    private static readonly Regex TokenPattern = new(
        @"[\p{L}\p{M}\p{N}_]+(?:['’\-‐‑‒–][\p{L}\p{M}\p{N}_]+)*|[\uD800-\uDBFF][\uDC00-\uDFFF]|[^\s]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly HashSet<char> ClosingPunctuation =
        new(",.;:!?%)]}»”’\"—–".ToCharArray());

    private static readonly HashSet<char> OpeningPunctuation =
        new("([{«“".ToCharArray());

    private readonly Dictionary<string, int> _tokenIds;
    private readonly string[] _vocabulary;
    private readonly bool[] _isWordToken;
    private readonly SuffixArray _suffixArray;
    private readonly ContinuationCounter _continuations = new();
    private readonly Dictionary<int, int> _unigramCounts = new();
    private readonly Dictionary<int, int> _wordUnigramCounts = new();
    private readonly SentenceShape[] _sentenceShapes;

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
            if (IsWordToken(surface))
            {
                _wordUnigramCounts[id] = _wordUnigramCounts.GetValueOrDefault(id) + 1;
            }
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

        _isWordToken = _vocabulary.Select(IsWordToken).ToArray();
        _suffixArray = new SuffixArray(tokenSequence.ToArray());
        (_sentenceCount, _sentenceShapes) = BuildSentenceShapes(text);
        Vocabulary = Array.AsReadOnly(_vocabulary);
    }

    private readonly int _sentenceCount;

    /// <summary>The number of word and punctuation tokens in the corpus.</summary>
    public int WordCount => _suffixArray.Length;

    /// <summary>The number of distinct case-insensitive tokens found in the corpus.</summary>
    public int VocabularySize => _vocabulary.Length;

    /// <summary>The number of complete sentences with a recognized ending in the corpus.</summary>
    public int SentenceCount => _sentenceCount;

    /// <summary>The number of distinct bounded clause-length and punctuation layouts found in the corpus.</summary>
    public int SentenceShapeCount => _sentenceShapes.Length;

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
    /// Generated-token spacing is normalized; punctuation is attached conventionally.
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
        var current = PrepareContext(contextWords);
        if (contextWords.Count > 0 && current.Count == 0)
        {
            return context;
        }

        var output = new StringBuilder(context);
        for (int step = 0; step < maxTokens; step++)
        {
            var probabilities = BuildDistribution(current, topK);
            if (probabilities.Count == 0)
            {
                break;
            }

            int next = Sampling.SampleFromDistribution(probabilities, temperature, rng);
            AppendToken(output, _vocabulary[next]);
            AddContextToken(current, next);
        }

        return output.ToString();
    }

    /// <summary>
    /// Generates one sentence using a clause-length and punctuation layout sampled from the corpus.
    /// Short phrase continuations are interpolated across corpus context lengths; word-level
    /// continuation counts provide the fallback when no observed phrase chunk fits.
    /// </summary>
    public string GenerateStructuredText(
        string context = "",
        int maxWords = 40,
        double temperature = 0.8,
        int? topK = 30,
        Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (maxWords < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxWords));
        }
        if (!double.IsFinite(temperature) || temperature < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature));
        }

        var contextWords = Tokenize(context);
        var current = PrepareContext(contextWords);
        if (contextWords.Count > 0 && current.Count == 0)
        {
            return context;
        }

        Random random = rng ?? Random.Shared;
        SentenceShape? shape = SampleSentenceShape(random, maxWords);
        if (shape is null || _wordUnigramCounts.Count == 0)
        {
            return context;
        }

        var output = new StringBuilder(context);
        for (int clause = 0; clause < shape.ClauseLengths.Length; clause++)
        {
            int remainingWords = shape.ClauseLengths[clause];
            while (remainingWords > 0)
            {
                int[]? phrase = remainingWords > 1
                    ? SamplePhrase(current, Math.Min(3, remainingWords), temperature, random)
                    : null;

                if (phrase is not null)
                {
                    foreach (int token in phrase)
                    {
                        AppendToken(output, _vocabulary[token]);
                        AddContextToken(current, token);
                    }

                    remainingWords -= phrase.Length;
                    continue;
                }

                var probabilities = BuildWordDistribution(current, topK);
                if (probabilities.Count == 0)
                {
                    return output.ToString();
                }

                int next = Sampling.SampleFromDistribution(probabilities, temperature, random);
                AppendToken(output, _vocabulary[next]);
                AddContextToken(current, next);
                remainingWords--;
            }

            string boundary = clause < shape.Separators.Length
                ? shape.Separators[clause]
                : shape.Ending;
            if (!_tokenIds.TryGetValue(boundary, out int boundaryId))
            {
                return output.ToString();
            }

            AppendToken(output, _vocabulary[boundaryId]);
            AddContextToken(current, boundaryId);
        }

        return output.ToString();
    }

    private List<int> PrepareContext(IReadOnlyList<string> words)
    {
        var current = EncodeTrailingKnownSuffix(words);
        int start = Math.Max(0, current.Count - MaximumContextWords);
        if (start > 0)
        {
            current.RemoveRange(0, start);
        }
        return current;
    }

    private void AddContextToken(List<int> context, int token)
    {
        context.Add(token);
        if (context.Count > MaximumContextWords)
        {
            context.RemoveAt(0);
        }
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

    private int[]? SamplePhrase(
        List<int> context, int maxPhraseWords, double temperature, Random random)
    {
        int maximumContext = Math.Min(3, context.Count);
        for (int phraseLength = maxPhraseWords; phraseLength >= 2; phraseLength--)
        {
            var probabilities = new Dictionary<string, double>(StringComparer.Ordinal);
            var phraseTokens = new Dictionary<string, int[]>(StringComparer.Ordinal);

            // Build from broad to specific, allowing a unique long context to
            // retain alternatives supplied by shorter corpus contexts.
            for (int contextLength = 1; contextLength <= maximumContext; contextLength++)
            {
                var counts = GetPhraseCounts(context, contextLength, phraseLength);
                if (counts.Count == 0)
                {
                    continue;
                }

                long total = counts.Values.Sum(candidate => (long)candidate.Count);
                double contextWeight = total / (double)(total + counts.Count + 1);
                double lowerWeight = 1.0 - contextWeight;

                foreach (string key in probabilities.Keys.ToArray())
                {
                    probabilities[key] *= lowerWeight;
                }

                foreach (var (key, candidate) in counts)
                {
                    double maximumLikelihood = candidate.Count / (double)total;
                    probabilities[key] = probabilities.GetValueOrDefault(key)
                        + contextWeight * maximumLikelihood;
                    phraseTokens[key] = candidate.Tokens;
                }
            }

            if (probabilities.Count == 0)
            {
                continue;
            }

            string[] keys = probabilities.Keys.ToArray();
            var distribution = new Dictionary<int, double>(keys.Length);
            for (int i = 0; i < keys.Length; i++)
            {
                distribution[i] = probabilities[keys[i]];
            }

            int selected = Sampling.SampleFromDistribution(distribution, temperature, random);
            return phraseTokens[keys[selected]];
        }

        return null;
    }

    private Dictionary<string, PhraseCandidate> GetPhraseCounts(
        List<int> context, int contextLength, int phraseLength)
    {
        var suffix = context.GetRange(context.Count - contextLength, contextLength);
        var positions = _suffixArray.SearchSampled(suffix, limit: 1024);
        var candidates = new Dictionary<string, PhraseCandidate>(StringComparer.Ordinal);

        foreach (int position in positions)
        {
            int start = position + contextLength;
            if (start + phraseLength > _suffixArray.Length)
            {
                continue;
            }

            var phrase = new int[phraseLength];
            bool valid = true;
            for (int i = 0; i < phraseLength; i++)
            {
                int token = _suffixArray.GetToken(start + i);
                if (!_isWordToken[token])
                {
                    valid = false;
                    break;
                }
                phrase[i] = token;
            }

            if (!valid)
            {
                continue;
            }

            string key = string.Join(",", phrase);
            if (candidates.TryGetValue(key, out var existing))
            {
                candidates[key] = existing with { Count = existing.Count + 1 };
            }
            else
            {
                candidates.Add(key, new PhraseCandidate(phrase, 1));
            }
        }

        return candidates;
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

    private Dictionary<int, double> BuildDistribution(List<int> context, int? topK)
    {
        var distribution = ToProbabilities(_unigramCounts);
        var orders = GetBackoffOrders(context.Count);

        for (int i = orders.Count - 1; i >= 0; i--)
        {
            int order = orders[i];
            var suffix = context.GetRange(context.Count - order, order);
            var counts = _continuations.GetExact(_suffixArray, suffix, DefaultContinuationLimit);
            if (counts is not { Count: > 0 })
            {
                continue;
            }

            long total = counts.Values.Sum(count => (long)count);
            double contextWeight = total / (double)(total + counts.Count + 1);
            distribution = Interpolate(counts, total, contextWeight, distribution);
        }

        return LimitAndNormalize(distribution, topK);
    }

    private Dictionary<int, double> BuildWordDistribution(List<int> context, int? topK)
    {
        var distribution = BuildDistribution(context, topK: null);
        foreach (int token in distribution.Keys.Where(token => !_isWordToken[token]).ToArray())
        {
            distribution.Remove(token);
        }

        if (distribution.Count == 0)
        {
            distribution = ToProbabilities(_wordUnigramCounts);
        }

        return LimitAndNormalize(distribution, topK);
    }

    private static Dictionary<int, double> LimitAndNormalize(
        Dictionary<int, double> distribution, int? topK)
    {
        if (topK is int limit && distribution.Count > limit)
        {
            distribution = distribution
                .OrderByDescending(pair => pair.Value)
                .Take(Math.Clamp(limit, 1, distribution.Count))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
        }

        double total = distribution.Values.Sum();
        if (total <= 0.0)
        {
            return new Dictionary<int, double>();
        }

        foreach (int token in distribution.Keys.ToArray())
        {
            distribution[token] /= total;
        }
        return distribution;
    }

    private static List<int> GetBackoffOrders(int contextLength)
    {
        var orders = new List<int>(5);
        int order = contextLength;

        for (int level = 0; level < MaximumBackoffLevels && order > 0; level++)
        {
            orders.Add(order);
            if (order == 1)
            {
                break;
            }

            order = Math.Max(1, order / 2);
        }

        if (orders.Count > 0 && orders[^1] != 1)
        {
            orders.Add(1);
        }

        return orders;
    }

    private static Dictionary<int, double> Interpolate(
        IReadOnlyDictionary<int, int> counts,
        long total,
        double contextWeight,
        Dictionary<int, double> lowerOrder)
    {
        double lowerOrderWeight = 1.0 - contextWeight;
        var probabilities = new Dictionary<int, double>(lowerOrder.Count + counts.Count);

        foreach (var (token, probability) in lowerOrder)
        {
            probabilities[token] = probability * lowerOrderWeight;
        }

        foreach (var (token, count) in counts)
        {
            double maximumLikelihood = count / (double)total;
            probabilities[token] = probabilities.GetValueOrDefault(token)
                + contextWeight * maximumLikelihood;
        }

        return probabilities;
    }

    private Dictionary<int, double> ToProbabilities(IReadOnlyDictionary<int, int> counts)
    {
        double total = counts.Values.Sum(count => (double)count);
        if (total <= 0.0)
        {
            return new Dictionary<int, double>();
        }

        var probabilities = new Dictionary<int, double>(counts.Count);
        foreach (var (token, count) in counts)
        {
            probabilities[token] = count / total;
        }
        return probabilities;
    }

    private static bool IsWordToken(string token) =>
        token.Any(character => char.IsLetterOrDigit(character) || character == '_');

    private static bool IsSentenceEnding(string token) => token is "." or "?" or "!";
    private static bool IsClauseSeparator(string token) => token is "," or ";" or ":" or "—" or "–";

    private static (int SentenceCount, SentenceShape[] Shapes) BuildSentenceShapes(string text)
    {
        var counts = new Dictionary<string, SentenceShape>(StringComparer.Ordinal);
        var clauseLengths = new List<int>();
        var separators = new List<string>();
        int clauseWords = 0;
        int sentenceWords = 0;
        int sentenceCount = 0;

        foreach (string token in Tokenize(text))
        {
            if (IsSentenceEnding(token))
            {
                if (clauseWords > 0)
                {
                    clauseLengths.Add(clauseWords);
                }

                if (sentenceWords > 0)
                {
                    sentenceCount++;
                    if (sentenceWords <= MaximumIndexedSentenceWords && clauseLengths.Count > 0)
                    {
                        string key = $"{string.Join(",", clauseLengths)}|{string.Join("|", separators)}|{token}";
                        if (counts.TryGetValue(key, out var existing))
                        {
                            counts[key] = existing with { Count = existing.Count + 1 };
                        }
                        else
                        {
                            counts.Add(key, new SentenceShape(
                                clauseLengths.ToArray(), separators.ToArray(), token, 1));
                        }
                    }
                }

                clauseLengths.Clear();
                separators.Clear();
                clauseWords = 0;
                sentenceWords = 0;
            }
            else if (IsClauseSeparator(token) && clauseWords > 0)
            {
                clauseLengths.Add(clauseWords);
                separators.Add(token);
                clauseWords = 0;
            }
            else if (IsWordToken(token))
            {
                clauseWords++;
                sentenceWords++;
            }
        }

        return (sentenceCount, counts.Values.ToArray());
    }

    private SentenceShape? SampleSentenceShape(Random random, int maxWords)
    {
        int total = 0;
        foreach (var shape in _sentenceShapes)
        {
            if (shape.WordCount <= maxWords)
            {
                total += shape.Count;
            }
        }

        if (total == 0)
        {
            return null;
        }

        int selected = random.Next(total);
        foreach (var shape in _sentenceShapes)
        {
            if (shape.WordCount > maxWords)
            {
                continue;
            }

            if (selected < shape.Count)
            {
                return shape;
            }
            selected -= shape.Count;
        }

        return null;
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

    private sealed record PhraseCandidate(int[] Tokens, int Count);

    private sealed record SentenceShape(
        int[] ClauseLengths, string[] Separators, string Ending, int Count)
    {
        internal int WordCount => ClauseLengths.Sum();
    }
}
