namespace Infinigram;

internal static class TokenPredictor
{
    private const int MaximumBackoffLevels = 8;
    private const double MaximumBackoffMass = 0.05;

    internal static IReadOnlyDictionary<int, double>? Predict(
        NGramModel model, IReadOnlyList<int> context, int topK = 50, double smoothing = 0.0)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (topK < 1)
        {
            topK = 1;
        }

        var probabilities = GetDistribution(model, context, smoothing);
        if (probabilities is null)
        {
            return null;
        }

        if (probabilities.Count > topK)
        {
            return probabilities
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        return probabilities;
    }

    internal static Dictionary<int, double>? GetDistribution(
        NGramModel model, IReadOnlyList<int> context, double smoothing)
    {
        if (context.Count == 0)
        {
            return GetUnigramDistribution(model, smoothing);
        }

        var exactCounts = model.ContinuationsExact(context);
        int longestMatchLength;
        if (exactCounts is { Count: > 0 })
        {
            longestMatchLength = model.MaxLength is int max && max > 0
                ? Math.Min(context.Count, max)
                : context.Count;

            long exactTotal = exactCounts.Values.Sum(count => (long)count);
            double exactWeight = exactTotal / (double)(exactTotal + exactCounts.Count + 1);
            if (1.0 - exactWeight <= MaximumBackoffMass)
            {
                return ToProbabilities(exactCounts, model.VocabSize, smoothing);
            }
        }
        else
        {
            var (_, matchLength) = model.LongestSuffix(context);
            longestMatchLength = matchLength;
        }

        if (longestMatchLength == 0)
        {
            return GetUnigramDistribution(model, smoothing);
        }

        var orders = GetBackoffOrders(longestMatchLength);
        var evidence = new List<(IReadOnlyDictionary<int, int> Counts, long Total, double ContextWeight)>(orders.Count);
        Dictionary<int, double>? lowerOrderDistribution = null;

        if (exactCounts is { Count: > 0 })
        {
            long total = exactCounts.Values.Sum(count => (long)count);
            double contextWeight = total / (double)(total + exactCounts.Count + 1);
            evidence.Add((exactCounts, total, contextWeight));
        }

        foreach (int order in orders)
        {
            if (exactCounts is { Count: > 0 } && order == longestMatchLength)
            {
                continue;
            }

            var suffix = GetSuffix(context, order);
            var counts = model.ContinuationsExact(suffix);
            if (counts is not { Count: > 0 })
            {
                continue;
            }

            long total = counts.Values.Sum(count => (long)count);
            if (total == 0)
            {
                continue;
            }

            // Witten-Bell interpolation uses N observations and T distinct
            // continuation types. One additional escape type keeps a single
            // observation from receiving half of the total prediction mass.
            double contextWeight = total / (double)(total + counts.Count + 1);

            if (1.0 - contextWeight <= MaximumBackoffMass)
            {
                if (evidence.Count == 0)
                {
                    return ToProbabilities(counts, model.VocabSize, smoothing);
                }

                lowerOrderDistribution = ToProbabilities(counts, model.VocabSize, smoothing: 0.0);
                break;
            }

            evidence.Add((counts, total, contextWeight));
        }

        // Most normal predictions have a strong exact context. Avoid copying
        // the unigram table unless an actual backoff path needs that prior.
        var distribution = lowerOrderDistribution ?? GetUnigramDistribution(model, smoothing);
        if (distribution is null)
        {
            return null;
        }

        // Build from broad to specific so each longer context refines the
        // distribution produced by its shorter suffix.
        for (int i = evidence.Count - 1; i >= 0; i--)
        {
            var level = evidence[i];
            distribution = Interpolate(level.Counts, level.Total, level.ContextWeight, distribution);
        }

        return distribution;
    }

    private static Dictionary<int, double>? GetUnigramDistribution(NGramModel model, double smoothing)
    {
        var counts = model.Continuations(Array.Empty<int>());
        return counts is null ? null : ToProbabilities(counts, model.VocabSize, smoothing);
    }

    private static List<int> GetBackoffOrders(int longestMatchLength)
    {
        var orders = new List<int>(MaximumBackoffLevels + 1);
        int order = longestMatchLength;

        for (int level = 0; level < MaximumBackoffLevels && order > 0; level++)
        {
            orders.Add(order);
            if (order == 1)
            {
                break;
            }

            order = Math.Max(1, order / 2);
        }

        // Always include the one-token context as the broadest n-gram level.
        if (orders.Count > 0 && orders[^1] != 1)
        {
            orders.Add(1);
        }

        return orders;
    }

    private static int[] GetSuffix(IReadOnlyList<int> context, int length)
    {
        var suffix = new int[length];
        int start = context.Count - length;
        for (int i = 0; i < length; i++)
        {
            suffix[i] = context[start + i];
        }

        return suffix;
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

    private static Dictionary<int, double> ToProbabilities(
        IReadOnlyDictionary<int, int> counts, int vocabSize, double smoothing)
    {
        double total = counts.Values.Sum(count => (double)count);
        double smoothedTotal = total + smoothing * vocabSize;
        var probabilities = new Dictionary<int, double>(counts.Count);

        foreach (var (token, count) in counts)
        {
            probabilities[token] = (count + smoothing) / smoothedTotal;
        }

        if (smoothing > 0.0)
        {
            double unseen = smoothing / smoothedTotal;
            for (int token = 0; token < vocabSize; token++)
            {
                probabilities.TryAdd(token, unseen);
            }
        }

        return probabilities;
    }

    internal static IReadOnlyDictionary<int, double>? PredictLogprobs(
        NGramModel model, IReadOnlyList<int> context, int topK = 50, double smoothing = 1e-10)
    {
        var probabilities = Predict(model, context, topK, smoothing);
        if (probabilities is null)
        {
            return null;
        }

        var logprobs = new Dictionary<int, double>(probabilities.Count);
        foreach (var (token, probability) in probabilities)
        {
            logprobs[token] = Math.Log(probability);
        }

        return logprobs;
    }
}
