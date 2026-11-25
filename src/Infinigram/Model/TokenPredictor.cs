namespace Infinigram;

internal static class TokenPredictor
{
    private const int StrongEvidenceThreshold = 64;
    private const double BackoffStrength = 4.0;

    internal static IReadOnlyDictionary<int, double>? Predict(NGramModel model, IReadOnlyList<int> context, int topK = 50, double smoothing = 0.0)
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
        var counts = model.Continuations(context);
        if (counts is null && context.Count > 0)
        {
            counts = model.Continuations(Array.Empty<int>());
        }
        if (counts is null)
        {
            return null;
        }

        var distribution = ToProbabilities(counts, model.VocabSize, smoothing);
        if (context.Count == 0)
        {
            return distribution;
        }

        int evidence = counts.Values.Sum();
        if (evidence >= StrongEvidenceThreshold)
        {
            return distribution;
        }

        var shorterContext = context.Count > 1
            ? context.Skip(1).ToArray()
            : Array.Empty<int>();
        var shorterCounts = model.Continuations(shorterContext);
        if (shorterCounts is null)
        {
            return distribution;
        }

        var backoff = ToProbabilities(shorterCounts, model.VocabSize, smoothing);
        double contextWeight = evidence / (evidence + BackoffStrength);
        double backoffWeight = 1.0 - contextWeight;

        foreach (var (token, probability) in backoff)
        {
            distribution[token] = distribution.GetValueOrDefault(token) * contextWeight
                + probability * backoffWeight;
        }

        if (contextWeight < 1.0)
        {
            foreach (var token in distribution.Keys.ToArray())
            {
                if (!backoff.ContainsKey(token))
                {
                    distribution[token] *= contextWeight;
                }
            }
        }

        return distribution;
    }

    private static Dictionary<int, double> ToProbabilities(
        Dictionary<int, int> counts, int vocabSize, double smoothing)
    {
        double total = counts.Values.Sum();
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
