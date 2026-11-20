namespace Infinigram;

internal static class InfinigramPredictor
{
    internal static IReadOnlyDictionary<int, double>? Predict(Infinigram model, IReadOnlyList<int> context, int topK = 50, double smoothing = 0.0)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (topK < 1)
        {
            topK = 1;
        }

        var counts = model.Continuations(context);
        if (counts is null)
        {
            return null;
        }

        double total = 0;
        foreach (var c in counts.Values)
        {
            total += c;
        }
        double smoothedTotal = total + smoothing * model.VocabSize;

        var probs = new Dictionary<int, double>(counts.Count);
        foreach (var (token, count) in counts)
        {
            probs[token] = (count + smoothing) / smoothedTotal;
        }

        if (smoothing > 0.0)
        {
            double unseen = smoothing / smoothedTotal;
            for (int t = 0; t < model.VocabSize; t++)
            {
                probs.TryAdd(t, unseen);
            }
        }

        if (probs.Count > topK)
        {
            probs = probs
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .ToDictionary(kv => kv.Key, kv => kv.Value);
        }
        return probs;
    }

    internal static IReadOnlyDictionary<int, double>? PredictLogprobs(Infinigram model, IReadOnlyList<int> context, int topK = 50,
        double smoothing = 1e-10)
    {
        var probs = model.Predict(context, topK, smoothing);
        if (probs is null)
        {
            return null;
        }
        var logprobs = new Dictionary<int, double>(probs.Count);
        foreach (var (token, p) in probs)
        {
            logprobs[token] = Math.Log(p);
        }
        return logprobs;
    }
}
