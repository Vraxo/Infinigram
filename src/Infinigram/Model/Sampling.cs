
namespace Infinigram;

public static class Sampling
{
    private const double LogEpsilon = 1e-10;

    public static int SampleFromDistribution(
        IReadOnlyDictionary<int, double> probs,
        double temperature = 1.0,
        Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(probs);
        if (probs.Count == 0)
        {
            return 0;
        }
        if (temperature == 0.0)
        {
            int bestToken = 0;
            double bestProb = double.NegativeInfinity;
            foreach (var (token, p) in probs)
            {
                if (p > bestProb)
                {
                    bestProb = p;
                    bestToken = token;
                }
            }
            return bestToken;
        }

        var (tokens, weights) = ApplyTemperature(probs, temperature);
        return WeightedChoice(tokens, weights, rng ?? Random.Shared);
    }

    public static List<int> SampleMultiple(
        IReadOnlyDictionary<int, double> probs,
        int k,
        double temperature = 1.0,
        bool withReplacement = true,
        Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(probs);
        if (probs.Count == 0 || k <= 0)
        {
            return new List<int>();
        }
        if (temperature == 0.0)
        {
            return probs
                .OrderByDescending(kv => kv.Value)
                .Take(k)
                .Select(kv => kv.Key)
                .ToList();
        }

        var (tokens, weights) = ApplyTemperature(probs, temperature);
        var random = rng ?? Random.Shared;

        double total = weights.Sum();
        if (total <= 0.0)
        {
            return new List<int>();
        }

        if (withReplacement)
        {
            var picks = new List<int>(k);
            for (int i = 0; i < k; i++)
            {
                picks.Add(WeightedChoice(tokens, weights, random));
            }
            return picks;
        }

        var resultList = new List<int>();
        var remainingTokens = new List<int>(tokens);
        var remainingWeights = new List<double>(weights);
        int sampleCount = Math.Min(k, remainingTokens.Count);
        for (int i = 0; i < sampleCount; i++)
        {
            if (remainingWeights.Sum() <= 0.0)
            {
                break;
            }
            int idx = WeightedChoiceIndex(remainingWeights, random);
            resultList.Add(remainingTokens[idx]);
            int last = remainingTokens.Count - 1;
            remainingTokens[idx] = remainingTokens[last];
            remainingWeights[idx] = remainingWeights[last];
            remainingTokens.RemoveAt(last);
            remainingWeights.RemoveAt(last);
        }
        return resultList;
    }

    public static Dictionary<int, double> TopKFilter(IReadOnlyDictionary<int, double> probs, int k)
    {
        ArgumentNullException.ThrowIfNull(probs);
        if (probs.Count == 0 || k <= 0)
        {
            return new Dictionary<int, double>();
        }

        var sorted = probs.OrderByDescending(kv => kv.Value).Take(k).ToList();
        double total = sorted.Sum(kv => kv.Value);
        if (total == 0.0)
        {
            double uniform = 1.0 / sorted.Count;
            return sorted.ToDictionary(kv => kv.Key, _ => uniform);
        }
        return sorted.ToDictionary(kv => kv.Key, kv => kv.Value / total);
    }

    public static Dictionary<int, double> TopPFilter(IReadOnlyDictionary<int, double> probs, double p)
    {
        ArgumentNullException.ThrowIfNull(probs);
        if (probs.Count == 0 || p <= 0.0)
        {
            return new Dictionary<int, double>();
        }
        if (p >= 1.0)
        {
            return new Dictionary<int, double>(probs);
        }

        var sorted = probs.OrderByDescending(kv => kv.Value);
        var result = new Dictionary<int, double>();
        double cumsum = 0.0;
        foreach (var (token, prob) in sorted)
        {
            result[token] = prob;
            cumsum += prob;
            if (cumsum >= p)
            {
                break;
            }
        }

        double totalResult = result.Values.Sum();
        if (totalResult > 0.0)
        {
            foreach (var token in result.Keys.ToList())
            {
                result[token] /= totalResult;
            }
        }
        return result;
    }

    private static (List<int> Tokens, List<double> Weights) ApplyTemperature(
        IReadOnlyDictionary<int, double> probs,
        double temperature)
    {
        var tokens = new List<int>(probs.Count);
        var weights = new List<double>(probs.Count);
        foreach (var (token, p) in probs)
        {
            tokens.Add(token);
            weights.Add(p);
        }

        if (temperature == 1.0)
        {
            return (tokens, weights);
        }

        var logProbs = weights.Select(p => Math.Log(p + LogEpsilon) / temperature).ToList();
        double maxLog = logProbs.Max();
        for (int i = 0; i < weights.Count; i++)
        {
            weights[i] = Math.Exp(logProbs[i] - maxLog);
        }
        return (tokens, weights);
    }

    private static int WeightedChoice(List<int> tokens, List<double> weights, Random rng)
    {
        double total = weights.Sum();
        double r = rng.NextDouble() * total;
        double acc = 0.0;
        for (int i = 0; i < weights.Count; i++)
        {
            acc += weights[i];
            if (r <= acc)
            {
                return tokens[i];
            }
        }
        return tokens[^1];
    }

    private static int WeightedChoiceIndex(List<double> weights, Random rng)
    {
        double total = weights.Sum();
        double r = rng.NextDouble() * total;
        double acc = 0.0;
        for (int i = 0; i < weights.Count; i++)
        {
            acc += weights[i];
            if (r <= acc)
            {
                return i;
            }
        }
        return weights.Count - 1;
    }
}
