namespace Infinigram;

internal static class SequenceScorer
{
    internal static double Score(NGramModel model, IReadOnlyList<int> sequence, double smoothing = 1e-10) =>
        Score(model, sequence, Array.Empty<int>(), smoothing);

    internal static double Score(
        NGramModel model, IReadOnlyList<int> sequence, IReadOnlyList<int> initialContext, double smoothing = 1e-10)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        ArgumentNullException.ThrowIfNull(initialContext);
        if (sequence.Count == 0)
        {
            return 0.0;
        }

        var context = new List<int>(initialContext.Count + sequence.Count);
        context.AddRange(initialContext);
        double totalLogProbability = 0.0;

        foreach (int token in sequence)
        {
            var probabilities = TokenPredictor.GetDistribution(model, context, smoothing);
            if (probabilities is null ||
                !probabilities.TryGetValue(token, out double probability) ||
                probability <= 0.0)
            {
                return double.NegativeInfinity;
            }

            totalLogProbability += Math.Log(probability);
            context.Add(token);
        }

        return totalLogProbability;
    }
}
