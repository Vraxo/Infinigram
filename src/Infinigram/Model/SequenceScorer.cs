namespace Infinigram;

internal static class SequenceScorer
{
    internal static double Score(NGramModel model, IReadOnlyList<int> sequence, double smoothing = 1e-10)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        if (sequence.Count == 0)
        {
            return 0.0;
        }

        double uniformLogProbability = Math.Log(1.0 / model.VocabSize);
        double totalLogProbability = uniformLogProbability;

        for (int i = 1; i < sequence.Count; i++)
        {
            var context = sequence.Take(i).ToArray();
            var probabilities = TokenPredictor.GetDistribution(model, context, smoothing);
            if (probabilities is null)
            {
                totalLogProbability += uniformLogProbability;
                continue;
            }

            if (!probabilities.TryGetValue(sequence[i], out double probability) || probability <= 0.0)
            {
                return double.NegativeInfinity;
            }

            totalLogProbability += Math.Log(probability);
        }

        return totalLogProbability;
    }
}
