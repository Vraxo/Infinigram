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

        double uniformLp = Math.Log(1.0 / model.VocabSize);
        double totalLp = uniformLp;

        for (int i = 1; i < sequence.Count; i++)
        {
            var context = sequence.Take(i).ToList();
            var counts = model.Continuations(context);
            if (counts is null)
            {
                totalLp += uniformLp;
                continue;
            }

            double total = 0;
            foreach (var c in counts.Values)
            {
                total += c;
            }
            counts.TryGetValue(sequence[i], out int count);
            double prob = (count + smoothing) / (total + smoothing * model.VocabSize);
            if (prob <= 0)
            {
                return double.NegativeInfinity;
            }
            totalLp += Math.Log(prob);
        }

        return totalLp;
    }
}
