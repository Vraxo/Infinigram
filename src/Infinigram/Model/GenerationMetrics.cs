namespace Infinigram;

/// <summary>Measures exact, contiguous reuse of generated tokens from a model's corpus.</summary>
public readonly record struct GenerationOverlap(int GeneratedTokens, int MatchedTokens, int LongestMatch)
{
    /// <summary>The fraction of generated tokens covered by non-overlapping corpus matches.</summary>
    public double MatchedRatio => GeneratedTokens == 0 ? 0.0 : (double)MatchedTokens / GeneratedTokens;
}

/// <summary>Utilities for evaluating how much generated output reuses its training corpus.</summary>
public readonly record struct GenerationEvaluation(GenerationOverlap Overlap, double MeanLogProbability);

public static class GenerationMetrics
{
    /// <summary>
    /// Evaluates exact corpus reuse and average model log-probability per generated token.
    /// Log-probability is a model-relative consistency signal, not a human fluency score.
    /// </summary>
    public static GenerationEvaluation Evaluate(
        NGramModel model, IReadOnlyList<int> generatedTokens, int minimumSpan = 32,
        double smoothing = 1e-10)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(generatedTokens);
        if (smoothing < 0 || !double.IsFinite(smoothing))
            throw new ArgumentOutOfRangeException(nameof(smoothing));

        var overlap = MeasureOverlap(model, generatedTokens, minimumSpan);
        double meanLogProbability = generatedTokens.Count == 0
            ? 0.0
            : model.Score(generatedTokens, smoothing) / generatedTokens.Count;

        return new GenerationEvaluation(overlap, meanLogProbability);
    }

    /// <summary>
    /// Measures the fraction of generated tokens covered by exact corpus spans at least
    /// <paramref name="minimumSpan"/> tokens long. Matches are counted left-to-right and
    /// do not overlap; short common fragments are intentionally ignored.
    /// </summary>
    public static GenerationOverlap MeasureOverlap(
        NGramModel model, IReadOnlyList<int> generatedTokens, int minimumSpan = 32)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(generatedTokens);
        if (minimumSpan < 1)
            throw new ArgumentOutOfRangeException(nameof(minimumSpan));

        int count = generatedTokens.Count;
        if (count == 0)
            return new GenerationOverlap(0, 0, 0);

        int matched = 0;
        int longest = 0;
        int start = 0;

        while (start <= count - minimumSpan)
        {
            int matchLength = FindLongestMatch(model, generatedTokens, start, minimumSpan);
            if (matchLength == 0)
            {
                start++;
                continue;
            }

            matched += matchLength;
            longest = Math.Max(longest, matchLength);
            start += matchLength;
        }

        return new GenerationOverlap(count, matched, longest);
    }

    private static int FindLongestMatch(
        NGramModel model, IReadOnlyList<int> tokens, int start, int minimumSpan)
    {
        int remaining = tokens.Count - start;
        if (remaining < minimumSpan)
            return 0;

        int low = minimumSpan;
        int high = remaining;
        int best = 0;

        while (low <= high)
        {
            int length = low + ((high - low) / 2);
            var candidate = new int[length];
            for (int i = 0; i < length; i++)
                candidate[i] = tokens[start + i];

            if (model.Count(candidate) > 0)
            {
                best = length;
                low = length + 1;
            }
            else
            {
                high = length - 1;
            }
        }

        return best;
    }
}
