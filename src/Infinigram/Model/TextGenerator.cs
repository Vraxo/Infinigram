using System.Runtime.InteropServices;

namespace Infinigram;

internal static class TextGenerator
{
    private const int MinimumNoveltyContext = 3;
    private const int MaximumNoveltyContext = 12;

    internal static List<int> Generate(
        NGramModel model, IReadOnlyList<int>? context = null, int maxTokens = 100,
        double temperature = 1.0, int? topK = null, double smoothing = 0.0,
        IEnumerable<IReadOnlyList<int>>? stop = null, Random? rng = null,
        double repetitionPenalty = 1.0, int noveltyWindow = 256)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (maxTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxTokens));
        if (!double.IsFinite(repetitionPenalty) || repetitionPenalty < 1.0)
            throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "Repetition penalty must be finite and at least 1.");
        if (noveltyWindow < 0) throw new ArgumentOutOfRangeException(nameof(noveltyWindow));

        int effectiveTopK = topK is null ? model.VocabSize : Math.Clamp(topK.Value, 1, model.VocabSize);
        var currentContext = context?.ToList() ?? new List<int>();
        var generated = new List<int>();
        var stopLists = stop?.Where(s => s is { Count: > 0 }).Select(s => s.ToList()).ToList();

        for (int step = 0; step < maxTokens; step++)
        {
            var probabilities = model.Predict(currentContext, topK: effectiveTopK, smoothing: smoothing);
            if (probabilities is null || probabilities.Count == 0) break;

            var adjusted = repetitionPenalty == 1.0 || noveltyWindow == 0
                ? probabilities
                : PenalizeRepeatedContinuations(probabilities, currentContext, noveltyWindow, repetitionPenalty);

            int nextToken = Sampling.SampleFromDistribution(adjusted, temperature, rng);
            generated.Add(nextToken);
            currentContext.Add(nextToken);

            if (stopLists is not null)
            {
                foreach (var stopSequence in stopLists)
                {
                    int stopLength = stopSequence.Count;
                    if (generated.Count >= stopLength &&
                        CollectionsMarshal.AsSpan(generated)[^stopLength..].SequenceEqual(CollectionsMarshal.AsSpan(stopSequence)))
                    {
                        generated.RemoveRange(generated.Count - stopLength, stopLength);
                        return generated;
                    }
                }
            }
        }
        return generated;
    }

    internal static string GenerateText(
        NGramModel model, string context = "", int maxTokens = 100, double temperature = 1.0,
        int? topK = 50, double smoothing = 0.0, IEnumerable<string>? stop = null, Random? rng = null,
        double repetitionPenalty = 1.0, int noveltyWindow = 256)
    {
        var seedTokens = string.IsNullOrEmpty(context) ? new List<int>() : model.Tokenizer.Encode(context).ToList();
        var stopTokens = stop?.Where(s => s.Length > 0).Select(s => model.Tokenizer.Encode(s).ToList()).ToList();
        var tokens = model.Generate(context: seedTokens, maxTokens: maxTokens, temperature: temperature,
            topK: topK, smoothing: smoothing, stop: stopTokens, rng: rng,
            repetitionPenalty: repetitionPenalty, noveltyWindow: noveltyWindow);
        return model.Tokenizer.Decode(tokens);
    }

    private static IReadOnlyDictionary<int, double> PenalizeRepeatedContinuations(
        IReadOnlyDictionary<int, double> probabilities, IReadOnlyList<int> history, int window, double penalty)
    {
        int historyStart = Math.Max(0, history.Count - window);
        int maxContext = Math.Min(MaximumNoveltyContext, history.Count - historyStart);
        if (maxContext < MinimumNoveltyContext) return probabilities;

        var adjusted = new Dictionary<int, double>(probabilities.Count);
        foreach (var (candidate, probability) in probabilities)
        {
            int repetitions = 0;
            for (int length = maxContext; length >= MinimumNoveltyContext; length--)
            {
                int currentStart = history.Count - length;
                for (int previous = historyStart; previous + length < currentStart; previous++)
                {
                    if (Matches(history, previous, currentStart, length) &&
                        history[previous + length] == candidate)
                        repetitions++;
                }
            }
            adjusted[candidate] = repetitions == 0 ? probability : probability / Math.Pow(penalty, repetitions);
        }
        return adjusted;
    }

    private static bool Matches(IReadOnlyList<int> tokens, int left, int right, int length)
    {
        for (int i = 0; i < length; i++)
            if (tokens[left + i] != tokens[right + i]) return false;
        return true;
    }
}
