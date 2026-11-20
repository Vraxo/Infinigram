using System.Runtime.InteropServices;

namespace Infinigram;

internal static class TextGenerator
{
    internal static List<int> Generate(
        NGramModel model,
        IReadOnlyList<int>? context = null,
        int maxTokens = 100,
        double temperature = 1.0,
        int? topK = null,
        double smoothing = 0.0,
        IEnumerable<IReadOnlyList<int>>? stop = null,
        Random? rng = null)
    {
        int effectiveTopK = topK is null ? model.VocabSize : Math.Clamp(topK.Value, 1, model.VocabSize);

        var currentContext = context?.ToList() ?? new List<int>();
        var generated = new List<int>();

        var stopLists = stop?
            .Where(s => s is { Count: > 0 })
            .Select(s => s.ToList())
            .ToList();

        for (int step = 0; step < maxTokens; step++)
        {
            var probs = model.Predict(currentContext, topK: effectiveTopK, smoothing: smoothing);
            if (probs is null || probs.Count == 0)
            {
                break;
            }

            int nextToken = Sampling.SampleFromDistribution(probs, temperature, rng);
            generated.Add(nextToken);
            currentContext.Add(nextToken);

            if (stopLists is not null)
            {
                foreach (var stopSeq in stopLists)
                {
                    int stopLength = stopSeq.Count;
                    if (generated.Count >= stopLength &&
                        CollectionsMarshal.AsSpan(generated)[^stopLength..].SequenceEqual(CollectionsMarshal.AsSpan(stopSeq)))
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
        NGramModel model,
        string context = "",
        int maxTokens = 100,
        double temperature = 1.0,
        int? topK = 50,
        double smoothing = 0.0,
        IEnumerable<string>? stop = null,
        Random? rng = null)
    {
        var seedTokens = string.IsNullOrEmpty(context)
            ? new List<int>()
            : model.Tokenizer.Encode(context).ToList();
        var stopTokens = stop?
            .Where(s => s.Length > 0)
            .Select(s => model.Tokenizer.Encode(s).ToList())
            .ToList();

        var tokens = model.Generate(
            context: seedTokens,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            smoothing: smoothing,
            stop: stopTokens,
            rng: rng);
        return model.Tokenizer.Decode(tokens);
    }
}
