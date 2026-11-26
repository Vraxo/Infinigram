using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static class GenerationExperimentRunner
{
    private static readonly double[] Temperatures = [0.7, 1.0, 1.3];
    private static readonly double[] RepetitionPenalties = [1.0, 1.15, 1.5];
    private const int PromptLength = 48;
    private const int OutputLength = 96;
    private const int Seed = 1729;

    internal static void Run(string[] args)
    {
        if (args.Contains("--help", StringComparer.Ordinal) || args.Contains("-h", StringComparer.Ordinal))
        {
            Console.WriteLine("Usage: dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-generation [--output path]");
            return;
        }

        string outputPath = GetOutputPath(args);
        string corpusDirectory = Path.Combine(AppContext.BaseDirectory, "Resources", "Corpora");
        string[] corpusPaths = Directory.Exists(corpusDirectory)
            ? Directory.GetFiles(corpusDirectory, "*.txt", SearchOption.AllDirectories)
                .Order(StringComparer.OrdinalIgnoreCase).ToArray()
            : [];

        if (corpusPaths.Length == 0)
            throw new DirectoryNotFoundException($"No .txt corpora found in '{corpusDirectory}'.");

        var rows = new List<string>
        {
            "corpus,prompt,temperature,repetition_penalty,seed,generated_tokens,matched_ratio,longest_match,mean_log_probability,generated_text"
        };

        foreach (string corpusPath in corpusPaths)
        {
            byte[] corpus = File.ReadAllBytes(corpusPath);
            if (corpus.Length < PromptLength + OutputLength)
            {
                Console.Error.WriteLine($"Skipping {Path.GetFileName(corpusPath)}: corpus is too small.");
                continue;
            }

            using var model = new NGramModel(corpus);
            int[] promptStarts = [corpus.Length / 10, corpus.Length / 2, corpus.Length * 9 / 10];

            for (int promptIndex = 0; promptIndex < promptStarts.Length; promptIndex++)
            {
                int start = Math.Clamp(promptStarts[promptIndex], 0, corpus.Length - PromptLength - 1);
                int[] prompt = corpus.AsSpan(start, PromptLength).ToArray().Select(value => (int)value).ToArray();
                int runSeed = Seed + promptIndex;

                foreach (double temperature in Temperatures)
                {
                    foreach (double penalty in RepetitionPenalties)
                    {
                        int[] generated = model.Generate(
                            prompt,
                            maxTokens: OutputLength,
                            temperature: temperature,
                            topK: 40,
                            rng: new Random(runSeed),
                            repetitionPenalty: penalty,
                            noveltyWindow: 256).ToArray();

                        GenerationEvaluation evaluation = GenerationMetrics.Evaluate(
                            model, prompt, generated, minimumSpan: 32);
                        string generatedText = model.Tokenizer.Decode(generated);

                        rows.Add(string.Join(",",
                            Csv(Path.GetFileName(corpusPath)),
                            promptIndex.ToString(CultureInfo.InvariantCulture),
                            temperature.ToString("0.##", CultureInfo.InvariantCulture),
                            penalty.ToString("0.##", CultureInfo.InvariantCulture),
                            runSeed.ToString(CultureInfo.InvariantCulture),
                            generated.Length.ToString(CultureInfo.InvariantCulture),
                            evaluation.Overlap.MatchedRatio.ToString("0.0000", CultureInfo.InvariantCulture),
                            evaluation.Overlap.LongestMatch.ToString(CultureInfo.InvariantCulture),
                            evaluation.MeanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                            Csv(generatedText)));
                    }
                }
            }

            Console.WriteLine($"Evaluated {Path.GetFileName(corpusPath)}.");
        }

        string fullOutputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath)!);
        File.WriteAllText(fullOutputPath, string.Join(Environment.NewLine, rows) + Environment.NewLine, new UTF8Encoding(false));
        Console.WriteLine($"Wrote {rows.Count - 1} experiment rows to {fullOutputPath}");
        Console.WriteLine("Compare matched_ratio/longest_match for copying and mean_log_probability for model-relative consistency; none is a human fluency score.");
    }

    private static string GetOutputPath(string[] args)
    {
        int index = Array.IndexOf(args, "--output");
        if (index < 0)
            return "generation-evaluation.csv";
        if (index + 1 >= args.Length || string.IsNullOrWhiteSpace(args[index + 1]))
            throw new ArgumentException("--output requires a file path.");
        return args[index + 1];
    }

    private static string Csv(string value) => "\"" + value.Replace("\"", "\"\"") + "\"";
}
