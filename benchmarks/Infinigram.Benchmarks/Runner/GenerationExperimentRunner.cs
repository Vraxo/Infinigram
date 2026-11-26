using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static class GenerationExperimentRunner
{
    private static readonly double[] Temperatures = [0.7, 1.0, 1.3];
    private static readonly double[] RepetitionPenalties = [1.0, 1.15, 1.5];
    private static readonly (string Name, int VocabularySize)[] Tokenizers =
    [
        ("byte", 256),
        ("bpe-1024", 1024),
        ("bpe-4096", 4096)
    ];

    private const int PromptLength = 48;
    private const int OutputLength = 96;
    private const int MinimumSpan = 32;
    private const int Seed = 1729;
    private const double TrainingFraction = 0.8;

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
            "corpus,tokenizer,vocab_size,training_bytes,prompt,prompt_offset,temperature,repetition_penalty,seed,generated_tokens,matched_ratio,longest_match,mean_log_probability,generated_text"
        };

        foreach (string corpusPath in corpusPaths)
        {
            string corpus = File.ReadAllText(corpusPath, Encoding.UTF8);
            int trainingCharacters = (int)(corpus.Length * TrainingFraction);
            int validationStart = trainingCharacters;
            int validationSpan = corpus.Length - validationStart - PromptLength - OutputLength;
            if (validationSpan < 0 || trainingCharacters == 0)
            {
                Console.Error.WriteLine($"Skipping {Path.GetFileName(corpusPath)}: corpus is too small for an 80/20 holdout.");
                continue;
            }

            string trainingText = corpus[..trainingCharacters];
            byte[] trainingBytes = Encoding.UTF8.GetBytes(trainingText);
            int[] promptStarts =
            [
                validationStart + validationSpan / 10,
                validationStart + validationSpan / 2,
                validationStart + validationSpan * 9 / 10
            ];

            foreach ((string tokenizerName, int vocabularySize) in Tokenizers)
            {
                ITokenizer tokenizer = tokenizerName == "byte"
                    ? new ByteTokenizer()
                    : BpeTokenizer.Train(trainingText, vocabularySize);

                int[] trainingTokens = tokenizer.Encode(trainingText).ToArray();
                using var model = new NGramModel(trainingTokens, tokenizer);

                double tokensPerByte = trainingTokens.Length / (double)trainingBytes.Length;
                int outputTokens = Math.Max(1, (int)Math.Round(OutputLength * tokensPerByte));
                int minimumSpanTokens = Math.Max(1, (int)Math.Round(MinimumSpan * tokensPerByte));

                for (int promptIndex = 0; promptIndex < promptStarts.Length; promptIndex++)
                {
                    int start = promptStarts[promptIndex];
                    string promptText = corpus.Substring(start, PromptLength);
                    int[] prompt = tokenizer.Encode(promptText).ToArray();
                    int runSeed = Seed + promptIndex;

                    foreach (double temperature in Temperatures)
                    {
                        foreach (double penalty in RepetitionPenalties)
                        {
                            int[] generated = model.Generate(
                                prompt,
                                maxTokens: outputTokens,
                                temperature: temperature,
                                topK: 40,
                                rng: new Random(runSeed),
                                repetitionPenalty: penalty,
                                noveltyWindow: 256).ToArray();

                            GenerationEvaluation evaluation = GenerationMetrics.Evaluate(
                                model, prompt, generated, minimumSpan: minimumSpanTokens);
                            string generatedText = tokenizer.Decode(generated);

                            rows.Add(string.Join(",",
                                Csv(Path.GetFileName(corpusPath)),
                                tokenizerName,
                                vocabularySize.ToString(CultureInfo.InvariantCulture),
                                trainingBytes.Length.ToString(CultureInfo.InvariantCulture),
                                promptIndex.ToString(CultureInfo.InvariantCulture),
                                (start - validationStart).ToString(CultureInfo.InvariantCulture),
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

                Console.WriteLine($"Evaluated {Path.GetFileName(corpusPath)} with {tokenizerName} ({trainingTokens.Length:N0} training tokens).");
            }
        }

        string fullOutputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath)!);
        File.WriteAllText(fullOutputPath, string.Join(Environment.NewLine, rows) + Environment.NewLine, new UTF8Encoding(false));
        Console.WriteLine($"Wrote {rows.Count - 1} experiment rows to {fullOutputPath}");
        Console.WriteLine("Prompts come from the held-out final 20%; overlap is measured only against the training partition.");
        Console.WriteLine("Mean log-probability is a model-relative consistency signal, not a human fluency score; compare it only within a tokenizer.");
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
