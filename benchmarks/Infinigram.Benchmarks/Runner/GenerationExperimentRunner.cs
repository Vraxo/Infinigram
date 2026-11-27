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
            "corpus,tokenizer,vocab_size,training_bytes,prompt,prompt_offset,temperature,repetition_penalty,seed,generated_tokens,metric_tokenizer,matched_ratio,longest_match,mean_log_probability,generated_text"
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
            const int promptCount = 24;
            int[] promptStarts = Enumerable.Range(0, promptCount)
                .Select(index => validationStart +
                    (int)((index + 0.5) * validationSpan / promptCount))
                .ToArray();

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
                                tokenizerName,
                                evaluation.Overlap.MatchedRatio.ToString("0.0000", CultureInfo.InvariantCulture),
                                evaluation.Overlap.LongestMatch.ToString(CultureInfo.InvariantCulture),
                                evaluation.MeanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                                Csv(generatedText)));
                        }
                    }
                }

                Console.WriteLine($"Evaluated {Path.GetFileName(corpusPath)} with {tokenizerName} ({trainingTokens.Length:N0} training tokens).");
            }

            // The word index is an additional view of the same training text.
            // Score its output with the byte model so overlap and likelihood use byte units.
            var byteTokenizer = new ByteTokenizer();
            using var byteModel = new NGramModel(trainingBytes, byteTokenizer);
            var wordIndex = byteModel.CreateWordIndex();

            for (int promptIndex = 0; promptIndex < promptStarts.Length; promptIndex++)
            {
                (string promptText, int promptStart) =
                    GetWordAlignedPrompt(corpus, promptStarts[promptIndex], PromptLength);
                int[] prompt = byteTokenizer.Encode(promptText).ToArray();
                int runSeed = Seed + promptIndex;

                foreach (double temperature in Temperatures)
                {
                    const double penalty = 1.0;
                    string fullText = wordIndex.GenerateText(
                        promptText,
                        maxTokens: 24,
                        temperature: temperature,
                        topK: 40,
                        rng: new Random(runSeed));

                    string generatedText = fullText.StartsWith(promptText, StringComparison.Ordinal)
                        ? fullText[promptText.Length..]
                        : string.Empty;
                    int[] generatedBytes = byteTokenizer.Encode(generatedText).ToArray();
                    GenerationEvaluation evaluation = GenerationMetrics.Evaluate(
                        byteModel, prompt, generatedBytes, minimumSpan: MinimumSpan);

                    rows.Add(string.Join(",",
                        Csv(Path.GetFileName(corpusPath)),
                        "word-index",
                        wordIndex.VocabularySize.ToString(CultureInfo.InvariantCulture),
                        trainingBytes.Length.ToString(CultureInfo.InvariantCulture),
                        promptIndex.ToString(CultureInfo.InvariantCulture),
                        (promptStart - validationStart).ToString(CultureInfo.InvariantCulture),
                        temperature.ToString("0.##", CultureInfo.InvariantCulture),
                        penalty.ToString("0.##", CultureInfo.InvariantCulture),
                        runSeed.ToString(CultureInfo.InvariantCulture),
                        CorpusWordIndex.Tokenize(generatedText).Count.ToString(CultureInfo.InvariantCulture),
                        "byte",
                        evaluation.Overlap.MatchedRatio.ToString("0.0000", CultureInfo.InvariantCulture),
                        evaluation.Overlap.LongestMatch.ToString(CultureInfo.InvariantCulture),
                        evaluation.MeanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                        Csv(generatedText)));

                    string structuredFullText = wordIndex.GenerateStructuredText(
                        promptText,
                        maxWords: 24,
                        temperature: temperature,
                        topK: 40,
                        rng: new Random(runSeed));
                    string structuredText = structuredFullText.StartsWith(promptText, StringComparison.Ordinal)
                        ? structuredFullText[promptText.Length..]
                        : string.Empty;
                    int[] structuredBytes = byteTokenizer.Encode(structuredText).ToArray();
                    GenerationEvaluation structuredEvaluation = GenerationMetrics.Evaluate(
                        byteModel, prompt, structuredBytes, minimumSpan: MinimumSpan);

                    rows.Add(string.Join(",",
                        Csv(Path.GetFileName(corpusPath)),
                        "word-phrase-structured",
                        wordIndex.VocabularySize.ToString(CultureInfo.InvariantCulture),
                        trainingBytes.Length.ToString(CultureInfo.InvariantCulture),
                        promptIndex.ToString(CultureInfo.InvariantCulture),
                        (promptStart - validationStart).ToString(CultureInfo.InvariantCulture),
                        temperature.ToString("0.##", CultureInfo.InvariantCulture),
                        penalty.ToString("0.##", CultureInfo.InvariantCulture),
                        runSeed.ToString(CultureInfo.InvariantCulture),
                        CorpusWordIndex.Tokenize(structuredText).Count.ToString(CultureInfo.InvariantCulture),
                        "byte",
                        structuredEvaluation.Overlap.MatchedRatio.ToString("0.0000", CultureInfo.InvariantCulture),
                        structuredEvaluation.Overlap.LongestMatch.ToString(CultureInfo.InvariantCulture),
                        structuredEvaluation.MeanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                        Csv(structuredText)));

                    string sentenceFullText = wordIndex.GenerateSentence(
                        promptText,
                        maxWords: 24,
                        temperature: temperature,
                        topK: 40,
                        rng: new Random(runSeed));
                    string sentenceText = sentenceFullText.StartsWith(promptText, StringComparison.Ordinal)
                        ? sentenceFullText[promptText.Length..]
                        : string.Empty;
                    int[] sentenceBytes = byteTokenizer.Encode(sentenceText).ToArray();
                    GenerationEvaluation sentenceEvaluation = GenerationMetrics.Evaluate(
                        byteModel, prompt, sentenceBytes, minimumSpan: MinimumSpan);

                    rows.Add(string.Join(",",
                        Csv(Path.GetFileName(corpusPath)),
                        "word-next-token-sentence",
                        wordIndex.VocabularySize.ToString(CultureInfo.InvariantCulture),
                        trainingBytes.Length.ToString(CultureInfo.InvariantCulture),
                        promptIndex.ToString(CultureInfo.InvariantCulture),
                        (promptStart - validationStart).ToString(CultureInfo.InvariantCulture),
                        temperature.ToString("0.##", CultureInfo.InvariantCulture),
                        penalty.ToString("0.##", CultureInfo.InvariantCulture),
                        runSeed.ToString(CultureInfo.InvariantCulture),
                        CorpusWordIndex.Tokenize(sentenceText).Count.ToString(CultureInfo.InvariantCulture),
                        "byte",
                        sentenceEvaluation.Overlap.MatchedRatio.ToString("0.0000", CultureInfo.InvariantCulture),
                        sentenceEvaluation.Overlap.LongestMatch.ToString(CultureInfo.InvariantCulture),
                        sentenceEvaluation.MeanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                        Csv(sentenceText)));
                }
            }

            Console.WriteLine(
                $"Evaluated {Path.GetFileName(corpusPath)} with word-index and phrase-structured generation " +
                $"({wordIndex.WordCount:N0} word tokens, {wordIndex.SentenceShapeCount:N0} observed shapes).");
        }

        string fullOutputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath)!);
        File.WriteAllText(fullOutputPath, string.Join(Environment.NewLine, rows) + Environment.NewLine, new UTF8Encoding(false));
        Console.WriteLine($"Wrote {rows.Count - 1} experiment rows to {fullOutputPath}");
        Console.WriteLine("Prompts come from the held-out final 20%; overlap is measured only against the training partition.");
        Console.WriteLine("Mean log-probability is model-relative, not a human fluency score; compare rows with the same metric_tokenizer.");
        Console.WriteLine("Word-index rows use byte-level overlap and likelihood metrics; generated_tokens counts word/punctuation units.");
    }

    private static (string Text, int Start) GetWordAlignedPrompt(
        string corpus, int requestedStart, int desiredLength)
    {
        int start = Math.Clamp(requestedStart, 0, corpus.Length);
        while (start < corpus.Length && start > 0 && !char.IsWhiteSpace(corpus[start - 1]))
            start++;
        while (start < corpus.Length && char.IsWhiteSpace(corpus[start]))
            start++;

        int end = Math.Min(corpus.Length, start + desiredLength);
        while (end < corpus.Length && !char.IsWhiteSpace(corpus[end]))
            end++;

        return (corpus[start..end].TrimEnd(), start);
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
