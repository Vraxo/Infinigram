using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static class PredictionEvaluationRunner
{
    private static readonly int[] ContextWindows = [0, 1, 3, 6, 12];

    private const double TrainingFraction = 0.8;
    private const int SampleCount = 512;
    private const int SizeMatchedSegmentCount = 16;

    internal static void Run(string[] args)
    {
        if (args.Contains("--help", StringComparer.Ordinal) ||
            args.Contains("-h", StringComparer.Ordinal))
        {
            Console.WriteLine(
                "Usage: dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-prediction [--output path]");
            return;
        }

        string outputPath = GetOutputPath(args);
        string corpusDirectory = Path.Combine(
            AppContext.BaseDirectory, "Resources", "Corpora");
        string[] corpusPaths = Directory.Exists(corpusDirectory)
            ? Directory.GetFiles(corpusDirectory, "*.txt", SearchOption.AllDirectories)
                .Order(StringComparer.OrdinalIgnoreCase)
                .ToArray()
            : [];

        if (corpusPaths.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"No .txt corpora found in '{corpusDirectory}'.");
        }

        var corpora = corpusPaths
            .Select(path => new CorpusData(
                Path.GetFileName(path),
                File.ReadAllText(path, Encoding.UTF8)))
            .Select(corpus => corpus with
            {
                Tokens = CorpusWordIndex.Tokenize(corpus.Text).ToArray()
            })
            .ToArray();

        var rows = new List<string>
        {
            "evaluation_mode,training_corpus,evaluation_corpus,training_tokens,evaluation_tokens,sampled_tokens,sampled_words,context_window_tokens,token_coverage,word_coverage,top1_accuracy,word_top1_accuracy,mean_log_probability,mean_word_log_probability,perplexity,word_perplexity"
        };

        EvaluateWithinCorpus(corpora, rows);
        EvaluateCrossCorpus(corpora, rows);
        EvaluateCombinedCorpus(corpora, rows);
        EvaluateSizeMatchedCrossCorpus(corpora, rows);
        EvaluateBalancedCrossCorpus(corpora, rows);

        string fullOutputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath)!);
        File.WriteAllText(
            fullOutputPath,
            string.Join(Environment.NewLine, rows) + Environment.NewLine,
            new UTF8Encoding(false));

        Console.WriteLine(
            $"Wrote {rows.Count - 1} prediction evaluation rows to {fullOutputPath}");
        Console.WriteLine(
            $"Each evaluation samples up to {SampleCount} evenly spaced next-token targets.");
        Console.WriteLine(
            "Within-corpus rows train on the first 80% and evaluate the final 20%; cross-corpus rows train on one full corpus and evaluate a different full corpus; combined-training rows add all other full corpora to the target corpus training split.");
        Console.WriteLine(
            "Predictions use teacher forcing. Contexts use word-index tokens, including punctuation. Log-probability and perplexity exclude out-of-vocabulary targets; coverage reports those separately.");
        Console.WriteLine(
            "Top-1 accuracy counts out-of-vocabulary targets as incorrect. The word metrics include lexical tokens only.");
    }

    private static void EvaluateWithinCorpus(
        IReadOnlyList<CorpusData> corpora,
        List<string> rows)
    {
        foreach (CorpusData corpus in corpora)
        {
            int trainingEnd = (int)(corpus.Text.Length * TrainingFraction);
            while (trainingEnd < corpus.Text.Length &&
                   !char.IsWhiteSpace(corpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }
            while (trainingEnd < corpus.Text.Length &&
                   char.IsWhiteSpace(corpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }

            if (trainingEnd <= 0 || trainingEnd >= corpus.Text.Length)
            {
                Console.Error.WriteLine(
                    $"Skipping {corpus.Name}: corpus is too small for a held-out split.");
                continue;
            }

            string trainingText = corpus.Text[..trainingEnd];
            string validationText = corpus.Text[trainingEnd..];
            string[] trainingTokens = CorpusWordIndex.Tokenize(trainingText).ToArray();
            string[] validationTokens = CorpusWordIndex.Tokenize(validationText).ToArray();
            if (trainingTokens.Length == 0 || validationTokens.Length == 0)
            {
                Console.Error.WriteLine(
                    $"Skipping {corpus.Name}: split produced no tokens.");
                continue;
            }

            using var model = new NGramModel(trainingText);
            var wordIndex = model.CreateWordIndex();
            var vocabulary = wordIndex.Vocabulary.ToHashSet(
                StringComparer.OrdinalIgnoreCase);

            Evaluate(
                rows,
                wordIndex,
                vocabulary,
                "within-corpus",
                corpus.Name,
                corpus.Name,
                trainingTokens.Length,
                trainingTokens,
                validationTokens);
        }
    }

    private static void EvaluateSizeMatchedCrossCorpus(
        IReadOnlyList<CorpusData> corpora,
        List<string> rows)
    {
        if (corpora.Count < 2)
        {
            return;
        }

        foreach (CorpusData evaluationCorpus in corpora)
        {
            int trainingEnd = (int)(evaluationCorpus.Text.Length * TrainingFraction);
            while (trainingEnd < evaluationCorpus.Text.Length &&
                   !char.IsWhiteSpace(evaluationCorpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }
            while (trainingEnd < evaluationCorpus.Text.Length &&
                   char.IsWhiteSpace(evaluationCorpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }

            if (trainingEnd <= 0 || trainingEnd >= evaluationCorpus.Text.Length)
            {
                continue;
            }

            int trainingBudget = CorpusWordIndex.Tokenize(
                evaluationCorpus.Text[..trainingEnd]).Count;
            if (trainingBudget == 0)
            {
                continue;
            }

            foreach (CorpusData trainingCorpus in corpora)
            {
                if (ReferenceEquals(trainingCorpus, evaluationCorpus) ||
                    trainingCorpus.Tokens.Length == 0)
                {
                    continue;
                }

                int sampledTokens = Math.Min(trainingBudget, trainingCorpus.Tokens.Length);
                string trainingText = BuildSizeMatchedTrainingText(
                    trainingCorpus.Tokens,
                    sampledTokens);
                int trainingTokenCount = CorpusWordIndex.Tokenize(trainingText).Count;

                using var model = new NGramModel(trainingText);
                var wordIndex = model.CreateWordIndex();
                var vocabulary = wordIndex.Vocabulary.ToHashSet(
                    StringComparer.OrdinalIgnoreCase);

                Evaluate(
                    rows,
                    wordIndex,
                    vocabulary,
                    "cross-corpus-size-matched",
                    trainingCorpus.Name,
                    evaluationCorpus.Name,
                    trainingTokenCount,
                    [],
                    evaluationCorpus.Tokens);
            }
        }
    }

    private static string BuildSizeMatchedTrainingText(
        IReadOnlyList<string> tokens,
        int tokenBudget)
    {
        if (tokenBudget >= tokens.Count)
        {
            return string.Join(" ", tokens);
        }

        int segmentCount = Math.Min(SizeMatchedSegmentCount, tokenBudget);
        int sampledTokenBudget = tokenBudget - segmentCount + 1;
        int baseSegmentLength = sampledTokenBudget / segmentCount;
        int extraTokens = sampledTokenBudget % segmentCount;
        var segments = new List<string>(segmentCount);
        int sourceRange = tokens.Count - sampledTokenBudget;

        int sampledBefore = 0;
        for (int segment = 0; segment < segmentCount; segment++)
        {
            int segmentLength = baseSegmentLength + (segment < extraTokens ? 1 : 0);
            int start = segmentCount == 1
                ? 0
                : (int)((long)segment * sourceRange / (segmentCount - 1))
                    + sampledBefore;

            segments.Add(string.Join(" ", tokens.Skip(start).Take(segmentLength)));
            sampledBefore += segmentLength;
        }

        return string.Join("\0", segments);
    }

    private static void EvaluateBalancedCrossCorpus(
        IReadOnlyList<CorpusData> corpora,
        List<string> rows)
    {
        if (corpora.Count < 2)
        {
            return;
        }

        int trainingBudget = corpora.Min(corpus => corpus.Tokens.Length);
        if (trainingBudget == 0)
        {
            return;
        }

        foreach (CorpusData trainingCorpus in corpora)
        {
            string trainingText = BuildSizeMatchedTrainingText(
                trainingCorpus.Tokens,
                trainingBudget);
            int trainingTokenCount = CorpusWordIndex.Tokenize(trainingText).Count;

            using var model = new NGramModel(trainingText);
            var wordIndex = model.CreateWordIndex();
            var vocabulary = wordIndex.Vocabulary.ToHashSet(
                StringComparer.OrdinalIgnoreCase);

            foreach (CorpusData evaluationCorpus in corpora)
            {
                if (ReferenceEquals(trainingCorpus, evaluationCorpus) ||
                    evaluationCorpus.Tokens.Length == 0)
                {
                    continue;
                }

                Evaluate(
                    rows,
                    wordIndex,
                    vocabulary,
                    "cross-corpus-balanced",
                    trainingCorpus.Name,
                    evaluationCorpus.Name,
                    trainingTokenCount,
                    [],
                    evaluationCorpus.Tokens);
            }
        }
    }

    private static void EvaluateCombinedCorpus(
        IReadOnlyList<CorpusData> corpora,
        List<string> rows)
    {
        if (corpora.Count < 2)
        {
            return;
        }

        foreach (CorpusData evaluationCorpus in corpora)
        {
            int trainingEnd = (int)(evaluationCorpus.Text.Length * TrainingFraction);
            while (trainingEnd < evaluationCorpus.Text.Length &&
                   !char.IsWhiteSpace(evaluationCorpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }
            while (trainingEnd < evaluationCorpus.Text.Length &&
                   char.IsWhiteSpace(evaluationCorpus.Text[trainingEnd]))
            {
                trainingEnd++;
            }

            if (trainingEnd <= 0 || trainingEnd >= evaluationCorpus.Text.Length)
            {
                continue;
            }

            string targetTrainingText = evaluationCorpus.Text[..trainingEnd];
            string validationText = evaluationCorpus.Text[trainingEnd..];
            string[] initialContext = CorpusWordIndex.Tokenize(targetTrainingText).ToArray();
            string[] validationTokens = CorpusWordIndex.Tokenize(validationText).ToArray();
            if (initialContext.Length == 0 || validationTokens.Length == 0)
            {
                continue;
            }

            CorpusData[] additionalCorpora = corpora
                .Where(corpus => !ReferenceEquals(corpus, evaluationCorpus) &&
                    corpus.Text.Length > 0)
                .ToArray();
            if (additionalCorpora.Length == 0)
            {
                continue;
            }

            var trainingTexts = new List<string> { targetTrainingText };
            trainingTexts.AddRange(additionalCorpora.Select(corpus => corpus.Text));
            string combinedTrainingText = string.Join("\0", trainingTexts);
            int trainingTokenCount = CorpusWordIndex.Tokenize(combinedTrainingText).Count;
            string trainingDescription =
                $"{evaluationCorpus.Name} (80%) + " +
                string.Join(" + ", additionalCorpora.Select(corpus => $"{corpus.Name} (full)"));

            using var model = new NGramModel(combinedTrainingText);
            var wordIndex = model.CreateWordIndex();
            var vocabulary = wordIndex.Vocabulary.ToHashSet(
                StringComparer.OrdinalIgnoreCase);

            Evaluate(
                rows,
                wordIndex,
                vocabulary,
                "combined-training",
                trainingDescription,
                evaluationCorpus.Name,
                trainingTokenCount,
                initialContext,
                validationTokens);
        }
    }

    private static void EvaluateCrossCorpus(
        IReadOnlyList<CorpusData> corpora,
        List<string> rows)
    {
        if (corpora.Count < 2)
        {
            return;
        }

        foreach (CorpusData trainingCorpus in corpora)
        {
            if (trainingCorpus.Text.Length == 0 || trainingCorpus.Tokens.Length == 0)
            {
                continue;
            }

            using var model = new NGramModel(trainingCorpus.Text);
            var wordIndex = model.CreateWordIndex();
            var vocabulary = wordIndex.Vocabulary.ToHashSet(
                StringComparer.OrdinalIgnoreCase);

            foreach (CorpusData evaluationCorpus in corpora)
            {
                if (ReferenceEquals(trainingCorpus, evaluationCorpus) ||
                    evaluationCorpus.Tokens.Length == 0)
                {
                    continue;
                }

                Evaluate(
                    rows,
                    wordIndex,
                    vocabulary,
                    "cross-corpus",
                    trainingCorpus.Name,
                    evaluationCorpus.Name,
                    trainingCorpus.Tokens.Length,
                    [],
                    evaluationCorpus.Tokens);
            }
        }
    }

    private static void Evaluate(
        List<string> rows,
        CorpusWordIndex wordIndex,
        HashSet<string> vocabulary,
        string evaluationMode,
        string trainingCorpus,
        string evaluationCorpus,
        int trainingTokenCount,
        IReadOnlyList<string> initialContext,
        IReadOnlyList<string> evaluationTokens)
    {
        int[] samplePositions = GetSamplePositions(evaluationTokens.Count);
        if (samplePositions.Length == 0)
        {
            return;
        }

        foreach (int contextWindow in ContextWindows)
        {
            var history = initialContext
                .Skip(Math.Max(0, initialContext.Count - contextWindow))
                .ToList();

            int correctTokens = 0;
            int coveredTokens = 0;
            int wordTargets = 0;
            int correctWords = 0;
            int coveredWords = 0;
            double totalLogProbability = 0.0;
            double totalWordLogProbability = 0.0;
            int nextSample = 0;

            for (int position = 0; position < evaluationTokens.Count; position++)
            {
                string target = evaluationTokens[position];
                if (nextSample < samplePositions.Length &&
                    position == samplePositions[nextSample])
                {
                    var predictions = wordIndex.PredictNext(
                        string.Join(" ", history),
                        contextWindow);
                    string bestPrediction = GetMostLikelyToken(predictions);
                    bool isWord = IsWordToken(target);
                    bool isCovered = vocabulary.Contains(target);
                    bool isCorrect = string.Equals(
                        bestPrediction, target, StringComparison.OrdinalIgnoreCase);

                    if (isCorrect)
                    {
                        correctTokens++;
                    }

                    if (isCovered)
                    {
                        coveredTokens++;
                        totalLogProbability += Math.Log(predictions[target]);
                    }

                    if (isWord)
                    {
                        wordTargets++;
                        if (isCorrect)
                        {
                            correctWords++;
                        }
                        if (isCovered)
                        {
                            coveredWords++;
                            totalWordLogProbability += Math.Log(predictions[target]);
                        }
                    }

                    nextSample++;
                }

                if (contextWindow > 0)
                {
                    history.Add(target);
                    if (history.Count > contextWindow)
                    {
                        history.RemoveAt(0);
                    }
                }
            }

            double tokenCoverage = Ratio(coveredTokens, samplePositions.Length);
            double wordCoverage = Ratio(coveredWords, wordTargets);
            double top1Accuracy = Ratio(correctTokens, samplePositions.Length);
            double wordTop1Accuracy = Ratio(correctWords, wordTargets);
            double meanLogProbability = coveredTokens == 0
                ? 0.0
                : totalLogProbability / coveredTokens;
            double meanWordLogProbability = coveredWords == 0
                ? 0.0
                : totalWordLogProbability / coveredWords;

            rows.Add(string.Join(",",
                Csv(evaluationMode),
                Csv(trainingCorpus),
                Csv(evaluationCorpus),
                trainingTokenCount.ToString(CultureInfo.InvariantCulture),
                evaluationTokens.Count.ToString(CultureInfo.InvariantCulture),
                samplePositions.Length.ToString(CultureInfo.InvariantCulture),
                wordTargets.ToString(CultureInfo.InvariantCulture),
                contextWindow.ToString(CultureInfo.InvariantCulture),
                tokenCoverage.ToString("0.0000", CultureInfo.InvariantCulture),
                wordCoverage.ToString("0.0000", CultureInfo.InvariantCulture),
                top1Accuracy.ToString("0.0000", CultureInfo.InvariantCulture),
                wordTop1Accuracy.ToString("0.0000", CultureInfo.InvariantCulture),
                meanLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                meanWordLogProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                Math.Exp(-meanLogProbability).ToString("0.0000", CultureInfo.InvariantCulture),
                Math.Exp(-meanWordLogProbability).ToString("0.0000", CultureInfo.InvariantCulture)));

            Console.WriteLine(
                $"{evaluationMode}: {trainingCorpus} -> {evaluationCorpus}, " +
                $"context={contextWindow}, top-1={top1Accuracy:P1}, " +
                $"word top-1={wordTop1Accuracy:P1}, coverage={tokenCoverage:P1}, " +
                $"mean log-probability={meanLogProbability:F3}");
        }
    }

    private static int[] GetSamplePositions(int tokenCount)
    {
        int count = Math.Min(SampleCount, tokenCount);
        if (count == 1)
        {
            return [0];
        }

        return Enumerable.Range(0, count)
            .Select(index => (int)((long)index * (tokenCount - 1) / (count - 1)))
            .ToArray();
    }

    private static string GetMostLikelyToken(
        IReadOnlyDictionary<string, double> predictions)
    {
        string bestToken = string.Empty;
        double bestProbability = double.NegativeInfinity;
        foreach (var (token, probability) in predictions)
        {
            if (probability > bestProbability)
            {
                bestToken = token;
                bestProbability = probability;
            }
        }

        return bestToken;
    }

    private static bool IsWordToken(string token) =>
        token.Any(character => char.IsLetterOrDigit(character) || character == '_');

    private static double Ratio(int numerator, int denominator) =>
        denominator == 0 ? 0.0 : numerator / (double)denominator;

    private static string GetOutputPath(string[] args)
    {
        int index = Array.IndexOf(args, "--output");
        if (index < 0)
        {
            return "prediction-evaluation.csv";
        }
        if (index + 1 >= args.Length || string.IsNullOrWhiteSpace(args[index + 1]))
        {
            throw new ArgumentException("--output requires a file path.");
        }

        return args[index + 1];
    }

    private static string Csv(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";

    private sealed record CorpusData(string Name, string Text)
    {
        internal string[] Tokens { get; init; } = [];
    }
}
