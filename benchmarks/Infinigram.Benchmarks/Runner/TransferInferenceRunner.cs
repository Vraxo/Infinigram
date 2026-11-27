using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static class TransferInferenceRunner
{
    private const int DefaultSampleCount = 2048;
    private const int DefaultBootstrapReplicates = 2000;
    private const int DefaultBlockCount = 32;
    private const int DefaultContextWindow = 3;
    private const double TrainingFraction = 0.8;

    internal static void Run(string[] args)
    {
        if (args.Contains("--help", StringComparer.Ordinal) ||
            args.Contains("-h", StringComparer.Ordinal))
        {
            Console.WriteLine(
                "Usage: dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-transfer-inference " +
                "[--sample-count N] [--bootstrap-replicates N] [--seed N] " +
                "[--context-window N] [--output path] [--report-output path]");
            return;
        }

        int sampleCount = GetIntArgument(args, "--sample-count", DefaultSampleCount, 1);
        int bootstrapReplicates = GetIntArgument(
            args, "--bootstrap-replicates", DefaultBootstrapReplicates, 100);
        int contextWindow = GetIntArgument(args, "--context-window", DefaultContextWindow, 0, 12);
        long seed = GetLongArgument(args, "--seed", 20261127);
        string csvPath = GetPathArgument(args, "--output", "transfer-inference.csv");
        string reportPath = GetPathArgument(args, "--report-output", "transfer-inference.md");

        CorpusData[] corpora = LoadCorpora();
        int blockCount = Math.Min(DefaultBlockCount, sampleCount);
        var samples = corpora.ToDictionary(
            corpus => corpus.Index,
            corpus => CreateTargetSample(corpus, sampleCount, blockCount, seed));

        var rows = new List<string>
        {
            "evaluation_mode,training_corpus,evaluation_corpus,block,sampled_words,correct_words,covered_words,word_top1_accuracy,word_coverage"
        };

        PairResult[,] balanced = EvaluateGloballyBalanced(
            corpora, samples, contextWindow, rows);
        PairResult[,] sizeMatched = EvaluateTargetSizeMatched(
            corpora, samples, contextWindow, rows);

        var balancedSummary = AnalyzeMode(
            balanced, corpora, blockCount, bootstrapReplicates,
            unchecked((long)((ulong)seed ^ 0x51A7UL)));
        var sizeMatchedSummary = AnalyzeMode(
            sizeMatched, corpora, blockCount, bootstrapReplicates,
            unchecked((long)((ulong)seed ^ 0xA13FUL)));

        WriteCsv(csvPath, rows);
        WriteText(reportPath, BuildReport(
            corpora, sampleCount, blockCount, bootstrapReplicates, seed,
            contextWindow, balancedSummary, sizeMatchedSummary));

        Console.WriteLine($"Wrote {rows.Count - 1} block-level rows to {Path.GetFullPath(csvPath)}");
        Console.WriteLine($"Wrote inference report to {Path.GetFullPath(reportPath)}");
        PrintSummary("Globally balanced", balancedSummary);
        PrintSummary("Target-size-matched", sizeMatchedSummary);
    }

    private static CorpusData[] LoadCorpora()
    {
        string directory = Path.Combine(
            AppContext.BaseDirectory, "Resources", "Corpora");
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException(
                $"No corpus directory found at '{directory}'.");
        }

        string[] paths = Directory.GetFiles(directory, "*.txt", SearchOption.AllDirectories)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (paths.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"No .txt corpora found in '{directory}'.");
        }

        var corpora = paths.Select(path =>
        {
            string fileName = Path.GetFileName(path);
            string text = File.ReadAllText(path, Encoding.UTF8);
            return new CorpusData(
                fileName,
                text,
                CorpusWordIndex.Tokenize(text).ToArray(),
                GetGroup(fileName));
        }).ToArray();

        int fictionCount = corpora.Count(corpus => corpus.Group == CorpusGroup.Fiction);
        int nonfictionCount = corpora.Count(corpus => corpus.Group == CorpusGroup.Nonfiction);
        if (fictionCount < 2 || nonfictionCount < 2)
        {
            throw new InvalidOperationException(
                "Transfer inference requires at least two fiction and two nonfiction corpora.");
        }

        return corpora.Select((corpus, index) => corpus with { Index = index }).ToArray();
    }

    private static CorpusGroup GetGroup(string fileName) =>
        fileName.ToLowerInvariant() switch
        {
            "war-and-peace.txt" or
            "pride-and-prejudice.txt" or
            "the-adventures-of-sherlock-holmes.txt" => CorpusGroup.Fiction,

            "on-the-origin-of-species.txt" or
            "the-steam-engine-explained-and-illustrated.txt" or
            "the-federalist-papers.txt" or
            "personal-memoirs-of-u-s-grant.txt" or
            "life-on-the-mississippi.txt" => CorpusGroup.Nonfiction,

            _ => throw new InvalidOperationException(
                $"Corpus '{fileName}' has no declared fiction/nonfiction classification. " +
                "Add it to GetGroup before running transfer inference.")
        };

    private static TargetSample CreateTargetSample(
        CorpusData corpus,
        int sampleCount,
        int blockCount,
        long seed)
    {
        var buckets = Enumerable.Range(0, blockCount)
            .Select(_ => new List<int>())
            .ToArray();

        for (int position = 0; position < corpus.Tokens.Length; position++)
        {
            if (!IsWordToken(corpus.Tokens[position]))
            {
                continue;
            }

            int block = Math.Min(
                blockCount - 1,
                (int)((long)position * blockCount / corpus.Tokens.Length));
            buckets[block].Add(position);
        }

        var random = new StableRandom(MixSeed(seed, corpus.FileName));
        var points = new List<SamplePoint>(sampleCount);
        int baseQuota = sampleCount / blockCount;
        int remainder = sampleCount % blockCount;

        for (int block = 0; block < blockCount; block++)
        {
            List<int> positions = buckets[block];
            int quota = Math.Min(
                positions.Count,
                baseQuota + (block < remainder ? 1 : 0));

            for (int index = 0; index < quota; index++)
            {
                int selected = index + random.NextInt(positions.Count - index);
                (positions[index], positions[selected]) =
                    (positions[selected], positions[index]);
                points.Add(new SamplePoint(positions[index], block));
            }
        }

        if (points.Count == 0)
        {
            throw new InvalidOperationException(
                $"Corpus '{corpus.FileName}' contains no lexical target positions.");
        }

        return new TargetSample(
            points.OrderBy(point => point.Position).ToArray(),
            blockCount);
    }

    private static PairResult[,] EvaluateGloballyBalanced(
        IReadOnlyList<CorpusData> corpora,
        IReadOnlyDictionary<int, TargetSample> samples,
        int contextWindow,
        List<string> rows)
    {
        int trainingBudget = corpora.Min(corpus => corpus.Tokens.Length);
        var results = new PairResult[corpora.Count, corpora.Count];

        foreach (CorpusData source in corpora)
        {
            string trainingText = BuildSizeMatchedTrainingText(
                source.Tokens, trainingBudget);
            using var model = new NGramModel(trainingText);
            CorpusWordIndex wordIndex = model.CreateWordIndex();
            var vocabulary = wordIndex.Vocabulary.ToHashSet(
                StringComparer.OrdinalIgnoreCase);

            foreach (CorpusData target in corpora)
            {
                if (source.Index == target.Index)
                {
                    continue;
                }

                PairResult result = EvaluatePair(
                    "cross-corpus-balanced", source, target,
                    wordIndex, vocabulary, samples[target.Index], contextWindow);
                results[source.Index, target.Index] = result;
                AddRows(rows, result);
            }
        }

        return results;
    }

    private static PairResult[,] EvaluateTargetSizeMatched(
        IReadOnlyList<CorpusData> corpora,
        IReadOnlyDictionary<int, TargetSample> samples,
        int contextWindow,
        List<string> rows)
    {
        var results = new PairResult[corpora.Count, corpora.Count];

        foreach (CorpusData target in corpora)
        {
            int trainingEnd = GetTrainingEnd(target.Text);
            int trainingBudget = CorpusWordIndex.Tokenize(
                target.Text[..trainingEnd]).Count;
            if (trainingBudget <= 0)
            {
                throw new InvalidOperationException(
                    $"Corpus '{target.FileName}' has no usable 80% training split.");
            }

            foreach (CorpusData source in corpora)
            {
                if (source.Index == target.Index)
                {
                    continue;
                }

                string trainingText = BuildSizeMatchedTrainingText(
                    source.Tokens, Math.Min(trainingBudget, source.Tokens.Length));
                using var model = new NGramModel(trainingText);
                CorpusWordIndex wordIndex = model.CreateWordIndex();
                var vocabulary = wordIndex.Vocabulary.ToHashSet(
                    StringComparer.OrdinalIgnoreCase);

                PairResult result = EvaluatePair(
                    "cross-corpus-size-matched", source, target,
                    wordIndex, vocabulary, samples[target.Index], contextWindow);
                results[source.Index, target.Index] = result;
                AddRows(rows, result);
            }
        }

        return results;
    }

    private static PairResult EvaluatePair(
        string mode,
        CorpusData source,
        CorpusData target,
        CorpusWordIndex wordIndex,
        HashSet<string> vocabulary,
        TargetSample sample,
        int contextWindow)
    {
        var blocks = Enumerable.Range(0, sample.BlockCount)
            .Select(_ => new BlockObservation())
            .ToArray();
        var history = new List<string>(contextWindow);
        int consumed = 0;

        foreach (SamplePoint point in sample.Points)
        {
            while (consumed < point.Position)
            {
                if (contextWindow > 0)
                {
                    history.Add(target.Tokens[consumed]);
                    if (history.Count > contextWindow)
                    {
                        history.RemoveAt(0);
                    }
                }
                consumed++;
            }

            string context = contextWindow == 0
                ? string.Empty
                : string.Join(" ", history);
            IReadOnlyDictionary<string, double> predictions =
                wordIndex.PredictNext(context, contextWindow);
            string bestPrediction = GetMostLikelyToken(predictions);
            string expected = target.Tokens[point.Position];
            bool covered = vocabulary.Contains(expected);
            bool correct = string.Equals(
                bestPrediction, expected, StringComparison.OrdinalIgnoreCase);

            BlockObservation observation = blocks[point.Block];
            observation.Total++;
            if (covered)
            {
                observation.Covered++;
            }
            if (correct)
            {
                observation.Correct++;
            }
        }

        return new PairResult(mode, source, target, blocks);
    }

    private static string BuildSizeMatchedTrainingText(
        IReadOnlyList<string> tokens,
        int tokenBudget)
    {
        if (tokenBudget >= tokens.Count)
        {
            return string.Join(" ", tokens);
        }

        int segmentCount = Math.Min(16, tokenBudget);
        int sampledTokenBudget = tokenBudget - segmentCount + 1;
        int baseSegmentLength = sampledTokenBudget / segmentCount;
        int extraTokens = sampledTokenBudget % segmentCount;
        var segments = new List<string>(segmentCount);
        int sourceRange = tokens.Count - sampledTokenBudget;
        int sampledBefore = 0;

        for (int segment = 0; segment < segmentCount; segment++)
        {
            int segmentLength = baseSegmentLength +
                (segment < extraTokens ? 1 : 0);
            int start = segmentCount == 1
                ? 0
                : (int)((long)segment * sourceRange / (segmentCount - 1))
                    + sampledBefore;

            segments.Add(string.Join(
                " ", tokens.Skip(start).Take(segmentLength)));
            sampledBefore += segmentLength;
        }

        return string.Join("\0", segments);
    }

    private static int GetTrainingEnd(string text)
    {
        int trainingEnd = (int)(text.Length * TrainingFraction);
        while (trainingEnd < text.Length && !char.IsWhiteSpace(text[trainingEnd]))
        {
            trainingEnd++;
        }
        while (trainingEnd < text.Length && char.IsWhiteSpace(text[trainingEnd]))
        {
            trainingEnd++;
        }

        return trainingEnd;
    }

    private static InferenceSummary AnalyzeMode(
        PairResult[,] results,
        IReadOnlyList<CorpusData> corpora,
        int blockCount,
        int bootstrapReplicates,
        long seed)
    {
        double fictionToFiction = GetPointEstimate(
            results, corpora, CorpusGroup.Fiction, CorpusGroup.Fiction);
        double fictionToNonfiction = GetPointEstimate(
            results, corpora, CorpusGroup.Fiction, CorpusGroup.Nonfiction);
        double nonfictionToFiction = GetPointEstimate(
            results, corpora, CorpusGroup.Nonfiction, CorpusGroup.Fiction);
        double nonfictionToNonfiction = GetPointEstimate(
            results, corpora, CorpusGroup.Nonfiction, CorpusGroup.Nonfiction);

        var random = new StableRandom(unchecked((ulong)seed));
        var ffSamples = new double[bootstrapReplicates];
        var fnSamples = new double[bootstrapReplicates];
        var nfSamples = new double[bootstrapReplicates];
        var nnSamples = new double[bootstrapReplicates];
        var differenceSamples = new double[bootstrapReplicates];

        for (int replicate = 0; replicate < bootstrapReplicates; replicate++)
        {
            ffSamples[replicate] = BootstrapCell(
                results, corpora, CorpusGroup.Fiction, CorpusGroup.Fiction,
                blockCount, random);
            fnSamples[replicate] = BootstrapCell(
                results, corpora, CorpusGroup.Fiction, CorpusGroup.Nonfiction,
                blockCount, random);
            nfSamples[replicate] = BootstrapCell(
                results, corpora, CorpusGroup.Nonfiction, CorpusGroup.Fiction,
                blockCount, random);
            nnSamples[replicate] = BootstrapCell(
                results, corpora, CorpusGroup.Nonfiction, CorpusGroup.Nonfiction,
                blockCount, random);
            differenceSamples[replicate] =
                nnSamples[replicate] - ffSamples[replicate];
        }

        return new InferenceSummary(
            new Estimate(fictionToFiction, GetInterval(ffSamples)),
            new Estimate(fictionToNonfiction, GetInterval(fnSamples)),
            new Estimate(nonfictionToFiction, GetInterval(nfSamples)),
            new Estimate(nonfictionToNonfiction, GetInterval(nnSamples)),
            new Estimate(
                nonfictionToNonfiction - fictionToFiction,
                GetInterval(differenceSamples)));
    }

    private static double GetPointEstimate(
        PairResult[,] results,
        IReadOnlyList<CorpusData> corpora,
        CorpusGroup sourceGroup,
        CorpusGroup targetGroup)
    {
        var scores = new List<double>();
        foreach (CorpusData target in corpora.Where(corpus => corpus.Group == targetGroup))
        {
            foreach (CorpusData source in corpora.Where(corpus => corpus.Group == sourceGroup))
            {
                if (source.Index != target.Index)
                {
                    scores.Add(results[source.Index, target.Index].Accuracy);
                }
            }
        }

        return scores.Average();
    }

    private static double BootstrapCell(
        PairResult[,] results,
        IReadOnlyList<CorpusData> corpora,
        CorpusGroup sourceGroup,
        CorpusGroup targetGroup,
        int blockCount,
        StableRandom random)
    {
        CorpusData[] sourceBooks = corpora
            .Where(corpus => corpus.Group == sourceGroup)
            .ToArray();
        CorpusData[] targetBooks = corpora
            .Where(corpus => corpus.Group == targetGroup)
            .ToArray();

        bool excludeSelf = sourceGroup == targetGroup;
        int sourceDrawCount = sourceBooks.Length - (excludeSelf ? 1 : 0);
        double targetTotal = 0.0;

        for (int targetDraw = 0; targetDraw < targetBooks.Length; targetDraw++)
        {
            CorpusData target = targetBooks[random.NextInt(targetBooks.Length)];
            CorpusData[] eligibleSources = excludeSelf
                ? sourceBooks.Where(source => source.Index != target.Index).ToArray()
                : sourceBooks;
            var selectedBlocks = new int[blockCount];
            for (int block = 0; block < blockCount; block++)
            {
                selectedBlocks[block] = random.NextInt(blockCount);
            }

            double sourceTotal = 0.0;
            for (int sourceDraw = 0; sourceDraw < sourceDrawCount; sourceDraw++)
            {
                CorpusData source = eligibleSources[
                    random.NextInt(eligibleSources.Length)];
                sourceTotal += GetBlockAccuracy(
                    results[source.Index, target.Index], selectedBlocks);
            }

            targetTotal += sourceTotal / sourceDrawCount;
        }

        return targetTotal / targetBooks.Length;
    }

    private static double GetBlockAccuracy(
        PairResult pair,
        IReadOnlyList<int> selectedBlocks)
    {
        double total = 0.0;
        int count = 0;
        foreach (int block in selectedBlocks)
        {
            BlockObservation observation = pair.Blocks[block];
            if (observation.Total == 0)
            {
                continue;
            }

            total += observation.Correct / (double)observation.Total;
            count++;
        }

        return count == 0 ? 0.0 : total / count;
    }

    private static ConfidenceInterval GetInterval(double[] values)
    {
        Array.Sort(values);
        return new ConfidenceInterval(
            Quantile(values, 0.025),
            Quantile(values, 0.975));
    }

    private static double Quantile(IReadOnlyList<double> sortedValues, double probability)
    {
        double index = (sortedValues.Count - 1) * probability;
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);
        double fraction = index - lower;
        return sortedValues[lower] +
            (sortedValues[upper] - sortedValues[lower]) * fraction;
    }

    private static void AddRows(List<string> rows, PairResult pair)
    {
        for (int block = 0; block < pair.Blocks.Length; block++)
        {
            BlockObservation observation = pair.Blocks[block];
            rows.Add(string.Join(",",
                Csv(pair.Mode),
                Csv(pair.Source.FileName),
                Csv(pair.Target.FileName),
                block.ToString(CultureInfo.InvariantCulture),
                observation.Total.ToString(CultureInfo.InvariantCulture),
                observation.Correct.ToString(CultureInfo.InvariantCulture),
                observation.Covered.ToString(CultureInfo.InvariantCulture),
                Ratio(observation.Correct, observation.Total).ToString(
                    "0.000000", CultureInfo.InvariantCulture),
                Ratio(observation.Covered, observation.Total).ToString(
                    "0.000000", CultureInfo.InvariantCulture)));
        }
    }

    private static string BuildReport(
        IReadOnlyList<CorpusData> corpora,
        int sampleCount,
        int blockCount,
        int bootstrapReplicates,
        long seed,
        int contextWindow,
        InferenceSummary balanced,
        InferenceSummary sizeMatched)
    {
        var report = new StringBuilder();
        report.AppendLine("# Repeated-sample transfer inference");
        report.AppendLine();
        report.AppendLine("## Evaluation configuration");
        report.AppendLine();
        report.AppendLine($"- **Corpora:** {corpora.Count} ({corpora.Count(corpus => corpus.Group == CorpusGroup.Fiction)} fiction; {corpora.Count(corpus => corpus.Group == CorpusGroup.Nonfiction)} nonfiction)");
        report.AppendLine($"- **Target sample:** up to {sampleCount:N0} randomly sampled lexical positions per target book, stratified across {blockCount} contiguous text blocks");
        report.AppendLine($"- **Context window:** {contextWindow} word-index tokens");
        report.AppendLine($"- **Bootstrap replicates:** {bootstrapReplicates:N0}");
        report.AppendLine($"- **Seed:** {seed.ToString(CultureInfo.InvariantCulture)} (SplitMix64 sampling)");
        report.AppendLine("- **Scoring:** lexical target top-1; punctuation remains eligible as the model's top prediction; out-of-vocabulary targets count as incorrect");
        report.AppendLine("- **Pairing:** every source model for a target is evaluated at the same sampled target positions");
        report.AppendLine();
        report.AppendLine("## Globally balanced transfer");
        AppendSummaryTable(report, balanced);
        report.AppendLine("## Target-size-matched transfer");
        AppendSummaryTable(report, sizeMatched);
        report.AppendLine("## Bootstrap interpretation");
        report.AppendLine();
        report.AppendLine(
            "Intervals are 95% percentile hierarchical bootstrap intervals. Each replicate resamples target books with replacement within target genre, resamples source books with replacement within source genre (excluding self-transfer for same-genre comparisons), and resamples contiguous target-text blocks with replacement. Block resampling is paired across source models for a selected target.");
        report.AppendLine();
        report.AppendLine(
            "The primary contrast is nonfiction→nonfiction minus fiction→fiction. A positive interval entirely above zero is consistent with a positive difference for this selected corpus set under this resampling design. It does not establish a general genre law: there are only three fiction and five nonfiction books, authors and historical periods are not independently balanced, and repeated blocks within a book are not independent books.");
        report.AppendLine();
        report.AppendLine(
            "The point estimates are averages over directed source/target book pairs. The bootstrap resamples book identities as clusters and target-text blocks within each selected book. The globally balanced mode gives every source model the same training-token budget; the target-size-matched mode caps each source to the target's 80% training-token budget.");
        report.AppendLine();
        report.AppendLine(
            "This is an inferential follow-up to the deterministic evenly spaced evaluation, not a change to Infinigram's production model. The raw CSV records per-pair block totals so the aggregates can be independently recomputed.");
        return report.ToString();
    }

    private static void AppendSummaryTable(StringBuilder report, InferenceSummary summary)
    {
        report.AppendLine();
        report.AppendLine("| Training source → evaluation text | Word top-1 | 95% bootstrap interval |");
        report.AppendLine("|---|---:|---:|");
        AppendEstimate(report, "Fiction → fiction", summary.FictionToFiction);
        AppendEstimate(report, "Fiction → nonfiction", summary.FictionToNonfiction);
        AppendEstimate(report, "Nonfiction → fiction", summary.NonfictionToFiction);
        AppendEstimate(report, "Nonfiction → nonfiction", summary.NonfictionToNonfiction);
        report.AppendLine();
        report.AppendLine(
            $"**Nonfiction→nonfiction minus fiction→fiction:** {FormatPercent(summary.SameGenreDifference.Value)} " +
            $"(95% interval {FormatInterval(summary.SameGenreDifference.Interval)})");
        report.AppendLine();
    }

    private static void AppendEstimate(
        StringBuilder report,
        string label,
        Estimate estimate)
    {
        report.AppendLine(
            $"| {label} | {FormatPercent(estimate.Value)} | {FormatInterval(estimate.Interval)} |");
    }

    private static void PrintSummary(string label, InferenceSummary summary)
    {
        Console.WriteLine(
            $"{label}: nonfiction→nonfiction {FormatPercent(summary.NonfictionToNonfiction.Value)}, " +
            $"95% CI {FormatInterval(summary.NonfictionToNonfiction.Interval)}; " +
            $"difference vs fiction→fiction {FormatPercent(summary.SameGenreDifference.Value)}, " +
            $"95% CI {FormatInterval(summary.SameGenreDifference.Interval)}");
    }

    private static string FormatPercent(double value) =>
        (value * 100.0).ToString("0.00", CultureInfo.InvariantCulture) + "%";

    private static string FormatInterval(ConfidenceInterval interval) =>
        $"[{FormatPercent(interval.Lower)}, {FormatPercent(interval.Upper)}]";

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

    private static ulong MixSeed(long seed, string value)
    {
        ulong hash = unchecked((ulong)seed) ^ 14695981039346656037UL;
        foreach (char character in value)
        {
            hash ^= character;
            hash *= 1099511628211UL;
        }
        return hash;
    }

    private static int GetIntArgument(
        string[] args, string name, int defaultValue, int minimum, int? maximum = null)
    {
        int index = Array.IndexOf(args, name);
        if (index < 0)
        {
            return defaultValue;
        }

        if (index + 1 >= args.Length ||
            !int.TryParse(
                args[index + 1],
                NumberStyles.None,
                CultureInfo.InvariantCulture,
                out int value) ||
            value < minimum ||
            (maximum is int max && value > max))
        {
            string range = maximum is int upper
                ? $"between {minimum} and {upper}"
                : $"at least {minimum}";
            throw new ArgumentException($"{name} requires an integer {range}.");
        }

        return value;
    }

    private static long GetLongArgument(string[] args, string name, long defaultValue)
    {
        int index = Array.IndexOf(args, name);
        if (index < 0)
        {
            return defaultValue;
        }

        if (index + 1 >= args.Length ||
            !long.TryParse(
                args[index + 1],
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out long value))
        {
            throw new ArgumentException($"{name} requires an integer seed.");
        }

        return value;
    }

    private static string GetPathArgument(
        string[] args, string name, string defaultValue)
    {
        int index = Array.IndexOf(args, name);
        if (index < 0)
        {
            return defaultValue;
        }

        if (index + 1 >= args.Length || string.IsNullOrWhiteSpace(args[index + 1]))
        {
            throw new ArgumentException($"{name} requires a file path.");
        }

        return args[index + 1];
    }

    private static string Csv(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";

    private static void WriteCsv(string path, IReadOnlyList<string> rows)
    {
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllText(
            fullPath,
            string.Join(Environment.NewLine, rows) + Environment.NewLine,
            new UTF8Encoding(false));
    }

    private static void WriteText(string path, string content)
    {
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllText(fullPath, content, new UTF8Encoding(false));
    }

    private sealed record CorpusData(
        string FileName,
        string Text,
        string[] Tokens,
        CorpusGroup Group)
    {
        internal int Index { get; init; }
    }

    private sealed record TargetSample(SamplePoint[] Points, int BlockCount);
    private sealed record SamplePoint(int Position, int Block);
    private sealed record PairResult(
        string Mode,
        CorpusData Source,
        CorpusData Target,
        BlockObservation[] Blocks)
    {
        internal double Accuracy =>
            Blocks.Sum(block => block.Correct) /
            (double)Math.Max(1, Blocks.Sum(block => block.Total));
    }

    private sealed class BlockObservation
    {
        internal int Total { get; set; }
        internal int Correct { get; set; }
        internal int Covered { get; set; }
    }

    private sealed record ConfidenceInterval(double Lower, double Upper);
    private sealed record Estimate(double Value, ConfidenceInterval Interval);
    private sealed record InferenceSummary(
        Estimate FictionToFiction,
        Estimate FictionToNonfiction,
        Estimate NonfictionToFiction,
        Estimate NonfictionToNonfiction,
        Estimate SameGenreDifference);

    private enum CorpusGroup
    {
        Fiction,
        Nonfiction
    }

    private sealed class StableRandom
    {
        private ulong _state;

        internal StableRandom(ulong seed)
        {
            _state = seed;
        }

        internal ulong NextUInt64()
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong value = _state;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
            return value ^ (value >> 31);
        }

        internal int NextInt(int exclusiveMaximum)
        {
            if (exclusiveMaximum <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum));
            }

            ulong bound = (uint)exclusiveMaximum;
            ulong threshold = unchecked(0UL - bound) % bound;
            ulong value;
            do
            {
                value = NextUInt64();
            }
            while (value < threshold);

            return (int)(value % bound);
        }
    }
}
