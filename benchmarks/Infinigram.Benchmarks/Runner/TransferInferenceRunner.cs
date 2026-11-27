using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static partial class TransferInferenceRunner
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
