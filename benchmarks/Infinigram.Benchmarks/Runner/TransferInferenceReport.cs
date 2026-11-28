using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static partial class TransferInferenceRunner
{
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
            "Intervals are 95% percentile hierarchical bootstrap intervals. Each replicate resamples target books with replacement within target genre and draws source-book bootstrap weights once within source genre, reusing those weights across all selected targets. Self-transfer is excluded when source and target genres match. Contiguous target-text blocks are resampled separately for each selected target, with the same block draw paired across its source models.");
        report.AppendLine();
        report.AppendLine(
            "The primary genre-match contrast is the difference-in-differences: (nonfiction→nonfiction − fiction→nonfiction) − (nonfiction→fiction − fiction→fiction). It measures whether nonfiction sources gain more on nonfiction targets than on fiction targets, relative to fiction sources. A positive interval entirely above zero is evidence of a source-genre × target-genre interaction within this selected corpus set. It does not establish a general genre law: there are only three fiction and five nonfiction books, authors and historical periods are not independently balanced, and repeated blocks within a book are not independent books.");
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
            $"**Diagonal contrast (nonfiction→nonfiction minus fiction→fiction):** {FormatPercent(summary.SameGenreDifference.Value)} " +
            $"(95% interval {FormatInterval(summary.SameGenreDifference.Interval)})");
        report.AppendLine(
            $"**Nonfiction-source advantage on fiction targets (nonfiction→fiction minus fiction→fiction):** {FormatPercent(summary.NonfictionSourceAdvantageOnFictionTargets.Value)} " +
            $"(95% interval {FormatInterval(summary.NonfictionSourceAdvantageOnFictionTargets.Interval)})");
        report.AppendLine(
            $"**Nonfiction-source advantage on nonfiction targets (nonfiction→nonfiction minus fiction→nonfiction):** {FormatPercent(summary.NonfictionSourceAdvantageOnNonfictionTargets.Value)} " +
            $"(95% interval {FormatInterval(summary.NonfictionSourceAdvantageOnNonfictionTargets.Interval)})");
        report.AppendLine(
            $"**Genre-match interaction (difference-in-differences):** {FormatPercent(summary.GenreMatchInteraction.Value)} " +
            $"(95% interval {FormatInterval(summary.GenreMatchInteraction.Interval)})");
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
            $"95% CI {FormatInterval(summary.SameGenreDifference.Interval)}; " +
            $"genre-match interaction {FormatPercent(summary.GenreMatchInteraction.Value)}, " +
            $"95% CI {FormatInterval(summary.GenreMatchInteraction.Interval)}");
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

}
