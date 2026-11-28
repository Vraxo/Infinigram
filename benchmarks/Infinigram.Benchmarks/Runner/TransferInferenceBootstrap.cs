using System.Globalization;
using System.Text;

namespace Infinigram.Benchmarks.Runner;

internal static partial class TransferInferenceRunner
{
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
        var diagonalSamples = new double[bootstrapReplicates];
        var fictionTargetAdvantageSamples = new double[bootstrapReplicates];
        var nonfictionTargetAdvantageSamples = new double[bootstrapReplicates];
        var interactionSamples = new double[bootstrapReplicates];

        for (int replicate = 0; replicate < bootstrapReplicates; replicate++)
        {
            BootstrapReplicate sample = ResampleAllCells(
                results, corpora, blockCount, random);

            ffSamples[replicate] = sample.FictionToFiction;
            fnSamples[replicate] = sample.FictionToNonfiction;
            nfSamples[replicate] = sample.NonfictionToFiction;
            nnSamples[replicate] = sample.NonfictionToNonfiction;
            diagonalSamples[replicate] =
                sample.NonfictionToNonfiction - sample.FictionToFiction;
            fictionTargetAdvantageSamples[replicate] =
                sample.NonfictionToFiction - sample.FictionToFiction;
            nonfictionTargetAdvantageSamples[replicate] =
                sample.NonfictionToNonfiction - sample.FictionToNonfiction;
            interactionSamples[replicate] =
                nonfictionTargetAdvantageSamples[replicate] -
                fictionTargetAdvantageSamples[replicate];
        }

        return new InferenceSummary(
            new Estimate(fictionToFiction, GetInterval(ffSamples)),
            new Estimate(fictionToNonfiction, GetInterval(fnSamples)),
            new Estimate(nonfictionToFiction, GetInterval(nfSamples)),
            new Estimate(nonfictionToNonfiction, GetInterval(nnSamples)),
            new Estimate(
                nonfictionToNonfiction - fictionToFiction,
                GetInterval(diagonalSamples)),
            new Estimate(
                nonfictionToFiction - fictionToFiction,
                GetInterval(fictionTargetAdvantageSamples)),
            new Estimate(
                nonfictionToNonfiction - fictionToNonfiction,
                GetInterval(nonfictionTargetAdvantageSamples)),
            new Estimate(
                (nonfictionToNonfiction - fictionToNonfiction) -
                (nonfictionToFiction - fictionToFiction),
                GetInterval(interactionSamples)));
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

    private static BootstrapReplicate ResampleAllCells(
        PairResult[,] results,
        IReadOnlyList<CorpusData> corpora,
        int blockCount,
        StableRandom random)
    {
        CorpusData[] fictionBooks = corpora
            .Where(corpus => corpus.Group == CorpusGroup.Fiction)
            .ToArray();
        CorpusData[] nonfictionBooks = corpora
            .Where(corpus => corpus.Group == CorpusGroup.Nonfiction)
            .ToArray();

        CorpusData[] selectedFictionTargets = ResampleBooks(fictionBooks, random);
        CorpusData[] selectedNonfictionTargets = ResampleBooks(nonfictionBooks, random);

        int[] fictionSourceWeights = ResampleSourceWeights(
            fictionBooks, selectedFictionTargets, random);
        int[] nonfictionSourceWeights = ResampleSourceWeights(
            nonfictionBooks, selectedNonfictionTargets, random);

        (double FictionSources, double NonfictionSources) fictionTargets =
            EvaluateResampledTargets(
                results,
                selectedFictionTargets,
                fictionBooks,
                fictionSourceWeights,
                nonfictionBooks,
                nonfictionSourceWeights,
                blockCount,
                random);

        (double FictionSources, double NonfictionSources) nonfictionTargets =
            EvaluateResampledTargets(
                results,
                selectedNonfictionTargets,
                fictionBooks,
                fictionSourceWeights,
                nonfictionBooks,
                nonfictionSourceWeights,
                blockCount,
                random);

        return new BootstrapReplicate(
            fictionTargets.FictionSources,
            nonfictionTargets.FictionSources,
            fictionTargets.NonfictionSources,
            nonfictionTargets.NonfictionSources);
    }

    private static CorpusData[] ResampleBooks(
        IReadOnlyList<CorpusData> books,
        StableRandom random)
    {
        var selected = new CorpusData[books.Count];
        for (int index = 0; index < selected.Length; index++)
        {
            selected[index] = books[random.NextInt(books.Count)];
        }

        return selected;
    }

    private static int[] ResampleSourceWeights(
        IReadOnlyList<CorpusData> sourceBooks,
        IReadOnlyList<CorpusData> sameGenreTargets,
        StableRandom random)
    {
        var weights = new int[sourceBooks.Count];
        bool hasEligibleSource;
        do
        {
            Array.Clear(weights);
            for (int draw = 0; draw < sourceBooks.Count; draw++)
            {
                weights[random.NextInt(sourceBooks.Count)]++;
            }

            hasEligibleSource = true;
            foreach (CorpusData target in sameGenreTargets)
            {
                int eligibleWeight = 0;
                for (int sourceIndex = 0; sourceIndex < sourceBooks.Count; sourceIndex++)
                {
                    if (sourceBooks[sourceIndex].Index != target.Index)
                    {
                        eligibleWeight += weights[sourceIndex];
                    }
                }

                if (eligibleWeight == 0)
                {
                    hasEligibleSource = false;
                    break;
                }
            }
        }
        while (!hasEligibleSource);

        return weights;
    }

    private static (double FictionSources, double NonfictionSources)
        EvaluateResampledTargets(
            PairResult[,] results,
            IReadOnlyList<CorpusData> selectedTargets,
            IReadOnlyList<CorpusData> fictionSources,
            IReadOnlyList<int> fictionSourceWeights,
            IReadOnlyList<CorpusData> nonfictionSources,
            IReadOnlyList<int> nonfictionSourceWeights,
            int blockCount,
            StableRandom random)
    {
        double fictionTotal = 0.0;
        double nonfictionTotal = 0.0;

        foreach (CorpusData target in selectedTargets)
        {
            var selectedBlocks = new int[blockCount];
            for (int block = 0; block < blockCount; block++)
            {
                selectedBlocks[block] = random.NextInt(blockCount);
            }

            fictionTotal += GetSourceMean(
                results, target, fictionSources, fictionSourceWeights, selectedBlocks);
            nonfictionTotal += GetSourceMean(
                results, target, nonfictionSources, nonfictionSourceWeights, selectedBlocks);
        }

        return (
            fictionTotal / selectedTargets.Count,
            nonfictionTotal / selectedTargets.Count);
    }

    private static double GetSourceMean(
        PairResult[,] results,
        CorpusData target,
        IReadOnlyList<CorpusData> sourceBooks,
        IReadOnlyList<int> sourceWeights,
        IReadOnlyList<int> selectedBlocks)
    {
        double total = 0.0;
        int weightTotal = 0;

        for (int sourceIndex = 0; sourceIndex < sourceBooks.Count; sourceIndex++)
        {
            CorpusData source = sourceBooks[sourceIndex];
            int weight = sourceWeights[sourceIndex];
            if (weight == 0 || source.Index == target.Index)
            {
                continue;
            }

            total += weight * GetBlockAccuracy(
                results[source.Index, target.Index], selectedBlocks);
            weightTotal += weight;
        }

        return weightTotal == 0 ? 0.0 : total / weightTotal;
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

    private sealed record BootstrapReplicate(
        double FictionToFiction,
        double FictionToNonfiction,
        double NonfictionToFiction,
        double NonfictionToNonfiction);
}
