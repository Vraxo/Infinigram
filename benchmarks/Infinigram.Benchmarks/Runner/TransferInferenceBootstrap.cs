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

        var selectedTargets = new CorpusData[targetBooks.Length];
        for (int index = 0; index < selectedTargets.Length; index++)
        {
            selectedTargets[index] = targetBooks[random.NextInt(targetBooks.Length)];
        }

        // Draw source-book weights once per replicate so the same source-book
        // resample is shared across every target book in this comparison.
        var sourceWeights = new int[sourceBooks.Length];
        bool hasEligibleSource;
        do
        {
            Array.Clear(sourceWeights);
            for (int draw = 0; draw < sourceBooks.Length; draw++)
            {
                sourceWeights[random.NextInt(sourceBooks.Length)]++;
            }

            hasEligibleSource = true;
            foreach (CorpusData target in selectedTargets)
            {
                int eligibleWeight = 0;
                for (int sourceIndex = 0; sourceIndex < sourceBooks.Length; sourceIndex++)
                {
                    if (sourceBooks[sourceIndex].Index != target.Index)
                    {
                        eligibleWeight += sourceWeights[sourceIndex];
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

        double targetTotal = 0.0;
        foreach (CorpusData target in selectedTargets)
        {
            var selectedBlocks = new int[blockCount];
            for (int block = 0; block < blockCount; block++)
            {
                selectedBlocks[block] = random.NextInt(blockCount);
            }

            double sourceTotal = 0.0;
            int eligibleWeightTotal = 0;
            for (int sourceIndex = 0; sourceIndex < sourceBooks.Length; sourceIndex++)
            {
                CorpusData source = sourceBooks[sourceIndex];
                int weight = sourceWeights[sourceIndex];
                if (weight == 0 || source.Index == target.Index)
                {
                    continue;
                }

                sourceTotal += weight * GetBlockAccuracy(
                    results[source.Index, target.Index], selectedBlocks);
                eligibleWeightTotal += weight;
            }

            targetTotal += sourceTotal / eligibleWeightTotal;
        }

        return targetTotal / selectedTargets.Length;
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

}
