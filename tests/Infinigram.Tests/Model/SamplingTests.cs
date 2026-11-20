using AwesomeAssertions;

namespace Infinigram.Tests;

public class SampleFromDistributionTests
{
    [Fact]
    public void EmptyDistributionReturnsZero()
    {
        Sampling.SampleFromDistribution(new Dictionary<int, double>()).Should().Be(0);
    }

    [Fact]
    public void GreedySamplingReturnsMax()
    {
        var probs = new Dictionary<int, double> { [65] = 0.3, [66] = 0.5, [67] = 0.2 };
        Sampling.SampleFromDistribution(probs, temperature: 0).Should().Be(66);
    }

    [Fact]
    public void StandardSamplingReturnsValidToken()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.2 };
        int result = Sampling.SampleFromDistribution(probs, temperature: 1.0, rng: new Random(123));
        probs.Should().ContainKey(result);
    }

    [Fact]
    public void HighTemperatureSamplingReturnsValidTokens()
    {
        var probs = new Dictionary<int, double> { [65] = 0.9, [66] = 0.1 };
        var rng = new Random(123);
        for (int i = 0; i < 100; i++)
        {
            probs.Should().ContainKey(Sampling.SampleFromDistribution(probs, temperature: 2.0, rng: rng));
        }
    }

    [Fact]
    public void LowTemperatureSamplingFavorsHighProbability()
    {
        var probs = new Dictionary<int, double> { [65] = 0.9, [66] = 0.05, [67] = 0.05 };
        var rng = new Random(123);
        int bCount = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Sampling.SampleFromDistribution(probs, temperature: 0.1, rng: rng) == 65)
            {
                bCount++;
            }
        }
        bCount.Should().BeGreaterThan(90);
    }

    [Fact]
    public void SingleTokenDistributionAlwaysReturnsThatToken()
    {
        var probs = new Dictionary<int, double> { [42] = 1.0 };
        Sampling.SampleFromDistribution(probs, temperature: 1.0, rng: new Random(123)).Should().Be(42);
    }
}

public class SampleMultipleTests
{
    [Fact]
    public void EmptyDistributionReturnsEmpty()
    {
        Sampling.SampleMultiple(new Dictionary<int, double>(), k: 5).Should().BeEmpty();
    }

    [Fact]
    public void ZeroSamplesReturnsEmpty()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.5 };
        Sampling.SampleMultiple(probs, k: 0).Should().BeEmpty();
    }

    [Fact]
    public void ReturnsKSamplesWithReplacement()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.2 };
        var result = Sampling.SampleMultiple(probs, k: 10, rng: new Random(123));
        result.Should().HaveCount(10);
        result.Should().OnlyContain(t => probs.ContainsKey(t));
    }

    [Fact]
    public void GreedyReturnsTopK()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.2 };
        var result = Sampling.SampleMultiple(probs, k: 2, temperature: 0);
        result.Should().Equal(new[] { 65, 66 });
    }

    [Fact]
    public void WithoutReplacementReturnsUnique()
    {
        var probs = new Dictionary<int, double> { [65] = 0.4, [66] = 0.3, [67] = 0.2, [68] = 0.1 };
        var result = Sampling.SampleMultiple(probs, k: 3, withReplacement: false, rng: new Random(123));
        result.Should().HaveCount(3);
        result.Distinct().Should().HaveCount(3);
        result.Should().OnlyContain(t => probs.ContainsKey(t));
    }

    [Fact]
    public void WithoutReplacementKExceedsVocab()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.5 };
        var result = Sampling.SampleMultiple(probs, k: 10, withReplacement: false, rng: new Random(123));
        result.OrderBy(t => t).Should().Equal(new[] { 65, 66 });
    }

    [Fact]
    public void AllZeroWeightsReturnsEmpty()
    {
        var probs = new Dictionary<int, double> { [65] = 0.0, [66] = 0.0 };
        Sampling.SampleMultiple(probs, k: 2, withReplacement: false, rng: new Random(123)).Should().BeEmpty();
        Sampling.SampleMultiple(probs, k: 2, withReplacement: true, rng: new Random(123)).Should().BeEmpty();
    }

    [Fact]
    public void WithoutReplacementMixedZeroStopsEarly()
    {
        var probs = new Dictionary<int, double> { [65] = 1.0, [66] = 0.0, [67] = 0.0 };
        var result = Sampling.SampleMultiple(probs, k: 3, withReplacement: false, rng: new Random(123));
        result.Should().Equal(new[] { 65 });
    }
}

public class TopKFilterTests
{
    [Fact]
    public void FiltersToKTokens()
    {
        var probs = new Dictionary<int, double> { [65] = 0.4, [66] = 0.3, [67] = 0.2, [68] = 0.1 };
        var result = Sampling.TopKFilter(probs, k: 2);
        result.Should().HaveCount(2);
        result.Should().ContainKey(65);
        result.Should().ContainKey(66);
    }

    [Fact]
    public void NormalizesToOne()
    {
        var probs = new Dictionary<int, double> { [65] = 0.4, [66] = 0.3, [67] = 0.2, [68] = 0.1 };
        var result = Sampling.TopKFilter(probs, k: 2);
        Math.Abs(result.Values.Sum() - 1.0).Should().BeLessThanOrEqualTo(1e-10);
    }

    [Fact]
    public void EmptyInputReturnsEmpty()
    {
        Sampling.TopKFilter(new Dictionary<int, double>(), k: 5).Should().BeEmpty();
    }

    [Fact]
    public void ZeroKReturnsEmpty()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.5 };
        Sampling.TopKFilter(probs, k: 0).Should().BeEmpty();
    }

    [Fact]
    public void KLargerThanVocabReturnsAll()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.5 };
        Sampling.TopKFilter(probs, k: 10).Should().HaveCount(2);
    }

    [Fact]
    public void AllZeroFallsBackToUniform()
    {
        var probs = new Dictionary<int, double> { [65] = 0.0, [66] = 0.0, [67] = 0.0 };
        var result = Sampling.TopKFilter(probs, k: 2);
        result.Should().HaveCount(2);
        result.Values.Should().OnlyContain(v => Math.Abs(v - 0.5) < 1e-10);
    }
}

public class TopPFilterTests
{
    [Fact]
    public void FiltersByCumulativeProbability()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.15, [68] = 0.05 };
        var result = Sampling.TopPFilter(probs, p: 0.8);
        result.Should().ContainKey(65);
        result.Should().ContainKey(66);
    }

    [Fact]
    public void NormalizesToOne()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.15, [68] = 0.05 };
        var result = Sampling.TopPFilter(probs, p: 0.8);
        Math.Abs(result.Values.Sum() - 1.0).Should().BeLessThanOrEqualTo(1e-10);
    }

    [Fact]
    public void POneReturnsAll()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.3, [67] = 0.2 };
        Sampling.TopPFilter(probs, p: 1.0).Should().HaveCount(3);
    }

    [Fact]
    public void PZeroReturnsEmpty()
    {
        var probs = new Dictionary<int, double> { [65] = 0.5, [66] = 0.5 };
        Sampling.TopPFilter(probs, p: 0).Should().BeEmpty();
    }

    [Fact]
    public void EmptyInputReturnsEmpty()
    {
        Sampling.TopPFilter(new Dictionary<int, double>(), p: 0.5).Should().BeEmpty();
    }
}
