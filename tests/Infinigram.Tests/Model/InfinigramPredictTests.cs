using AwesomeAssertions;

namespace Infinigram.Tests;

public class PredictTests
{
    [Fact]
    public void ReturnsProbabilities()
    {
        using var model = new NGramModel(Enumerable.Repeat(new List<int> { 1, 2, 3 }, 10).SelectMany(x => x).ToList());
        var probs = model.Predict(new List<int> { 1, 2 }, topK: 256);

        probs.Should().NotBeNull();
        probs!.Should().NotBeEmpty();
        probs.Values.Should().OnlyContain(p => p >= 0 && p <= 1);
        Math.Abs(probs!.Values.Sum() - 1.0).Should().BeLessThanOrEqualTo(0.01);
    }

    [Fact]
    public void TopKLimitsResults()
    {
        using var model = new NGramModel(Enumerable.Range(0, 100).SelectMany(i => Enumerable.Repeat(i, 5)).ToList());
        var probs = model.Predict(new List<int> { 0 }, topK: 5);
        probs!.Count.Should().BeLessThanOrEqualTo(5);
    }

    [Fact]
    public void SmoothingAssignsProbabilityToUnseenTokens()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 1, 2, 4 });
        var probs = model.Predict(new List<int> { 1, 2 }, topK: 256, smoothing: 0.1);

        probs.Should().NotBeNull();
        probs!.Should().ContainKey(5);
        probs![5].Should().BeGreaterThan(0);
    }

    [Fact]
    public void EmptyContextReturnsUnigramDistribution()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var probs = model.Predict(new List<int>(), topK: 256);

        probs.Should().NotBeNull();
        probs!.Should().NotBeEmpty();
        probs![1].Should().BeGreaterThan(0);
        probs![2].Should().BeGreaterThan(0);
        probs![3].Should().BeGreaterThan(0);
    }

    [Fact]
    public void PredictionsAreConsistentAcrossCalls()
    {
        using var model = new NGramModel(Enumerable.Repeat(new List<int> { 1, 2, 3 }, 5).SelectMany(x => x).ToList());
        var first = model.Predict(new List<int> { 1, 2 });
        var second = model.Predict(new List<int> { 1, 2 });

        first.Should().NotBeNull();
        second.Should().NotBeNull();
        first!.Should().BeEquivalentTo(second);
    }

    [Fact]
    public void PredictNoMatchReturnsNull()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        model.Predict(new List<int> { 99, 98 }).Should().BeNull();
    }

    [Fact]
    public void TopKReturnsCorrectItems()
    {
        using var model = new NGramModel(TestData.B("ab ab ab ab ac ac ad").ToArray());
        var probs = model.Predict(TestData.B("a"), topK: 2);

        probs.Should().NotBeNull();
        probs!.Count.Should().BeLessThanOrEqualTo(2);
        var topToken = probs!.OrderByDescending(kv => kv.Value).First().Key;
        topToken.Should().BeOneOf(new[] { (int)'b', (int)' ' });
    }

    [Fact]
    public void TopKReturnsCorrectCount()
    {
        using var model = new NGramModel(Enumerable.Range(0, 50).SelectMany(v => Enumerable.Repeat(v, 10)).ToList());

        foreach (var k in new[] { 1, 3, 5, 10 })
        {
            var probs = model.Predict(new List<int> { 0 }, topK: k);
            probs!.Count.Should().BeLessThanOrEqualTo(k);
        }
    }

    [Fact]
    public void ReturnsAllWhenFewerContinuationsThanTopK()
    {
        using var model = new NGramModel(TestData.B("abc abc abc").ToArray());
        var probs = model.Predict(TestData.B("ab"), topK: 256);

        probs.Should().NotBeNull();
        probs!.Should().NotBeEmpty();
        Math.Abs(probs!.Values.Sum() - 1.0).Should().BeLessThanOrEqualTo(0.01);
    }

    [Fact]
    public void PredictUsesLimitedContinuationsAndCompletes()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the ", 10000))).ToArray());
        var probs = model.Predict(TestData.B("the"), topK: 10);

        probs.Should().NotBeNull();
        probs.Should().ContainKey((int)' ');
    }

    [Fact]
    public void ContinuationLimitSamplesAcrossTheEntireSuffixRange()
    {
        const int occurrencesPerContinuation = NGramModel.DefaultContinuationLimit;
        var corpus = new int[occurrencesPerContinuation * 2 * 2];

        for (int i = 0; i < occurrencesPerContinuation; i++)
        {
            corpus[i * 2] = 0;
            corpus[i * 2 + 1] = 1;
        }

        for (int i = 0; i < occurrencesPerContinuation; i++)
        {
            int offset = (occurrencesPerContinuation + i) * 2;
            corpus[offset] = 0;
            corpus[offset + 1] = 2;
        }

        using var model = new NGramModel(corpus);
        var continuations = model.Continuations(new[] { 0 });

        continuations.Should().NotBeNull();
        continuations!.Should().ContainKey(1);
        continuations.Should().ContainKey(2);
        continuations[1].Should().Be(occurrencesPerContinuation / 2);
        continuations[2].Should().Be(occurrencesPerContinuation / 2);
        continuations.Values.Sum().Should().Be(NGramModel.DefaultContinuationLimit);
    }
}

public class PredictLogprobsTests
{
    [Fact]
    public void LogprobsAreNonPositive()
    {
        using var model = new NGramModel(TestData.B("the cat sat ").ToArray());
        var logprobs = model.PredictLogprobs(TestData.B("the"));

        logprobs.Should().NotBeNull();
        logprobs!.Values.Should().OnlyContain(lp => lp <= 0);
    }

    [Fact]
    public void LogprobsAreNullWhenNoMatch()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        model.PredictLogprobs(new List<int> { 200, 201 }).Should().BeNull();
    }
}

public class SearchLimitModelTests
{
    [Fact]
    public void SearchRespectsLimit()
    {
        using var model = new NGramModel(TestData.B("the cat sat. the cat ran. the cat jumped.").ToArray());

        model.Search(TestData.B("the")).Should().HaveCount(3);
        model.Search(TestData.B("the"), limit: 2).Should().HaveCount(2);
    }
}

public class UpdateTests
{
    [Fact]
    public void UpdateExtendsCorpus()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        int initialSize = model.N;

        model.Update(new List<int> { 4, 5, 6 });

        model.N.Should().Be(initialSize + 3);
        model.Count(new List<int> { 4, 5, 6 }).Should().Be(1);
    }

    [Fact]
    public void VocabularyStaysFixedAt256Bytes()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        model.Update(new List<int> { 4, 5 });

        model.VocabSize.Should().Be(256);
        model.MaxLength.Should().BeNull();
    }

    [Fact]
    public void UpdateEnablesNewPredictions()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var context = new List<int> { 2, 3 };

        _ = model.Continuations(context);
        model.Update(new List<int> { 2, 3, 4 });

        var after = model.Continuations(context);
        after.Should().NotBeNull();
        after!.Should().ContainKey(4);
    }

    [Fact]
    public void RepeatedUpdatesAccumulate()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        for (int i = 0; i < 20; i++)
        {
            model.Update(new List<int> { 65 + (i % 26) });
        }

        model.N.Should().Be(23);
        var probs = model.Predict(new List<int> { 1, 2, 3 });
        probs.Should().NotBeNull();
    }

    [Fact]
    public void UpdateWithStringEncodes()
    {
        using var model = new NGramModel("abc");
        model.Update(" def");

        model.N.Should().Be(7);
        model.Count(TestData.B(" def")).Should().Be(1);
    }
}
