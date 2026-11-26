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
    public void RareSpecificContextInterpolatesWithBroaderSuffixEvidence()
    {
        int[] corpus =
        [
            9, 1, 2, 3,
            8, 2, 4,
            7, 2, 4,
            6, 2, 4,
            5, 2, 4,
        ];
        using var model = new NGramModel(corpus);
        var probabilities = model.Predict(new[] { 1, 2 }, topK: 256)
            ?? throw new InvalidOperationException("Expected a prediction distribution.");

        probabilities[3].Should().BeGreaterThan(0.55);
        probabilities[4].Should().BeGreaterThan(0.30);
        Math.Abs(probabilities.Values.Sum() - 1.0).Should().BeLessThanOrEqualTo(0.01);
    }

    [Fact]
    public void PredictNoMatchFallsBackToUnigrams()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var probabilities = model.Predict(new List<int> { 99, 98 }, topK: 256);

        probabilities.Should().NotBeNull();
        probabilities!.Should().ContainKey(1);
        probabilities.Should().ContainKey(2);
        probabilities.Should().ContainKey(3);
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
    public void LogprobsUseUnigramFallbackWhenContextHasNoMatch()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        var logprobs = model.PredictLogprobs(new List<int> { 200, 201 }, topK: 256);

        logprobs.Should().NotBeNull();
        logprobs!.Should().ContainKey((int)'a');
        logprobs.Should().ContainKey((int)'b');
        logprobs.Should().ContainKey((int)'c');
        logprobs.Values.Should().OnlyContain(value => value <= 0);
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

public class EdgeCaseTests
{
    [Fact]
    public void SingleTokenCorpus()
    {
        using var model = new NGramModel(new List<int> { 1 });
        model.N.Should().Be(1);

        var probs = model.Predict(new List<int>());
        probs.Should().NotBeNull();
        probs.Should().NotBeEmpty();
    }

    [Fact]
    public void RepeatedPatternCorpusPredictsCommonToken()
    {
        using var model = new NGramModel(Enumerable.Repeat(new List<int> { 1, 2 }, 100).SelectMany(x => x).ToList());
        var probs = model.Predict(new List<int> { 1 });

        probs.Should().NotBeNull();
        probs!.Should().ContainKey(2);
        probs![2].Should().BeGreaterThanOrEqualTo(0.4);
    }

    [Fact]
    public void LongContext()
    {
        using var model = new NGramModel(TestData.Range(0, 100));
        var probs = model.Predict(TestData.Range(0, 50));

        probs.Should().NotBeNull();
        probs.Should().NotBeEmpty();
    }

    [Fact]
    public void ContextLongerThanCorpus()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var probs = model.Predict(TestData.Range(1, 10));

        (probs is null || probs.Count > 0).Should().BeTrue();
    }
}

public class IntegrationTests
{
    [Fact]
    public void WikipediaStyleCorpus()
    {
        const int the = 1, cat = 2, sat = 3, on = 4, mat = 5;
        var corpus = new List<int> { the, cat, sat, on, the, mat, the, cat, sat, on, the };
        using var model = new NGramModel(corpus);

        var probs = model.Predict(new List<int> { the, cat });

        probs.Should().NotBeNull();
        probs!.Should().ContainKey(sat);
        probs![sat].Should().BeGreaterThanOrEqualTo(0.15);
    }

    [Fact]
    public void CodeCompletionScenario()
    {
        const int DEF = 1, FOO = 2, LP = 3, RP = 4, COLON = 5, RETURN = 6, NUM = 7;
        var corpus = new List<int>
        {
            DEF, FOO, LP, RP, COLON, RETURN, NUM,
            DEF, FOO, LP, RP, COLON, RETURN,
        };
        using var model = new NGramModel(corpus);

        var probs = model.Predict(new List<int> { DEF, FOO, LP, RP, COLON, RETURN });

        probs.Should().NotBeNull();
        probs!.Should().ContainKey(NUM);
    }

    [Fact]
    public void UpdateAndQueryCycle()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });

        for (int i = 0; i < 10; i++)
        {
            var probs = model.Predict(new List<int> { i % 3 + 1 });
            probs.Should().NotBeNull();
            probs!.Should().NotBeEmpty();

            model.Update(new List<int> { i % 3 + 1, (i + 1) % 3 + 1 });
        }

        model.Predict(new List<int> { 1 }).Should().NotBeNull();
    }
}

public class GetTokenTests
{
    [Fact]
    public void ValidPositions()
    {
        using var model = new NGramModel(new List<int> { 65, 66, 67 });
        model.GetToken(0).Should().Be(65);
        model.GetToken(1).Should().Be(66);
        model.GetToken(2).Should().Be(67);
    }

    [Fact]
    public void OutOfBoundsPositive()
    {
        using var model = new NGramModel(new List<int> { 65, 66, 67 });
        Action act = () => model.GetToken(10);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*out of range*");
    }

    [Fact]
    public void OutOfBoundsNegative()
    {
        using var model = new NGramModel(new List<int> { 65, 66, 67 });
        Action act = () => model.GetToken(-1);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*out of range*");
    }

    [Fact]
    public void AtBoundary()
    {
        using var model = new NGramModel(new List<int> { 65, 66, 67 });
        model.GetToken(2).Should().Be(67);
        Action act = () => model.GetToken(3);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

public class GenerateTests
{
    [Fact]
    public void ReturnsTokenIds()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the cat sat on the mat ", 50))).ToArray());
        var result = model.Generate(TestData.B("the "), maxTokens: 10, temperature: 0.5, rng: new Random(123));

        result.Should().OnlyContain(t => t >= 0 && t <= 255);
    }

    [Fact]
    public void RespectsMaxTokens()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("abcabcabc", 100))).ToArray());
        var result = model.Generate(TestData.B("a"), maxTokens: 5);
        result.Count.Should().BeLessThanOrEqualTo(5);
    }

    [Fact]
    public void EmptyContextGenerates()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("hello world ", 50))).ToArray());
        var result = model.Generate(maxTokens: 10, temperature: 0);
        result.Should().NotBeEmpty();
    }

    [Fact]
    public void GreedyIsDeterministic()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the cat sat on the mat ", 100))).ToArray());
        var first = model.Generate(TestData.B("the cat "), maxTokens: 10, temperature: 0);
        var second = model.Generate(TestData.B("the cat "), maxTokens: 10, temperature: 0);
        first.Should().Equal(second);
    }

    [Fact]
    public void StopSequenceIsStripped()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("hello world ", 100))).ToArray());
        var stop = TestData.B(" world");
        var result = model.Generate(TestData.B("hello"), maxTokens: 50, temperature: 0,
            stop: new List<IReadOnlyList<int>> { stop });

        int stopLength = stop.Count;
        bool endsWithStop = result.Count >= stopLength &&
            result.GetRange(result.Count - stopLength, stopLength).SequenceEqual(stop);
        endsWithStop.Should().BeFalse();
    }

    [Fact]
    public void MultiTokenStopStripsOnlyMatchedSuffix()
    {
        using var model = new NGramModel(string.Concat(Enumerable.Repeat("abcdef", 10)));
        var result = model.Generate(TestData.B("a"), maxTokens: 10, temperature: 0,
            stop: new List<IReadOnlyList<int>> { TestData.B("cz"), TestData.B("cd") });

        result.Should().Equal(TestData.B("b"));
    }

    [Fact]
    public void TakesTokenizedInput()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the quick brown fox ", 50))).ToArray());
        var result = model.Generate(model.Tokenizer.Encode("the "), maxTokens: 10, temperature: 0);
        result.Should().NotBeEmpty();
    }

    [Fact]
    public void ZeroMaxTokensReturnsEmpty()
    {
        using var model = new NGramModel(TestData.B("hello").ToArray());
        model.Generate(TestData.B("h"), maxTokens: 0).Should().BeEmpty();
    }

    [Fact]
    public void GenerateTextDecodesToString()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the cat sat on the mat ", 50))).ToArray());
        var text = model.GenerateText("the ", maxTokens: 10, temperature: 0);
        text.Should().NotBeNull();
    }
}

public class ScoreTests
{
    [Fact]
    public void ReturnsNegativeForTypicalInput()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the cat sat on the mat ", 50))).ToArray());
        var result = model.Score(TestData.B("the cat"));
        double.IsFinite(result).Should().BeTrue();
    }

    [Fact]
    public void LogProbabilitiesAreNonPositive()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("abcabc", 100))).ToArray());
        model.Score(TestData.B("abc")).Should().BeLessThanOrEqualTo(0.0);
    }

    [Fact]
    public void EmptySequenceScoresZero()
    {
        using var model = new NGramModel(TestData.B("hello").ToArray());
        model.Score(new List<int>()).Should().Be(0.0);
    }

    [Fact]
    public void InDistributionScoresHigherThanRandom()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog ", 100))).ToArray());
        var inDistribution = model.Score(TestData.B("the quick brown"));
        var random = model.Score(Enumerable.Range(0, 15).ToList());
        inDistribution.Should().BeGreaterThan(random);
    }

    [Fact]
    public void TokenizedInputScoresNegative()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("hello world ", 50))).ToArray());
        model.Score(model.Tokenizer.Encode("hello")).Should().BeLessThan(0.0);
    }

    [Fact]
    public void LongerSequenceAccumulatesMoreLogProb()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("abcdefghijklmnop ", 100))).ToArray());
        var shortScore = model.Score(TestData.B("abc"));
        var longScore = model.Score(TestData.B("abcdefgh"));
        longScore.Should().BeLessThan(shortScore);
    }
}


public class ContinuationSamplingTests
{
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


public class ConfidenceAwareBackoffTests
{
    [Fact]
    public void SparseContextBacksOffToBetterSupportedShorterContext()
    {
        var corpus = new List<int> { 1, 2, 7 };
        for (int i = 0; i < 20; i++)
        {
            corpus.AddRange(new[] { 4, 2, 8 });
        }

        using var model = new NGramModel(corpus);
        var probabilities = model.Predict(new[] { 1, 2 }, topK: 256);

        probabilities.Should().NotBeNull();
        probabilities![8].Should().BeGreaterThan(probabilities[7]);
    }

    [Fact]
    public void StronglySupportedContextKeepsItsSpecificDistribution()
    {
        var corpus = new List<int>();
        for (int i = 0; i < 100; i++)
        {
            corpus.AddRange(new[] { 1, 2, 7 });
            corpus.AddRange(new[] { 4, 2, 8 });
        }

        using var model = new NGramModel(corpus);
        var probabilities = model.Predict(new[] { 1, 2 }, topK: 256);

        probabilities.Should().NotBeNull();
        probabilities!.Should().ContainKey(7);
        probabilities.Should().NotContainKey(8);
    }

    [Fact]
    public void SequenceScoringUsesConfidenceAwareBackoff()
    {
        var corpus = new List<int> { 1, 2, 7 };
        for (int i = 0; i < 20; i++)
        {
            corpus.AddRange(new[] { 4, 2, 8 });
        }

        using var model = new NGramModel(corpus);

        model.Score(new[] { 1, 2, 8 }).Should().BeGreaterThan(model.Score(new[] { 1, 2, 7 }));
    }
}


public class UnigramDistributionTests
{
    [Fact]
    public void EmptyContextReflectsCorpusTokenFrequencies()
    {
        using var model = new NGramModel(new[] { 1, 1, 1, 1, 2, 2, 3 });

        var probabilities = model.Predict(Array.Empty<int>(), topK: 256);

        probabilities.Should().NotBeNull();
        probabilities![1].Should().BeApproximately(4.0 / 7.0, 1e-10);
        probabilities[2].Should().BeApproximately(2.0 / 7.0, 1e-10);
        probabilities[3].Should().BeApproximately(1.0 / 7.0, 1e-10);
    }

    [Fact]
    public void UnseenContextFallsBackToUnigramDistribution()
    {
        using var model = new NGramModel(new[] { 1, 1, 1, 1, 2, 2, 3 });

        var probabilities = model.Predict(new[] { 99, 98 }, topK: 256);

        probabilities.Should().NotBeNull();
        probabilities![1].Should().BeGreaterThan(probabilities[2]);
        probabilities[2].Should().BeGreaterThan(probabilities[3]);
    }

    [Fact]
    public void UpdatingCorpusUpdatesUnigramDistribution()
    {
        using var model = new NGramModel(new[] { 1, 2, 2 });
        model.Update(new[] { 1, 1, 1 });

        var probabilities = model.Predict(Array.Empty<int>(), topK: 256);

        probabilities.Should().NotBeNull();
        probabilities![1].Should().BeApproximately(4.0 / 6.0, 1e-10);
        probabilities[2].Should().BeApproximately(2.0 / 6.0, 1e-10);
    }

    [Fact]
    public void SequenceScoringUsesUnigramProbabilityForFirstToken()
    {
        using var model = new NGramModel(new[] { 1, 1, 1, 1, 2, 2, 3 });

        model.Score(new[] { 1 }).Should().BeGreaterThan(model.Score(new[] { 3 }));
    }
}
