using AwesomeAssertions;

namespace Infinigram.Tests;

public class NoveltyGenerationTests
{
    [Fact]
    public void RepetitionPenaltyPrefersUnseenContinuation()
    {
        using var model = new NGramModel("abcXabcXabcY");
        var context = model.Tokenizer.Encode("abcXabc").ToList();

        var ordinary = model.Generate(context, maxTokens: 1, temperature: 0);
        var novel = model.Generate(context, maxTokens: 1, temperature: 0, repetitionPenalty: 10);

        model.Tokenizer.Decode(ordinary).Should().Be("X");
        model.Tokenizer.Decode(novel).Should().Be("Y");
    }

    [Fact]
    public void DefaultPenaltyPreservesExistingGenerationBehavior()
    {
        using var model = new NGramModel("abcXabcXabcY");
        var context = model.Tokenizer.Encode("abcXabc").ToList();

        var defaultResult = model.Generate(context, maxTokens: 8, temperature: 0, rng: new Random(7));
        var explicitDefault = model.Generate(context, maxTokens: 8, temperature: 0,
            rng: new Random(7), repetitionPenalty: 1.15);

        explicitDefault.Should().Equal(defaultResult);
    }

    [Fact]
    public void NoveltyWindowCanDisableRepetitionPenalty()
    {
        using var model = new NGramModel("abcXabcXabcY");
        var context = model.Tokenizer.Encode("abcXabc").ToList();

        var result = model.Generate(context, maxTokens: 1, temperature: 0,
            repetitionPenalty: 10, noveltyWindow: 0);

        model.Tokenizer.Decode(result).Should().Be("X");
    }

    [Fact]
    public void GenerationDoesNotMutateCallerContext()
    {
        using var model = new NGramModel("abcXabcXabcY");
        var context = model.Tokenizer.Encode("abcXabc").ToList();
        var original = context.ToArray();

        model.Generate(context, maxTokens: 8, repetitionPenalty: 2);

        context.Should().Equal(original);
    }

    [Fact]
    public void RejectsInvalidNoveltySettings()
    {
        using var model = new NGramModel("some training text");

        Action invalidPenalty = () => model.Generate(repetitionPenalty: 0.9);
        Action infinitePenalty = () => model.Generate(repetitionPenalty: double.PositiveInfinity);
        Action negativeWindow = () => model.Generate(noveltyWindow: -1);
        Action negativeLength = () => model.Generate(maxTokens: -1);

        invalidPenalty.Should().Throw<ArgumentOutOfRangeException>();
        infinitePenalty.Should().Throw<ArgumentOutOfRangeException>();
        negativeWindow.Should().Throw<ArgumentOutOfRangeException>();
        negativeLength.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GenerateTextForwardsNoveltySettings()
    {
        using var model = new NGramModel("abcXabcXabcY");

        model.GenerateText("abcXabc", maxTokens: 1, temperature: 0,
            repetitionPenalty: 10).Should().Be("Y");
    }
}

public class GenerationMetricsTests
{
    [Fact]
    public void MeasuresLongExactCorpusReuse()
    {
        const string passage = "A distinct passage that should be recognized exactly.";
        using var model = new NGramModel("prefix " + passage + " suffix");
        var generated = model.Tokenizer.Encode(passage + " plus newly written words");

        var overlap = GenerationMetrics.MeasureOverlap(model, generated, minimumSpan: 16);

        overlap.GeneratedTokens.Should().Be(generated.Count);
        overlap.LongestMatch.Should().Be(model.Tokenizer.Encode(passage + " ").Count);
        overlap.MatchedTokens.Should().BeGreaterThanOrEqualTo(overlap.LongestMatch);
        overlap.MatchedRatio.Should().BeGreaterThan(0.5);
    }

    [Fact]
    public void IgnoresShortCommonFragments()
    {
        using var model = new NGramModel("the cat sat on the mat");
        var generated = model.Tokenizer.Encode("the");

        var overlap = GenerationMetrics.MeasureOverlap(model, generated, minimumSpan: 8);

        overlap.MatchedTokens.Should().Be(0);
        overlap.LongestMatch.Should().Be(0);
        overlap.MatchedRatio.Should().Be(0);
    }

    [Fact]
    public void ReturnsZeroWhenNoLongSpanExists()
    {
        using var model = new NGramModel("abcdefgh");
        var generated = model.Tokenizer.Encode("12345678");

        var overlap = GenerationMetrics.MeasureOverlap(model, generated, minimumSpan: 4);

        overlap.MatchedTokens.Should().Be(0);
        overlap.LongestMatch.Should().Be(0);
    }

    [Fact]
    public void HandlesEmptyOutput()
    {
        using var model = new NGramModel("training corpus");

        GenerationMetrics.MeasureOverlap(model, Array.Empty<int>())
            .Should().Be(new GenerationOverlap(0, 0, 0));
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        using var model = new NGramModel("training corpus");
        var tokens = model.Tokenizer.Encode("output");

        Action nullModel = () => GenerationMetrics.MeasureOverlap(null!, tokens);
        Action nullTokens = () => GenerationMetrics.MeasureOverlap(model, null!);
        Action invalidSpan = () => GenerationMetrics.MeasureOverlap(model, tokens, 0);

        nullModel.Should().Throw<ArgumentNullException>();
        nullTokens.Should().Throw<ArgumentNullException>();
        invalidSpan.Should().Throw<ArgumentOutOfRangeException>();
    }
}
