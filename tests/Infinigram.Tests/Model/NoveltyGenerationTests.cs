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
