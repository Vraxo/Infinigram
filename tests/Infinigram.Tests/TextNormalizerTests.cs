using AwesomeAssertions;

namespace Infinigram.Tests;

public class TextNormalizerTests
{
    [Fact]
    public void DefaultsNormalizeUnicodeAndLineEndingsOnly()
    {
        string result = TextNormalizer.Normalize("E\u0301\r\n  Hi\rThere\n", new TextNormalizationOptions());
        result.Should().Be("É\n  Hi\nThere\n");
    }

    [Fact]
    public void OptionsCanBeDisabledOrCombined()
    {
        const string input = " E\u0301\r\n\tHELLO  ";
        var none = new TextNormalizationOptions { UnicodeNfc = false, NormalizeLineEndings = false };
        TextNormalizer.Normalize(input, none).Should().Be(input);

        var cleanup = new TextNormalizationOptions { Lowercase = true, CollapseWhitespace = true };
        TextNormalizer.Normalize(input, cleanup).Should().Be(" é hello ");
        TextNormalizer.Normalize(TextNormalizer.Normalize(input, cleanup), cleanup).Should().Be(" é hello ");
    }

    [Fact]
    public void NormalizingCorpusAndQueryMakesEquivalentTextMatch()
    {
        var options = new TextNormalizationOptions { Lowercase = true };
        using var model = new Infinigram(TextNormalizer.Normalize("CAFÉ\r\nCafe\u0301", options));
        var query = TextNormalizer.Normalize("CAFE\u0301", options);
        model.Count(model.Tokenizer.Encode(query)).Should().Be(2);
    }

    [Fact]
    public void NullArgumentsThrow()
    {
        Action nullText = () => TextNormalizer.Normalize(null!, new TextNormalizationOptions());
        Action nullOptions = () => TextNormalizer.Normalize("text", null!);
        nullText.Should().Throw<ArgumentNullException>();
        nullOptions.Should().Throw<ArgumentNullException>();
    }
}
