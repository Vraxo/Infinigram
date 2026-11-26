using AwesomeAssertions;

namespace Infinigram.Tests;

public class CorpusWordIndexTests
{
    [Fact]
    public void BuildsVocabularyAndSequenceFromCorpus()
    {
        using var model = new NGramModel("The cat sat. The cat ran. The dog slept.");
        var words = model.CreateWordIndex();

        words.WordCount.Should().Be(12);
        words.VocabularySize.Should().Be(9);
        words.Vocabulary.Should().Contain("cat");
        words.Vocabulary.Should().Contain(".");
    }

    [Fact]
    public void ContinuationsCountObservedNextWords()
    {
        using var model = new NGramModel("The cat sat. The cat ran. The dog slept.");
        var words = model.CreateWordIndex();

        var continuations = words.Continuations("the cat");

        continuations.Should().NotBeNull();
        continuations!.Should().ContainKey("sat");
        continuations.Should().ContainKey("ran");
        continuations["sat"].Should().Be(1);
        continuations["ran"].Should().Be(1);
    }

    [Fact]
    public void ContextMatchingIsCaseInsensitive()
    {
        using var model = new NGramModel("The cat sat. The cat ran.");
        var words = model.CreateWordIndex();

        var continuations = words.Continuations("THE CAT");

        continuations.Should().NotBeNull();
        continuations!.Should().ContainKeys("sat", "ran");
    }

    [Fact]
    public void UnknownTrailingTokenDoesNotSkipBackToAnEarlierKnownWord()
    {
        using var model = new NGramModel("the cat sat. the dog ran.");
        var words = model.CreateWordIndex();

        words.Continuations("the unknown").Should().BeNull();
    }

    [Fact]
    public void GenerationUsesOnlyCorpusVocabulary()
    {
        using var model = new NGramModel("The cat sat. The cat ran.");
        var words = model.CreateWordIndex();

        string generated = words.GenerateText("The cat", maxTokens: 3, temperature: 0.0);

        generated.Should().StartWith("The cat ");
        CorpusWordIndex.Tokenize(generated)
            .Should().OnlyContain(token => words.Vocabulary.Contains(token, StringComparer.OrdinalIgnoreCase));
    }

    [Fact]
    public void EmptyContextUsesCorpusUnigrams()
    {
        using var model = new NGramModel("one two three.");
        var words = model.CreateWordIndex();

        string generated = words.GenerateText(maxTokens: 4, temperature: 0.0);

        CorpusWordIndex.Tokenize(generated)
            .Should().NotBeEmpty()
            .And.OnlyContain(token => words.Vocabulary.Contains(token, StringComparer.OrdinalIgnoreCase));
    }

    [Fact]
    public void IndexCreatedAfterUpdateIncludesAppendedCorpus()
    {
        using var model = new NGramModel("the cat sat.");
        var beforeUpdate = model.CreateWordIndex();

        model.Update(" the cat ran.");
        var afterUpdate = model.CreateWordIndex();

        beforeUpdate.Continuations("the cat")!["sat"].Should().Be(1);
        afterUpdate.Continuations("the cat")!.Should().ContainKey("ran");
    }

    [Fact]
    public void TokenizeKeepsWordsAndPunctuationSeparate()
    {
        CorpusWordIndex.Tokenize("Hello, world!").Should().Equal("Hello", ",", "world", "!");
        CorpusWordIndex.Tokenize("don't stop").Should().Equal("don't", "stop");
    }
}
