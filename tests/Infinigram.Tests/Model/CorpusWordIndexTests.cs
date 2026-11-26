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
        words.VocabularySize.Should().Be(7);
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
    public void GenerationInterpolatesLongAndShortCorpusContexts()
    {
        using var model = new NGramModel(
            "alpha beta gamma delta one. omega beta gamma delta two. zeta beta gamma delta three.");
        var words = model.CreateWordIndex();

        var generated = Enumerable.Range(1, 64)
            .Select(seed => words.GenerateText(
                "alpha beta gamma delta",
                maxTokens: 1,
                temperature: 1.0,
                topK: null,
                rng: new Random(seed)))
            .Select(text => CorpusWordIndex.Tokenize(text).Last())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        generated.Should().Contain("one");
        generated.Should().Contain("two");
        generated.Should().Contain("three");
    }

    [Fact]
    public void ExtractsDistinctSentenceAndClauseShapesFromCorpus()
    {
        using var model = new NGramModel("The cat sleeps. A dog runs, jumps, and plays!");
        var words = model.CreateWordIndex();

        words.SentenceCount.Should().Be(2);
        words.SentenceShapeCount.Should().Be(2);
    }

    [Fact]
    public void StructuredGenerationUsesOnlyObservedSentenceLayouts()
    {
        using var model = new NGramModel("The cat sleeps. A dog runs, jumps, and plays!");
        var words = model.CreateWordIndex();

        var generated = Enumerable.Range(1, 128)
            .Select(seed => words.GenerateStructuredText(
                maxWords: 10,
                temperature: 1.0,
                topK: null,
                rng: new Random(seed)))
            .Where(text => text.Length > 0)
            .ToArray();

        generated.Any(text => text.EndsWith(".", StringComparison.Ordinal)).Should().BeTrue();
        generated.Any(text => text.EndsWith("!", StringComparison.Ordinal)).Should().BeTrue();

        foreach (string text in generated)
        {
            var tokens = CorpusWordIndex.Tokenize(text);
            int wordCount = tokens.Count(token =>
                token.Any(character => char.IsLetterOrDigit(character) || character == '_'));
            int commaCount = tokens.Count(token => token == ",");

            if (text.EndsWith('.'))
            {
                wordCount.Should().Be(3);
                commaCount.Should().Be(0);
            }
            else
            {
                text.Should().EndWith("!");
                wordCount.Should().Be(6);
                commaCount.Should().Be(2);
            }

            tokens.Should().OnlyContain(token =>
                words.Vocabulary.Contains(token, StringComparer.OrdinalIgnoreCase));
        }
    }

    [Fact]
    public void StructuredGenerationPreservesPromptAndRejectsUnknownTrailingContext()
    {
        using var model = new NGramModel("the cat sleeps. a dog runs, jumps, and plays!");
        var words = model.CreateWordIndex();

        words.GenerateStructuredText("unseen context").Should().Be("unseen context");

        string generated = words.GenerateStructuredText(
            "the cat",
            maxWords: 10,
            temperature: 0.0,
            rng: new Random(42));

        generated.Should().StartWith("the cat");
        (generated.EndsWith(".", StringComparison.Ordinal) ||
         generated.EndsWith("!", StringComparison.Ordinal)).Should().BeTrue();
    }

    [Fact]
    public void TokenizeKeepsWordsAndPunctuationSeparate()
    {
        CorpusWordIndex.Tokenize("Hello, world!").Should().Equal("Hello", ",", "world", "!");
        CorpusWordIndex.Tokenize("don't stop").Should().Equal("don't", "stop");
    }
}
