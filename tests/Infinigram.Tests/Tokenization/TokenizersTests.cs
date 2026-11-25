using AwesomeAssertions;

namespace Infinigram.Tests;

public class ByteTokenizerTests
{
    [Fact]
    public void VocabSizeIs256()
    {
        new ByteTokenizer().VocabSize.Should().Be(256);
    }

    [Fact]
    public void EncodeAscii()
    {
        new ByteTokenizer().Encode("Hello").Should().Equal(new[] { 72, 101, 108, 108, 111 });
    }

    [Fact]
    public void DecodeAscii()
    {
        new ByteTokenizer().Decode(new List<int> { 72, 101, 108, 108, 111 }).Should().Be("Hello");
    }

    [Theory]
    [InlineData("Hello")]
    [InlineData("café")]
    [InlineData("日本語")]
    [InlineData("🦊")]
    [InlineData("")]
    [InlineData("aaaaaaaaaa")]
    public void RoundTripUtf8(string text)
    {
        var tok = new ByteTokenizer();
        tok.Decode(tok.Encode(text)).Should().Be(text);
    }

    [Fact]
    public void RoundTripLongString()
    {
        var tok = new ByteTokenizer();
        var text = new string('a', 10000);
        tok.Decode(tok.Encode(text)).Should().Be(text);
    }

    [Fact]
    public void EncodeMultibyteCodepoint()
    {
        new ByteTokenizer().Encode("€").Should().Equal(new[] { 0xE2, 0x82, 0xAC });
    }

    [Fact]
    public void DecodeInvalidBytesReplaces()
    {
        new ByteTokenizer().Decode(new List<int> { 0x80 }).Should().Be("\uFFFD");
    }

    [Fact]
    public void DecodeTokenOutOfByteRangeThrows()
    {
        Action act = () => new ByteTokenizer().Decode(new List<int> { 256 });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ImplementsITokenizer()
    {
        typeof(ByteTokenizer).Should().BeAssignableTo<ITokenizer>();
    }

    [Fact]
    public void EqualityAndHash()
    {
        new ByteTokenizer().Should().Be(new ByteTokenizer());
        new ByteTokenizer().GetHashCode().Should().Be(new ByteTokenizer().GetHashCode());
    }

    [Fact]
    public void ToStringContainsTypeName()
    {
        new ByteTokenizer().ToString().Should().Contain("ByteTokenizer");
    }
}


public class BpeTokenizerTests
{
    [Fact]
    public void LearnsMergesThatCompressRepeatedText()
    {
        const string corpus = "abcabcabcabc";
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 259, minFrequency: 2);

        tokenizer.MergeCount.Should().BeGreaterThan(0);
        tokenizer.Encode(corpus).Count.Should().BeLessThan(corpus.Length);
        tokenizer.Decode(tokenizer.Encode(corpus)).Should().Be(corpus);
    }

    [Theory]
    [InlineData("café 世界 🦊")]
    [InlineData("unseen words and punctuation!?")]
    [InlineData("")]
    public void RoundTripsKnownAndUnseenUtf8(string text)
    {
        var tokenizer = BpeTokenizer.Train(
            "hello world hello world. hello there!", vocabSize: 300);

        tokenizer.Decode(tokenizer.Encode(text)).Should().Be(text);
    }

    [Fact]
    public void TrainingIsDeterministic()
    {
        const string corpus = "the quick brown fox jumps over the lazy dog. ";
        var first = BpeTokenizer.Train(corpus, vocabSize: 300);
        var second = BpeTokenizer.Train(corpus, vocabSize: 300);

        first.Should().Be(second);
        first.Encode(corpus).Should().Equal(second.Encode(corpus));
    }

    [Fact]
    public void VocabularyAlwaysIncludesEveryByte()
    {
        var tokenizer = BpeTokenizer.Train("abcabcabc", vocabSize: 256);

        tokenizer.VocabSize.Should().Be(256);
        tokenizer.Decode(Enumerable.Range(0, 256).ToArray())
            .Should().Be(Encoding.UTF8.GetString(Enumerable.Range(0, 256).Select(i => (byte)i).ToArray()));
    }

    [Fact]
    public void DecodeRejectsTokenOutsideVocabulary()
    {
        var tokenizer = BpeTokenizer.Train("abcabcabc", vocabSize: 256);

        Action act = () => tokenizer.Decode(new[] { 256 });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void TrainingRejectsInvalidArguments()
    {
        Action tooSmall = () => BpeTokenizer.Train("text", vocabSize: 255);
        Action empty = () => BpeTokenizer.Train("");
        Action invalidFrequency = () => BpeTokenizer.Train("text", minFrequency: 0);

        tooSmall.Should().Throw<ArgumentOutOfRangeException>();
        empty.Should().Throw<ArgumentException>();
        invalidFrequency.Should().Throw<ArgumentOutOfRangeException>();
    }
}
