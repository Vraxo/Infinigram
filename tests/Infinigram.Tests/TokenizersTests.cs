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
