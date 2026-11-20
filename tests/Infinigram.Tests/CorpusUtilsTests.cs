// Port of Python/tests/test_corpus_utils.py and Python/tests/test_utf8_and_documents.py.
using System.Text.Json;
using AwesomeAssertions;

namespace Infinigram.Tests;

public class BuildCorpusFromDocumentsTests
{
    [Fact]
    public void SingleDocument()
    {
        var result = CorpusUtils.BuildCorpusFromDocuments(new[] { "hello" });
        result.Should().Equal(TestData.B("hello"));
    }

    [Fact]
    public void MultipleDocumentsWithDefaultSeparator()
    {
        var result = CorpusUtils.BuildCorpusFromDocuments(new[] { "hello", "world" });
        result.Should().Equal(TestData.B("hello\n\nworld"));
    }

    [Fact]
    public void CustomSeparator()
    {
        var result = CorpusUtils.BuildCorpusFromDocuments(new[] { "a", "b", "c" }, separator: new byte[] { 0x00 });
        result.Should().Equal(new[] { 97, 0, 98, 0, 99 });
    }

    [Fact]
    public void EmptyDocumentListReturnsEmpty()
    {
        CorpusUtils.BuildCorpusFromDocuments(Array.Empty<string>()).Should().BeEmpty();
    }

    [Fact]
    public void NoSeparatorAfterLastDocument()
    {
        var docs = new[] { new List<int> { 1, 2 }, new List<int> { 3, 4 } };
        var result = CorpusUtils.BuildCorpusFromDocuments(docs, separator: new byte[] { 0x00 });
        result.Should().Equal(new[] { 1, 2, 0, 3, 4 });
    }

    [Fact]
    public void PreEncodedDocuments()
    {
        var docs = new[] { new List<int> { 1, 2, 3 }, new List<int> { 4, 5, 6 } };
        var result = CorpusUtils.BuildCorpusFromDocuments(docs, separator: new byte[] { 0x00 });
        result.Should().Equal(new[] { 1, 2, 3, 0, 4, 5, 6 });
    }

    [Fact]
    public void InvalidByteValueThrows()
    {
        var docs = new[] { new List<int> { 256, 1, 2 } }; // 256 is not a byte
        Action act = () => CorpusUtils.BuildCorpusFromDocuments(docs);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void TextDocumentsAreUtf8Encoded()
    {
        var result = CorpusUtils.BuildCorpusFromDocuments(new[] { "café" });
        result.Should().Equal(TestData.B("café"));
    }
}

public class IterJsonLinesTests
{
    private static string WriteTempJsonl(TempDir temp, string name, string contents)
    {
        var path = Path.Combine(temp.Path, name);
        File.WriteAllText(path, contents);
        return path;
    }

    [Fact]
    public void IterSimpleJsonl()
    {
        using var temp = new TempDir();
        var path = WriteTempJsonl(temp, "simple.jsonl",
            "{\"text\": \"doc1\"}\n{\"text\": \"doc2\"}\n{\"text\": \"doc3\"}\n");

        CorpusUtils.IterJsonLines(path).Should().Equal(new[] { "doc1", "doc2", "doc3" });
    }

    [Fact]
    public void IterCustomField()
    {
        using var temp = new TempDir();
        var path = WriteTempJsonl(temp, "custom.jsonl",
            "{\"content\": \"doc1\"}\n{\"content\": \"doc2\"}\n");

        CorpusUtils.IterJsonLines(path, textField: "content").Should().Equal(new[] { "doc1", "doc2" });
    }

    [Fact]
    public void IterSkipsMissingField()
    {
        using var temp = new TempDir();
        var path = WriteTempJsonl(temp, "missing.jsonl",
            "{\"text\": \"doc1\"}\n{\"other\": \"ignored\"}\n{\"text\": \"doc3\"}\n");

        CorpusUtils.IterJsonLines(path).Should().Equal(new[] { "doc1", "doc3" });
    }

    [Fact]
    public void IterSkipsBlankLines()
    {
        using var temp = new TempDir();
        var path = WriteTempJsonl(temp, "blank.jsonl",
            "{\"text\": \"doc1\"}\n\n{\"text\": \"doc2\"}\n");

        CorpusUtils.IterJsonLines(path).Should().Equal(new[] { "doc1", "doc2" });
    }
}

public class BuildCorpusFromJsonLinesTests
{
    [Fact]
    public void BuildsCorpusWithSeparator()
    {
        using var temp = new TempDir();
        var path = Path.Combine(temp.Path, "corpus.jsonl");
        File.WriteAllText(path, "{\"text\": \"hello\"}\n{\"text\": \"world\"}\n");

        var corpus = CorpusUtils.BuildCorpusFromJsonLines(path, separator: new byte[] { 0x00 });
        corpus.Should().Equal(TestData.B("hello\0world"));
    }

    [Fact]
    public void HonorsMaxDocuments()
    {
        using var temp = new TempDir();
        var path = Path.Combine(temp.Path, "corpus.jsonl");
        File.WriteAllText(path, "{\"text\": \"doc1\"}\n{\"text\": \"doc2\"}\n{\"text\": \"doc3\"}\n");

        var corpus = CorpusUtils.BuildCorpusFromJsonLines(path, maxDocuments: 2, separator: new byte[] { 0x00 });
        corpus.Should().Equal(TestData.B("doc1\0doc2"));
    }
}

public class SaveCorpusToJsonLinesTests
{
    [Fact]
    public void SaveAndReloadRoundTrips()
    {
        using var temp = new TempDir();
        var path = Path.Combine(temp.Path, "roundtrip.jsonl");
        var docs = new[] { "hello", "world", "test" };

        CorpusUtils.SaveCorpusToJsonLines(docs, path);
        CorpusUtils.IterJsonLines(path).Should().Equal(docs);
    }

    [Fact]
    public void SaveCustomField()
    {
        using var temp = new TempDir();
        var path = Path.Combine(temp.Path, "custom.jsonl");

        CorpusUtils.SaveCorpusToJsonLines(new[] { "doc1", "doc2" }, path, textField: "content");

        using var doc = JsonDocument.Parse(File.ReadLines(path).First());
        doc.RootElement.TryGetProperty("content", out var field).Should().BeTrue();
        field.GetString().Should().Be("doc1");
    }

    [Fact]
    public void SaveUtf8Content()
    {
        using var temp = new TempDir();
        var path = Path.Combine(temp.Path, "utf8.jsonl");
        var docs = new[] { "café", "naïve" };

        CorpusUtils.SaveCorpusToJsonLines(docs, path);
        CorpusUtils.IterJsonLines(path).Should().Equal(docs);
    }
}

public class ByteTokenizerModelRoundTripTests
{
    // From Python/tests/test_utf8_and_documents.py — tokenizer round-trips through the model.

    [Fact]
    public void RoundTripAscii()
    {
        using var model = new Infinigram("The quick brown fox");
        model.Tokenizer.Decode(model.Tokenizer.Encode("The quick brown fox")).Should().Be("The quick brown fox");
    }

    [Fact]
    public void RoundTripUnicode()
    {
        using var model = new Infinigram("cafe 世界");
        model.Tokenizer.Decode(model.Tokenizer.Encode("cafe 世界")).Should().Be("cafe 世界");
    }

    [Fact]
    public void ByteModelGeneratesTokenIds()
    {
        using var model = new Infinigram(TestData.B("hello hello hello").ToArray());
        var output = model.Generate(TestData.B("hell"), maxTokens: 5, temperature: 0);
        output.Should().OnlyContain(t => t >= 0 && t <= 255);
        model.Tokenizer.Decode(output).Should().NotBeEmpty();
    }
}

public class DocumentSeparatorTests
{
    [Fact]
    public void TextDocumentsContainSeparator()
    {
        var corpus = CorpusUtils.BuildCorpusFromDocuments(new[] { "Hello world", "Goodbye world" });
        corpus.Should().OnlyContain(b => b >= 0 && b <= 255);

        var sep = TestData.B("\n\n");
        ContainsSubsequence(corpus, sep).Should().BeTrue();
    }

    [Fact]
    public void PreEncodedDocumentsWithNullSeparator()
    {
        var corpus = CorpusUtils.BuildCorpusFromDocuments(
            new[] { new List<int> { 1, 2, 3 }, new List<int> { 4, 5, 6 } },
            separator: new byte[] { 0x00 });
        corpus.Should().Equal(new[] { 1, 2, 3, 0, 4, 5, 6 });
    }

    [Fact]
    public void CustomStringSeparator()
    {
        var corpus = CorpusUtils.BuildCorpusFromDocuments(new[] { "doc1", "doc2" }, separator: TestData.B("<SEP>").Select(i => (byte)i).ToArray());
        ContainsSubsequence(corpus, TestData.B("<SEP>")).Should().BeTrue();
    }

    [Fact]
    public void SingleDocumentNoSeparator()
    {
        var corpus = CorpusUtils.BuildCorpusFromDocuments(new[] { "Hello world" });
        corpus.Should().Equal(TestData.B("Hello world"));
    }

    private static bool ContainsSubsequence(List<int> corpus, List<int> pattern)
    {
        for (int i = 0; i + pattern.Count <= corpus.Count; i++)
        {
            bool match = true;
            for (int j = 0; j < pattern.Count; j++)
            {
                if (corpus[i + j] != pattern[j])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                return true;
            }
        }
        return false;
    }
}
