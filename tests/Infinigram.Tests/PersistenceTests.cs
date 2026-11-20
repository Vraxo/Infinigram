using AwesomeAssertions;

namespace Infinigram.Tests;

public class PersistenceTests : IDisposable
{
    private readonly TempDir _temp = new();

    public void Dispose() => _temp.Dispose();

    [Fact]
    public void BuildThenLoadIsQueryable()
    {
        var modelPath = _temp.SubDir("model");
        using (var built = Infinigram.Build("the cat sat on the mat the cat ran", modelPath, verbose: false))
        {
            built.ModelPath.Should().NotBeNull();
        }

        using var loaded = Infinigram.Load(modelPath);
        loaded.N.Should().Be("the cat sat on the mat the cat ran".Length);
        loaded.ModelPath.Should().NotBeNull();
        loaded.Tokenizer.Should().BeOfType<ByteTokenizer>();

        loaded.Count(TestData.B("the")).Should().Be(3);
        loaded.Predict(TestData.B("the cat")).Should().NotBeNull();
    }

    [Fact]
    public void UpdatedModelReloadsWithMatchingInferenceResults()
    {
        var modelPath = _temp.SubDir("updated-model");
        var context = TestData.B("abc");
        using var built = Infinigram.Build("abcabc", modelPath, verbose: false);
        _ = built.Continuations(context);
        built.Update("dabc");

        using var loaded = Infinigram.Load(modelPath);
        loaded.Count(context).Should().Be(built.Count(context));
        loaded.LongestSuffix(context).Should().Be(built.LongestSuffix(context));
        loaded.Continuations(context).Should().BeEquivalentTo(built.Continuations(context));
        loaded.Predict(context).Should().BeEquivalentTo(built.Predict(context));
        loaded.GenerateText("abc", maxTokens: 5, temperature: 0)
            .Should().Be(built.GenerateText("abc", maxTokens: 5, temperature: 0));
        loaded.Score(TestData.B("abcd")).Should().Be(built.Score(TestData.B("abcd")));
    }

    [Fact]
    public void LoadProducesSamePredictions()
    {
        var modelPath = _temp.SubDir("model");
        var context = TestData.B("the cat");

        using var inMemory = new Infinigram(TestData.B("the cat sat on the mat the cat ran on the mat"));
        using (Infinigram.Build(TestData.B("the cat sat on the mat the cat ran on the mat"), modelPath, verbose: false))
        {
        }

        using var loaded = Infinigram.Load(modelPath);
        var expected = inMemory.Predict(context, topK: 256);
        var actual = loaded.Predict(context, topK: 256);

        expected.Should().NotBeNull();
        actual.Should().NotBeNull();
        actual.Should().BeEquivalentTo(expected);
    }

    [Fact]
    public void ListModelsFindsBuiltModel()
    {
        var modelPath = _temp.SubDir("listmodels");
        using (Infinigram.Build("hello world hello", Path.Combine(modelPath, "greeting"), verbose: false))
        {
        }

        var models = Infinigram.ListModels(modelPath);
        models.Should().Contain("greeting");
    }

    [Fact]
    public void UpdateIsPersistedAcrossReload()
    {
        var modelPath = _temp.SubDir("model");
        using (var built = Infinigram.Build("hello", modelPath, verbose: false))
        {
            built.Update(TestData.B(" help").ToArray());
        }

        using var loaded = Infinigram.Load(modelPath);
        loaded.N.Should().Be("hello help".Length);
        loaded.Count(TestData.B("help")).Should().Be(1);
    }

    [Fact]
    public void MetadataAndMaxLengthRoundTripAfterUpdate()
    {
        var modelPath = _temp.SubDir("metadata");
        using (var model = Infinigram.Build("abc", modelPath, maxLength: 2, verbose: false))
        {
            model.Update("de");
        }

        using var meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "meta.json")));
        meta.RootElement.GetProperty("version").GetInt32().Should().Be(1);
        meta.RootElement.GetProperty("engine").GetString().Should().Be("infinigram-csharp");
        meta.RootElement.GetProperty("n").GetInt32().Should().Be(5);
        meta.RootElement.GetProperty("vocab_size").GetInt32().Should().Be(256);
        meta.RootElement.GetProperty("max_length").GetInt32().Should().Be(2);
        meta.RootElement.GetProperty("tokenizer").GetProperty("kind").GetString().Should().Be("byte");

        using var loaded = Infinigram.Load(modelPath);
        using var overridden = Infinigram.Load(modelPath, maxLength: 4);
        loaded.MaxLength.Should().Be(2);
        overridden.MaxLength.Should().Be(4);
        loaded.Count(TestData.B("de")).Should().Be(1);
    }

    [Fact]
    public void CorruptMetadataAndMismatchedTokenCountAreRejected()
    {
        var modelPath = _temp.SubDir("corrupt-metadata");
        using (Infinigram.Build("abc", modelPath, verbose: false)) { }
        string metadata = Path.Combine(modelPath, "meta.json");
        File.WriteAllText(metadata, "{");
        Action malformed = () => Infinigram.Load(modelPath);
        malformed.Should().Throw<InvalidDataException>();

        File.WriteAllText(metadata, "{\"version\":1,\"n\":4}");
        Action wrongLength = () => Infinigram.Load(modelPath);
        wrongLength.Should().Throw<InvalidDataException>().WithMessage("*meta.json says n=4*");
    }

    [Fact]
    public void CustomTokenizerMustBeSuppliedAndMatchVocabulary()
    {
        var modelPath = _temp.SubDir("custom");
        var tokenizer = new TestTokenizer(256);
        using (Infinigram.Build("abc", modelPath, tokenizer, verbose: false)) { }

        Action missing = () => Infinigram.Load(modelPath);
        missing.Should().Throw<InvalidOperationException>();
        Action mismatch = () => Infinigram.Load(modelPath, new TestTokenizer(512));
        mismatch.Should().Throw<InvalidDataException>();
        using var loaded = Infinigram.Load(modelPath, tokenizer);
        loaded.Tokenizer.Should().BeSameAs(tokenizer);
        loaded.Count(TestData.B("abc")).Should().Be(1);
    }

    private sealed class TestTokenizer(int vocabularySize) : ITokenizer
    {
        private readonly ByteTokenizer _inner = new();
        public int VocabSize => vocabularySize;
        public IReadOnlyList<int> Encode(string text) => _inner.Encode(text);
        public string Decode(IReadOnlyList<int> tokens) => _inner.Decode(tokens);
    }

    [Fact]
    public void LoadMissingModelThrows()
    {
        Action act = () => Infinigram.Load(Path.Combine(_temp.Path, "does-not-exist"));
        act.Should().Throw<FileNotFoundException>();
    }

    [Fact]
    public void LoadUnsupportedFormatVersionThrows()
    {
        var modelPath = _temp.SubDir("model");
        using (Infinigram.Build("hello world", modelPath, verbose: false))
        {
        }

        File.WriteAllText(Path.Combine(modelPath, "meta.json"), "{\"version\": 2, \"n\": 11}");

        Action act = () => Infinigram.Load(modelPath);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void LoadTruncatedIndexThrows()
    {
        var modelPath = _temp.SubDir("model");
        using (Infinigram.Build("hello world", modelPath, verbose: false))
        {
        }

        File.WriteAllBytes(Path.Combine(modelPath, "corpus.bin"), new byte[] { 3, 0, 0, 0, 65, 66 });

        Action act = () => Infinigram.Load(modelPath);
        act.Should().Throw<InvalidDataException>();
    }
}
