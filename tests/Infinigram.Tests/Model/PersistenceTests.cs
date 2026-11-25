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
        using (var built = NGramModel.Build("the cat sat on the mat the cat ran", modelPath, verbose: false))
        {
            built.ModelPath.Should().NotBeNull();
        }

        using var loaded = NGramModel.Load(modelPath);
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
        using var built = NGramModel.Build("abcabc", modelPath, verbose: false);
        _ = built.Continuations(context);
        built.Update("dabc");

        using var loaded = NGramModel.Load(modelPath);
        loaded.Count(context).Should().Be(built.Count(context));
        loaded.LongestSuffix(context).Should().Be(built.LongestSuffix(context));
        loaded.Continuations(context).Should().BeEquivalentTo(built.Continuations(context));
        loaded.Predict(context).Should().BeEquivalentTo(built.Predict(context));
        loaded.GenerateText("abc", maxTokens: 5, temperature: 0)
            .Should().Be(built.GenerateText("abc", maxTokens: 5, temperature: 0));
        loaded.Score(TestData.B("abcd")).Should().Be(built.Score(TestData.B("abcd")));
    }

    [Fact]
    public void BuildFromFileAndLoadRoundTripWithCustomTokenizer()
    {
        string sourcePath = Path.Combine(_temp.Path, "corpus.txt");
        File.WriteAllText(sourcePath, "abcabc");
        string modelPath = _temp.SubDir("from-file");
        var tokenizer = new TestTokenizer(512);

        using (var built = NGramModel.BuildFromFile(sourcePath, modelPath, tokenizer, maxLength: 3, verbose: false))
        {
            built.ModelPath.Should().Be(Path.GetFullPath(modelPath));
            built.Count(tokenizer.Encode("abc")).Should().Be(2);
            built.Update("abc");
        }

        using var loaded = NGramModel.Load(modelPath, tokenizer);
        loaded.MaxLength.Should().Be(3);
        loaded.Count(tokenizer.Encode("abc")).Should().Be(3);
        loaded.N.Should().Be(9);
    }

    [Fact]
    public void LoadProducesSamePredictions()
    {
        var modelPath = _temp.SubDir("model");
        var context = TestData.B("the cat");

        using var inMemory = new NGramModel(TestData.B("the cat sat on the mat the cat ran on the mat"));
        using (NGramModel.Build(TestData.B("the cat sat on the mat the cat ran on the mat"), modelPath, verbose: false))
        {
        }

        using var loaded = NGramModel.Load(modelPath);
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
        using (NGramModel.Build("hello world hello", Path.Combine(modelPath, "greeting"), verbose: false))
        {
        }

        var models = NGramModel.ListModels(modelPath);
        models.Should().Contain("greeting");
    }

    [Fact]
    public void UpdateIsPersistedAcrossReload()
    {
        var modelPath = _temp.SubDir("model");
        using (var built = NGramModel.Build("hello", modelPath, verbose: false))
        {
            built.Update(TestData.B(" help").ToArray());
        }

        using var loaded = NGramModel.Load(modelPath);
        loaded.N.Should().Be("hello help".Length);
        loaded.Count(TestData.B("help")).Should().Be(1);
    }

    [Fact]
    public void MetadataAndMaxLengthRoundTripAfterUpdate()
    {
        var modelPath = _temp.SubDir("metadata");
        using (var model = NGramModel.Build("abc", modelPath, maxLength: 2, verbose: false))
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

        using var loaded = NGramModel.Load(modelPath);
        using var overridden = NGramModel.Load(modelPath, maxLength: 4);
        loaded.MaxLength.Should().Be(2);
        overridden.MaxLength.Should().Be(4);
        loaded.Count(TestData.B("de")).Should().Be(1);
    }

    [Fact]
    public void CorruptMetadataAndMismatchedTokenCountAreRejected()
    {
        var modelPath = _temp.SubDir("corrupt-metadata");
        using (NGramModel.Build("abc", modelPath, verbose: false)) { }
        string metadata = Path.Combine(modelPath, "meta.json");
        File.WriteAllText(metadata, "{");
        Action malformed = () => NGramModel.Load(modelPath);
        malformed.Should().Throw<InvalidDataException>();

        File.WriteAllText(metadata, "{\"version\":1,\"n\":4}");
        Action wrongLength = () => NGramModel.Load(modelPath);
        wrongLength.Should().Throw<InvalidDataException>().WithMessage("*meta.json says n=4*");
    }

    [Fact]
    public void CustomTokenizerMustBeSuppliedAndMatchVocabulary()
    {
        var modelPath = _temp.SubDir("custom");
        var tokenizer = new TestTokenizer(256);
        using (NGramModel.Build("abc", modelPath, tokenizer, verbose: false)) { }

        Action missing = () => NGramModel.Load(modelPath);
        missing.Should().Throw<InvalidOperationException>();
        Action mismatch = () => NGramModel.Load(modelPath, new TestTokenizer(512));
        mismatch.Should().Throw<InvalidDataException>();
        using var loaded = NGramModel.Load(modelPath, tokenizer);
        loaded.Tokenizer.Should().BeSameAs(tokenizer);
        loaded.Count(TestData.B("abc")).Should().Be(1);
    }

    [Fact]
    public void BpeTokenizerIsSavedAndRestoredWithTheModel()
    {
        string phrase = "the quick brown fox jumps over the lazy dog. ";
        string corpus = string.Concat(Enumerable.Repeat(phrase, 20));
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 320);
        tokenizer.MergeCount.Should().BeGreaterThan(0);

        string modelPath = _temp.SubDir("bpe-model");
        var context = tokenizer.Encode("the quick ");

        using var built = NGramModel.Build(corpus, modelPath, tokenizer, verbose: false);
        built.Update("unseen words 🦊");

        using var metadata = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(modelPath, "meta.json")));
        var tokenizerMetadata = metadata.RootElement.GetProperty("tokenizer");
        tokenizerMetadata.GetProperty("kind").GetString().Should().Be("bpe");
        tokenizerMetadata.GetProperty("merges").GetArrayLength().Should().Be(tokenizer.MergeCount);

        using var loaded = NGramModel.Load(modelPath);
        loaded.Tokenizer.Should().BeOfType<BpeTokenizer>();
        loaded.Tokenizer.Should().Be(tokenizer);
        loaded.N.Should().Be(built.N);
        loaded.Predict(context, topK: 16).Should().BeEquivalentTo(built.Predict(context, topK: 16));
        loaded.GenerateText("the quick ", maxTokens: 8, temperature: 0)
            .Should().Be(built.GenerateText("the quick ", maxTokens: 8, temperature: 0));
    }

    [Fact]
    public void LoadingBpeModelRejectsDifferentMergeRules()
    {
        string modelPath = _temp.SubDir("bpe-mismatch");
        var tokenizer = BpeTokenizer.Train("hello hello hello world world", vocabSize: 300);
        var different = BpeTokenizer.Train(new string('z', 200), vocabSize: 300);

        using (NGramModel.Build("hello hello hello world world", modelPath, tokenizer, verbose: false)) { }

        Action act = () => NGramModel.Load(modelPath, different);
        act.Should().Throw<ArgumentException>().WithMessage("*different BPE merges*");
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
        Action act = () => NGramModel.Load(Path.Combine(_temp.Path, "does-not-exist"));
        act.Should().Throw<FileNotFoundException>();
    }

    [Fact]
    public void LoadUnsupportedFormatVersionThrows()
    {
        var modelPath = _temp.SubDir("model");
        using (NGramModel.Build("hello world", modelPath, verbose: false))
        {
        }

        File.WriteAllText(Path.Combine(modelPath, "meta.json"), "{\"version\": 2, \"n\": 11}");

        Action act = () => NGramModel.Load(modelPath);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void LoadTruncatedIndexThrows()
    {
        var modelPath = _temp.SubDir("model");
        using (NGramModel.Build("hello world", modelPath, verbose: false))
        {
        }

        File.WriteAllBytes(Path.Combine(modelPath, "corpus.bin"), new byte[] { 3, 0, 0, 0, 65, 66 });

        Action act = () => NGramModel.Load(modelPath);
        act.Should().Throw<InvalidDataException>();
    }
}
