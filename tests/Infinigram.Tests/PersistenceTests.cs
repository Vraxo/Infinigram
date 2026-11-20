// Persistent model (Infinigram.Build/Load/ListModels) round-trip tests.
// Exercises the C# little-endian int32 format; Python's .npy indexes are not
// readable by this engine and are intentionally not tested here.
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

        // Simulate a Python-built (.npy, version 2) index marker.
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

        // Corrupt corpus.bin: header claims 3 ints, file holds fewer.
        File.WriteAllBytes(Path.Combine(modelPath, "corpus.bin"), new byte[] { 3, 0, 0, 0, 65, 66 });

        Action act = () => Infinigram.Load(modelPath);
        act.Should().Throw<InvalidDataException>();
    }
}
