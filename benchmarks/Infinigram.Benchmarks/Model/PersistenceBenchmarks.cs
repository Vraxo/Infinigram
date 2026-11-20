using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[MemoryDiagnoser]
public class PersistenceBenchmarks
{
    [Params(32768, 1048576)]
    public int CorpusBytes { get; set; }

    private byte[] _corpus = null!;
    private string _directory = null!;
    private string _savedPath = null!;
    private string _buildPath = null!;
    private string _updatePath = null!;
    private NGramModel? _updateModel;
    private readonly byte[] _append = " benchmark update"u8.ToArray();

    [GlobalSetup]
    public void Setup()
    {
        _corpus = CorpusData.Create(CorpusBytes);
        _directory = Path.Combine(Path.GetTempPath(), "infinigram-benchmarks", Guid.NewGuid().ToString("N"));
        _savedPath = Path.Combine(_directory, "saved");
        _buildPath = Path.Combine(_directory, "build");
        _updatePath = Path.Combine(_directory, "update");
        using var model = NGramModel.Build(_corpus, _savedPath, verbose: false);
    }

    [Benchmark]
    public int BuildAndSave()
    {
        using var model = NGramModel.Build(_corpus, _buildPath, verbose: false);
        return model.N;
    }

    [Benchmark]
    public int Load()
    {
        using var model = NGramModel.Load(_savedPath);
        return model.N;
    }

    [IterationSetup(Target = nameof(UpdateAndSave))]
    public void PrepareUpdate() => _updateModel = NGramModel.Build(_corpus, _updatePath, verbose: false);

    [Benchmark]
    public int UpdateAndSave()
    {
        _updateModel!.Update(_append);
        return _updateModel.N;
    }

    [IterationCleanup(Target = nameof(UpdateAndSave))]
    public void CleanupUpdate()
    {
        _updateModel?.Dispose();
        _updateModel = null;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _updateModel?.Dispose();
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
