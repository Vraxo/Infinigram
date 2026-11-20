using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[MemoryDiagnoser]
public class BuildBenchmarks
{
    [Params(32768, 1048576)]
    public int CorpusBytes { get; set; }

    private byte[] _corpus = null!;

    [GlobalSetup]
    public void Setup() => _corpus = CorpusData.Create(CorpusBytes);

    [Benchmark]
    public int BuildInMemory()
    {
        using var model = new NGramModel(_corpus);
        return model.N;
    }
}
