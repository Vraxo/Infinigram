using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[MemoryDiagnoser]
public class PredictionBenchmarks
{
    [Params(32768, 1048576)]
    public int CorpusBytes { get; set; }

    [Params(8, 128)]
    public int ContextBytes { get; set; }

    [Params(5, 50)]
    public int TopK { get; set; }

    private NGramModel _model = null!;
    private int[] _context = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = CorpusData.Create(CorpusBytes);
        _model = new NGramModel(corpus);
        _context = CorpusData.Context(corpus, ContextBytes);
        if (_model.Predict(_context, topK: TopK) is null)
            throw new InvalidOperationException("Expected context to have continuations.");
    }

    [Benchmark]
    public IReadOnlyDictionary<int, double>? WarmCache() => _model.Predict(_context, topK: TopK);

    [Benchmark]
    public IReadOnlyDictionary<int, double>? ClearCacheThenPredict()
    {
        _model.ClearCache();
        return _model.Predict(_context, topK: TopK);
    }

    [GlobalCleanup]
    public void Cleanup() => _model?.Dispose();
}
