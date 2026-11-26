using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[ShortRunJob]
[MemoryDiagnoser]
public class PredictionHotPathBenchmarks
{
    private NGramModel _model = null!;
    private int[] _context = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = CorpusData.Create(32768);
        _model = new NGramModel(corpus);
        _context = CorpusData.Context(corpus, 128);

        if (_model.Predict(_context, topK: 50) is null)
        {
            throw new InvalidOperationException("Expected context to have continuations.");
        }
    }

    [Benchmark]
    public IReadOnlyDictionary<int, double>? WarmCache() =>
        _model.Predict(_context, topK: 50);

    [Benchmark]
    public IReadOnlyDictionary<int, double>? ClearCacheThenPredict()
    {
        _model.ClearCache();
        return _model.Predict(_context, topK: 50);
    }

    [GlobalCleanup]
    public void Cleanup() => _model?.Dispose();
}
