using System.Text;
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
    private int[] _commonContext = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = CorpusData.Create(32768);
        _model = new NGramModel(corpus);
        _context = CorpusData.Context(corpus, 128);
        _commonContext = Encoding.UTF8.GetBytes("alpha ").Select(value => (int)value).ToArray();

        if (_model.Predict(_context, topK: 50) is null ||
            _model.Predict(_commonContext, topK: 50) is null)
        {
            throw new InvalidOperationException("Expected both benchmark contexts to have continuations.");
        }
    }

    [Benchmark]
    public IReadOnlyDictionary<int, double>? WarmSparseContext() =>
        _model.Predict(_context, topK: 50);

    [Benchmark]
    public IReadOnlyDictionary<int, double>? ClearCacheThenPredictSparseContext()
    {
        _model.ClearCache();
        return _model.Predict(_context, topK: 50);
    }

    [Benchmark]
    public IReadOnlyDictionary<int, double>? WarmCommonContext() =>
        _model.Predict(_commonContext, topK: 50);

    [Benchmark]
    public IReadOnlyDictionary<int, double>? ClearCacheThenPredictCommonContext()
    {
        _model.ClearCache();
        return _model.Predict(_commonContext, topK: 50);
    }

    [GlobalCleanup]
    public void Cleanup() => _model?.Dispose();
}
