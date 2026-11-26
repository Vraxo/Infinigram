using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[MemoryDiagnoser]
public class GenerationBenchmarks
{
    [Params(32768, 1048576)]
    public int CorpusBytes { get; set; }

    [Params(32, 128)]
    public int OutputBytes { get; set; }

    private NGramModel _model = null!;
    private int[] _context = null!;
    private int[] _generated = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = CorpusData.Create(CorpusBytes);
        _model = new NGramModel(corpus);
        _context = CorpusData.Context(corpus, 16);
        _generated = _model.Generate(_context, maxTokens: OutputBytes, temperature: 0).ToArray();
        if (_generated.Length != OutputBytes)
            throw new InvalidOperationException("Generation stopped before the configured output length.");
        _model.ClearCache();
    }

    [Benchmark]
    public List<int> GenerateGreedy()
    {
        _model.ClearCache();
        return _model.Generate(_context, maxTokens: OutputBytes, temperature: 0);
    }

    [Benchmark]
    public GenerationOverlap MeasureCorpusOverlap() =>
        GenerationMetrics.MeasureOverlap(_model, _generated, minimumSpan: 8);

    [GlobalCleanup]
    public void Cleanup() => _model?.Dispose();
}
