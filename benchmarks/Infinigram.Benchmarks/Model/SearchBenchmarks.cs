using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks.Model;

[MemoryDiagnoser]
public class SearchBenchmarks
{
    [Params(32768, 1048576)]
    public int CorpusBytes { get; set; }

    [Params("Common", "Rare", "Missing", "Long")]
    public string Pattern { get; set; } = null!;

    private NGramModel _model = null!;
    private int[] _pattern = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = CorpusData.Create(CorpusBytes);
        _model = new NGramModel(corpus);
        _pattern = Pattern switch
        {
            "Common" => CorpusData.Encode(CorpusData.CommonPattern),
            "Rare" => CorpusData.Encode(CorpusData.RarePattern),
            "Missing" => CorpusData.Encode(CorpusData.MissingPattern),
            "Long" => CorpusData.Context(corpus, 128),
            _ => throw new InvalidOperationException("Unknown pattern."),
        };
        int count = _model.Count(_pattern);
        if ((Pattern == "Missing" && count != 0) || (Pattern == "Rare" && count != 1) ||
            (Pattern == "Common" && count < 2) || (Pattern == "Long" && count == 0))
            throw new InvalidOperationException("Corpus does not satisfy benchmark assumptions.");
    }

    [Benchmark]
    public int Count() => _model.Count(_pattern);

    [Benchmark]
    public List<int> SearchLimited() => _model.Search(_pattern, limit: 100);

    [GlobalCleanup]
    public void Cleanup() => _model?.Dispose();
}
