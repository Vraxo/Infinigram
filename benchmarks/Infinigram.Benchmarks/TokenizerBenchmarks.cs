using BenchmarkDotNet.Attributes;
using Infinigram;

namespace Infinigram.Benchmarks;

[MemoryDiagnoser]
public class TokenizerBenchmarks
{
    private const string Sample =
        "Infinigram builds variable-length n-gram language models over suffix arrays. " +
        "The corpus is the model, and predictions are generated from observed continuations. " +
        "Byte-level tokenization provides a compact, deterministic baseline for UTF-8 text. " +
        "A learned byte-pair tokenizer can represent frequent sequences with fewer tokens.";

    private readonly string _corpus = string.Join(
        Environment.NewLine,
        Enumerable.Repeat(Sample, 128));

    private readonly ByteTokenizer _byteTokenizer = new();
    private BpeTokenizer _bpeTokenizer = null!;
    private IReadOnlyList<int> _byteTokens = null!;
    private IReadOnlyList<int> _bpeTokens = null!;

    [Params(1024, 4096)]
    public int VocabularySize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _bpeTokenizer = BpeTokenizer.Train(_corpus, VocabularySize);
        _byteTokens = _byteTokenizer.Encode(_corpus);
        _bpeTokens = _bpeTokenizer.Encode(_corpus);
    }

    [Benchmark(Baseline = true)]
    public int ByteEncode() => _byteTokenizer.Encode(_corpus).Count;

    [Benchmark]
    public int BpeEncode() => _bpeTokenizer.Encode(_corpus).Count;

    [Benchmark]
    public int ByteDecode() => _byteTokenizer.Decode(_byteTokens).Length;

    [Benchmark]
    public int BpeDecode() => _bpeTokenizer.Decode(_bpeTokens).Length;

    [Benchmark]
    public int BpeTrain() => BpeTokenizer.Train(_corpus, VocabularySize).MergeCount;
}
