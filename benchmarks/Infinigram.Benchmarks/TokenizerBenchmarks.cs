using System.Text;
using BenchmarkDotNet.Attributes;
using Infinigram.Benchmarks.Core;

namespace Infinigram.Benchmarks;

[MemoryDiagnoser]
public class TokenizerBenchmarks
{
    private const int TrainingSampleLength = 128 * 1024;
    private const int TrainingSegmentCount = 8;

    private readonly ByteTokenizer _byteTokenizer = new();
    private string _corpus = null!;
    private string _trainingCorpus = null!;
    private BpeTokenizer _bpeTokenizer = null!;
    private IReadOnlyList<int> _byteTokens = null!;
    private IReadOnlyList<int> _bpeTokens = null!;

    [ParamsSource(nameof(CorpusFiles))]
    public string CorpusFile { get; set; } = null!;

    [Params(1024, 4096)]
    public int VocabularySize { get; set; }

    public static IEnumerable<string> CorpusFiles() =>
        TokenizerCorpusData.AvailableCorpora();

    [GlobalSetup]
    public void Setup()
    {
        _corpus = TokenizerCorpusData.Read(CorpusFile);
        _trainingCorpus = CreateTrainingSample(_corpus);
        _bpeTokenizer = BpeTokenizer.Train(_trainingCorpus, VocabularySize);
        _byteTokens = _byteTokenizer.Encode(_corpus);
        _bpeTokens = _bpeTokenizer.Encode(_corpus);

        if (_byteTokenizer.Decode(_byteTokens) != _corpus ||
            _bpeTokenizer.Decode(_bpeTokens) != _corpus)
        {
            throw new InvalidOperationException($"Tokenizer round-trip failed for corpus '{CorpusFile}'.");
        }

        int utf8Bytes = Encoding.UTF8.GetByteCount(_corpus);
        int trainingBytes = Encoding.UTF8.GetByteCount(_trainingCorpus);
        int reduction = _byteTokens.Count == 0
            ? 0
            : (int)Math.Round((1d - (double)_bpeTokens.Count / _byteTokens.Count) * 100d);

        Console.WriteLine(
            $"Corpus: {CorpusFile} | UTF-8 bytes: {utf8Bytes:N0} | " +
            $"Training sample: {_trainingCorpus.Length:N0} chars / {trainingBytes:N0} bytes | " +
            $"Byte tokens: {_byteTokens.Count:N0} | BPE tokens: {_bpeTokens.Count:N0} | " +
            $"Token reduction: {reduction}% | Learned merges: {_bpeTokenizer.MergeCount:N0} | " +
            $"Used BPE tokens: {_bpeTokens.Distinct().Count():N0}");
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
    public int BpeTrain() => BpeTokenizer.Train(_trainingCorpus, VocabularySize).MergeCount;

    private static string CreateTrainingSample(string corpus)
    {
        if (corpus.Length <= TrainingSampleLength)
        {
            return corpus;
        }

        int segmentLength = TrainingSampleLength / TrainingSegmentCount;
        int finalStart = corpus.Length - segmentLength;
        var sample = new StringBuilder(TrainingSampleLength);

        for (int segment = 0; segment < TrainingSegmentCount; segment++)
        {
            int start = (int)((long)finalStart * segment / (TrainingSegmentCount - 1));
            sample.Append(corpus, start, segmentLength);
        }

        return sample.ToString();
    }
}
