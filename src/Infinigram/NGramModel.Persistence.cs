using System.Diagnostics;

namespace Infinigram;

public sealed partial class NGramModel
{
    public static NGramModel Build(string corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        var tok = tokenizer ?? new ByteTokenizer();
        return BuildFromTokens(tok.Encode(corpus).ToList(), modelPath, tok, maxLength, verbose);
    }

    public static NGramModel Build(byte[] corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        var tok = tokenizer ?? new ByteTokenizer();
        if (tok.VocabSize > 256)
        {
            throw new ArgumentException(
                $"bytes corpora are only supported with byte-level tokenizers; " +
                $"tokenizer has VocabSize={tok.VocabSize}. Pass IReadOnlyList<int> instead.",
                nameof(corpus));
        }
        return BuildFromTokens(corpus.Select(b => (int)b).ToList(), modelPath, tok, maxLength, verbose);
    }

    public static NGramModel Build(IReadOnlyList<int> corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        return BuildFromTokens(corpus.ToList(), modelPath, tokenizer ?? new ByteTokenizer(), maxLength, verbose);
    }

    public static NGramModel BuildFromFile(string corpusFilePath, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpusFilePath);
        ArgumentNullException.ThrowIfNull(modelPath);
        if (!File.Exists(corpusFilePath))
        {
            throw new FileNotFoundException($"Corpus file not found: {corpusFilePath}", corpusFilePath);
        }

        var tok = tokenizer ?? new ByteTokenizer();
        List<int> tokens;
        if (tok.VocabSize <= 256)
        {
            if (verbose)
            {
                Console.WriteLine($"Loading corpus from {corpusFilePath}...");
            }
            tokens = File.ReadAllBytes(corpusFilePath).Select(b => (int)b).ToList();
        }
        else
        {
            tokens = tok.Encode(File.ReadAllText(corpusFilePath)).ToList();
        }
        return BuildFromTokens(tokens, modelPath, tok, maxLength, verbose);
    }

    private static NGramModel BuildFromTokens(List<int> tokens, string modelPath, ITokenizer tokenizer,
        int? maxLength, bool verbose)
    {
        if (tokens.Count == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(tokens));
        }

        var stopwatch = Stopwatch.StartNew();
        string fullPath = Path.GetFullPath(modelPath);
        var model = new NGramModel(tokenizer, maxLength);

        var validated = CorpusValidator.Validate(tokens, model.VocabSize, "Corpus");
        int n = validated.Length;
        if (verbose)
        {
            Console.WriteLine($"Building model at {fullPath}");
            Console.WriteLine($"Corpus: {n:N0} tokens (vocab={tokenizer.VocabSize})");
            Console.WriteLine("Building suffix array...");
        }

        var sa = new SuffixArray(validated);
        ModelStorage.Save(fullPath, sa, model.Tokenizer, model.VocabSize, model.MaxLength);

        model._sa = sa;
        model._modelDir = fullPath;

        if (verbose)
        {
            double elapsed = stopwatch.Elapsed.TotalSeconds;
            double rate = elapsed > 0 ? n / elapsed / 1e6 : double.PositiveInfinity;
            Console.WriteLine($"Built in {elapsed:F1}s ({rate:F2} MT/sec)");
        }
        return model;
    }

    public static NGramModel Load(string modelPath, ITokenizer? tokenizer = null, int? maxLength = null)
    {
        var (suffixArray, resolved, effectiveMaxLength, directory) = ModelStorage.Load(modelPath, tokenizer, maxLength);
        return new NGramModel(resolved, effectiveMaxLength)
        {
            _sa = suffixArray,
            _modelDir = directory,
        };
    }

    public static List<string> ListModels(string? modelsDir = null) => ModelStorage.ListModels(modelsDir);
}
