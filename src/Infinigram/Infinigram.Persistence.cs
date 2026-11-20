using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Infinigram;

public sealed partial class Infinigram
{
    private static string DefaultModelsDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".infinigram", "models");

    private static readonly JsonSerializerOptions JsonWriteOptions = new() { WriteIndented = true };

    public static Infinigram Build(string corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        var tok = tokenizer ?? new ByteTokenizer();
        return BuildFromTokens(tok.Encode(corpus).ToList(), modelPath, tok, maxLength, verbose);
    }

    public static Infinigram Build(byte[] corpus, string modelPath, ITokenizer? tokenizer = null,
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

    public static Infinigram Build(IReadOnlyList<int> corpus, string modelPath, ITokenizer? tokenizer = null,
        int? maxLength = null, bool verbose = true)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(modelPath);
        return BuildFromTokens(corpus.ToList(), modelPath, tokenizer ?? new ByteTokenizer(), maxLength, verbose);
    }

    public static Infinigram BuildFromFile(string corpusFilePath, string modelPath, ITokenizer? tokenizer = null,
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

    private static Infinigram BuildFromTokens(List<int> tokens, string modelPath, ITokenizer tokenizer,
        int? maxLength, bool verbose)
    {
        if (tokens.Count == 0)
        {
            throw new ArgumentException("Corpus must not be empty.", nameof(tokens));
        }

        var stopwatch = Stopwatch.StartNew();
        string fullPath = Path.GetFullPath(modelPath);
        var model = new Infinigram(tokenizer, maxLength);

        var validated = model.ValidateCorpus(tokens, "Corpus");
        int n = validated.Length;
        if (verbose)
        {
            Console.WriteLine($"Building model at {fullPath}");
            Console.WriteLine($"Corpus: {n:N0} tokens (vocab={tokenizer.VocabSize})");
            Console.WriteLine("Building suffix array...");
        }

        var sa = new SuffixArray(validated);
        sa.Save(fullPath);

        model._sa = sa;
        model._modelDir = fullPath;
        model.WriteMeta(fullPath, n);

        if (verbose)
        {
            double elapsed = stopwatch.Elapsed.TotalSeconds;
            double rate = elapsed > 0 ? n / elapsed / 1e6 : double.PositiveInfinity;
            Console.WriteLine($"Built in {elapsed:F1}s ({rate:F2} MT/sec)");
        }
        return model;
    }

    public static Infinigram Load(string modelPath, ITokenizer? tokenizer = null, int? maxLength = null)
    {
        ArgumentNullException.ThrowIfNull(modelPath);

        string dir = modelPath;
        if (!Path.IsPathRooted(modelPath) && !Directory.Exists(modelPath))
        {
            string candidate = Path.Combine(DefaultModelsDir, modelPath);
            if (Directory.Exists(candidate))
            {
                dir = candidate;
            }
        }

        if (!Directory.Exists(dir))
        {
            throw new FileNotFoundException(
                $"Model not found at {dir}. Use Infinigram.Build(...) to create a new model.");
        }

        string metaPath = Path.Combine(dir, "meta.json");
        if (!File.Exists(metaPath))
        {
            throw new FileNotFoundException(
                $"Model not found at {dir} (missing meta.json). Use Infinigram.Build(...) to create a new model.");
        }

        JsonObject meta;
        try
        {
            meta = JsonNode.Parse(File.ReadAllText(metaPath)) as JsonObject
                   ?? throw new JsonException("meta.json is not a JSON object");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Corrupt meta.json at {metaPath}: {ex.Message}", ex);
        }

        int version = meta.TryGetPropertyValue("version", out var versionNode) && versionNode is not null
            ? versionNode.GetValue<int>()
            : 0;
        if (version != 1)
        {
            throw new NotSupportedException(
                $"Unsupported model format version {version} at {metaPath}. This engine reads C#-built " +
                "indexes (version 1). If this is a Python-built model (.npy files, version 2), it is " +
                "not supported yet — rebuild it with Infinigram.Build.");
        }

        ITokenizer resolved = ResolveTokenizer(meta, metaPath, tokenizer);

        if (meta.TryGetPropertyValue("vocab_size", out var vocabNode) && vocabNode is not null)
        {
            int metaVocab = vocabNode.GetValue<int>();
            if (metaVocab != resolved.VocabSize)
            {
                throw new InvalidDataException(
                    $"Tokenizer mismatch: model at {dir} was built with vocab_size={metaVocab}, " +
                    $"but the resolved tokenizer has VocabSize={resolved.VocabSize}.");
            }
        }

        int? metaMaxLength = null;
        if (meta.TryGetPropertyValue("max_length", out var maxLengthNode) && maxLengthNode is not null)
        {
            metaMaxLength = maxLengthNode.GetValue<int>();
        }

        var model = new Infinigram(resolved, maxLength ?? metaMaxLength);
        string fullDir = Path.GetFullPath(dir);
        model._sa = SuffixArray.Load(fullDir);
        model._modelDir = fullDir;

        if (meta.TryGetPropertyValue("n", out var nNode) && nNode is not null)
        {
            int metaN = nNode.GetValue<int>();
            if (metaN != model.N)
            {
                throw new InvalidDataException(
                    $"Corrupt model at {dir}: meta.json says n={metaN} but the corpus has {model.N} tokens.");
            }
        }
        return model;
    }

    public static List<string> ListModels(string? modelsDir = null)
    {
        string dir = modelsDir ?? DefaultModelsDir;
        if (!Directory.Exists(dir))
        {
            return new List<string>();
        }
        return Directory
            .EnumerateDirectories(dir)
            .Where(d => File.Exists(Path.Combine(d, "meta.json")))
            .Select(Path.GetFileName)
            .Where(name => name is not null)
            .Select(name => name!)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToList();
    }

    private static ITokenizer ResolveTokenizer(JsonObject meta, string metaPath, ITokenizer? supplied)
    {
        string? kind = null;
        string? customName = null;
        if (meta.TryGetPropertyValue("tokenizer", out var tokenizerNode) && tokenizerNode is JsonObject tokConfig)
        {
            kind = tokConfig.TryGetPropertyValue("kind", out var kindNode) ? kindNode?.GetValue<string>() : null;
            if (tokConfig.TryGetPropertyValue("name", out var nameNode))
            {
                customName = nameNode?.GetValue<string>();
            }
        }

        switch (kind)
        {
            case "byte":
                if (supplied is not null && supplied is not ByteTokenizer)
                {
                    throw new ArgumentException(
                        "Tokenizer mismatch: model was built with a byte tokenizer but a different " +
                        "tokenizer was supplied. Pass the matching tokenizer or omit tokenizer:.");
                }
                return new ByteTokenizer();

            case "custom":
                if (supplied is null)
                {
                    throw new InvalidOperationException(
                        $"Model at {metaPath} was built with a custom tokenizer ({customName}). " +
                        "Pass tokenizer: when loading.");
                }
                return supplied;

            case null:
                return supplied ?? new ByteTokenizer();

            default:
                if (supplied is null)
                {
                    throw new NotSupportedException(
                        $"Unknown tokenizer kind '{kind}' in {metaPath}. Pass tokenizer: explicitly.");
                }
                return supplied;
        }
    }

    private void WriteMeta(string dir, int n)
    {
        var tokenizerConfig = Tokenizer is ByteTokenizer
            ? new Dictionary<string, object?> { ["kind"] = "byte", ["encoding"] = "utf-8", ["errors"] = "replace" }
            : new Dictionary<string, object?> { ["kind"] = "custom", ["name"] = Tokenizer.GetType().FullName };

        var meta = new Dictionary<string, object?>
        {
            ["version"] = 1,
            ["engine"] = "infinigram-csharp",
            ["n"] = n,
            ["vocab_size"] = VocabSize,
            ["max_length"] = MaxLength,
            ["tokenizer"] = tokenizerConfig,
        };
        File.WriteAllText(
            Path.Combine(dir, "meta.json"),
            JsonSerializer.Serialize(meta, JsonWriteOptions));
    }

}
