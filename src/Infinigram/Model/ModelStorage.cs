using System.Text.Json;
using System.Text.Json.Nodes;

namespace Infinigram;

internal static class ModelStorage
{
    private static string DefaultModelsDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".infinigram", "models");

    private static readonly JsonSerializerOptions JsonWriteOptions = new() { WriteIndented = true };

    internal static void Save(string directory, SuffixArray suffixArray, ITokenizer tokenizer, int vocabSize, int? maxLength)
    {
        suffixArray.Save(directory);
        WriteMeta(directory, suffixArray.Length, tokenizer, vocabSize, maxLength);
    }

    internal static (SuffixArray SuffixArray, ITokenizer Tokenizer, int? MaxLength, string Directory) Load(
        string modelPath, ITokenizer? tokenizer, int? maxLength)
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
                $"Model not found at {dir}. Use NGramModel.Build(...) to create a new model.");
        }

        string metaPath = Path.Combine(dir, "meta.json");
        if (!File.Exists(metaPath))
        {
            throw new FileNotFoundException(
                $"Model not found at {dir} (missing meta.json). Use NGramModel.Build(...) to create a new model.");
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
                "not supported yet — rebuild it with NGramModel.Build.");
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

        string fullDir = Path.GetFullPath(dir);
        var suffixArray = SuffixArray.Load(fullDir);

        if (meta.TryGetPropertyValue("n", out var nNode) && nNode is not null)
        {
            int metaN = nNode.GetValue<int>();
            if (metaN != suffixArray.Length)
            {
                throw new InvalidDataException(
                    $"Corrupt model at {dir}: meta.json says n={metaN} but the corpus has {suffixArray.Length} tokens.");
            }
        }
        return (suffixArray, resolved, maxLength ?? metaMaxLength, fullDir);
    }

    internal static List<string> ListModels(string? modelsDir)
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
        JsonObject? tokenizerConfig = null;
        if (meta.TryGetPropertyValue("tokenizer", out var tokenizerNode) && tokenizerNode is JsonObject tokConfig)
        {
            tokenizerConfig = tokConfig;
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

            case "bpe":
                if (tokenizerConfig?["merges"] is not JsonArray mergeNodes)
                {
                    throw new InvalidDataException(
                        $"Model at {metaPath} has an invalid BPE tokenizer configuration (missing merges).");
                }

                var mergePairs = new List<(int Left, int Right)>(mergeNodes.Count);
                foreach (JsonNode? mergeNode in mergeNodes)
                {
                    if (mergeNode is not JsonArray pair || pair.Count != 2 ||
                        pair[0] is null || pair[1] is null)
                    {
                        throw new InvalidDataException(
                            $"Model at {metaPath} has an invalid BPE merge entry.");
                    }

                    try
                    {
                        mergePairs.Add((pair[0]!.GetValue<int>(), pair[1]!.GetValue<int>()));
                    }
                    catch (Exception ex) when (ex is InvalidOperationException or FormatException)
                    {
                        throw new InvalidDataException(
                            $"Model at {metaPath} has an invalid BPE merge entry.", ex);
                    }
                }

                BpeTokenizer restored;
                try
                {
                    restored = BpeTokenizer.FromMerges(mergePairs);
                }
                catch (ArgumentException ex)
                {
                    throw new InvalidDataException(
                        $"Model at {metaPath} has an invalid BPE tokenizer configuration: {ex.Message}", ex);
                }

                if (supplied is not null &&
                    (supplied is not BpeTokenizer suppliedBpe || !suppliedBpe.Equals(restored)))
                {
                    throw new ArgumentException(
                        "Tokenizer mismatch: model was built with different BPE merges. " +
                        "Omit tokenizer: to restore the tokenizer saved with the model.");
                }

                return restored;

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

    private static void WriteMeta(string dir, int n, ITokenizer tokenizer, int vocabSize, int? maxLength)
    {
        var tokenizerConfig = tokenizer switch
        {
            ByteTokenizer => new Dictionary<string, object?>
            {
                ["kind"] = "byte",
                ["encoding"] = "utf-8",
                ["errors"] = "replace",
            },
            BpeTokenizer bpe => new Dictionary<string, object?>
            {
                ["kind"] = "bpe",
                ["encoding"] = "utf-8",
                ["merges"] = bpe.MergePairs
                    .Select(static pair => new[] { pair.Left, pair.Right })
                    .ToArray(),
            },
            _ => new Dictionary<string, object?>
            {
                ["kind"] = "custom",
                ["name"] = tokenizer.GetType().FullName,
            },
        };

        var meta = new Dictionary<string, object?>
        {
            ["version"] = 1,
            ["engine"] = "infinigram-csharp",
            ["n"] = n,
            ["vocab_size"] = vocabSize,
            ["max_length"] = maxLength,
            ["tokenizer"] = tokenizerConfig,
        };
        File.WriteAllText(
            Path.Combine(dir, "meta.json"),
            JsonSerializer.Serialize(meta, JsonWriteOptions));
    }
}
