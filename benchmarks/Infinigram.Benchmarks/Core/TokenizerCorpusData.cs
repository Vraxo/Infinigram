using System.Text;

namespace Infinigram.Benchmarks.Core;

internal static class TokenizerCorpusData
{
    private const string SyntheticCorpus = """
        Infinigram builds variable-length n-gram language models over suffix arrays.
        The corpus is the model, and predictions are generated from observed continuations.
        Byte-level tokenization provides a deterministic baseline for UTF-8 text.
        A learned byte-pair tokenizer can represent frequent sequences with fewer tokens.

        public static IReadOnlyList<int> Encode(ReadOnlySpan<byte> input)
        {
            var tokens = new List<int>(input.Length);
            foreach (byte value in input)
            {
                tokens.Add(value);
            }
            return tokens;
        }

        русский текст проверяет кириллицу; Ελληνικά δοκιμάζουν ελληνικούς χαρακτήρες.
        فارسی و العربية و 日本語 و 中文 و 한국어 و emoji 😀🌍 проверяют Unicode.
        Repeated phrases, punctuation, contractions, numbers (2026), and whitespace.
        """;

    internal static IEnumerable<string> AvailableCorpora()
    {
        string directory = FindCorpusDirectory();
        string[] files = Directory.Exists(directory)
            ? Directory.GetFiles(directory, "*.txt", SearchOption.AllDirectories)
                .Select(path => Path.GetRelativePath(directory, path))
                .Order(StringComparer.OrdinalIgnoreCase)
                .ToArray()
            : [];

        return new[] { "__synthetic__" }.Concat(files);
    }

    internal static string Read(string corpusFile)
    {
        if (corpusFile == "__synthetic__")
        {
            return string.Join(
                Environment.NewLine,
                Enumerable.Repeat(SyntheticCorpus, 128));
        }

        string directory = FindCorpusDirectory();
        string path = Path.GetFullPath(Path.Combine(directory, corpusFile));
        string root = Path.GetFullPath(directory) + Path.DirectorySeparatorChar;

        if (!path.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Corpus paths must remain inside Resources/Corpora.");
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Corpus file was not found: {path}", path);
        }

        string content = File.ReadAllText(path, new UTF8Encoding(false, true));
        if (string.IsNullOrWhiteSpace(content))
        {
            throw new InvalidDataException($"Corpus file is empty: {path}");
        }

        return content;
    }

    private static string FindCorpusDirectory()
    {
        foreach (string start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            DirectoryInfo? current = new(start);
            while (current is not null)
            {
                string candidate = Path.Combine(current.FullName, "Resources", "Corpora");
                if (Directory.Exists(candidate))
                {
                    return candidate;
                }

                current = current.Parent;
            }
        }

        return Path.Combine(Environment.CurrentDirectory, "Resources", "Corpora");
    }
}
