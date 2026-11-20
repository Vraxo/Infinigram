
using System.Text;
using System.Text.Json;

namespace Infinigram;

public static class CorpusUtils
{
    public static List<int> BuildCorpusFromDocuments(
        IEnumerable<string> documents,
        byte[]? separator = null,
        Encoding? encoding = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        var separatorBytes = separator ?? new byte[] { 10, 10 };
        var enc = encoding ?? Encoding.UTF8;

        var corpus = new List<int>();
        bool first = true;
        foreach (var doc in documents)
        {
            ArgumentNullException.ThrowIfNull(doc);
            if (!first)
            {
                corpus.AddRange(separatorBytes.Select(b => (int)b));
            }
            first = false;
            corpus.AddRange(enc.GetBytes(doc).Select(b => (int)b));
        }
        return corpus;
    }

    public static List<int> BuildCorpusFromDocuments(
        IEnumerable<IReadOnlyList<int>> documents,
        byte[]? separator = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        var separatorBytes = (separator ?? new byte[] { 10, 10 }).Select(b => (int)b).ToList();

        var corpus = new List<int>();
        bool first = true;
        int docIndex = 0;
        foreach (var doc in documents)
        {
            ArgumentNullException.ThrowIfNull(doc);
            for (int i = 0; i < doc.Count; i++)
            {
                int t = doc[i];
                if (t < 0 || t > 255)
                {
                    throw new ArgumentException(
                        $"Document {docIndex} contains invalid byte values: {t}");
                }
            }

            if (!first)
            {
                corpus.AddRange(separatorBytes);
            }
            first = false;
            corpus.AddRange(doc);
            docIndex++;
        }
        return corpus;
    }

    public static IEnumerable<string> IterJsonLines(
        string path,
        string textField = "text",
        Encoding? encoding = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var enc = encoding ?? Encoding.UTF8;

        using var reader = new StreamReader(path, enc);
        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            using var doc = JsonDocument.Parse(line);
            if (doc.RootElement.TryGetProperty(textField, out var element) &&
                element.ValueKind == JsonValueKind.String)
            {
                var text = element.GetString();
                if (text is not null)
                {
                    yield return text;
                }
            }
        }
    }

    public static List<int> BuildCorpusFromJsonLines(
        string path,
        string textField = "text",
        byte[]? separator = null,
        int? maxDocuments = null,
        Encoding? encoding = null)
    {
        var documents = new List<string>();
        foreach (var text in IterJsonLines(path, textField, encoding))
        {
            if (maxDocuments is int max && documents.Count >= max)
            {
                break;
            }
            documents.Add(text);
        }
        return BuildCorpusFromDocuments(documents, separator, encoding);
    }

    public static void SaveCorpusToJsonLines(
        IEnumerable<string> documents,
        string path,
        string textField = "text",
        Encoding? encoding = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(path);
        var enc = encoding ?? Encoding.UTF8;

        using var writer = new StreamWriter(path, append: false, enc);
        foreach (var doc in documents)
        {
            ArgumentNullException.ThrowIfNull(doc);
            var payload = JsonSerializer.Serialize(new Dictionary<string, string> { [textField] = doc });
            writer.Write(payload);
            writer.Write('\n');
        }
    }
}
