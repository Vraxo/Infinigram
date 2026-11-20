using System.Text;

namespace Infinigram.Tests;

/// <summary>Shared helpers for building token-id sequences and temp directories.</summary>
internal static class TestData
{
    /// <summary>Encode an ASCII/UTF-8 string to a token-id list.</summary>
    public static List<int> B(string text) =>
        Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();

    /// <summary>Token-id list from raw byte values.</summary>
    public static List<int> B(params byte[] bytes) =>
        bytes.Select(b => (int)b).ToList();

    /// <summary>Token-ids for a byte range [start, start+count).</summary>
    public static List<int> Range(int start, int count) =>
        Enumerable.Range(start, count).ToList();
}

/// <summary>A unique temporary directory, recursively deleted on dispose.</summary>
internal sealed class TempDir : IDisposable
{
    public string Path { get; } =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), "infinigram_tests_" + Guid.NewGuid().ToString("N"));

    public TempDir()
    {
        Directory.CreateDirectory(Path);
    }

    public string SubDir(string name)
    {
        var full = System.IO.Path.Combine(Path, name);
        Directory.CreateDirectory(full);
        return full;
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // best effort cleanup
        }
        catch (UnauthorizedAccessException)
        {
            // best effort cleanup
        }
    }
}
