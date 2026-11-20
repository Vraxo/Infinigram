using System.Text;

namespace Infinigram.Tests;

internal static class TestData
{
    public static List<int> B(string text) =>
        Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();

    public static List<int> B(params byte[] bytes) =>
        bytes.Select(b => (int)b).ToList();

    public static List<int> Range(int start, int count) =>
        Enumerable.Range(start, count).ToList();
}

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
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
