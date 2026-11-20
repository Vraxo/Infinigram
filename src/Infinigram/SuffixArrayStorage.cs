using System.Text;

namespace Infinigram;

internal static class SuffixArrayStorage
{
    internal static void Save(string directory, int[] corpus, int[] suffixArray)
    {
        ArgumentNullException.ThrowIfNull(directory);
        Directory.CreateDirectory(directory);
        WriteIntArray(Path.Combine(directory, "corpus.bin"), corpus);
        WriteIntArray(Path.Combine(directory, "suffix_array.bin"), suffixArray);
    }

    internal static (int[] Corpus, int[] Positions) Load(string directory)
    {
        ArgumentNullException.ThrowIfNull(directory);
        string corpusPath = Path.Combine(directory, "corpus.bin");
        string saPath = Path.Combine(directory, "suffix_array.bin");
        if (!File.Exists(corpusPath) || !File.Exists(saPath))
        {
            throw new FileNotFoundException(
                $"Index not found at {directory} (expected corpus.bin and suffix_array.bin). " +
                "Note: indexes written by the Python implementation (.npy files) are not " +
                "readable by this engine yet.");
        }

        int[] corpus = ReadIntArray(corpusPath);
        int[] sa = ReadIntArray(saPath);
        if (sa.Length != corpus.Length)
        {
            throw new InvalidDataException(
                $"Corrupt index at {directory}: corpus has {corpus.Length} tokens but the " +
                $"suffix array has {sa.Length} entries.");
        }
        return (corpus, sa);
    }

    private static void WriteIntArray(string path, int[] data)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        writer.Write(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            writer.Write(data[i]);
        }
    }

    private static int[] ReadIntArray(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
        int count = reader.ReadInt32();
        if (count < 0 || count != (stream.Length - sizeof(int)) / sizeof(int))
        {
            throw new InvalidDataException($"Corrupt index file {path} (bad element count {count}).");
        }
        var data = new int[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = reader.ReadInt32();
        }
        return data;
    }

}
