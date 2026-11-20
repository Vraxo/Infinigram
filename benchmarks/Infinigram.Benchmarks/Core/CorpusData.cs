using System.Text;

namespace Infinigram.Benchmarks.Core;

internal static class CorpusData
{
    internal const string CommonPattern = "alpha ";
    internal const string RarePattern = " UniqueMarkerXyZ ";
    internal const string MissingPattern = " MissingMarkerQz ";

    internal static byte[] Create(int size)
    {
        string[] words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota"];
        var builder = new StringBuilder(size + 64);
        var random = new Random(1729);
        while (builder.Length < size - RarePattern.Length)
        {
            builder.Append(words[random.Next(words.Length)]).Append(' ');
        }
        builder.Length = size - RarePattern.Length;
        builder.Append(RarePattern);
        return Encoding.UTF8.GetBytes(builder.ToString());
    }

    internal static int[] Encode(string value) => Encoding.UTF8.GetBytes(value).Select(b => (int)b).ToArray();

    internal static int[] Context(byte[] corpus, int length) =>
        corpus.Skip(corpus.Length / 2).Take(length).Select(b => (int)b).ToArray();
}
