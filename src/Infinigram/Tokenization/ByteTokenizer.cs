using System.Text;

namespace Infinigram;

public sealed class ByteTokenizer : ITokenizer, IEquatable<ByteTokenizer>
{
    private static readonly Encoding Utf8Replacement =
        new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

    public int VocabSize => 256;

    public string Name => "byte";

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Utf8Replacement.GetBytes(text).Select(b => (int)b).ToList();
    }

    public string Decode(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        var bytes = new byte[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            int t = tokens[i];
            if (t < 0 || t > 255)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokens), $"Token id {t} is outside the byte range [0, 255].");
            }
            bytes[i] = (byte)t;
        }
        return Utf8Replacement.GetString(bytes);
    }

    public bool Equals(ByteTokenizer? other) => other is not null;

    public override bool Equals(object? obj) => obj is ByteTokenizer;

    public override int GetHashCode() => typeof(ByteTokenizer).GetHashCode();

    public override string ToString() => "ByteTokenizer(encoding='utf-8')";
}
