// Tokenizer abstraction for Infinigram.
//
// Infinigram indexes sequences of token ids. An <see cref="ITokenizer"/> converts
// between strings and token-id sequences at build and query time. The library ships
// with <see cref="ByteTokenizer"/> (identity over UTF-8 bytes, vocab=256) and accepts
// any implementation of the interface (e.g. adapters around tiktoken or
// HuggingFace tokenizers).
//
// The tokenizer is a property of the model: the index records which tokenizer
// produced it so load-time mismatches are caught instead of silently scrambling
// results.

using System.Text;

namespace Infinigram;

/// <summary>
/// Minimal contract for any tokenizer Infinigram can use.
/// </summary>
public interface ITokenizer
{
    /// <summary>Encode text to token ids.</summary>
    IReadOnlyList<int> Encode(string text);

    /// <summary>Decode token ids back to text.</summary>
    string Decode(IReadOnlyList<int> tokens);

    /// <summary>Size of the vocabulary (token ids are in <c>[0, VocabSize)</c>).</summary>
    int VocabSize { get; }
}

/// <summary>
/// Identity tokenizer over UTF-8 bytes.
/// </summary>
/// <remarks>
/// Maps each UTF-8 byte to itself as a token id in <c>[0, 256)</c>. Round-trips
/// losslessly for valid UTF-8; on decode, invalid bytes are replaced with U+FFFD
/// (matching Python's <c>errors='replace'</c> default).
/// </remarks>
public sealed class ByteTokenizer : ITokenizer, IEquatable<ByteTokenizer>
{
    private static readonly Encoding Utf8Replacement =
        new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

    /// <inheritdoc />
    public int VocabSize => 256;

    /// <summary>Canonical name used in model metadata.</summary>
    public string Name => "byte";

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Utf8Replacement.GetBytes(text).Select(b => (int)b).ToList();
    }

    /// <inheritdoc />
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
