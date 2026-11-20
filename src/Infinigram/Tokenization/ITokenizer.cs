namespace Infinigram;

public interface ITokenizer
{
    IReadOnlyList<int> Encode(string text);

    string Decode(IReadOnlyList<int> tokens);

    int VocabSize { get; }
}
