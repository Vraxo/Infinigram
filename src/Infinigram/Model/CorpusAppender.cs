namespace Infinigram;

internal static class CorpusAppender
{
    internal static SuffixArray Append(SuffixArray suffixArray, int[] tokens)
    {
        var existing = suffixArray.Corpus as int[] ?? suffixArray.Corpus.ToArray();
        var updated = new int[existing.Length + tokens.Length];
        Array.Copy(existing, updated, existing.Length);
        Array.Copy(tokens, 0, updated, existing.Length, tokens.Length);
        return new SuffixArray(updated);
    }
}
