namespace Infinigram;

internal static class CorpusValidator
{
    internal static int[] Validate(IReadOnlyList<int> tokens, int vocabSize, string what)
    {
        if (tokens.Count == 0)
        {
            return Array.Empty<int>();
        }

        var arr = tokens as int[] ?? tokens.ToArray();
        int min = int.MaxValue, max = int.MinValue;
        foreach (int t in arr)
        {
            if (t < min) min = t;
            if (t > max) max = t;
        }

        if (min < 0 || max >= vocabSize)
        {
            string message = vocabSize == 256
                ? $"{what} must contain only bytes (0-255), but found {(max >= 256 ? max : min)}"
                : $"{what} contains token ids outside [0, {vocabSize}); min={min}, max={max}";
            throw new ArgumentOutOfRangeException(what, message);
        }
        return arr;
    }
}
