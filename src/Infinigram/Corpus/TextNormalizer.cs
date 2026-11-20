using System.Text;

namespace Infinigram;

public static class TextNormalizer
{
    public static string Normalize(string text, TextNormalizationOptions options)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(options);

        if (options.NormalizeLineEndings)
        {
            text = text.Replace("\r\n", "\n").Replace('\r', '\n');
        }
        if (options.Lowercase)
        {
            text = text.ToLowerInvariant();
        }
        if (options.UnicodeNfc)
        {
            text = text.Normalize(NormalizationForm.FormC);
        }
        if (options.CollapseWhitespace)
        {
            var result = new StringBuilder(text.Length);
            bool previousWhitespace = false;
            foreach (char c in text)
            {
                if (char.IsWhiteSpace(c))
                {
                    if (!previousWhitespace) result.Append(' ');
                    previousWhitespace = true;
                }
                else
                {
                    result.Append(c);
                    previousWhitespace = false;
                }
            }
            text = result.ToString();
        }
        return text;
    }
}
