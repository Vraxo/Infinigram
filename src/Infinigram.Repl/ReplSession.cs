using System.Text;
using Model = Infinigram.Infinigram;

namespace Infinigram.Repl;

public static class ReplSession
{
    public static int Run(TextReader input, TextWriter output, TextWriter error, string corpusPath,
        TextNormalizationOptions? normalization = null)
    {
        Model model;
        try
        {
            model = normalization is null
                ? new Model(File.ReadAllBytes(corpusPath))
                : new Model(TextNormalizer.Normalize(
                    File.ReadAllText(corpusPath, new UTF8Encoding(false, true)), normalization));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            error.WriteLine($"Could not load corpus '{corpusPath}': {ex.Message}");
            return 1;
        }

        using (model)
        {
            output.WriteLine("Enter a prompt to generate a continuation. End input to exit.");
            while (true)
            {
                output.Write("infinigram> ");
                output.Flush();
                string? prompt = input.ReadLine();
                if (prompt is null) break;
                string context = normalization is null ? prompt : TextNormalizer.Normalize(prompt, normalization);
                output.WriteLine(model.GenerateText(context, maxTokens: 80, temperature: 0));
            }
        }
        return 0;
    }
}
