using System.Text;
using Spectre.Console;
using Model = Infinigram.NGramModel;

namespace Infinigram.Repl;

public static class ReplSession
{
    public static int Run(TextReader input, TextWriter output, TextWriter error, string corpusPath,
        TextNormalizationOptions? normalization = null)
    {
        var console = CreateConsole(output);
        var errorConsole = CreateConsole(error);
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
            errorConsole.Markup("[red]Could not load corpus[/]");
            error.WriteLine($" '{corpusPath}': {ex.Message}");
            return 1;
        }

        using (model)
        {
            console.MarkupLine("[grey]Enter a prompt to generate a continuation. End input to exit.[/]");
            while (true)
            {
                console.Markup("[bold cyan]infinigram> [/]");
                output.Flush();
                string? prompt = input.ReadLine();
                if (prompt is null) break;
                string context = normalization is null ? prompt : TextNormalizer.Normalize(prompt, normalization);
                output.WriteLine(model.GenerateText(context, maxTokens: 80, temperature: 0));
            }
        }
        return 0;
    }

    private static IAnsiConsole CreateConsole(TextWriter writer)
    {
        var destination = new AnsiConsoleOutput(writer);
        return AnsiConsole.Create(new AnsiConsoleSettings
        {
            Out = destination,
            Ansi = destination.IsTerminal ? AnsiSupport.Detect : AnsiSupport.No,
        });
    }
}
