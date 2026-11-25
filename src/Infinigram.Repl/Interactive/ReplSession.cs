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
            model = IsTerminal(output) && console.Profile.Capabilities.Interactive
                ? console.Status().Start("Reading corpus...", status =>
                    LoadModel(corpusPath, normalization, () => status.Status = "Building suffix array..."))
                : LoadModel(corpusPath, normalization);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            if (IsTerminal(error)) errorConsole.Markup("[red]Could not load corpus[/]");
            else error.Write("Could not load corpus");
            error.WriteLine($" '{corpusPath}': {ex.Message}");
            return 1;
        }

        using (model)
        {
            if (IsTerminal(output)) console.MarkupLine("[grey]Enter a prompt to generate a continuation. End input to exit.[/]");
            else output.WriteLine("Enter a prompt to generate a continuation. End input to exit.");
            while (true)
            {
                if (IsTerminal(output)) console.Markup("[bold cyan]infinigram> [/]");
                else output.Write("infinigram> ");
                output.Flush();
                string? prompt = input.ReadLine();
                if (prompt is null) break;
                string context = normalization is null ? prompt : TextNormalizer.Normalize(prompt, normalization);
                output.WriteLine(model.GenerateText(context, maxTokens: 80, temperature: 0));
            }
        }
        return 0;
    }

    private static Model LoadModel(string corpusPath, TextNormalizationOptions? normalization, Action? onIndexing = null)
    {
        if (normalization is null)
        {
            var bytes = File.ReadAllBytes(corpusPath);
            onIndexing?.Invoke();
            return new Model(bytes);
        }

        var text = File.ReadAllText(corpusPath, new UTF8Encoding(false, true));
        text = TextNormalizer.Normalize(text, normalization);
        onIndexing?.Invoke();
        return new Model(text);
    }

    private static IAnsiConsole CreateConsole(TextWriter writer)
    {
        var destination = new AnsiConsoleOutput(writer);
        return AnsiConsole.Create(new AnsiConsoleSettings
        {
            Out = destination,
            Ansi = IsTerminal(writer) ? AnsiSupport.Detect : AnsiSupport.No,
        });
    }

    private static bool IsTerminal(TextWriter writer)
    {
        if (ReferenceEquals(writer, Console.Out))
        {
            return !Console.IsOutputRedirected && new AnsiConsoleOutput(writer).IsTerminal;
        }

        if (ReferenceEquals(writer, Console.Error))
        {
            return !Console.IsErrorRedirected && new AnsiConsoleOutput(writer).IsTerminal;
        }

        return false;
    }
}
