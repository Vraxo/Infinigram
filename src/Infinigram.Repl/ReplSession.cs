using System.Globalization;
using System.Text.Json;
using Model = Infinigram.Infinigram;

namespace Infinigram.Repl;

public sealed class ReplSession(TextReader input, TextWriter output, TextWriter error)
{
    private Model? _model;

    public int Run(string[] args)
    {
        try
        {
            if (args.Length == 1 && args[0] is "--help" or "-h")
            {
                PrintHelp();
                return 0;
            }

            if (args.Length != 0)
            {
                if (args.Length != 2 || args[0] is not ("--file" or "--model"))
                {
                    error.WriteLine("Usage: infinigram-repl [--file <corpus.txt> | --model <directory>]");
                    return 2;
                }

                try
                {
                    ReplaceModel(args[0] == "--file" ? OpenFile(args[1]) : Model.Load(args[1]));
                }
                catch (Exception ex) when (IsUserError(ex))
                {
                    error.WriteLine($"Error: {ex.Message}");
                    return 1;
                }
            }

            output.WriteLine("Infinigram REPL. Type :help for commands, :quit to exit.");
            while (true)
            {
                output.Write("infinigram> ");
                output.Flush();
                string? line = input.ReadLine();
                if (line is null) break;
                if (line.Length == 0) continue;

                try
                {
                    if (!Execute(line)) break;
                }
                catch (Exception ex) when (IsUserError(ex))
                {
                    error.WriteLine($"Error: {ex.Message}");
                }
            }
            return 0;
        }
        finally
        {
            _model?.Dispose();
            _model = null;
        }
    }

    private bool Execute(string line)
    {
        if (!line.StartsWith(':'))
        {
            Generate(line);
            return true;
        }

        int separator = line.IndexOf(' ');
        string command = separator < 0 ? line : line[..separator];
        string argument = separator < 0 ? "" : line[(separator + 1)..];
        switch (command.ToLowerInvariant())
        {
            case ":quit":
            case ":exit": return false;
            case ":help": PrintHelp(); break;
            case ":text": ReplaceModel(new Model(RequireArgument(argument))); break;
            case ":open": ReplaceModel(OpenFile(RequireArgument(argument))); break;
            case ":load": ReplaceModel(Model.Load(RequireArgument(argument))); break;
            case ":info":
                var model = RequireModel();
                output.WriteLine($"Tokens: {model.N}; vocabulary: {model.VocabSize}; model: {model.ModelPath ?? "in memory"}");
                break;
            case ":count":
                model = RequireModel();
                output.WriteLine(model.Count(model.Tokenizer.Encode(RequireArgument(argument))));
                break;
            case ":search":
                model = RequireModel();
                var positions = model.Search(model.Tokenizer.Encode(RequireArgument(argument)), limit: 50);
                output.WriteLine(positions.Count == 0 ? "No matches." : $"Byte offsets (up to 50, suffix order): {string.Join(", ", positions)}");
                break;
            case ":predict":
                model = RequireModel();
                var probabilities = model.Predict(model.Tokenizer.Encode(argument), topK: 10);
                if (probabilities is null || probabilities.Count == 0) output.WriteLine("No predictions.");
                else
                {
                    foreach (var (token, probability) in probabilities.OrderByDescending(pair => pair.Value))
                    {
                        string decoded = model.Tokenizer.Decode(new[] { token });
                        output.WriteLine($"{token}: {JsonSerializer.Serialize(decoded)}  {probability.ToString("P2", CultureInfo.InvariantCulture)}");
                    }
                }
                break;
            case ":generate": Generate(argument); break;
            case ":score":
                model = RequireModel();
                output.WriteLine(model.Score(model.Tokenizer.Encode(RequireArgument(argument))).ToString("G", CultureInfo.InvariantCulture));
                break;
            default: error.WriteLine("Unknown command. Type :help for commands."); break;
        }
        return true;
    }

    private void Generate(string context)
    {
        var model = RequireModel();
        output.WriteLine(model.GenerateText(context, maxTokens: 80, temperature: 0));
    }

    private static Model OpenFile(string path) => new(File.ReadAllBytes(path));

    private Model RequireModel() => _model ?? throw new InvalidOperationException("No corpus loaded. Use :text, :open, or :load.");

    private static string RequireArgument(string value) => value.Length == 0
        ? throw new ArgumentException("This command requires an argument. See :help.")
        : value;

    private void ReplaceModel(Model model)
    {
        var previous = _model;
        _model = model;
        previous?.Dispose();
        output.WriteLine($"Loaded {model.N} tokens.");
    }

    private static bool IsUserError(Exception ex) =>
        ex is ArgumentException or IOException or UnauthorizedAccessException or InvalidOperationException or NotSupportedException;

    private void PrintHelp()
    {
        output.WriteLine("""
            Usage: infinigram-repl [--file <corpus.txt> | --model <directory>]
            :text <text>       Build an in-memory index from text
            :open <path>       Build an in-memory index from a file's bytes
            :load <path>       Load a saved .NET model
            :info              Show model information
            :count <text>      Count occurrences
            :search <text>     Find byte offsets (first 50 in suffix order)
            :predict [text]    Show the top 10 next-byte predictions
            :generate [text]   Generate up to 80 bytes (greedy)
            :score <text>      Score a sequence (natural log probability)
            :help              Show this help
            :quit              Exit (EOF also exits)
            Any other line generates a continuation using that line as context.
            Text after the first command separator is preserved, including spaces.
            Predictions are individual UTF-8 bytes, so non-ASCII bytes may display as replacement characters.
            """);
    }
}
