using AwesomeAssertions;
using Infinigram.Repl;
using Model = Infinigram.Infinigram;

namespace Infinigram.Tests;

public class ReplTests
{
    [Fact]
    public void TextQueriesPreserveSpacesAndExitOnEof()
    {
        var output = new StringWriter();
        var error = new StringWriter();
        var repl = new ReplSession(new StringReader(":text ab ab\n:count ab\n:count  ab\n:search ab\n:info\n"), output, error);

        repl.Run(Array.Empty<string>()).Should().Be(0);
        output.ToString().Should().Contain("Loaded 5 tokens.");
        output.ToString().Should().Contain("infinigram> 2");
        output.ToString().Should().Contain("infinigram> 1");
        output.ToString().Should().Contain("Byte offsets (up to 50, suffix order): 3, 0");
        output.ToString().Should().Contain("Tokens: 5; vocabulary: 256; model: in memory");
        error.ToString().Should().BeEmpty();
    }

    [Fact]
    public void FailedReplacementKeepsPreviousModelAndCommandsContinue()
    {
        var output = new StringWriter();
        var error = new StringWriter();
        var repl = new ReplSession(new StringReader(":count a\n:text aaa\n:open /nonexistent/infinigram-file\n:count a\n:unknown\n:quit\n:count a\n"), output, error);

        repl.Run(Array.Empty<string>()).Should().Be(0);
        output.ToString().Should().Contain("infinigram> 3");
        error.ToString().Should().Contain("No corpus loaded").And.Contain("Unknown command").And.Contain("Error:");
    }

    [Fact]
    public void FileAndSavedModelCanBeLoadedAtStartup()
    {
        using var temp = new TempDir();
        var corpusPath = Path.Combine(temp.Path, "corpus with spaces.txt");
        File.WriteAllText(corpusPath, "banana");
        var modelPath = Path.Combine(temp.Path, "model");
        using (Model.Build("banana", modelPath, verbose: false)) { }

        foreach (var args in new[] { new[] { "--file", corpusPath }, new[] { "--model", modelPath } })
        {
            var output = new StringWriter();
            var error = new StringWriter();
            var repl = new ReplSession(new StringReader(":count ana\n:quit\n"), output, error);
            repl.Run(args).Should().Be(0);
            output.ToString().Should().Contain("infinigram> 2");
            error.ToString().Should().BeEmpty();
        }
    }

    [Fact]
    public void FileAndModelCanBeSwitchedDuringSession()
    {
        using var temp = new TempDir();
        var corpusPath = Path.Combine(temp.Path, "file with spaces.txt");
        File.WriteAllText(corpusPath, "banana");
        var modelPath = Path.Combine(temp.Path, "model with spaces");
        using (Model.Build("abcabc", modelPath, verbose: false)) { }
        var output = new StringWriter();
        var error = new StringWriter();
        var repl = new ReplSession(new StringReader($":open {corpusPath}\n:count ana\n:load {modelPath}\n:count abc\n:quit\n"), output, error);

        repl.Run(Array.Empty<string>()).Should().Be(0);
        output.ToString().Split("infinigram> 2").Length.Should().Be(3);
        error.ToString().Should().BeEmpty();
    }

    [Fact]
    public void InvalidArgumentsAndMissingStartupFileReturnErrors()
    {
        var output = new StringWriter();
        var error = new StringWriter();
        var repl = new ReplSession(new StringReader(""), output, error);
        repl.Run(new[] { "--file" }).Should().Be(2);
        repl.Run(new[] { "--file", Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString()) }).Should().Be(1);
        error.ToString().Should().Contain("Usage:").And.Contain("Error:");
    }

    [Fact]
    public void PredictAndGenerateAreAvailable()
    {
        var output = new StringWriter();
        var error = new StringWriter();
        var repl = new ReplSession(new StringReader(":text ababab\n:predict a\n:generate a\n:quit\n"), output, error);
        repl.Run(Array.Empty<string>()).Should().Be(0);
        output.ToString().Should().Contain("98: \"b\"");
        output.ToString().Should().Contain("infinigram> b");
        error.ToString().Should().BeEmpty();
    }
}
