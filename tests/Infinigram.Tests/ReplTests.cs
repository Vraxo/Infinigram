using AwesomeAssertions;
using Infinigram.Repl;
using Model = Infinigram.Infinigram;

namespace Infinigram.Tests;

public class ReplTests
{
    [Fact]
    public void LoadsCorpusAndGeneratesForEachLineUntilEof()
    {
        using var temp = new TempDir();
        string path = Path.Combine(temp.Path, "corpus.txt");
        File.WriteAllText(path, "abcabcabc");
        var output = new StringWriter();
        var error = new StringWriter();
        using var model = new Model(File.ReadAllBytes(path));
        string[] prompts = { "a", "a ", ":quit" };
        string input = string.Join('\n', prompts) + "\n";

        ReplSession.Run(new StringReader(input), output, error, path).Should().Be(0);

        string expected = "Enter a prompt to generate a continuation. End input to exit." + Environment.NewLine;
        foreach (string prompt in prompts)
        {
            expected += "infinigram> " + model.GenerateText(prompt, maxTokens: 80, temperature: 0) + Environment.NewLine;
        }
        expected += "infinigram> ";
        output.ToString().Should().Be(expected);
        error.ToString().Should().BeEmpty();
    }

    [Fact]
    public void MissingOrEmptyFileFailsBeforePrompting()
    {
        using var temp = new TempDir();
        string missing = Path.Combine(temp.Path, "missing.txt");
        string empty = Path.Combine(temp.Path, "empty.txt");
        File.WriteAllText(empty, "");
        foreach (string path in new[] { missing, empty })
        {
            var output = new StringWriter();
            var error = new StringWriter();
            ReplSession.Run(new StringReader("hello\n"), output, error, path).Should().Be(1);
            output.ToString().Should().BeEmpty();
            error.ToString().Should().Contain("Could not load corpus").And.Contain(path);
        }
    }
}
