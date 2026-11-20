namespace Infinigram;

public sealed record TextNormalizationOptions
{
    public bool UnicodeNfc { get; init; } = true;
    public bool NormalizeLineEndings { get; init; } = true;
    public bool Lowercase { get; init; }
    public bool CollapseWhitespace { get; init; }
}
