using AwesomeAssertions;

namespace Infinigram.Tests;

public class SuffixArrayBuildTests
{
    [Fact]
    public void BuildHandlesEmptySingleAndRepeatedTokens()
    {
        new SuffixArray(Array.Empty<int>()).Positions.Should().BeEmpty();
        new SuffixArray(new[] { int.MinValue }).Positions.Should().Equal(0);
        new SuffixArray(new[] { 7, 7, 7 }).Positions.Should().Equal(2, 1, 0);
        new SuffixArray(new[] { int.MaxValue, int.MinValue }).Positions.Should().Equal(1, 0);
    }

    [Fact]
    public void BuildFromBytes()
    {
        var sa = new SuffixArray(TestData.B("the cat sat on the mat"));
        sa.Length.Should().Be(22);
    }

    [Fact]
    public void BuildFromIntList()
    {
        var tokens = new List<int> { 1, 100, 2, 500, 1, 100, 3, 1, 100, 2 };
        var sa = new SuffixArray(tokens);
        sa.Length.Should().Be(10);
    }

    [Fact]
    public void BuildFromArrayEqualsBuildFromList()
    {
        var tokens = TestData.Range(0, 50);
        var fromList = new SuffixArray(tokens);
        var fromArray = new SuffixArray(tokens.ToArray());
        fromList.Positions.Should().Equal(fromArray.Positions);
    }

    [Fact]
    public void ConstructionProducesSortedUniquePermutation()
    {
        var corpus = new List<int> { 1, 2, 3, 1, 2, 4 };
        var sa = new SuffixArray(corpus);
        sa.Positions.Should().HaveCount(corpus.Count);
        sa.Positions.OrderBy(p => p).Should().Equal(Enumerable.Range(0, corpus.Count));

        for (int i = 1; i < sa.Length; i++)
        {
            var prev = Suffix(corpus, sa.Positions[i - 1]);
            var curr = Suffix(corpus, sa.Positions[i]);
            LexicographicCompare(prev, curr).Should().BeLessThanOrEqualTo(0);
        }
    }

    private static List<int> Suffix(List<int> corpus, int position) =>
        corpus.GetRange(position, corpus.Count - position);

    private static int LexicographicCompare(List<int> a, List<int> b)
    {
        for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
        {
            if (a[i] != b[i])
            {
                return a[i].CompareTo(b[i]);
            }
        }
        return a.Count.CompareTo(b.Count);
    }
}

public class SuffixArraySearchTests
{
    [Fact]
    public void SearchSimple()
    {
        var sa = new SuffixArray(TestData.B("the cat sat on the mat"));
        sa.Search(TestData.B("the")).OrderBy(p => p).Should().Equal(new[] { 0, 15 });
    }

    [Fact]
    public void SearchMissingReturnsEmpty()
    {
        var sa = new SuffixArray(TestData.B("abc"));
        sa.Search(TestData.B("xyz")).Should().BeEmpty();
    }

    [Fact]
    public void SearchTokenPattern()
    {
        var tokens = new List<int> { 1, 100, 2, 500, 1, 100, 3, 1, 100, 2 };
        var sa = new SuffixArray(tokens);
        sa.Search(new List<int> { 1, 100 }).OrderBy(p => p).Should().Equal(new[] { 0, 4, 7 });
    }

    [Fact]
    public void SearchLongerPattern()
    {
        var tokens = new List<int> { 1, 100, 2, 500, 1, 100, 3, 1, 100, 2 };
        var sa = new SuffixArray(tokens);
        sa.Search(new List<int> { 1, 100, 2 }).OrderBy(p => p).Should().Equal(new[] { 0, 7 });
    }

    [Fact]
    public void SearchMissingToken()
    {
        var sa = new SuffixArray(new List<int> { 1, 2, 3 });
        sa.Search(new List<int> { 999 }).Should().BeEmpty();
    }

    [Fact]
    public void SearchEmptyPatternReturnsEmpty()
    {
        var sa = new SuffixArray(TestData.B("abc"));
        sa.Search(new List<int>()).Should().BeEmpty();
    }

    [Fact]
    public void SearchSingleToken()
    {
        var sa = new SuffixArray(new List<int> { 1, 2, 1, 3, 1 });
        sa.Search(new List<int> { 1 }).Should().HaveCount(3);
    }

    [Fact]
    public void SearchLargeVocabTokens()
    {
        var rng = new Random(42);
        var tokens = Enumerable.Range(0, 200).Select(_ => rng.Next(0, 50_000)).ToList();
        var sa = new SuffixArray(tokens);
        var pattern = tokens.Take(3).ToList();
        sa.Search(pattern).Should().Contain(0);
    }

    [Fact]
    public void SearchWithLimitReturnsAtMostLimit()
    {
        var sa = new SuffixArray(TestData.B("ababab"));
        sa.Search(TestData.B("ab"), limit: 2).Should().HaveCount(2);
    }

    [Fact]
    public void SearchLimitZeroReturnsEmpty()
    {
        var sa = new SuffixArray(TestData.B("test data"));
        sa.Search(TestData.B("test"), limit: 0).Should().BeEmpty();
    }

    [Fact]
    public void SearchLimitOneReturnsExactlyOne()
    {
        var sa = new SuffixArray(TestData.B("test test test"));
        var results = sa.Search(TestData.B("test"), limit: 1);
        results.Should().HaveCount(1);
        new[] { 0, 5, 10 }.Should().Contain(results[0]);
    }

    [Fact]
    public void SearchLimitLargerThanMatchCountReturnsAll()
    {
        var sa = new SuffixArray(TestData.B("abc def abc"));
        sa.Search(TestData.B("abc"), limit: 10).Should().HaveCount(2);
    }

    [Fact]
    public void SearchLimitNullReturnsAll()
    {
        var sa = new SuffixArray(TestData.B(string.Join(' ', Enumerable.Repeat("test", 50))));
        sa.Search(TestData.B("test")).Should().HaveCount(50);
    }
}

public class SuffixArrayCountTests
{
    [Fact]
    public void CountBytes()
    {
        var sa = new SuffixArray(TestData.B("aaabaaaba"));
        sa.Count(TestData.B("a")).Should().Be(7);
        sa.Count(TestData.B("aa")).Should().Be(4);
    }

    [Fact]
    public void CountTokens()
    {
        var sa = new SuffixArray(new List<int> { 1, 100, 1, 100, 1 });
        sa.Count(new List<int> { 1 }).Should().Be(3);
        sa.Count(new List<int> { 1, 100 }).Should().Be(2);
    }

    [Fact]
    public void CountMissingReturnsZero()
    {
        var sa = new SuffixArray(TestData.B("abc"));
        sa.Count(TestData.B("zz")).Should().Be(0);
    }
}

public class SuffixArrayGetContextTests
{
    [Fact]
    public void GetContextReturnsWindowAroundPosition()
    {
        var sa = new SuffixArray(TestData.Range(10, 20));
        var (before, after) = sa.GetContext(5, window: 3);
        before.Should().Equal(new[] { 12, 13, 14 });
        after.Should().Equal(new[] { 15, 16, 17 });
    }

    [Fact]
    public void GetContextClampsToCorpusBoundaries()
    {
        var sa = new SuffixArray(TestData.Range(10, 20));
        var (before, after) = sa.GetContext(0, window: 5);
        before.Should().BeEmpty();
        after.Should().Equal(new[] { 10, 11, 12, 13, 14 });
    }

    [Fact]
    public void GetContextOutOfRangeThrows()
    {
        var sa = new SuffixArray(TestData.Range(10, 20));
        Action act = () => sa.GetContext(20, window: 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

public class SuffixArraySaveLoadTests : IDisposable
{
    private readonly TempDir _temp = new();

    public void Dispose() => _temp.Dispose();

    [Fact]
    public void SaveLoadRoundTripPreservesSearchResults()
    {
        var corpus = TestData.B("the cat sat on the mat the cat ran on the dog");
        var original = new SuffixArray(corpus);

        var dir = _temp.SubDir("index");
        original.Save(dir);

        var reloaded = SuffixArray.Load(dir);
        reloaded.Length.Should().Be(original.Length);
        reloaded.Corpus.Should().Equal(original.Corpus);
        reloaded.Positions.Should().Equal(original.Positions);
        reloaded.Search(TestData.B("the")).OrderBy(p => p)
            .Should().Equal(original.Search(TestData.B("the")).OrderBy(p => p));
        reloaded.Count(TestData.B("cat")).Should().Be(original.Count(TestData.B("cat")));
    }

    [Fact]
    public void SaveUsesExistingLittleEndianFormat()
    {
        var dir = _temp.SubDir("format");
        new SuffixArray(new[] { 256, -1 }).Save(dir);

        File.ReadAllBytes(Path.Combine(dir, "corpus.bin")).Should().Equal(
            new byte[] { 2, 0, 0, 0, 0, 1, 0, 0, 255, 255, 255, 255 });
        File.ReadAllBytes(Path.Combine(dir, "suffix_array.bin")).Should().Equal(
            new byte[] { 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 });
    }

    [Fact]
    public void LoadRejectsMismatchedArrayLengths()
    {
        var dir = _temp.SubDir("mismatch");
        new SuffixArray(new[] { 1, 2 }).Save(dir);
        File.WriteAllBytes(Path.Combine(dir, "suffix_array.bin"), new byte[] { 0, 0, 0, 0 });

        Action act = () => SuffixArray.Load(dir);
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void LoadMissingIndexThrows()
    {
        var dir = _temp.SubDir("missing");
        Action act = () => SuffixArray.Load(dir);
        act.Should().Throw<FileNotFoundException>();
    }
}

public class SuffixArrayBinarySearchTests
{

    [Fact]
    public void ComparisonFindsExistingPattern()
    {
        var sa = new SuffixArray(TestData.B("hello world hello"));
        var positions = sa.Search(TestData.B("hello"));
        positions.Should().HaveCount(2);
        positions.OrderBy(p => p).Should().Equal(new[] { 0, 12 });
    }

    [Fact]
    public void ComparisonEmptyPatternReturnsEmpty()
    {
        var sa = new SuffixArray(TestData.B("test data"));
        sa.Search(TestData.B(string.Empty)).Should().BeEmpty();
    }

    [Fact]
    public void ComparisonPatternAtEndOfCorpus()
    {
        var sa = new SuffixArray(TestData.B("start middle end"));
        var positions = sa.Search(TestData.B("end"));
        positions.Should().HaveCount(1);
        positions[0].Should().Be(13);
    }

    [Fact]
    public void ComparisonPatternLongerThanSuffix()
    {
        var sa = new SuffixArray(TestData.B("abc"));
        sa.Search(TestData.B("abcdefgh")).Should().BeEmpty();
    }

    [Fact]
    public void ComparisonSingleBytePattern()
    {
        var sa = new SuffixArray(TestData.B("abacada"));
        sa.Search(TestData.B("a")).OrderBy(p => p).Should().Equal(new[] { 0, 2, 4, 6 });
    }

    [Fact]
    public void ComparisonFullCorpusAsPattern()
    {
        var corpus = TestData.B("unique_corpus_text");
        var sa = new SuffixArray(corpus);
        sa.Search(corpus).Should().Equal(new[] { 0 });
    }

    [Fact]
    public void ComparisonBinaryData()
    {
        var corpus = new List<int> { 0, 128, 255, 0, 128, 255, 42 };
        var sa = new SuffixArray(corpus);
        var positions = sa.Search(new List<int> { 0, 128, 255 });
        positions.Should().HaveCount(2);
        positions.OrderBy(p => p).Should().Equal(new[] { 0, 3 });
    }

    [Fact]
    public void ComparisonAdjacentSimilarPatterns()
    {
        var sa = new SuffixArray(TestData.B("abc abd abe"));
        sa.Search(TestData.B("abc")).Should().HaveCount(1);
        sa.Search(TestData.B("abd")).Should().HaveCount(1);
        sa.Search(TestData.B("abe")).Should().HaveCount(1);
        sa.Search(TestData.B("abf")).Should().BeEmpty();
    }

    [Fact]
    public void ComparisonAllSameBytes()
    {
        var sa = new SuffixArray(TestData.B("aaaaaaa"));
        sa.Search(TestData.B("aaa")).OrderBy(p => p).Should().Equal(new[] { 0, 1, 2, 3, 4 });
    }

    [Fact]
    public void ComparisonPrefixMatchDoesNotOvercount()
    {
        var sa = new SuffixArray(TestData.B("abc abcd abcde"));
        sa.Search(TestData.B("abc")).Should().HaveCount(3);
        sa.Search(TestData.B("abcd")).Should().HaveCount(2);
        sa.Search(TestData.B("abcde")).Should().HaveCount(1);
    }
}
