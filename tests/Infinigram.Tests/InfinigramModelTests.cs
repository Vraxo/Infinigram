using AwesomeAssertions;

namespace Infinigram.Tests;

public class ModelInitializationTests
{
    [Fact]
    public void InitializesWithByteVocabulary()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 4, 5 });
        model.N.Should().Be(5);
        model.VocabSize.Should().Be(256);
    }

    [Fact]
    public void MaxLengthIsStored()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 4, 5 }, maxLength: 3);
        model.MaxLength.Should().Be(3);
    }

    [Theory]
    [InlineData(new[] { 0 })]
    [InlineData(new[] { 128 })]
    [InlineData(new[] { 255 })]
    public void ValidBytesAreAccepted(int[] corpus)
    {
        using var model = new NGramModel(corpus.ToList());
        model.N.Should().Be(1);
    }

    [Theory]
    [InlineData(new[] { 256 })]
    [InlineData(new[] { -1 })]
    [InlineData(new[] { 0, 100, 300, 1000 })]
    public void InvalidBytesAreRejected(int[] corpus)
    {
        Action act = () => new NGramModel(corpus.ToList());
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*must contain only bytes*");
    }

    [Fact]
    public void EmptyCorpusThrows()
    {
        Action act = () => new NGramModel(new List<int>());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void InvalidUpdateDoesNotChangeCorpusOrCachedContinuations()
    {
        using var model = new NGramModel(TestData.B("abab"));
        var context = TestData.B("a");
        var before = model.Continuations(context);
        int cacheCount = model.ContinuationsCacheCount;

        Action update = () => model.Update(new[] { 256 }.ToList());
        update.Should().Throw<ArgumentOutOfRangeException>();
        model.N.Should().Be(4);
        model.Count(context).Should().Be(2);
        model.ContinuationsCacheCount.Should().Be(cacheCount);
        model.Continuations(context).Should().BeEquivalentTo(before);
    }

    [Fact]
    public void UpdateRejectsOutOfRangeTokens()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });

        Action act1 = () => model.Update(new List<int> { 256 });
        act1.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*new_tokens must contain only bytes*");

        Action act2 = () => model.Update(new List<int> { -1 });
        act2.Should().Throw<ArgumentOutOfRangeException>();
    }
}

public class LongestSuffixTests
{
    [Fact]
    public void ExactMatch()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 4, 2, 3, 5 });
        var (position, length) = model.LongestSuffix(new List<int> { 1, 2, 3 });
        length.Should().Be(3);
        position.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public void PartialMatch()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 4, 5 });
        var (_, length) = model.LongestSuffix(new List<int> { 99, 2, 3 });
        length.Should().Be(2);
    }

    [Fact]
    public void NoMatch()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var (position, length) = model.LongestSuffix(new List<int> { 99, 98, 97 });
        length.Should().Be(0);
        position.Should().Be(-1);
    }

    [Fact]
    public void EmptyContext()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        var (_, length) = model.LongestSuffix(new List<int>());
        length.Should().Be(0);
    }

    [Fact]
    public void RespectsMaxLength()
    {
        using var model = new NGramModel(TestData.Range(1, 8), maxLength: 3);
        var (_, length) = model.LongestSuffix(TestData.Range(1, 8));
        length.Should().BeLessThanOrEqualTo(3);
    }
}

public class FindAllSuffixMatchesTests
{
    [Fact]
    public void Basic()
    {
        using var model = new NGramModel(TestData.B("the cat sat on the mat").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("the cat"));

        matches.Should().NotBeEmpty();
        var lengths = matches.Select(m => m.Length).ToList();
        lengths.Should().Equal(lengths.OrderByDescending(l => l));
        foreach (var (length, positions) in matches)
        {
            length.Should().BeGreaterThan(0);
            positions.Should().NotBeEmpty();
        }
    }

    [Fact]
    public void MultipleLengths()
    {
        using var model = new NGramModel(TestData.B("abc abcd abcde").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("abcde"));
        matches.Select(m => m.Length).Distinct().Should().HaveCountGreaterThan(1);
    }

    [Fact]
    public void PositionsAreValid()
    {
        var corpus = TestData.B("the cat sat. the cat ran.");
        using var model = new NGramModel(corpus.ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("the cat"));

        foreach (var (length, positions) in matches)
        {
            positions.Should().OnlyContain(p => p >= 0 && p < corpus.Count);
        }
    }

    [Fact]
    public void NoMatch()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        model.FindAllSuffixMatches(TestData.B("xyz")).Should().BeEmpty();
    }

    [Fact]
    public void EmptyContext()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        model.FindAllSuffixMatches(new List<int>()).Should().BeEmpty();
    }

    [Fact]
    public void RespectsMaxLength()
    {
        using var model = new NGramModel(TestData.B("abcdefgh").ToArray(), maxLength: 3);
        var matches = model.FindAllSuffixMatches(TestData.B("abcdefgh"));
        if (matches.Count > 0)
        {
            matches.Max(m => m.Length).Should().BeLessThanOrEqualTo(3);
        }
    }

    [Fact]
    public void MaxLengthOneClampsToSingleByte()
    {
        using var model = new NGramModel(TestData.B("abcdefgh").ToArray(), maxLength: 1);
        var matches = model.FindAllSuffixMatches(TestData.B("abcdefgh"));
        if (matches.Count > 0)
        {
            matches.Max(m => m.Length).Should().Be(1);
        }
    }

    [Fact]
    public void SingleByteCorpus()
    {
        using var model = new NGramModel(TestData.B("a").ToArray());

        var matchesA = model.FindAllSuffixMatches(TestData.B("a"));
        matchesA.Should().HaveCount(1);
        matchesA[0].Length.Should().Be(1);
        matchesA[0].Positions.Should().Equal(new[] { 0 });

        model.FindAllSuffixMatches(TestData.B("b")).Should().BeEmpty();
    }

    [Fact]
    public void SingleByteContextInLongerCorpus()
    {
        using var model = new NGramModel(TestData.B("abcabc").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("a"));
        matches.Should().HaveCount(1);
        var (length, positions) = matches[0];
        length.Should().Be(1);
        positions.OrderBy(p => p).Should().Equal(new[] { 0, 3 });
    }

    [Fact]
    public void ContextLongerThanCorpusFindsPartialMatches()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("xyzabc"));

        matches.Should().HaveCount(3);
        matches.Select(m => m.Length).Should().Equal(new[] { 3, 2, 1 });
    }

    [Fact]
    public void PositionsCarryTheSuffix()
    {
        var corpus = TestData.B("the cat sat on the mat");
        var context = TestData.B("the cat");
        using var model = new NGramModel(corpus.ToArray());

        var matches = model.FindAllSuffixMatches(context);
        foreach (var (length, positions) in matches)
        {
            var suffix = context.GetRange(context.Count - length, length);
            foreach (var pos in positions)
            {
                corpus.GetRange(pos, length).Should().Equal(suffix);
            }
        }
    }

    [Fact]
    public void PositionCountsMatchOccurrences()
    {
        var corpus = TestData.B("abab ab ab abab");
        using var model = new NGramModel(corpus.ToArray());

        var matches = model.FindAllSuffixMatches(TestData.B("ab"));
        matches.Should().HaveCount(2);

        var length2 = matches.Single(m => m.Length == 2);
        var expectedCount = CountOccurrences(corpus, TestData.B("ab"));
        length2.Positions.Should().HaveCount(expectedCount);
    }

    [Fact]
    public void TokenizedQueryMatchesBytesQuery()
    {
        using var model = new NGramModel(TestData.B("the cat sat on the mat").ToArray());
        var viaEncode = model.FindAllSuffixMatches(model.Tokenizer.Encode("the cat"));
        var viaBytes = model.FindAllSuffixMatches(TestData.B("the cat"));
        viaEncode.Should().HaveCount(viaBytes.Count);
        for (int i = 0; i < viaBytes.Count; i++)
        {
            viaEncode[i].Length.Should().Be(viaBytes[i].Length);
            viaEncode[i].Positions.Should().Equal(viaBytes[i].Positions);
        }
    }

    [Fact]
    public void RepeatedPatternPositionsAreAllEven()
    {
        using var model = new NGramModel(TestData.B(new string('a', 0) + string.Concat(Enumerable.Repeat("ab", 100))).ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("ab"));

        var length2 = matches.Single(m => m.Length == 2);
        length2.Positions.Should().HaveCount(100);
        length2.Positions.OrderBy(p => p).Should().Equal(Enumerable.Range(0, 100).Select(i => i * 2));
    }

    [Fact]
    public void OverlappingMatchesFound()
    {
        using var model = new NGramModel(TestData.B("aaaa").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("aa"));

        var length2 = matches.Single(m => m.Length == 2);
        length2.Positions.OrderBy(p => p).Should().Equal(new[] { 0, 1, 2 });
    }

    [Fact]
    public void OverlappingLongerPatternFound()
    {
        using var model = new NGramModel(TestData.B("aaaaa").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("aaa"));

        var length3 = matches.Single(m => m.Length == 3);
        length3.Positions.OrderBy(p => p).Should().Equal(new[] { 0, 1, 2 });
    }

    [Fact]
    public void AllLengthsFound()
    {
        using var model = new NGramModel(TestData.B("abcdefg").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("defg"));
        matches.Select(m => m.Length).Should().Equal(new[] { 4, 3, 2, 1 });
    }

    [Fact]
    public void PartialSuffixOnly()
    {
        using var model = new NGramModel(TestData.B("xyz").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("abc xyz"));

        matches.Should().NotBeEmpty();
        matches.Max(m => m.Length).Should().BeLessThanOrEqualTo(4);
    }

    [Fact]
    public void UnicodeAfterEncode()
    {
        using var model = new NGramModel(TestData.B("hello world").ToArray());
        var matches = model.FindAllSuffixMatches(model.Tokenizer.Encode("hello"));
        matches.Should().NotBeEmpty();
        matches.Max(m => m.Length).Should().Be(5);
    }

    [Fact]
    public void SpecialBytes()
    {
        using var model = new NGramModel(new List<int> { 0, 1, 255, 254, 0, 1 });
        var matches = model.FindAllSuffixMatches(new List<int> { 0, 1 });

        matches.Should().NotBeEmpty();
        var length2 = matches.Single(m => m.Length == 2);
        length2.Positions.OrderBy(p => p).Should().Equal(new[] { 0, 4 });
    }

    [Fact]
    public void NoFalsePositives()
    {
        var corpus = TestData.B("ab cd ef");
        var context = TestData.B("ab");
        using var model = new NGramModel(corpus.ToArray());

        var matches = model.FindAllSuffixMatches(context);
        foreach (var (length, positions) in matches)
        {
            var suffix = context.GetRange(context.Count - length, length);
            positions.Should().OnlyContain(p => corpus.GetRange(p, length).SequenceEqual(suffix));
        }
    }

    [Fact]
    public void PositionsAreLists()
    {
        using var model = new NGramModel(TestData.B("test test test").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("test"));
        foreach (var (_, positions) in matches)
        {
            positions.Should().NotBeNull();
            positions.Should().NotBeEmpty();
        }
    }

    [Fact]
    public void NewlinesInCorpus()
    {
        using var model = new NGramModel(TestData.B("line1\nline2\nline1").ToArray());
        var matches = model.FindAllSuffixMatches(TestData.B("line1"));

        matches.Should().NotBeEmpty();
        var length5 = matches.Single(m => m.Length == 5);
        length5.Positions.Should().HaveCount(2);
    }

    private static int CountOccurrences(List<int> corpus, List<int> pattern)
    {
        int count = 0;
        for (int i = 0; i + pattern.Count <= corpus.Count; i++)
        {
            if (corpus.GetRange(i, pattern.Count).SequenceEqual(pattern))
            {
                count++;
            }
        }
        return count;
    }
}

public class ContinuationsTests
{
    [Fact]
    public void ContinuationsBasic()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 1, 2, 4, 1, 2, 5 });
        var conts = model.Continuations(new List<int> { 1, 2 });

        conts.Should().NotBeNull();
        conts!.Should().ContainKey(3);
        conts.Should().ContainKey(4);
        conts.Should().ContainKey(5);
        conts![3].Should().Be(1);
        conts![4].Should().Be(1);
        conts![5].Should().Be(1);
    }

    [Fact]
    public void EmptyContextReturnsUniform()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 1, 3, 1 });
        var conts = model.Continuations(new List<int>());

        conts.Should().NotBeNull();
        conts!.Should().HaveCount(256);
        conts.Values.Should().OnlyContain(v => v == 1);
    }

    [Fact]
    public void NoMatchReturnsNull()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3 });
        model.Continuations(new List<int> { 99, 98, 97 }).Should().BeNull();
    }

    [Fact]
    public void MatchOnlyAtEndOfCorpusFallsBack()
    {
        using var model = new NGramModel(new List<int> { 1, 2, 3, 4, 5 });
        var conts = model.Continuations(new List<int> { 4, 5 });
        (conts is null || conts.Count > 0).Should().BeTrue();
    }

    [Fact]
    public void LimitCapsSampleCount()
    {
        using var model = new NGramModel(TestData.B(new string('x', 0) + string.Concat(Enumerable.Repeat("ab", 500))).ToArray());

        var full = model.Continuations(TestData.B("a"), limit: null);
        var limited = model.Continuations(TestData.B("a"), limit: 100);

        full![98].Should().Be(500);
        limited!.Values.Sum().Should().BeLessThanOrEqualTo(100);
        limited.Should().ContainKey(98);
    }

    [Fact]
    public void DefaultLimitIsApplied()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("ab", 20000))).ToArray());
        var conts = model.Continuations(TestData.B("a"));
        conts.Should().NotBeNull();
        conts!.Values.Sum().Should().BeLessThanOrEqualTo(NGramModel.DefaultContinuationLimit);
    }

    [Fact]
    public void ExplicitLimitRespected()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("ab", 1000))).ToArray());
        var conts = model.Continuations(TestData.B("a"), limit: 10);
        conts![98].Should().BeLessThanOrEqualTo(10);
    }

    [Fact]
    public void LimitNoneUsesAllOccurrences()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("ab", 100))).ToArray());
        var conts = model.Continuations(TestData.B("a"), limit: null);
        conts![98].Should().Be(100);
    }

    [Fact]
    public void ContinuationsAreCached()
    {
        using var model = new NGramModel(TestData.B("the cat sat on the mat the cat ran").ToArray());
        var context = TestData.B("the cat");

        var first = model.Continuations(context);
        model.ContinuationsCacheCount.Should().BeGreaterThanOrEqualTo(1);

        var second = model.Continuations(context);
        first.Should().NotBeNull();
        second.Should().NotBeNull();
        first.Should().BeEquivalentTo(second);
    }

    [Fact]
    public void CachedResultsAreCopies()
    {
        using var model = new NGramModel(TestData.B("abcabd").ToArray());
        var context = TestData.B("ab");

        var first = model.Continuations(context);
        first![999] = 42;

        var second = model.Continuations(context);
        second.Should().NotContainKey(999);
    }

    [Fact]
    public void CacheClearedAfterUpdate()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        model.Continuations(TestData.B("ab"));
        model.ContinuationsCacheCount.Should().BeGreaterThan(0);

        model.Update(TestData.B("def").ToArray());
        model.ContinuationsCacheCount.Should().Be(0);
    }

    [Fact]
    public void CacheReflectsUpdatedCorpus()
    {
        using var model = new NGramModel(TestData.B("ab ab ab").ToArray());
        var context = TestData.B("ab");
        model.Continuations(context);

        model.Update(TestData.B(" abc").ToArray());
        var after = model.Continuations(context);

        after.Should().NotBeNull();
        after!.Should().NotBeEmpty();
    }

    [Fact]
    public void ClearCacheEmptiesCache()
    {
        using var model = new NGramModel(TestData.B("the cat sat on the mat").ToArray());
        model.Continuations(TestData.B("the"));
        model.Continuations(TestData.B("cat"));
        model.Continuations(TestData.B("sat"));
        model.ContinuationsCacheCount.Should().BeGreaterThanOrEqualTo(3);

        model.ClearCache();
        model.ContinuationsCacheCount.Should().Be(0);
    }

    [Fact]
    public void CacheEvictionStaysBounded()
    {
        using var model = new NGramModel(Enumerable.Range(0, 256).SelectMany(v => Enumerable.Repeat(v, 10)).ToList());
        model.ContinuationsCacheCapacity = 10;

        for (int i = 0; i < 25; i++)
        {
            model.Continuations(new List<int> { i, (i + 1) % 256 });
        }

        model.ContinuationsCacheCount.Should().BeLessThanOrEqualTo(model.ContinuationsCacheCapacity);
    }

    [Fact]
    public void CacheEvictionKeepsResultsCorrect()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("abcdefghijklmnopqrstuvwxyz", 5))).ToArray());
        model.ContinuationsCacheCapacity = 5;

        for (int i = 0; i < 20; i++)
        {
            model.Continuations(new List<int> { 'a' + i });
        }

        var result = model.Continuations(TestData.B("a"));
        result.Should().NotBeNull();
        result!.Should().ContainKey((int)'b');
    }

    [Fact]
    public void DifferentLimitsAreSeparateCacheEntries()
    {
        using var model = new NGramModel(TestData.B(string.Concat(Enumerable.Repeat("ab", 500))).ToArray());
        model.Continuations(TestData.B("a"), limit: 10);
        model.Continuations(TestData.B("a"), limit: 100);

        model.ContinuationsCacheCount.Should().BeGreaterThanOrEqualTo(2);
    }

    [Fact]
    public void OodContextsAreCachedAsMisses()
    {
        using var model = new NGramModel(TestData.B("abc").ToArray());
        model.Continuations(new List<int> { 0xFF, 0xFE, 0xFD });
        model.Continuations(new List<int> { 0xFA, 0xFB, 0xFC });

        model.ContinuationsCacheCount.Should().Be(2);
        model.Continuations(new List<int> { 0xFF, 0xFE, 0xFD }).Should().BeNull();
    }

    [Fact]
    public void ContinuationCountsMatchManualLoop()
    {
        var corpus = TestData.B("the cat sat on the mat the cat ran on the dog");
        using var model = new NGramModel(corpus.ToArray());

        var context = TestData.B("the");
        var modelConts = model.Continuations(context, limit: null);

        var reference = new Dictionary<int, int>();
        for (int i = 0; i + context.Count <= corpus.Count; i++)
        {
            if (corpus.GetRange(i, context.Count).SequenceEqual(context))
            {
                int next = i + context.Count;
                if (next < corpus.Count)
                {
                    reference[corpus[next]] = reference.TryGetValue(corpus[next], out var c) ? c + 1 : 1;
                }
            }
        }

        modelConts.Should().NotBeNull();
        modelConts.Should().BeEquivalentTo(reference);
    }

    [Fact]
    public void ContinuationCountsMatchManualLoopDiverseCorpus()
    {
        var corpus = TestData.B("ab1 ab2 ab3 ab4 ab5 cd1 cd2 ef1");
        using var model = new NGramModel(corpus.ToArray());

        foreach (var ctx in new[] { "ab", "cd", "ef" })
        {
            var context = TestData.B(ctx);
            var modelConts = model.Continuations(context, limit: null);

            var reference = new Dictionary<int, int>();
            for (int i = 0; i + context.Count <= corpus.Count; i++)
            {
                if (corpus.GetRange(i, context.Count).SequenceEqual(context))
                {
                    int next = i + context.Count;
                    if (next < corpus.Count)
                    {
                        reference[corpus[next]] = reference.TryGetValue(corpus[next], out var c) ? c + 1 : 1;
                    }
                }
            }

            modelConts.Should().NotBeNull();
            modelConts.Should().BeEquivalentTo(reference);
        }
    }

    [Fact]
    public void ContinuationCountsCoverEveryByteValue()
    {
        var pattern = new List<int> { 253, 254 };
        var corpusParts = new List<int>();
        for (int b = 0; b < 256; b++)
        {
            corpusParts.AddRange(new[] { 253, 254, b, 0 });
        }

        using var model = new NGramModel(corpusParts);
        var conts = model.Continuations(pattern, limit: null);

        conts.Should().NotBeNull();
        conts!.Should().HaveCount(256);
        foreach (var b in Enumerable.Range(0, 256))
        {
            conts![b].Should().BeGreaterThanOrEqualTo(1);
        }
    }

    [Fact]
    public void SingleOccurrenceHasCountOne()
    {
        using var model = new NGramModel(TestData.B("unique_pattern_xyz").ToArray());
        var conts = model.Continuations(TestData.B("unique_pattern_xy"));

        conts![(int)'z'].Should().Be(1);
        conts.Values.Sum().Should().Be(1);
    }

    [Fact]
    public void AllMatchesAtEndStillReturnsDictionary()
    {
        using var model = new NGramModel(TestData.B("xyzxyz").ToArray());
        var conts = model.Continuations(TestData.B("xyz"));

        conts.Should().NotBeNull();
        conts!.Should().NotBeEmpty();
    }

    [Fact]
    public void CountMatchesKnownValues()
    {
        var corpus = TestData.B("abc abc abc");
        using var model = new NGramModel(corpus.ToArray());

        var conts = model.Continuations(TestData.B("ab"));
        conts![(int)'c'].Should().Be(3);
        conts.Values.Sum().Should().Be(3);
    }
}
