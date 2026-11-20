// Port of Python/tests/test_infinigram.py — TestResourceLifecycle (close/dispose).
// The mmap/fd-leak variant is not applicable (models are in-memory);
// repeated Update rebuilds are covered in InfinigramPredictTests.
using AwesomeAssertions;

namespace Infinigram.Tests;

public class ResourceLifecycleTests
{
    [Fact]
    public void DisposeIsIdempotent()
    {
        var model = new Infinigram(TestData.B("hello world").ToArray());
        model.Dispose();
        model.Dispose(); // Should not raise
    }

    [Fact]
    public void QueriesAfterDisposeThrow()
    {
        var model = new Infinigram(TestData.B("hello world").ToArray());
        var context = TestData.B("hello");
        model.Dispose();

        Action predict = () => model.Predict(context);
        predict.Should().Throw<ObjectDisposedException>();

        Action search = () => model.Search(context);
        search.Should().Throw<ObjectDisposedException>();

        Action count = () => model.Count(context);
        count.Should().Throw<ObjectDisposedException>();

        Action getToken = () => model.GetToken(0);
        getToken.Should().Throw<ObjectDisposedException>();

        Action continuations = () => model.Continuations(context);
        continuations.Should().Throw<ObjectDisposedException>();

        Action longestSuffix = () => model.LongestSuffix(context);
        longestSuffix.Should().Throw<ObjectDisposedException>();

        Action score = () => model.Score(context);
        score.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public void UpdateAfterDisposeThrows()
    {
        var model = new Infinigram(TestData.B("hello world").ToArray());
        model.Dispose();

        Action update = () => model.Update(TestData.B(" more").ToArray());
        update.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public void RepeatedUpdatesAreConsistent()
    {
        using var model = new Infinigram(TestData.B("hello").ToArray());
        for (int i = 0; i < 20; i++)
        {
            model.Update(new List<int> { 65 + (i % 26) });
        }

        model.N.Should().Be(25); // 5 original + 20 appended
        model.Predict(TestData.B("hello")).Should().NotBeNull();
    }

    [Fact]
    public void DisposeClearsTheCache()
    {
        var model = new Infinigram(TestData.B("the cat sat on the mat").ToArray());
        model.Continuations(TestData.B("the"));
        model.ContinuationsCacheCount.Should().BeGreaterThan(0);

        model.Dispose();
        model.ContinuationsCacheCount.Should().Be(0);
    }
}
