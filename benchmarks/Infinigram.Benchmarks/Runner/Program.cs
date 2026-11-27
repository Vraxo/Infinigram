using BenchmarkDotNet.Running;
using Infinigram.Benchmarks.Model;
using Infinigram.Benchmarks.Runner;

if (args.Contains("--evaluate-generation", StringComparer.Ordinal))
{
    GenerationExperimentRunner.Run(args);
    return;
}

if (args.Contains("--evaluate-prediction", StringComparer.Ordinal))
{
    PredictionEvaluationRunner.Run(args);
    return;
}

BenchmarkSwitcher.FromAssembly(typeof(BuildBenchmarks).Assembly).Run(args);
