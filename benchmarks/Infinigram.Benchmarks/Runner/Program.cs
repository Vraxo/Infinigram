using BenchmarkDotNet.Running;
using Infinigram.Benchmarks.Model;

BenchmarkSwitcher.FromAssembly(typeof(BuildBenchmarks).Assembly).Run(args);
