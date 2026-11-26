# Generation experiments

Run the deterministic decoding comparison from the repository root:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-generation
```

Choose a different CSV destination with:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-generation --output path/to/results.csv
```

The runner discovers UTF-8 `.txt` files copied from `Resources/Corpora`. For each corpus it uses three fixed prompt positions and compares temperatures 0.7, 1.0, and 1.3 against repetition penalties 1.0, 1.15, and 1.5. Every configuration for a given prompt uses the same random seed, with a top-40 sampling limit, a 96-token output target, and a 256-token novelty history window.

The CSV records the generated text, token count, exact long-span corpus overlap, longest copied span, and mean per-token log-probability. The latter is only a model-relative consistency signal, not a human fluency measure. This is a controlled comparison of decoding settings, not a claim that one configuration produces better prose. Results are only as representative as the available corpus; currently the repository includes one real book corpus.
