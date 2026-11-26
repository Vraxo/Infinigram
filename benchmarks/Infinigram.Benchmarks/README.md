# Generation experiments

Run the deterministic decoding comparison from the repository root:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-generation
```

Choose a different CSV destination with:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-generation --output path/to/results.csv
```

The runner discovers UTF-8 `.txt` files copied from `Resources/Corpora`. It trains each tokenizer/model on the first 80% of each corpus and samples three fixed prompts from the held-out final 20%. It compares byte tokens with BPE vocabularies of 1,024 and 4,096 tokens. Prompt and output budgets are normalized approximately by UTF-8 byte length, so BPE uses fewer tokens for a similar text length.

For every tokenizer and prompt, it compares temperatures 0.7, 1.0, and 1.3 and repetition penalties 1.0, 1.15, and 1.5. Every configuration for a given prompt uses the same random seed and a top-40 sampling limit.

The CSV records tokenizer, vocabulary size, training size, generated text, token count, exact long-span overlap with the training partition, longest copied span, and mean conditional log-probability given the held-out prompt. Overlap spans are normalized approximately to 32 UTF-8 bytes; log-probability is per token and should only be compared within the same tokenizer. Neither metric is a human fluency score. Results are only as representative as the available corpus; currently the repository includes one real book corpus.
