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

## Held-out next-token evaluation

Measure the word-index model's predictive performance on unseen text:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-prediction
```

Choose a different CSV destination with:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-prediction --output path/to/prediction.csv
```

The runner reports two evaluation modes. Within-corpus evaluation trains on the first 80% of each UTF-8 corpus and evaluates up to 512 evenly spaced targets from the held-out final 20%. The split is moved to a whitespace boundary so a word is never divided between training and validation. Cross-corpus evaluation trains on one complete corpus and evaluates up to 512 evenly spaced targets from every other complete corpus. No evaluation text is included in the cross-corpus training model. Both modes use teacher forcing: earlier evaluation tokens are supplied as context, but the model never trains on them.

It compares context windows of 0, 1, 3, 6, and 12 word-index tokens. A window of zero is the unconditional unigram baseline. Context lengths count both words and punctuation tokens.

The CSV identifies the evaluation mode, training corpus, and evaluation corpus, then reports token and lexical-word top-1 accuracy, vocabulary coverage, mean conditional log-probability, and perplexity. Top-1 accuracy counts out-of-vocabulary targets as incorrect. Log-probability and perplexity are calculated only for targets present in the training vocabulary; coverage is reported separately so those scores cannot hide OOV failures. Word-only metrics exclude punctuation targets.

This evaluates predictive consistency, not generated-text quality or semantic understanding. Within-corpus results measure prediction inside a work; cross-corpus results measure transfer between distinct works and should be interpreted alongside vocabulary coverage. Both remain limited by the number and variety of available corpora.
