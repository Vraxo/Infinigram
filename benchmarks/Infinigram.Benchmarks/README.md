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

The CSV records tokenizer, vocabulary size, training size, generated text, token count, exact long-span overlap with the training partition, longest copied span, and mean conditional log-probability given the held-out prompt. Overlap spans are normalized approximately to 32 UTF-8 bytes; log-probability is per token and should only be compared within the same tokenizer. Neither metric is a human fluency score. Results are only as representative as the available corpora. The current repository set contains eight historical English works: three fiction, three expository nonfiction, and two narrative nonfiction texts.

## Held-out next-token evaluation

Measure the word-index model's predictive performance on unseen text:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-prediction
```

Choose the number of evenly spaced target positions and a different CSV destination with:

```sh
dotnet run --project benchmarks/Infinigram.Benchmarks -- --evaluate-prediction --sample-count 2048 --output path/to/prediction.csv
```

The manual default is 512 sampled positions per evaluation sequence. The automated prediction workflow uses 2,048. Sample positions are evenly spaced across the evaluation sequence; because positions include punctuation, the number of lexical word targets is lower than the configured sample count.

The runner discovers all UTF-8 `.txt` corpora recursively and reports five evaluation modes:

- **Within-corpus:** train on the first 80% of a corpus and evaluate the held-out final 20%. The split is moved to a whitespace boundary so a word is never divided between training and validation.
- **Cross-corpus:** train on one complete corpus and evaluate every other complete corpus.
- **Size-matched cross-corpus:** for each target, cap the source at the target's 80% training-token count. Larger sources are represented by 16 evenly spaced contiguous segments separated by a sentinel, preventing contexts from crossing segment boundaries.
- **Globally balanced cross-corpus:** cap every source model at the smallest corpus's full word-index token count. This gives every source model the same training-token budget and is the cleanest source-to-source comparison.
- **Combined-training:** train on the target's first 80% plus every other complete corpus, then evaluate the target's held-out final 20%. Corpus boundaries are separated by a sentinel.

All modes use teacher forcing: earlier evaluation tokens are supplied as context, but the model never trains on evaluation text. Context windows are 0, 1, 3, 6, and 12 word-index tokens. A window of zero is the unconditional unigram baseline; context lengths count both words and punctuation.

The CSV identifies the evaluation mode, training corpus, and evaluation corpus, then reports token and lexical-word top-1 accuracy, vocabulary coverage, mean conditional log-probability, and perplexity. Top-1 accuracy counts out-of-vocabulary targets as incorrect. Log-probability and perplexity are calculated only for targets present in the training vocabulary; coverage is reported separately. Word-only metrics exclude punctuation targets.

The globally balanced mode is preferred for comparing source corpora because all source models have the same token budget. Compare combined-training results with the same target's within-corpus baseline to measure whether additional corpora help. More training text can increase coverage without improving top-1 accuracy or perplexity, so interpret all metrics together.

This evaluates predictive consistency, not generated-text quality or semantic understanding. Within-corpus results measure prediction inside a work; cross-corpus results measure transfer between distinct works; combined-training results measure the effect of adding external books to an otherwise unchanged target-corpus training split. Results remain limited by the number and variety of available corpora.
