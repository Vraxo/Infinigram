# Six-corpus transfer evaluation

**Run:** [Prediction benchmarks — 27 November 2026](https://github.com/Vraxo/Infinigram/actions/runs/46009374840)  
**Evaluated main commit:** `02c8b9522c4e254521cb2fe7deb7aded5a8bde95`  
**Evaluation configuration:** 2,048 evenly spaced target positions; context windows 0, 1, 3, 6, and 12  
**Output:** 510 CSV data rows across five evaluation modes

## Corpus groups

| Group | Corpus |
|---|---|
| Fiction | War and Peace |
| Fiction | Pride and Prejudice |
| Fiction | The Adventures of Sherlock Holmes |
| Nonfiction | On the Origin of Species |
| Nonfiction | The Steam Engine Explained and Illustrated |
| Nonfiction | The Federalist Papers |

The fiction and nonfiction groups each contain three works. The three nonfiction works cover natural science, mechanical engineering, and political/constitutional argument.

## Globally balanced transfer

Every source model in `cross-corpus-balanced` used exactly **126,146 word-index tokens**. At context window 3, the mean lexical-word top-1 accuracy across directed source/target pairs was:

| Training source → evaluation text | Pair count | Word top-1 | Word coverage | Word perplexity |
|---|---:|---:|---:|---:|
| Fiction → fiction | 6 | 9.43% | 89.21% | 368.0 |
| Fiction → nonfiction | 9 | 10.61% | 83.58% | 488.7 |
| Nonfiction → fiction | 9 | 9.09% | 82.94% | 627.9 |
| Nonfiction → nonfiction | 6 | **13.53%** | 85.13% | 366.2 |

The nonfiction-to-nonfiction top-1 average is 4.10 percentage points above fiction-to-fiction. All six directed transfers between the three nonfiction works scored between **11.41% and 15.44%**.

The difference persists across longer context windows:

| Context window | Fiction → fiction | Nonfiction → nonfiction | Difference |
|---:|---:|---:|---:|
| 1 | 8.96% | 13.84% | +4.88 pp |
| 3 | 9.43% | 13.53% | +4.10 pp |
| 6 | 9.48% | 13.47% | +3.99 pp |
| 12 | 9.61% | 13.63% | +4.03 pp |

## Target-specific interpretation

The advantage is not a universal advantage of nonfiction sources. When the target is nonfiction, the average top-1 score from nonfiction sources is higher than from fiction sources on all three targets:

| Evaluation target | Fiction sources | Nonfiction sources |
|---|---:|---:|
| On the Origin of Species | 9.51% | 11.72% |
| The Federalist Papers | 11.02% | 14.55% |
| The Steam Engine Explained and Illustrated | 11.31% | 14.34% |

When the target is fiction, the result is mixed:

| Evaluation target | Fiction sources | Nonfiction sources |
|---|---:|---:|
| Pride and Prejudice | 8.88% | 8.15% |
| The Adventures of Sherlock Holmes | 9.66% | 10.00% |
| War and Peace | 9.75% | 9.12% |

Across all external source models, the nonfiction targets also have higher average word top-1 accuracy than the fiction targets. This suggests that the target text itself matters: the three nonfiction works may contain more repetitive or structurally predictable language, and nonfiction sources appear to transfer better to those targets.

The top-1 difference should not be read as an across-the-board improvement in probability quality. Nonfiction-to-nonfiction word perplexity (366.2) is almost identical to fiction-to-fiction (368.0), while vocabulary coverage is lower. The metrics describe different aspects of prediction and should be considered together.

## Within-corpus and combined-training results

At context window 3, within-corpus lexical-word top-1 accuracy was:

| Corpus | Word top-1 |
|---|---:|
| On the Origin of Species | 21.37% |
| The Federalist Papers | 20.96% |
| The Steam Engine Explained and Illustrated | 19.88% |
| War and Peace | 15.51% |
| The Adventures of Sherlock Holmes | 13.36% |
| Pride and Prejudice | 12.90% |

Combined-training raised word coverage for all six targets, but lowered word top-1 for five of six and worsened word perplexity for all six relative to the within-corpus baseline. Broadly concatenating all available books is therefore not supported as a production-model change by this experiment.

## Conclusion and limitations

Across these six works, nonfiction-to-nonfiction transfer is consistently stronger in lexical-word top-1 accuracy than fiction-to-fiction transfer, and nonfiction sources transfer better to each nonfiction target. The result is not symmetric: nonfiction sources do not improve prediction on fiction targets as a group.

This is evidence about these six historical English works, not a general law about fiction and nonfiction. Genre is confounded with text structure, topic, author, period, and editorial conventions. The 2,048 positions are deterministic and evenly spaced rather than an independent random sample, so the results are descriptive rather than a formal significance test.

No production-model behavior was changed as a result of this evaluation.
