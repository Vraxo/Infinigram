# Eight-corpus transfer evaluation

**Run:** [Prediction benchmarks — 27 November 2026](https://github.com/Vraxo/Infinigram/actions/runs/46018515521)  
**Evaluated commit:** `21c450e2c1cd951b8b35c37aa9749d754310033c`  
**Evaluation configuration:** 2,048 evenly spaced target positions; context windows 0, 1, 3, 6, and 12  
**Output:** 920 CSV data rows across five evaluation modes

## Corpus set

The expanded set contains eight historical English works:

| Group | Corpus |
|---|---|
| Fiction | War and Peace |
| Fiction | Pride and Prejudice |
| Fiction | The Adventures of Sherlock Holmes |
| Expository nonfiction | On the Origin of Species |
| Expository nonfiction | The Steam Engine Explained and Illustrated |
| Expository nonfiction | The Federalist Papers |
| Narrative nonfiction | Personal Memoirs of U. S. Grant |
| Narrative nonfiction | Life on the Mississippi |

This gives three fiction sources and five nonfiction sources, including two first-person narrative memoirs alongside scientific, technical, and political/constitutional prose.

## Globally balanced transfer

Every source model in `cross-corpus-balanced` used exactly **126,146 word-index tokens**, the size of the smallest corpus. At context window 3, the mean lexical-target top-1 accuracy across directed source/target pairs was:

| Training source → evaluation text | Pair count | Word top-1 | Word coverage | Word perplexity |
|---|---:|---:|---:|---:|
| Fiction → fiction | 6 | 9.43% | 89.21% | 368.0 |
| Fiction → nonfiction | 15 | 10.14% | 84.45% | 470.9 |
| Nonfiction → fiction | 15 | 9.40% | 85.49% | 565.7 |
| Nonfiction → nonfiction | 20 | **11.83%** | 85.06% | 483.8 |

The nonfiction-to-nonfiction mean is **2.39 percentage points** above fiction-to-fiction. The direction remains consistent with the earlier six-corpus result, but the gap is smaller than the earlier 4.10-point difference after adding the two narrative nonfiction controls.

The result is not symmetric: nonfiction sources do not improve the average score on fiction targets as a group. Nonfiction-to-fiction (9.40%) is essentially the same as fiction-to-fiction (9.43%).

## Target-specific results

Nonfiction sources score higher than fiction sources on all five nonfiction targets:

| Evaluation target | Fiction sources | Nonfiction sources | Difference |
|---|---:|---:|---:|
| Life on the Mississippi | 8.04% | 8.70% | +0.66 pp |
| On the Origin of Species | 9.51% | 10.76% | +1.25 pp |
| Personal Memoirs of U. S. Grant | 10.82% | 12.37% | +1.55 pp |
| The Federalist Papers | 11.02% | 13.79% | +2.77 pp |
| The Steam Engine Explained and Illustrated | 11.31% | 13.51% | +2.21 pp |

The fiction targets remain mixed:

| Evaluation target | Fiction sources | Nonfiction sources | Difference |
|---|---:|---:|---:|
| Pride and Prejudice | 8.88% | 8.25% | −0.63 pp |
| The Adventures of Sherlock Holmes | 9.67% | 10.23% | +0.57 pp |
| War and Peace | 9.75% | 9.71% | −0.04 pp |

The new memoir targets matter: nonfiction sources retain an advantage on both, although it is modest for Twain's narrative.

## Expository and narrative nonfiction sources

Splitting nonfiction sources by broad text type gives a more useful view of the expanded set. The following are pair-weighted means at context window 3; self-transfer pairs are excluded.

| Source type | Fiction targets | Nonfiction targets |
|---|---:|---:|
| Fiction | 9.43% (6 pairs) | 10.14% (15 pairs) |
| Expository nonfiction | 9.09% (9 pairs) | 11.94% (12 pairs) |
| Narrative nonfiction | 9.85% (6 pairs) | 11.65% (8 pairs) |

Narrative nonfiction sources transfer to nonfiction targets at a level close to the expository sources. On fiction targets, the two memoir sources also perform competitively. This is descriptive evidence from only two narrative works, not a stable estimate for the whole subgenre.

## Context-window sensitivity

The balanced same-genre comparison persists at every nonzero context window:

| Context window | Fiction → fiction | Nonfiction → nonfiction | Difference |
|---:|---:|---:|---:|
| 1 | 8.96% | 12.08% | +3.13 pp |
| 3 | 9.43% | 11.83% | +2.39 pp |
| 6 | 9.48% | 11.80% | +2.32 pp |
| 12 | 9.61% | 11.81% | +2.20 pp |

The gap narrows as more context is available, but does not disappear in the tested range.

## Size-matched control

The separate `cross-corpus-size-matched` mode caps each source to the target's 80% training-token budget. Larger sources are drawn from 16 evenly spaced contiguous segments separated by a sentinel, so no context crosses a segment boundary.

At context window 3:

| Training source → evaluation text | Pair count | Word top-1 | Word coverage | Word perplexity |
|---|---:|---:|---:|---:|
| Fiction → fiction | 6 | 9.39% | 88.94% | 368.7 |
| Fiction → nonfiction | 15 | 10.28% | 85.00% | 482.6 |
| Nonfiction → fiction | 15 | 9.42% | 85.72% | 579.1 |
| Nonfiction → nonfiction | 20 | **11.98%** | 85.91% | 505.4 |

The same-genre difference is **2.59 percentage points**, close to the globally balanced result. The advantage therefore is not explained simply by giving nonfiction sources more training tokens.

The probability-quality metrics remain less encouraging. Nonfiction-to-nonfiction word perplexity is higher than fiction-to-fiction in both balanced and size-matched comparisons. The top-1 result should not be presented as a general improvement in probability quality.

## Within-corpus and combined-training results

At context window 3, within-corpus lexical-target top-1 accuracy was:

| Corpus | Word top-1 |
|---|---:|
| On the Origin of Species | 21.37% |
| The Federalist Papers | 20.96% |
| The Steam Engine Explained and Illustrated | 19.88% |
| Personal Memoirs of U. S. Grant | 18.41% |
| War and Peace | 15.51% |
| The Adventures of Sherlock Holmes | 13.36% |
| Pride and Prejudice | 12.90% |
| Life on the Mississippi | 10.01% |

Combined training increased word coverage for all eight targets, but reduced word top-1 for six of eight and worsened word perplexity for all eight relative to their within-corpus baselines. Broadly concatenating all available books is therefore still not supported as a production-model change by this evaluation.

## Interpretation and limitations

Across these eight works, nonfiction sources outperform fiction sources on every nonfiction target, and nonfiction-to-nonfiction transfer remains stronger than fiction-to-fiction transfer under both globally balanced and target-size-matched training budgets. Adding two narrative nonfiction works reduces the original six-corpus gap from 4.10 to 2.39 percentage points, which is an important qualification rather than a failure of the pattern.

The results do **not** establish that Infinigram has learned a general concept of genre. Genre remains confounded with topic, author, historical period, editorial conventions, and document structure. The corpus set is small and consists entirely of historical English works. The three fiction and five nonfiction sources also do not provide balanced coverage of every nonfiction subgenre.

The 2,048 target positions are deterministic and evenly spaced. They are descriptive samples, not independent replications, and this run reports no confidence intervals or formal significance test. Pair averages also reuse the same source and target books across multiple comparisons.

Word top-1 is measured only at lexical target positions, while the model's full word-and-punctuation vocabulary remains eligible for the top prediction. Coverage is reported separately, and word perplexity is calculated only for covered lexical targets.

**Conclusion:** the observed transfer pattern survives a meaningful expansion with narrative nonfiction controls, but its reduced size and the mixed fiction-target results argue for a narrow claim: these nonfiction sources transfer better among the tested nonfiction works. The next evaluation should use repeated, reproducible target resampling with uncertainty estimates and a broader, more balanced collection of contemporary and historical texts.

No production-model behavior was changed as a result of this evaluation.
