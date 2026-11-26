# Text corpora

Place UTF-8 plain-text corpus files (.txt) in this directory. The tokenizer
benchmark discovers them recursively and runs byte-tokenizer and BPE
measurements against each file.

Suggested layout:

    Resources/
      Corpora/
        war-and-peace.txt
        other-book.txt
        domain/
          technical-prose.txt

## Corpus requirements

- Use plain UTF-8 text without a byte-order mark where possible.
- Keep the source text unchanged; do not normalize or preprocess it for the benchmark.
- Record the source, edition, URL, and license/usage terms alongside the corpus
  (for example, in a neighbouring .source.md file).
- Only add material that may legally be redistributed in this repository.
- Do not commit generated models, tokenizer caches, or benchmark output here.

The benchmark reports UTF-8 size, byte-token and BPE-token counts, token
reduction, learned merge count, and used BPE vocabulary. It also measures
encode, decode, and training time/allocation. A small synthetic mixed-language
fixture remains available as a stable baseline when no real corpus is present.

Large corpora can make the repository and CI checkout significantly larger.
Prefer a small, representative set and keep held-out evaluation text separate
from training corpora.
