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
- Preserve source spelling, punctuation, and prose. Remove only distribution boilerplate when needed, and document extraction and cleanup steps below.
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


## Pride and Prejudice

- **Author:** Jane Austen
- **Original publication:** 1813
- **Source edition:** Project Gutenberg eBook #42671, edited by R. W. Chapman
- **Project Gutenberg record:** https://www.gutenberg.org/ebooks/42671
- **Preparation:** novel text retained; Gutenberg front/back matter, volume title pages, chapter headings, and illustration placeholders removed.

Project Gutenberg identifies eBook #42671 as public domain in the United States. The corpus contains the novel's prose rather than its distribution metadata.


## The Adventures of Sherlock Holmes

- **Author:** Arthur Conan Doyle
- **Original collection publication:** 1892
- **Source text:** Sherlock Holmes Collection, version 3.1 — https://sherlock-holm.es
- **Preparation:** all twelve story bodies and titles retained; title page, tables of contents, chapter-number markers, and collection/distribution footer removed.

## On the Origin of Species

- **Author:** Charles Darwin
- **Original publication:** 1859, first edition
- **Source edition:** Project Gutenberg eBook #1228
- **Project Gutenberg record:** https://www.gutenberg.org/ebooks/1228
- **Source repository copy:** https://github.com/GITenberg/On-the-Origin-of-Species-By-Means-of-Natural-Selection--13-Or-the-Preservation-of-Favoured-Rac__1228/blob/master/1228.txt
- **Preparation:** introduction and complete fourteen-chapter body retained; title/front matter, table of contents, index, and Project Gutenberg distribution/license text removed. Original chapter and section headings are retained.
