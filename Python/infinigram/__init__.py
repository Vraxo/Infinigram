"""Infinigram: Corpus-based language modeling via suffix arrays.

Index a corpus of token ids (byte-level by default, or any tokenizer) and
query it as a variable-length n-gram model with O(m log n) pattern matching,
autoregressive generation, and sequence scoring.
"""

from infinigram.infinigram import Infinigram
from infinigram.suffix_array import SuffixArray
from infinigram.tokenizers import ByteTokenizer, Tokenizer, TokenIds, vocab_size_of
from infinigram import corpus_utils
from infinigram import sampling
from infinigram import tokenizers

__version__ = "0.7.0"  # Removed REPL, REST server, evaluation/storage/vfs, predict_weighted, confidence, min_count.
__author__ = "Alex Towell"
__email__ = "lex@metafunctor.com"

__all__ = [
    "Infinigram",
    "SuffixArray",
    "ByteTokenizer",
    "Tokenizer",
    "TokenIds",
    "vocab_size_of",
    "corpus_utils",
    "sampling",
    "tokenizers",
]
