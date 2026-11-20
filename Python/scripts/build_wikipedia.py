"""Build an Infinigram index over the English Wikipedia corpus with cl100k_base.

Phase 1: Stream-tokenize the ~20 GB text file into a flat int32 token file on
disk, so peak RAM stays bounded by one text chunk (not the full corpus).

Phase 2: Hand the mmap'd token file to ``Infinigram.build``, which triggers
``ChunkedMmapSuffixArray`` (one SA per chunk; mergeable at query time).

Run:
    python scripts/build_wikipedia.py  # defaults below
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tiktoken

from infinigram import Infinigram


CORPUS_PATH = Path(__file__).resolve().parent.parent / "raw-data" / "wikipedia_en.txt"
INDEX_PATH = Path.home() / ".infinigram" / "models" / "wikipedia"
CACHE_DIR = Path.home() / ".infinigram" / "cache"
TEXT_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB per tokenization batch


def _split_at_whitespace(buf: bytes) -> tuple[bytes, bytes]:
    """Split ``buf`` at the last whitespace so BPE merges don't straddle chunks."""
    for i in range(len(buf) - 1, -1, -1):
        if buf[i] in (0x20, 0x0A, 0x0D, 0x09):
            return buf[: i + 1], buf[i + 1 :]
    return buf, b""


def stream_tokenize(
    corpus_path: Path,
    tokens_path: Path,
    tokenizer: tiktoken.Encoding,
    text_chunk_bytes: int = TEXT_CHUNK_BYTES,
) -> int:
    """Tokenize ``corpus_path`` into ``tokens_path`` as a flat int32 file.

    Reads text chunks at whitespace boundaries so BPE merges are not split.
    Returns the total token count.
    """
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_size = corpus_path.stat().st_size

    total_tokens = 0
    bytes_read = 0
    tail = b""
    start = time.time()

    with open(corpus_path, "rb") as src, open(tokens_path, "wb") as dst:
        while True:
            raw = src.read(text_chunk_bytes)
            if not raw:
                break
            buf = tail + raw
            # Only split mid-stream; on the last chunk (empty raw) flush the tail.
            text_bytes, tail = _split_at_whitespace(buf)
            if not text_bytes:
                tail = buf
                continue

            text = text_bytes.decode("utf-8", errors="replace")
            ids = tokenizer.encode(text, disallowed_special=())
            arr = np.asarray(ids, dtype=np.int32)
            dst.write(arr.tobytes())

            total_tokens += arr.size
            bytes_read += len(text_bytes)
            elapsed = time.time() - start
            mb_s = bytes_read / (elapsed * 1e6) if elapsed > 0 else 0
            pct = 100 * bytes_read / corpus_size
            print(
                f"\r  tokenized {bytes_read / 1e9:.2f} / {corpus_size / 1e9:.2f} GB "
                f"({pct:.1f}%), {total_tokens / 1e6:.1f}M tokens, {mb_s:.1f} MB/s",
                end="",
                flush=True,
            )

        if tail:
            text = tail.decode("utf-8", errors="replace")
            ids = tokenizer.encode(text, disallowed_special=())
            arr = np.asarray(ids, dtype=np.int32)
            dst.write(arr.tobytes())
            total_tokens += arr.size

    print()
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--tokens-cache", type=Path, default=CACHE_DIR / "wikipedia_cl100k.i32")
    parser.add_argument("--chunk-size-gb", type=float, default=1.0,
                        help="SA chunk size in GB of int32 corpus (default 1.0 = ~268M tokens)")
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Reuse existing tokens file at --tokens-cache")
    args = parser.parse_args()

    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")
    if args.index.exists() and any(args.index.iterdir()):
        raise FileExistsError(
            f"Index directory {args.index} is not empty; remove it first."
        )

    enc = tiktoken.get_encoding(args.encoding)
    vocab_size = enc.n_vocab
    corpus_gb = args.corpus.stat().st_size / 1e9

    print(f"Corpus:     {args.corpus}  ({corpus_gb:.2f} GB)")
    print(f"Tokenizer:  {args.encoding}  (vocab = {vocab_size:,})")
    print(f"Index:      {args.index}")
    print(f"Tokens cache: {args.tokens_cache}")
    print()

    # Phase 1: tokenize.
    if args.skip_tokenize and args.tokens_cache.exists():
        total_tokens = args.tokens_cache.stat().st_size // 4
        print(f"Phase 1: skipping tokenization, reusing {total_tokens:,} tokens")
    else:
        print("Phase 1: streaming tokenization...")
        t0 = time.time()
        total_tokens = stream_tokenize(args.corpus, args.tokens_cache, enc)
        elapsed = time.time() - t0
        print(f"  done: {total_tokens:,} tokens in {elapsed:.0f}s "
              f"({total_tokens / max(elapsed, 1) / 1e6:.1f} M tokens/s)")
        print(f"  compression: {corpus_gb * 1e9 / total_tokens:.2f} chars/token")
    print()

    # Phase 2: build index from the mmap'd token file.
    print("Phase 2: building suffix array (chunked)...")
    tokens = np.memmap(args.tokens_cache, dtype=np.int32, mode="r")
    assert tokens.shape[0] == total_tokens

    t0 = time.time()
    Infinigram.build(
        tokens,
        str(args.index),
        tokenizer=enc,
        chunk_size_gb=args.chunk_size_gb,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"Phase 2 done in {elapsed:.0f}s")
    print()
    print(f"Index built at {args.index}")
    print(f"Total wall time: phase 1 + phase 2.")


if __name__ == "__main__":
    main()
