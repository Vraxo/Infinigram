#!/usr/bin/env python3
"""
Corpus utilities for Infinigram.

Helpers for building byte-level corpora from documents, JSONL files, and so on.
All functions return ``List[int]`` of bytes (0-255); they are byte-only and not
suitable for token-level (tiktoken / HuggingFace) models. For those, encode
text yourself and pass the result to ``Infinigram(tokens, tokenizer=...)``.
"""

from typing import List, Union, Optional, Iterator
from pathlib import Path
import json


def build_corpus_from_documents(
    documents: Union[List[str], List[List[int]]],
    separator: bytes = b"\n\n",
    encoding: str = 'utf-8'
) -> List[int]:
    """
    Build a byte-level corpus from multiple documents with separators.

    Documents are separated by a byte sequence (default: double newline)
    to prevent spurious cross-document patterns.

    Args:
        documents: List of text strings or byte sequences
        separator: Byte sequence to insert between documents (default: b"\\n\\n")
        encoding: Text encoding if documents are strings (default: 'utf-8')

    Returns:
        Byte sequence with separators between documents

    Example:
        >>> docs = ["the cat sat", "the cat ran"]
        >>> corpus = build_corpus_from_documents(docs)
        >>> # Result: b"the cat sat\\n\\nthe cat ran"
        >>> # In bytes: [116, 104, 101, ..., 10, 10, ...]

        >>> # Byte-level documents
        >>> byte_docs = [[1, 2, 3], [4, 5, 6]]
        >>> corpus = build_corpus_from_documents(byte_docs, separator=b"\\x00")
        >>> # Result: [1, 2, 3, 0, 4, 5, 6]
    """
    separator_bytes = list(separator)
    corpus_bytes = []

    for i, doc in enumerate(documents):
        # Convert to bytes if needed
        if isinstance(doc, str):
            doc_bytes = list(doc.encode(encoding))
        elif isinstance(doc, (list, bytes)):
            doc_bytes = list(doc)
        else:
            raise TypeError(f"Document must be str, list, or bytes. Got {type(doc)}")

        # Validate byte range
        invalid = [b for b in doc_bytes if not (0 <= b <= 255)]
        if invalid:
            raise ValueError(
                f"Document {i} contains invalid byte values: {invalid[:10]}"
            )

        corpus_bytes.extend(doc_bytes)

        # Add separator between documents (not after last)
        if i < len(documents) - 1:
            corpus_bytes.extend(separator_bytes)

    return corpus_bytes


# =============================================================================
# Large Dataset Loading Utilities
# =============================================================================

def iter_jsonl(
    path: Union[str, Path],
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> Iterator[str]:
    """
    Iterate over text documents in a JSONL file.

    Args:
        path: Path to JSONL file
        text_field: Field name containing text (default: 'text')
        encoding: Text encoding (default: 'utf-8')

    Yields:
        Text content from each line

    Example:
        >>> for doc in iter_jsonl("corpus.jsonl"):
        ...     print(len(doc))
    """
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if text_field in obj:
                    yield obj[text_field]


def build_corpus_from_jsonl(
    path: Union[str, Path],
    text_field: str = 'text',
    separator: bytes = b"\n\n",
    max_documents: Optional[int] = None,
    encoding: str = 'utf-8'
) -> List[int]:
    """
    Build corpus from a JSONL file.

    Args:
        path: Path to JSONL file
        text_field: Field name containing text (default: 'text')
        separator: Byte sequence between documents (default: b"\\n\\n")
        max_documents: Maximum documents to load (None = all)
        encoding: Text encoding (default: 'utf-8')

    Returns:
        Byte corpus as List[int]

    Example:
        >>> corpus = build_corpus_from_jsonl("wiki.jsonl", max_documents=1000)
        >>> model = Infinigram(corpus)
    """
    documents = []
    for i, text in enumerate(iter_jsonl(path, text_field, encoding)):
        if max_documents is not None and i >= max_documents:
            break
        documents.append(text)

    return build_corpus_from_documents(documents, separator=separator, encoding=encoding)


def save_corpus_to_jsonl(
    documents: List[str],
    path: Union[str, Path],
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> None:
    """
    Save documents to a JSONL file.

    Args:
        documents: List of text documents
        path: Output path
        text_field: Field name for text (default: 'text')
        encoding: Text encoding (default: 'utf-8')

    Example:
        >>> docs = ["Document 1", "Document 2"]
        >>> save_corpus_to_jsonl(docs, "corpus.jsonl")
    """
    with open(path, 'w', encoding=encoding) as f:
        for doc in documents:
            obj = {text_field: doc}
            f.write(json.dumps(obj) + '\n')
