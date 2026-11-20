#!/usr/bin/env python3
"""
Tests for corpus_utils module.

Tests for loading, converting, and processing corpora.
"""

import json
import tempfile
from pathlib import Path
import pytest

from infinigram.corpus_utils import (
    build_corpus_from_documents,
    iter_jsonl,
    build_corpus_from_jsonl,
    save_corpus_to_jsonl,
)


class TestBuildCorpusFromDocuments:
    """Tests for build_corpus_from_documents."""

    def test_single_document(self):
        docs = ["hello"]
        result = build_corpus_from_documents(docs)
        expected = list(b"hello")
        assert result == expected

    def test_multiple_documents_with_separator(self):
        docs = ["hello", "world"]
        result = build_corpus_from_documents(docs, separator=b"\n\n")
        expected = list(b"hello\n\nworld")
        assert result == expected

    def test_custom_separator(self):
        docs = ["a", "b", "c"]
        result = build_corpus_from_documents(docs, separator=b"\x00")
        expected = list(b"a\x00b\x00c")
        assert result == expected

    def test_empty_documents(self):
        docs = []
        result = build_corpus_from_documents(docs)
        assert result == []

    def test_byte_list_documents(self):
        docs = [[1, 2, 3], [4, 5, 6]]
        result = build_corpus_from_documents(docs, separator=b"\x00")
        assert result == [1, 2, 3, 0, 4, 5, 6]

    def test_invalid_byte_value_raises(self):
        docs = [[256, 1, 2]]  # 256 is invalid
        with pytest.raises(ValueError):
            build_corpus_from_documents(docs)


class TestIterJsonl:
    """Tests for iter_jsonl."""

    def test_iter_simple_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"text": "doc2"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc2", "doc3"]

    def test_iter_custom_field(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"content": "doc1"}\n')
            f.write('{"content": "doc2"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name, text_field='content'))
            assert docs == ["doc1", "doc2"]

    def test_iter_skips_missing_field(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"other": "ignored"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc3"]

    def test_iter_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('\n')
            f.write('{"text": "doc2"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc2"]


class TestBuildCorpusFromJsonl:
    """Tests for build_corpus_from_jsonl."""

    def test_build_corpus_from_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "hello"}\n')
            f.write('{"text": "world"}\n')
            f.flush()

            corpus = build_corpus_from_jsonl(f.name, separator=b"\x00")
            expected = list(b"hello\x00world")
            assert corpus == expected

    def test_build_corpus_max_documents(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"text": "doc2"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            corpus = build_corpus_from_jsonl(f.name, max_documents=2, separator=b"\x00")
            expected = list(b"doc1\x00doc2")
            assert corpus == expected


class TestSaveCorpusToJsonl:
    """Tests for save_corpus_to_jsonl."""

    def test_save_and_reload(self):
        docs = ["hello", "world", "test"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path)

        # Reload and verify
        reloaded = list(iter_jsonl(path))
        assert reloaded == docs

    def test_save_custom_field(self):
        docs = ["doc1", "doc2"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path, text_field='content')

        # Verify field name
        with open(path, 'r') as f:
            obj = json.loads(f.readline())
            assert 'content' in obj
            assert obj['content'] == 'doc1'

    def test_save_utf8_content(self):
        docs = ["café", "naïve"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path)
        reloaded = list(iter_jsonl(path))
        assert reloaded == docs


