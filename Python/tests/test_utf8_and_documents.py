"""Tests for UTF-8 round-tripping through ByteTokenizer and document-separator handling."""

from infinigram import Infinigram, ByteTokenizer
from infinigram.corpus_utils import build_corpus_from_documents


class TestDocumentSeparators:
    def test_build_corpus_from_text_documents(self):
        docs = ["Hello world", "Goodbye world"]
        corpus = build_corpus_from_documents(docs)
        assert isinstance(corpus, list)
        assert all(0 <= b <= 255 for b in corpus)
        sep = list(b"\n\n")
        assert any(corpus[i:i + len(sep)] == sep for i in range(len(corpus) - len(sep)))

    def test_build_corpus_from_byte_documents(self):
        assert build_corpus_from_documents([[1, 2, 3], [4, 5, 6]], separator=b"\x00") == [
            1, 2, 3, 0, 4, 5, 6,
        ]

    def test_custom_separator(self):
        docs = ["doc1", "doc2"]
        corpus = build_corpus_from_documents(docs, separator=b"<SEP>")
        sep = list(b"<SEP>")
        assert any(corpus[i:i + len(sep)] == sep for i in range(len(corpus) - len(sep)))

    def test_no_separator_after_last_document(self):
        corpus = build_corpus_from_documents([[1, 2], [3, 4]], separator=b"\x00")
        assert corpus == [1, 2, 0, 3, 4]

    def test_single_document_no_separator(self):
        corpus = build_corpus_from_documents(["Hello world"])
        assert corpus == list(b"Hello world")


class TestByteTokenizerWithInfinigram:
    """ByteTokenizer is the default. These confirm it round-trips through the model."""

    def test_roundtrip_ascii(self):
        corpus = "The quick brown fox"
        model = Infinigram(corpus, tokenizer=ByteTokenizer())
        assert model.tokenizer.decode(model.tokenizer.encode(corpus)) == corpus

    def test_roundtrip_unicode(self):
        corpus = "cafe 世界"
        model = Infinigram(corpus, tokenizer=ByteTokenizer())
        assert model.tokenizer.decode(model.tokenizer.encode(corpus)) == corpus

    def test_byte_model_generates_token_ids(self):
        model = Infinigram(b"hello hello hello")
        out = model.generate(b"hell", max_tokens=5, temperature=0)
        assert isinstance(out, list)
        assert all(0 <= t <= 255 for t in out)
        # Decode via the model's tokenizer to recover the text.
        assert model.tokenizer.decode(out)
