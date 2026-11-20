"""Tokenizer abstraction for Infinigram.

Infinigram indexes sequences of token ids. A ``Tokenizer`` converts between
strings and token-id sequences at build and query time. The library ships with
``ByteTokenizer`` (identity over UTF-8 bytes, vocab=256) and accepts any object
satisfying the :class:`Tokenizer` protocol — notably ``tiktoken`` encodings and
``transformers`` fast tokenizers.

The tokenizer is a property of the model: the index records which tokenizer
produced it so that load-time mismatches are caught instead of silently
scrambling results.
"""

from typing import List, Protocol, runtime_checkable


# Canonical query/sequence type for the model API. Always pre-tokenized.
# Convert text via ``model.tokenizer.encode(text)`` at the call site.
TokenIds = List[int]


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal protocol for any tokenizer Infinigram can use.

    Requires ``encode(text) -> List[int]`` and ``decode(tokens) -> str``.
    Vocabulary size is read via :func:`vocab_size_of`, which accepts either
    ``.vocab_size`` (HuggingFace, ByteTokenizer) or ``.n_vocab`` (tiktoken).
    """

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...


def vocab_size_of(tokenizer: object) -> int:
    """Return the vocabulary size of a tokenizer.

    Checks ``.vocab_size`` first (HuggingFace, ByteTokenizer), then ``.n_vocab``
    (tiktoken). Raises ``AttributeError`` if neither is present.
    """
    for attr in ("vocab_size", "n_vocab"):
        size = getattr(tokenizer, attr, None)
        if size is not None:
            return int(size)
    raise AttributeError(
        f"Tokenizer {tokenizer!r} has neither 'vocab_size' nor 'n_vocab'"
    )


class ByteTokenizer:
    """Identity tokenizer over UTF-8 bytes.

    Maps each UTF-8 byte to itself as a token id in ``[0, 256)``. Satisfies the
    :class:`Tokenizer` protocol and is the default when no tokenizer is given
    to :class:`Infinigram`. Round-trips losslessly when ``errors='strict'``; use
    ``errors='replace'`` (default) to tolerate corrupt input on decode.
    """

    vocab_size: int = 256
    name: str = "byte"

    def __init__(self, encoding: str = "utf-8", errors: str = "replace"):
        self.encoding = encoding
        self.errors = errors

    def encode(self, text: str) -> List[int]:
        return list(text.encode(self.encoding))

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode(self.encoding, errors=self.errors)

    def __repr__(self) -> str:
        return f"ByteTokenizer(encoding={self.encoding!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ByteTokenizer)
            and self.encoding == other.encoding
            and self.errors == other.errors
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.encoding, self.errors))


def tokenizer_config(tokenizer: Tokenizer) -> dict:
    """Serialize a tokenizer to a config dict for index metadata.

    Recognizes ``ByteTokenizer`` and ``tiktoken`` encodings; other tokenizers
    are stored as ``{"kind": "custom", "name": repr(tokenizer)}`` — enough to
    warn on mismatch, not enough to reconstruct without user help.
    """
    if isinstance(tokenizer, ByteTokenizer):
        return {"kind": "byte", "encoding": tokenizer.encoding, "errors": tokenizer.errors}

    # tiktoken.Encoding has a `.name` attribute (e.g. "cl100k_base")
    name = getattr(tokenizer, "name", None)
    if name is not None and _is_tiktoken(tokenizer):
        return {"kind": "tiktoken", "name": name}

    return {"kind": "custom", "name": repr(tokenizer)}


def load_tokenizer(config: dict) -> Tokenizer:
    """Reconstruct a tokenizer from its serialized config.

    Raises ``ValueError`` if the config refers to a ``custom`` tokenizer
    (caller must supply one at load time) or an unknown kind.
    """
    kind = config.get("kind")
    if kind == "byte":
        return ByteTokenizer(
            encoding=config.get("encoding", "utf-8"),
            errors=config.get("errors", "replace"),
        )
    if kind == "tiktoken":
        import tiktoken  # imported lazily; dep is optional
        return tiktoken.get_encoding(config["name"])
    if kind == "custom":
        raise ValueError(
            f"Index was built with a custom tokenizer ({config.get('name')}). "
            "Pass tokenizer= when loading."
        )
    raise ValueError(f"Unknown tokenizer kind: {kind!r}")


def _is_tiktoken(tokenizer: object) -> bool:
    module = type(tokenizer).__module__ or ""
    return module.startswith("tiktoken")
