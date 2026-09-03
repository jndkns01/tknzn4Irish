"""Tokenizer loading + word-level encoding for CríochScore calculation.

Supports three tokenizer families:

* ``sentencepiece`` - a ``.model`` file loaded via
  ``sentencepiece.SentencePieceProcessor``, encoded with ``encode_as_pieces``.
* ``byte-level`` - a ``.json`` file loaded via the ``tokenizers`` library
  (e.g. a byte-level BPE tokenizer), whose GPT-2-style byte tokens are
  decoded back to readable Irish text before comparison.
* ``standard`` - a ``.json`` file loaded via the ``tokenizers`` library for
  any other tokenizer type (WordPiece, plain BPE, Unigram, ...). No byte
  decoding is applied - tokens are used as-is.

Any of the three may still use a "special token" marker (e.g. ``'##'`` for
WordPiece-style continuation marking, or ``'▁'`` / ``'Ġ'`` for
whitespace-marking schemes), declared independently via ``remove_specials``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ModelType = Literal["sentencepiece", "byte-level", "standard"]


def load_tokenizer(path: str | Path, model_type: ModelType):
    """Load a tokenizer file. ``model_type`` must be 'sentencepiece',
    'byte-level', or 'standard' (see module docstring)."""
    path = str(path)

    if model_type == "sentencepiece":
        import sentencepiece as spm

        tok = spm.SentencePieceProcessor()
        tok.load(path)
        return tok

    if model_type in ("byte-level", "standard"):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(path)

    raise ValueError(
        f"Unknown model_type: {model_type!r} (expected 'sentencepiece', 'byte-level', or 'standard')"
    )


def _bytes_to_unicode() -> dict[int, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(cs, bs))


_BYTE_DECODER = _bytes_to_unicode()


def decode_bpe_token(token: str) -> str:
    """Decode a GPT-2-style byte-level BPE token back to UTF-8 text."""
    token = token.replace("Ġ", "")  # strip leading space marker entirely
    return bytes([_BYTE_DECODER[ord(c)] for c in token]).decode("utf-8", errors="replace")


def get_token_lens(
    tokenizer, words: list[str], model_type: ModelType
) -> dict[str, tuple[int, list[str]]]:
    """Encode every word, returning ``{word: (num_tokens, [tokens...])}``."""
    token_lens: dict[str, tuple[int, list[str]]] = {}

    for word in words:
        if model_type == "sentencepiece":
            encoded = tokenizer.encode_as_pieces(word)
            token_lens[word] = (len(encoded), list(encoded))
        elif model_type == "byte-level":
            raw = tokenizer.encode(word).tokens
            decoded = [decode_bpe_token(t) for t in raw]
            token_lens[word] = (len(decoded), decoded)
        elif model_type == "standard":
            encoded = tokenizer.encode(word).tokens
            token_lens[word] = (len(encoded), list(encoded))
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    return token_lens


def remove_specials(
    token_lens: dict[str, tuple[int, list[str]]], special_token: str | None = None
) -> dict[str, tuple[int, list[str]]]:
    """Strip a tokenizer's special marker from the subword tokens so they
    line up with plain Irish text.

    - ``special_token`` is ``None`` / falsy: tokens are returned unchanged.
    - ``special_token == '##'``: WordPiece-style continuation marker,
      stripped from every token *after* the first.
    - ``special_token`` is a whitespace-marking style (``'▁'``, ``'_'``,
      ``'Ġ'``): stripped as a single leading character from the *first*
      token only.
    - anything else: treated the same as the whitespace-marking style
      above (stripped as a leading character from the first token), which
      covers most custom single-character markers.
    """
    if not special_token:
        return {k: v for k, v in token_lens.items()}

    new_dict: dict[str, tuple[int, list[str]]] = {}
    for word, (_, tokens_list) in token_lens.items():
        if not tokens_list:
            new_dict[word] = (0, [])
            continue

        if special_token == "##":
            processed = [tokens_list[0]]
            for tok in tokens_list[1:]:
                processed.append(tok[len(special_token):] if tok.startswith(special_token) else tok)
        else:
            first = tokens_list[0]
            processed = [first[1:] if first.startswith(special_token) else first]
            processed.extend(tokens_list[1:])

        new_dict[word] = (len(processed), processed)

    return new_dict
