"""Absorption analysis: how often a tokenizer swallows a morpheme into a
larger token instead of leaving it as its own piece.

Three measures, computed per tokenizer from the same ``score_info`` that the
CríochScore calculation produces (``scoring.find_morphological_info``):

* **Urú absorbed** - % of evaluated words with an urú whose first token is not
  exactly the urú.
* **Urú mapping** - the subset of those where the first token is exactly the
  urú plus the letter that follows it (the mutation fused with the next
  letter, e.g. ``bhf``).
* **Prefix / suffix absorption** - for words whose prefix (suffix) was
  "tokenized incorrectly", whether the affix is *fully absorbed* (it sits
  inside a longer token) or *split absorbed* (spread over several tokens that
  each carry part of it). Reported as a % of all evaluated words that have
  that affix.

This is a direct port of the analysis in the original notebook. The
"fully absorbed" test is deliberately not position-anchored (it checks every
token), to stay comparable with the published numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .affixes import SPECIAL_PREFIXES

CSV_FIELDS = [
    "tokenizer",
    "uru_absorbed_%",
    "uru_mapping_%",
    "suffix_fully_absorbed_%",
    "suffix_split_absorbed_%",
    "prefix_fully_absorbed_%",
    "prefix_split_absorbed_%",
    "n_words_evaluated",
    "n_uru",
    "n_prefix",
    "n_suffix",
]


def _get_partitions(s: str, parts: int):
    """Yield every way of cutting ``s`` into ``parts`` non-empty pieces."""
    if parts == 1:
        yield [s]
        return
    for i in range(1, len(s) - parts + 2):
        for rest in _get_partitions(s[i:], parts - 1):
            yield [s[:i]] + rest


def _suffix_absorption(suffix: str, tokens: list[str]) -> str | None:
    """Return 'fully', 'split' or None for a mis-tokenized suffix."""
    suffix = suffix.lower()

    for token in tokens:
        if suffix in token.lower() and token.lower() != suffix:
            return "fully"

    for n in range(2, len(tokens) + 1):
        last_n = tokens[-n:]
        for partition in _get_partitions(suffix, n):
            if all(last_n[i].lower().endswith(partition[i]) for i in range(n)):
                return "split"
    return None


def _prefix_absorption(prefix: str, tokens: list[str]) -> str | None:
    """Return 'fully', 'split' or None for a mis-tokenized prefix."""
    prefix = prefix.lower()

    for token in tokens:
        if prefix in token.lower() and token.lower() != prefix:
            return "fully"

    for n in range(2, len(tokens) + 1):
        first_n = tokens[:n]
        for partition in _get_partitions(prefix, n):
            if all(first_n[i].lower().startswith(partition[i]) for i in range(n)):
                return "split"
    return None


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator > 0 else 0.0


def compute_absorption(
    score_info: dict[str, list],
    combined_affix: dict[str, tuple[str, str, str]],
    special_prefixes: dict[str, list[str]] | None = None,
) -> dict[str, float | int]:
    """Compute urú, prefix and suffix absorption for one tokenizer.

    ``score_info`` is ``{word: [uru_info, prefix_info, suffix_info, tokens]}``
    where each ``*_info`` is ``(status, candidates)`` - exactly what
    ``find_morphological_info`` returns, or what its JSON output reloads to.
    Words absent from ``combined_affix`` are ignored. Denominators are the
    words actually evaluated, so results stay correct when words have been
    excluded (e.g. ``--exclude-single-token``).
    """
    special_prefixes = special_prefixes or SPECIAL_PREFIXES

    n_evaluated = 0
    n_uru = n_prefix = n_suffix = 0
    uru_absorbed = uru_mapping = 0
    prefix_full = prefix_split = 0
    suffix_full = suffix_split = 0

    for word, (uru_info, prefix_info, suffix_info, tokens) in score_info.items():
        if word not in combined_affix:
            continue
        n_evaluated += 1
        uru, prefix, suffix = combined_affix[word]

        # --- urú ---
        if uru != "":
            n_uru += 1
            first = tokens[0] if tokens else None
            if first != uru:
                uru_absorbed += 1
                next_chars = special_prefixes.get(uru)
                if (
                    next_chars is not None
                    and len(word) > len(uru)
                    and word[len(uru)] in next_chars
                    and first == uru + word[len(uru)]
                ):
                    uru_mapping += 1

        # --- suffix ---
        if suffix_info[0] != "no suffix":
            n_suffix += 1
            if suffix_info[0] == "tokenized incorrectly":
                kind = _suffix_absorption(suffix, tokens)
                if kind == "fully":
                    suffix_full += 1
                elif kind == "split":
                    suffix_split += 1

        # --- prefix ---
        if prefix_info[0] != "no prefix":
            n_prefix += 1
            if prefix_info[0] == "tokenized incorrectly":
                kind = _prefix_absorption(prefix, tokens)
                if kind == "fully":
                    prefix_full += 1
                elif kind == "split":
                    prefix_split += 1

    return {
        "uru_absorbed_%": _pct(uru_absorbed, n_uru),
        "uru_mapping_%": _pct(uru_mapping, n_uru),
        "suffix_fully_absorbed_%": round(_pct(suffix_full, n_suffix), 2),
        "suffix_split_absorbed_%": round(_pct(suffix_split, n_suffix), 2),
        "prefix_fully_absorbed_%": round(_pct(prefix_full, n_prefix), 2),
        "prefix_split_absorbed_%": round(_pct(prefix_split, n_prefix), 2),
        "n_words_evaluated": n_evaluated,
        "n_uru": n_uru,
        "n_prefix": n_prefix,
        "n_suffix": n_suffix,
    }


def write_absorption_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write one row per tokenizer. Each row needs a 'tokenizer' key plus the
    keys returned by ``compute_absorption``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Absorption results saved to {output_path}")
