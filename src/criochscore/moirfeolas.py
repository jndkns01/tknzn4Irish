"""Stage 1 (optional): build a MoirfEolas table from a corpus.

A MoirfEolas is a CSV with columns ``word, urú, prefix, suffix`` describing
the ideal morpheme boundaries for each word. It's the reference data that
Stage 2 (CríochScore calculation) scores a tokenizer against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .affixes import SPECIAL_PREFIXES, build_combined_affix, load_prefixes, load_suffixes


def load_corpus(path: str | Path) -> list[str]:
    """Load a one-word-per-line corpus file."""
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                tokens.append(word)
    return tokens


def build_moirfeolas(
    corpus_path: str | Path,
    prefixes_path: str | Path,
    suffixes_path: str | Path,
    output_path: str | Path,
    filter_english: bool = False,
) -> pd.DataFrame:
    """Run the full Stage 1 pipeline and write the resulting MoirfEolas CSV.

    Parameters
    ----------
    corpus_path: one-word-per-line text file.
    prefixes_path / suffixes_path: one-affix-per-line text files.
    output_path: where to write the resulting MoirfEolas CSV.
    filter_english: if True, run words through fastText language ID and
        drop anything identified as English (not Irish). Requires the
        ``fasttext`` and ``huggingface_hub`` packages.
    """
    tokens = load_corpus(corpus_path)
    print(f"Loaded {len(tokens)} words from corpus.")

    if filter_english:
        from .english_filter import find_english_words

        english_words = find_english_words(tokens)
        print(f"Removing {len(english_words)} words identified as English.")
        tokens = [t for t in tokens if t not in english_words]

    print(f"Building MoirfEolas from {len(tokens)} words.")

    all_prefixes = load_prefixes(prefixes_path)
    all_suffixes = load_suffixes(suffixes_path)

    combined_affix = build_combined_affix(tokens, all_prefixes, all_suffixes, SPECIAL_PREFIXES)

    boundaries = pd.DataFrame(
        [
            {"word": word, "urú": v[0], "prefix": v[1], "suffix": v[2]}
            for word, v in combined_affix.items()
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundaries.to_csv(output_path, index=False)

    print(f"MoirfEolas created with {len(boundaries)} entries -> {output_path}")
    return boundaries
