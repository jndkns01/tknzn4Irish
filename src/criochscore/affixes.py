"""Prefix/suffix/urú matching logic used to build a MoirfEolas table.

This is a direct, function-by-function refactor of the notebook cells that
matched words in a corpus against known Irish prefix/suffix wordlists and
the hardcoded ``special_prefixes`` (urú / séimhiú-triggering) map. The
behaviour is unchanged from the notebook - only the plumbing (file I/O,
paths) has moved out into the CLI layer.
"""

from __future__ import annotations

from pathlib import Path

# Hardcoded urú / special-prefix rules: prefix -> list of letters the prefix
# is only "real" in front of (e.g. "n-" only counts before a vowel or d/g).
SPECIAL_PREFIXES: dict[str, list[str]] = {
    "n-": ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
           "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "d", "g"],
    "m": ["b", "B"],
    "g": ["c", "C"],
    "n": ["d", "g", "D", "G"],
    "bh": ["f", "F"],
    "b": ["p", "P"],
    "d": ["t", "T"],
    "h-": ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
           "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú"],
    "h": ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
          "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú"],
    "t-": ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
           "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "S", "s"],
    "t": ["A", "E", "I", "O", "U", "Á", "É", "Í", "Ó", "Ú", "S"],
    "d'": ["fh", "Fh", "FH", "a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
           "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú"],
    "D'": ["fh", "Fh", "FH", "a", "A", "e", "E", "i", "I", "o", "O", "u", "U",
           "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú"],
}


def load_prefixes(path: str | Path) -> list[str]:
    """Load one-prefix-per-line file, dropping single-character entries."""
    prefixes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pref = line.strip()
            if pref:
                prefixes.append(pref)
    return [p for p in prefixes if len(p) > 1]


def load_suffixes(path: str | Path) -> list[str]:
    """Load one-suffix-per-line file, dropping the noisy 'r'/'t' entries."""
    suffixes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            suff = line.strip()
            if suff:
                suffixes.append(suff)
    return [s for s in suffixes if s not in ("r", "t")]


def _find_candidates(
    tokens: list[str],
    all_prefixes: list[str],
    all_suffixes: list[str],
    special_prefixes: dict[str, list[str]],
) -> tuple[list[tuple], list[tuple]]:
    suffixed: list[tuple] = []
    prefixed: list[tuple] = []

    for word in tokens:
        for suffix in all_suffixes:
            if word.endswith(suffix):
                suffixed.append((word, suffix))

        for prefix in all_prefixes:
            if word.startswith(prefix):
                prefixed.append((word, prefix))

        for sp_prefix_key, sp_following_list in special_prefixes.items():
            if word.startswith(sp_prefix_key):
                if sp_prefix_key == "d'" and word.startswith("d'fh"):
                    prefixed.append((word, "d'", "*"))
                    break
                idx_after_prefix = len(sp_prefix_key)
                if len(word) > idx_after_prefix:
                    if word[idx_after_prefix] in sp_following_list:
                        prefixed.append((word, sp_prefix_key, "*"))
                        break

    return suffixed, prefixed


def _resolve_prefix_candidates(prefixed: list[tuple], all_prefixes: list[str]) -> list[tuple]:
    updated_prefix = []
    for entry in prefixed:
        if entry[-1] == "*":
            for prefix in all_prefixes:
                uru_removed = entry[0][len(entry[1]):]
                if uru_removed.startswith(prefix):
                    updated_prefix.append((entry[0], entry[1], entry[2], prefix))
        else:
            updated_prefix.append(entry)
    return updated_prefix


def _build_candidate_dicts(
    suffixed: list[tuple], updated_prefix: list[tuple]
) -> tuple[dict, dict]:
    candidates: dict[str, list[str]] = {}
    for word, suffix in suffixed:
        candidates.setdefault(word, []).append(suffix)

    pref_candidates: dict[str, list] = {}
    for entry in updated_prefix:
        word = entry[0]
        if word not in pref_candidates:
            pref_candidates[word] = []
            if len(entry) in (3, 4):
                pref_candidates[word].append("".join(entry[1:3]))
                pref_candidates[word].append(entry[3])
            else:
                pref_candidates[word].append(entry[1])
        else:
            pref_candidates[word].append(entry[1])

    return candidates, pref_candidates


def _resolve_actual_affixes(candidates: dict, pref_candidates: dict) -> tuple[dict, dict]:
    actual_suffix: dict[str, str] = {}
    for word, options in candidates.items():
        suffix = max(options, key=len)
        if suffix != word:
            actual_suffix[word] = suffix

    actual_prefix: dict[str, str | list[str]] = {}
    for word, options in pref_candidates.items():
        if options[0][-1] != "*":
            prefix = max(options, key=len)
            if prefix != word:
                actual_prefix[word] = prefix
        else:
            uru = options[0][:-1]
            prefix = max(options[1:], key=len)
            if prefix != word:
                actual_prefix[word] = [uru, prefix]

    cleaned_prefixes: dict[str, str | list[str]] = {}
    for word, value in actual_prefix.items():
        if isinstance(value, list) and value[0] == value[1]:
            cleaned_prefixes[word] = value[0]
        else:
            cleaned_prefixes[word] = value

    return actual_suffix, cleaned_prefixes


def _combine_affixes(actual_suffix: dict, cleaned_prefixes: dict) -> dict[str, tuple]:
    combined: dict[str, tuple] = {}
    affix_keys = list(actual_suffix.keys())
    for word in cleaned_prefixes:
        if word not in affix_keys:
            affix_keys.append(word)

    for word in affix_keys:
        in_suffix = word in actual_suffix
        in_prefix = word in cleaned_prefixes

        if in_suffix and in_prefix:
            prefix_val = cleaned_prefixes[word]
            if isinstance(prefix_val, list):
                combined[word] = (prefix_val[0], prefix_val[1], actual_suffix[word])
            else:
                combined[word] = ("", prefix_val, actual_suffix[word])
        elif in_suffix:
            combined[word] = ("", "", actual_suffix[word])
        elif in_prefix:
            prefix_val = cleaned_prefixes[word]
            if isinstance(prefix_val, list):
                combined[word] = (prefix_val[0], prefix_val[1], "")
            else:
                combined[word] = ("", prefix_val, "")

    return combined


def _apply_special_prefixes(
    combined: dict[str, tuple], special_prefixes: dict[str, list[str]]
) -> dict[str, tuple]:
    for word in list(combined.keys()):
        uru, prefix, suffix = combined[word]
        if uru != "":
            continue

        for sprefix, allowed_letters in special_prefixes.items():
            if not word.startswith(sprefix):
                continue

            if len(sprefix) == 1:
                following_letters = word[1]
            else:
                max_len = max(len(a) for a in allowed_letters)
                following_letters = word[len(sprefix):len(sprefix) + max_len]

            if not any(following_letters.lower().startswith(a.lower()) for a in allowed_letters):
                continue

            new_uru = sprefix
            new_tuple = ("" if new_uru == prefix else prefix, suffix)
            new_tuple = (new_uru, new_tuple[0], new_tuple[1])

            combined_word_sp = new_uru + suffix
            combined_word_pr = prefix + suffix
            if combined_word_sp == word or combined_word_pr == word:
                del combined[word]
            else:
                combined[word] = new_tuple
            break

    return combined


def build_combined_affix(
    tokens: list[str],
    all_prefixes: list[str],
    all_suffixes: list[str],
    special_prefixes: dict[str, list[str]] | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Match every word in ``tokens`` against the prefix/suffix wordlists and
    the urú rules, returning ``{word: (uru, prefix, suffix)}``.

    This is the core of MoirfEolas creation - equivalent to notebook cells
    23-24 combined into one call.
    """
    special_prefixes = special_prefixes or SPECIAL_PREFIXES

    suffixed, prefixed = _find_candidates(tokens, all_prefixes, all_suffixes, special_prefixes)
    updated_prefix = _resolve_prefix_candidates(prefixed, all_prefixes)
    candidates, pref_candidates = _build_candidate_dicts(suffixed, updated_prefix)
    actual_suffix, cleaned_prefixes = _resolve_actual_affixes(candidates, pref_candidates)
    combined = _combine_affixes(actual_suffix, cleaned_prefixes)
    combined = _apply_special_prefixes(combined, special_prefixes)
    return combined
