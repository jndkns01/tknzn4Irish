"""Stage 2: classify how well a tokenizer's subword boundaries line up with
a word's urú/prefix/suffix, and aggregate that into the CríochScore."""

from __future__ import annotations

import json
from pathlib import Path

SCORE_MAP = {
    "tokenized exactly": 1,
    "tokenized in parts": 0.5,
    "tokenized incorrectly": 0,
    "no uru": 0,
    "no prefix": 0,
    "no suffix": 0,
}


def classify_suffix(suffix: str, tokens: list[str]) -> tuple[str, list[str]]:
    if suffix == "":
        return "no suffix", []
    if suffix == tokens[-1]:
        return "tokenized exactly", []

    candidates = ["".join(tokens[-i:]) for i in range(2, len(tokens) + 1)]
    if suffix in candidates:
        return "tokenized in parts", candidates
    return "tokenized incorrectly", []


def classify_prefix(prefix: str, tokens: list[str], uru: str) -> tuple[str, list[str]]:
    if prefix == "":
        return "no prefix", []

    start_idx = 1 if uru != "" else 0
    if start_idx < len(tokens) and prefix == tokens[start_idx]:
        return "tokenized exactly", []

    candidates = []
    running = ""
    for tok in tokens[start_idx:]:
        running += tok
        candidates.append(running)
        if len(running) >= len(prefix):
            break

    if prefix in candidates:
        return "tokenized in parts", candidates
    return "tokenized incorrectly", []


def classify_uru(uru: str, tokens: list[str]) -> tuple[str, list[str]]:
    if uru == "":
        return "no uru", []
    if uru == tokens[0]:
        return "tokenized exactly", []

    candidates = []
    running = ""
    for tok in tokens:
        running += tok
        candidates.append(running)
        if len(running) >= len(uru):
            break

    if uru in candidates:
        return "tokenized in parts", candidates
    return "tokenized incorrectly", []


def find_morphological_info(
    token_lens: dict[str, tuple[int, list[str]]],
    combined_affix: dict[str, tuple[str, str, str]],
    output_path: str | Path | None = None,
) -> dict[str, list]:
    """Classify uru/prefix/suffix alignment for every word present in both
    the tokenizer output and the MoirfEolas table."""
    score_info: dict[str, list] = {}

    for word, (_, tokens) in token_lens.items():
        if word not in combined_affix:
            continue

        uru, prefix, suffix = combined_affix[word]
        suffix_info = classify_suffix(suffix, tokens)
        prefix_info = classify_prefix(prefix, tokens, uru)
        uru_info = classify_uru(uru, tokens)

        score_info[word] = [uru_info, prefix_info, suffix_info, tokens]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(score_info, f, ensure_ascii=False, indent=2)
        print(f"CríochScore Morphological Summary saved to {output_path}")

    return score_info


_SCORE_NAMES = [
    "uru_only", "prefix_only", "suffix_only",
    "uru+prefix", "uru+suffix", "prefix+suffix", "all_affixes",
]


def calculate_morphological_score(
    score_info: dict[str, list], output_path: str | Path | None = None
) -> tuple[dict, float, dict, dict]:
    """Aggregate per-word classifications into the overall CríochScore plus
    per-category averages."""
    morphological_score_summary: dict[str, tuple] = {}
    score_details_by_word: dict[str, list] = {}

    total_overall_score = 0.0
    num_words_with_affixes = 0

    for word, (uru_info, prefix_info, suffix_info, _tokens) in score_info.items():
        has_uru = uru_info[0] != "no uru"
        has_prefix = prefix_info[0] != "no prefix"
        has_suffix = suffix_info[0] != "no suffix"

        uru_score = SCORE_MAP[uru_info[0]] if has_uru else None
        prefix_score = SCORE_MAP[prefix_info[0]] if has_prefix else None
        suffix_score = SCORE_MAP[suffix_info[0]] if has_suffix else None

        current_word_scores_detail: list[float | None] = [None] * 7
        calculated_word_score = 0.0
        num_present_affixes = 0

        if has_uru:
            calculated_word_score += uru_score
            num_present_affixes += 1
        if has_prefix:
            calculated_word_score += prefix_score
            num_present_affixes += 1
        if has_suffix:
            calculated_word_score += suffix_score
            num_present_affixes += 1

        if num_present_affixes > 0:
            calculated_word_score /= num_present_affixes
            total_overall_score += calculated_word_score
            num_words_with_affixes += 1

        if has_uru and not has_prefix and not has_suffix:
            current_word_scores_detail[0] = uru_score
        elif not has_uru and has_prefix and not has_suffix:
            current_word_scores_detail[1] = prefix_score
        elif not has_uru and not has_prefix and has_suffix:
            current_word_scores_detail[2] = suffix_score
        elif has_uru and has_prefix and not has_suffix:
            current_word_scores_detail[3] = (uru_score + prefix_score) / 2
        elif has_uru and not has_prefix and has_suffix:
            current_word_scores_detail[4] = (uru_score + suffix_score) / 2
        elif not has_uru and has_prefix and has_suffix:
            current_word_scores_detail[5] = (prefix_score + suffix_score) / 2
        elif has_uru and has_prefix and has_suffix:
            current_word_scores_detail[6] = (uru_score + prefix_score + suffix_score) / 3

        morphological_score_summary[word] = (
            [uru_info[0], prefix_info[0], suffix_info[0]],
            calculated_word_score,
        )
        score_details_by_word[word] = current_word_scores_detail

    final_score = total_overall_score / num_words_with_affixes if num_words_with_affixes > 0 else 0.0

    counts = {name: 0 for name in _SCORE_NAMES}
    totals = {name: 0.0 for name in _SCORE_NAMES}
    for detail in score_details_by_word.values():
        for i, val in enumerate(detail):
            if val is not None:
                totals[_SCORE_NAMES[i]] += val
                counts[_SCORE_NAMES[i]] += 1

    average_scores = {
        name: (totals[name] / counts[name] if counts[name] > 0 else None)
        for name in _SCORE_NAMES
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Overall CríochScore: {final_score * 100:.2f}%\n")
            for name in _SCORE_NAMES:
                if average_scores[name] is not None:
                    f.write(f"{name}: {average_scores[name] * 100:.2f}% (based on {counts[name]} valid values)\n")
                else:
                    f.write(f"{name}: None (based on {counts[name]} valid values)\n")
        print(f"CríochScore saved to {output_path}")

    return morphological_score_summary, final_score, score_details_by_word, average_scores
