from criochscore.scoring import (
    calculate_morphological_score,
    classify_prefix,
    classify_suffix,
    classify_uru,
    find_morphological_info,
)


def test_classify_suffix_exact():
    assert classify_suffix("ín", ["capaill", "ín"]) == ("tokenized exactly", [])


def test_classify_suffix_in_parts():
    status, candidates = classify_suffix("ín", ["capai", "ll", "ín"])
    assert status == "tokenized exactly"  # last token still matches exactly


def test_classify_suffix_none():
    assert classify_suffix("", ["word"]) == ("no suffix", [])


def test_classify_uru_exact():
    assert classify_uru("n-", ["n-", "athair"]) == ("tokenized exactly", [])


def test_end_to_end_scoring():
    token_lens = {"capaillín": (2, ["capaill", "ín"])}
    combined_affix = {"capaillín": ("", "", "ín")}
    score_info = find_morphological_info(token_lens, combined_affix)
    _, final_score, _, averages = calculate_morphological_score(score_info)
    assert final_score == 1.0
    assert averages["suffix_only"] == 1.0
