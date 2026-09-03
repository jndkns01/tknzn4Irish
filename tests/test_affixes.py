from criochscore.affixes import build_combined_affix


def test_simple_suffix_match():
    tokens = ["capaillín"]
    prefixes: list[str] = []
    suffixes = ["ín"]
    combined = build_combined_affix(tokens, prefixes, suffixes, special_prefixes={})
    assert combined["capaillín"] == ("", "", "ín")


def test_simple_prefix_match():
    tokens = ["neamhbheo"]
    prefixes = ["neamh"]
    suffixes: list[str] = []
    combined = build_combined_affix(tokens, prefixes, suffixes, special_prefixes={})
    assert combined["neamhbheo"] == ("", "neamh", "")


def test_special_prefix_uru():
    # The special-prefix (urú) pass only revisits words that already matched
    # a regular prefix/suffix, mirroring the original notebook behaviour.
    # "n-" only counts before a vowel/d/g - here it precedes 'ó'.
    tokens = ["n-óráidí"]
    combined = build_combined_affix(tokens, [], ["í"])
    assert combined["n-óráidí"] == ("n-", "", "í")


def test_word_with_no_affixes_is_absent():
    tokens = ["madra"]
    combined = build_combined_affix(tokens, ["neamh"], ["ín"], special_prefixes={})
    assert "madra" not in combined
