import csv
import json
import random

from criochscore.absorption import compute_absorption, write_absorption_csv
from criochscore.affixes import SPECIAL_PREFIXES
from criochscore.scoring import find_morphological_info


def _score_info(token_lens, combined_affix):
    return find_morphological_info(token_lens, combined_affix)


def test_uru_absorbed_and_mapping():
    combined = {
        "bhfear": ("bhf", "", ""),     # not in SPECIAL_PREFIXES -> absorbed, never "mapping"
        "mbád": ("m", "", ""),         # first token 'mb' = uru + next letter -> mapping
        "gcat": ("g", "", ""),         # first token 'g' == uru -> not absorbed
        "nathair": ("n", "", ""),      # 'n' only maps before d/g, so 'na' is absorbed, not mapping
    }
    token_lens = {
        "bhfear": (2, ["bhfe", "ar"]),
        "mbád": (2, ["mb", "ád"]),
        "gcat": (2, ["g", "cat"]),
        "nathair": (2, ["na", "thair"]),
    }
    r = compute_absorption(_score_info(token_lens, combined), combined)
    assert r["n_uru"] == 4
    assert r["uru_absorbed_%"] == 75.0
    assert r["uru_mapping_%"] == 25.0


def test_uru_unknown_key_does_not_crash():
    combined = {"xword": ("zz", "", "")}
    token_lens = {"xword": (2, ["xw", "ord"])}
    r = compute_absorption(_score_info(token_lens, combined), combined)
    assert r["uru_absorbed_%"] == 100.0
    assert r["uru_mapping_%"] == 0.0


def test_suffix_fully_and_split_absorbed():
    combined = {
        "capaillín": ("", "", "ín"),   # 'ín' lives inside longer token 'aillín' -> fully
        "madraí": ("", "", "aí"),      # 'aí' spread over ['ra', 'í']? -> split
        "bádóir": ("", "", "óir"),     # tokenized exactly -> not absorbed
    }
    token_lens = {
        "capaillín": (2, ["cap", "aillín"]),
        "madraí": (3, ["mad", "ra", "í"]),
        "bádóir": (2, ["bád", "óir"]),
    }
    r = compute_absorption(_score_info(token_lens, combined), combined)
    assert r["n_suffix"] == 3
    assert r["suffix_fully_absorbed_%"] == 33.33
    assert r["suffix_split_absorbed_%"] == 33.33


def test_prefix_fully_and_split_absorbed():
    combined = {
        "neamhbheo": ("", "neamh", ""),   # inside longer first token -> fully
        "antoisc": ("", "ant", ""),       # ['an','t' ...] -> split
    }
    token_lens = {
        "neamhbheo": (2, ["neamhbh", "eo"]),
        "antoisc": (3, ["an", "tois", "c"]),
    }
    r = compute_absorption(_score_info(token_lens, combined), combined)
    assert r["n_prefix"] == 2
    assert r["prefix_fully_absorbed_%"] == 50.0
    assert r["prefix_split_absorbed_%"] == 50.0


def test_denominator_is_words_evaluated():
    combined = {
        "mbád": ("m", "", ""),
        "gcat": ("g", "", ""),
    }
    # Only one word was evaluated (e.g. --exclude-single-token dropped the other)
    token_lens = {"mbád": (2, ["mb", "ád"])}
    r = compute_absorption(_score_info(token_lens, combined), combined)
    assert r["n_uru"] == 1
    assert r["uru_absorbed_%"] == 100.0


def test_roundtrips_through_summary_json(tmp_path):
    combined = {"capaillín": ("", "", "ín")}
    token_lens = {"capaillín": (2, ["cap", "aillín"])}
    score_info = _score_info(token_lens, combined)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(score_info, ensure_ascii=False), encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert compute_absorption(reloaded, combined) == compute_absorption(score_info, combined)


def test_write_csv(tmp_path):
    combined = {"capaillín": ("", "", "ín")}
    score_info = _score_info({"capaillín": (2, ["cap", "aillín"])}, combined)
    row = {"tokenizer": "tok"}
    row.update(compute_absorption(score_info, combined))
    out = tmp_path / "abs.csv"
    write_absorption_csv([row], out)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["tokenizer"] == "tok"
    assert rows[0]["suffix_fully_absorbed_%"] == "100.0"


# ---------------------------------------------------------------------------
# Equivalence with the original notebook code (copied verbatim as a reference)
# ---------------------------------------------------------------------------

def _nb_get_partitions(s, parts):
    if parts == 1:
        yield [s]
        return
    for i in range(1, len(s) - parts + 2):
        for rest in _nb_get_partitions(s[i:], parts - 1):
            yield [s[:i]] + rest


def _notebook_reference(score_info, combined_affix):
    """The notebook's uru_absorbed / uru_and_next / calculate_absorption logic,
    adapted only for data access (v[0][-1] -> suffix status, etc.)."""
    words_with_uru = {w: a[0] for w, a in combined_affix.items() if a[0] != ""}
    # The notebook divides by all uru words; in this test every word is evaluated.
    n_uru = len(words_with_uru)

    absorbed = nxt = 0
    for key, info in score_info.items():
        tokens = info[3]
        if key in words_with_uru and words_with_uru[key] != tokens[0]:
            absorbed += 1
            uru = words_with_uru[key]
            next_chars = SPECIAL_PREFIXES[uru]
            if len(key) > len(uru) and key[len(uru)] in next_chars:
                if tokens[0] == uru + key[len(uru)]:
                    nxt += 1

    sc = sf = ss = pc = pf = ps = 0
    for k, info in score_info.items():
        tokens = info[3]
        suffix_status, prefix_status = info[2][0], info[1][0]
        if suffix_status != "no suffix":
            sc += 1
        if suffix_status == "tokenized incorrectly":
            suffix = combined_affix[k][2].lower()
            fully = any(suffix in t.lower() and t.lower() != suffix for t in tokens)
            split = False
            if not fully:
                for n in range(2, len(tokens) + 1):
                    last_n = tokens[-n:]
                    for part in _nb_get_partitions(suffix, n):
                        if all(last_n[i].lower().endswith(part[i]) for i in range(n)):
                            split = True
                            break
                    if split:
                        break
            sf += fully
            ss += (not fully) and split
        if prefix_status != "no prefix":
            pc += 1
        if prefix_status == "tokenized incorrectly":
            prefix = combined_affix[k][1].lower()
            fully = any(prefix in t.lower() and t.lower() != prefix for t in tokens)
            split = False
            if not fully:
                for n in range(2, len(tokens) + 1):
                    first_n = tokens[:n]
                    for part in _nb_get_partitions(prefix, n):
                        if all(first_n[i].lower().startswith(part[i]) for i in range(n)):
                            split = True
                            break
                    if split:
                        break
            pf += fully
            ps += (not fully) and split

    def pct(a, b):
        return a / b * 100 if b else 0

    return {
        "uru_absorbed_%": pct(absorbed, n_uru),
        "uru_mapping_%": pct(nxt, n_uru),
        "suffix_fully_absorbed_%": round(pct(sf, sc), 2),
        "suffix_split_absorbed_%": round(pct(ss, sc), 2),
        "prefix_fully_absorbed_%": round(pct(pf, pc), 2),
        "prefix_split_absorbed_%": round(pct(ps, pc), 2),
    }


def test_matches_notebook_logic_on_random_data():
    rng = random.Random(0)
    urus = ["n-", "m", "g", "n", "bh", "b", "d", "h", "t", "d'"]
    prefixes = ["", "an", "neamh", "ró", "ant"]
    suffixes = ["", "ín", "ach", "aí", "óir"]
    letters = "abcdeílnmórt"

    combined, token_lens = {}, {}
    for i in range(400):
        uru = rng.choice(urus + [""] * 4)
        prefix = rng.choice(prefixes)
        suffix = rng.choice(suffixes)
        body = "".join(rng.choice(letters) for _ in range(rng.randint(2, 6)))
        word = f"{uru}{prefix}{body}{suffix}{i}"
        combined[word] = (uru, prefix, suffix)
        # random segmentation into 1-5 pieces
        cuts = sorted(rng.sample(range(1, len(word)), k=min(len(word) - 1, rng.randint(0, 4))))
        pieces = [word[a:b] for a, b in zip([0] + cuts, cuts + [len(word)])]
        token_lens[word] = (len(pieces), pieces)

    score_info = find_morphological_info(token_lens, combined)
    mine = compute_absorption(score_info, combined)
    ref = _notebook_reference(score_info, combined)
    for key, expected in ref.items():
        assert abs(mine[key] - expected) < 1e-9, key
