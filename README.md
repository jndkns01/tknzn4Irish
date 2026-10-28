# CríochScore and MoirfEolas

CríochScore measures how well a subword tokenizer's splits respect Irish
morpheme boundaries — urú (initial mutation), prefixes, and suffixes. It's
a two-stage pipeline, and the two stages are independent:

1. **MoirfEolas creation** *(optional)* — build a reference table of
   `word, urú, prefix, suffix` from a corpus of your own.
2. **CríochScore calculation** — score any trained tokenizer against a
   MoirfEolas table, either the one shipped in this repo or one you built
   yourself in step 1.

## Install

```bash
git clone <this-repo>
cd tknzn4Irish
pip install -e .
# or, if you want Stage 1's optional English-word filtering:
pip install -e ".[english-filter]"
```

This installs the `criochscore` command.

## Stage 1 (optional): build your own MoirfEolas

Skip this entirely if you're happy scoring against the MoirfEolas shipped
in `data/MoirfEolas/MoirfEolas.csv`.

```bash
criochscore build-moirfeolas \
    --corpus my_corpus.txt \
    --output my_moirfeolas.csv
```

- `--corpus` must be a **one-word-per-line** text file.
- `--prefixes` / `--suffixes` default to
  `data/MoirfEolas/Affixes/final_prefixes.txt` / `final_suffixes.txt` in
  this repo — pass your own paths to override.
- `--filter-english` runs each word through a fastText language-ID model
  first and drops anything identified as English rather than Irish
  (requires `pip install -e ".[english-filter]"`).

## Stage 2: calculate CríochScore

```bash
criochscore score \
    --moirfeolas data/MoirfEolas/MoirfEolas.csv \
    --tokenizer path/to/your_tokenizer.json \
    --model-type byte-level \
    --special-token "Ġ" \
    --output-dir results/
```

- `--moirfeolas` defaults to `data/MoirfEolas/MoirfEolas.csv` (the one
  shipped with this repo). Point it at your own CSV from Stage 1 instead
  if you built one.
- `--tokenizer` is a `.json` file (for `--model-type byte-level`, loaded
  via the `tokenizers` library) or a `.model` file (for
  `--model-type sentencepiece`, loaded via `sentencepiece`).
- `--model-type` and the special-token flag are **required** — if you
  omit either, the CLI will interactively prompt you for them rather
  than guessing:

  ```
  Is your tokenizer sentencepiece or byte-level? [sentencepiece/byte-level]:
  Does your tokenizer use a special token (e.g. '##', '▁', 'Ġ')? [y/n]:
  Enter the special token:
  ```

  Pass `--model-type` and either `--special-token "<token>"` or
  `--no-special-token` explicitly to skip the prompts (useful for
  scripting).
- `--exclude-single-token` skips scoring for any word the tokenizer splits
  into just one token (it can't exhibit boundary-splitting behaviour, so
  including it can dilute the score). Output filenames get an
  `_excl-single-token` suffix in this mode, so a normal run and an
  excluded run can both live in the same `--output-dir` without
  overwriting each other.
- Results are written to `--output-dir` (default: `results/` in your
  **current working directory**, not inside the repo — it's gitignored).
  Two files are written per run:
  - `<tokenizer_name>_morphological_summary.json` — per-word boundary
    classifications.
  - `<tokenizer_name>_criochscore.txt` — the aggregate CríochScore and
    per-category breakdown.

## Absorption analysis (urú, prefix, suffix)

Absorption measures how often a tokenizer swallows a morpheme into a larger
token instead of keeping it as its own piece:

- **Urú absorbed** — % of words with an urú whose first token is not exactly
  the urú. **Urú mapping** is the subset where the first token is exactly the
  urú plus the letter that follows it.
- **Prefix / suffix fully absorbed** — the affix sits inside a longer token.
  **Split absorbed** — the affix is spread over several tokens that each carry
  part of it. Both are percentages of all evaluated words that have that
  affix, and only words the CríochScore marked "tokenized incorrectly" can be
  absorbed.

Run it as part of scoring:

```bash
criochscore score --tokenizer ... --model-type ... --no-special-token --absorption
```

or afterwards, from the summary files `score` already wrote (no tokenizer
needed, so nothing is re-downloaded):

```bash
criochscore absorption \
    --summary results/bbpe_8k_morphological_summary.json \
              results/bbpe_16k_morphological_summary.json \
    --moirfeolas data/MoirfEolas/MoirfEolas.csv \
    --output-dir results/
```

One summary writes `<name>_absorption.csv`; several write a combined
`absorption_results.csv` with one row per tokenizer. Use the same MoirfEolas
the summaries were scored against. Percentages are of words actually
evaluated (so `--exclude-single-token` runs are handled correctly), and the
CSV includes the word counts used as denominators.

## Absorption analysis

Absorption measures how often a tokenizer swallows a morpheme into a
larger token instead of leaving it as its own piece:

- **Urú absorbed** — % of words with an urú/eclipsis whose first token is not exactly
  the urú. **Urú mapping** is the subset where the first token is exactly the
  urú/eclipsis plus the letter that follows it.
- **Prefix / suffix fully absorbed** — the affix sits inside a longer token i.e. merged/absorbed into subsequent/previous token.
- **Prefix / suffix split absorbed** — the affix is spread over several tokens i.e. part of the affix is absorbed/merged into subsequent/previous token.

Prefix/suffix absorption is only measured for words whose affix was
"tokenized incorrectly" in the CríochScore, and is reported as a percentage of
all evaluated words that have that affix. Denominators are the words actually
evaluated (so `--exclude-single-token` runs stay correct).

Run it alongside scoring:

```bash
criochscore score --tokenizer my_tokenizer.json --model-type standard \
    --no-special-token --absorption --output-dir results/
```

or afterwards, from saved results, with no tokenizer needed (several
`--summary` files give one combined CSV, e.g. for 8k/16k/32k/64k):

```bash
criochscore absorption \
    --summary results/tok_8k_morphological_summary.json results/tok_16k_morphological_summary.json \
    --moirfeolas data/MoirfEolas/MoirfEolas.csv \
    --output-dir results/
```

Pass the same `--moirfeolas` you scored against. Output is
`<tokenizer>_absorption.csv` (or `absorption_results.csv` for several
summaries), including the word counts behind each percentage.

## Data layout

```
data/MoirfEolas/
├── Affixes/
│   ├── final_prefixes.txt   # one prefix per line
│   └── final_suffixes.txt   # one suffix per line
└── MoirfEolas.csv           # word, urú, prefix, suffix
```

`data/MoirfEolas/Affixes/*.txt` and `MoirfEolas.csv` ship as placeholders
in this scaffold — replace them with your real wordlists / dataset.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Notes on tokenizer types

Four `--model-type` values are supported:

- `sentencepiece` — loaded via `sentencepiece.SentencePieceProcessor`,
  encoded with `encode_as_pieces`.
- `byte-level` — loaded via `tokenizers.Tokenizer.from_file`, with GPT-2
  style byte tokens decoded back to readable text before scoring.
- `standard` — loaded via `tokenizers.Tokenizer.from_file` like
  `byte-level`, but for anything else (WordPiece, plain BPE, Unigram, ...)
  — tokens are used as-is, no byte decoding.
- `huggingface` — loaded via `transformers.AutoTokenizer.from_pretrained`,
  encoded with `.tokenize()`. Use this for a pretrained model straight
  from the Hub, e.g.:

  ```bash
  pip install -e ".[huggingface]"

  criochscore score \
      --tokenizer DCU-NLP/bert-base-irish-cased-v1 \
      --model-type huggingface \
      --special-token "##" \
      --output-dir results/
  ```

  For `huggingface`, `--tokenizer` is a Hub model id or a local directory
  (whatever you'd normally pass to `from_pretrained`) rather than a single
  tokenizer file. `--do-lower-case` is passed straight through to
  `from_pretrained` if you need it.

Only `sentencepiece`, `byte-level`, and `huggingface` need to be
explicitly declared; anything else defaults to `standard` (the CLI
prompt offers it as `other` if you don't pass `--model-type`).

The special-token question is independent of `--model-type` — a
`standard` or `huggingface` WordPiece tokenizer, for example, would still
be declared with `--special-token "##"`.


**Note** 
This work appears in the paper [MoirfEolas and CríochScore: Developing Resources for and the Evaluation of Tokenization Alignment with Irish Morphology](https://arxiv.org/abs/2609.05022) which was presented at The Second Tokenization Workshhop (TokShop) at COLM 2026. Since uploading to arXiv, some errors have been found in the MoirfEolas dataset and have subsequently been resolved. This means that if using the tokenizers hosted in this repo along with the MoirfEolas dataset hosted here, results are slightly different to those reported but nonetheless still gather the same findings as discussed in the paper.
