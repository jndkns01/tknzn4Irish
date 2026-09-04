"""Command-line interface for criochscore.

    criochscore build-moirfeolas ...   # Stage 1 (optional)
    criochscore score ...              # Stage 2

Run `criochscore <command> --help` for details on each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIXES = REPO_ROOT / "data" / "MoirfEolas" / "Affixes" / "final_prefixes.txt"
DEFAULT_SUFFIXES = REPO_ROOT / "data" / "MoirfEolas" / "Affixes" / "final_suffixes.txt"
DEFAULT_MOIRFEOLAS = REPO_ROOT / "data" / "MoirfEolas" / "MoirfEolas.csv"


def _prompt_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def _prompt_model_type() -> str:
    prompt = (
        "Is your tokenizer sentencepiece, byte-level (e.g. byte-level BPE), "
        "a Hugging Face pretrained tokenizer, or another type (WordPiece/BPE/Unigram)? "
        "[sentencepiece/byte-level/huggingface/other]: "
    )
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("sentencepiece", "byte-level", "huggingface"):
            return answer
        if answer in ("other", "standard"):
            return "standard"
        print("Please answer 'sentencepiece', 'byte-level', 'huggingface', or 'other'.")


def _resolve_model_type(args: argparse.Namespace) -> str:
    if args.model_type:
        return args.model_type
    return _prompt_model_type()


def _resolve_special_token(args: argparse.Namespace) -> str | None:
    if args.special_token is not None:
        return args.special_token
    if args.no_special_token:
        return None
    has_special = _prompt_yes_no("Does your tokenizer use a special token (e.g. '##', '▁', 'Ġ')?")
    if not has_special:
        return None
    while True:
        token = input("Enter the special token: ").strip()
        if token:
            return token
        print("Special token can't be empty.")


def cmd_build_moirfeolas(args: argparse.Namespace) -> None:
    from .moirfeolas import build_moirfeolas

    build_moirfeolas(
        corpus_path=args.corpus,
        prefixes_path=args.prefixes,
        suffixes_path=args.suffixes,
        output_path=args.output,
        filter_english=args.filter_english,
    )


def cmd_score(args: argparse.Namespace) -> None:
    from .tokenizers_io import get_token_lens, load_tokenizer, remove_specials
    from .scoring import calculate_morphological_score, find_morphological_info

    model_type = _resolve_model_type(args)
    special_token = _resolve_special_token(args)

    moirfeolas = pd.read_csv(args.moirfeolas)
    for col in ("word", "urú", "prefix", "suffix"):
        if col not in moirfeolas.columns:
            sys.exit(f"Error: MoirfEolas file is missing required column '{col}'.")

    combined_affix = {
        row["word"]: (
            "" if pd.isna(row["urú"]) else str(row["urú"]),
            "" if pd.isna(row["prefix"]) else str(row["prefix"]),
            "" if pd.isna(row["suffix"]) else str(row["suffix"]),
        )
        for _, row in moirfeolas.iterrows()
    }
    words = list(combined_affix.keys())
    print(f"Loaded {len(words)} words from MoirfEolas.")

    tokenizer = load_tokenizer(args.tokenizer, model_type, do_lower_case=args.do_lower_case)
    token_lens = get_token_lens(tokenizer, words, model_type)
    token_lens = remove_specials(token_lens, special_token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face Hub ids look like "DCU-NLP/bert-base-irish-cased-v1" -
    # Path(...).stem safely gives "bert-base-irish-cased-v1" for those (no
    # dot to strip), and the usual filename-without-extension for local files.
    tokenizer_name = Path(args.tokenizer.rstrip("/")).stem
    summary_path = output_dir / f"{tokenizer_name}_morphological_summary.json"
    results_path = output_dir / f"{tokenizer_name}_criochscore.txt"

    score_info = find_morphological_info(token_lens, combined_affix, output_path=summary_path)
    _, final_score, _, average_scores = calculate_morphological_score(
        score_info, output_path=results_path
    )

    print(f"\nOverall CríochScore: {final_score * 100:.2f}%")
    for name, value in average_scores.items():
        if value is not None:
            print(f"  {name}: {value * 100:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="criochscore", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser(
        "build-moirfeolas",
        help="(optional) Build a MoirfEolas table from your own corpus.",
    )
    p_build.add_argument("--corpus", required=True, help="One-word-per-line text file.")
    p_build.add_argument("--prefixes", default=str(DEFAULT_PREFIXES),
                          help=f"Prefix wordlist (default: {DEFAULT_PREFIXES}).")
    p_build.add_argument("--suffixes", default=str(DEFAULT_SUFFIXES),
                          help=f"Suffix wordlist (default: {DEFAULT_SUFFIXES}).")
    p_build.add_argument("--filter-english", action="store_true",
                          help="Filter out words fastText identifies as English.")
    p_build.add_argument("--output", required=True, help="Path to write the resulting MoirfEolas CSV.")
    p_build.set_defaults(func=cmd_build_moirfeolas)

    p_score = subparsers.add_parser(
        "score",
        help="Calculate CríochScore for a tokenizer against a MoirfEolas table.",
    )
    p_score.add_argument("--moirfeolas", default=str(DEFAULT_MOIRFEOLAS),
                          help=f"MoirfEolas CSV to score against (default: {DEFAULT_MOIRFEOLAS}).")
    p_score.add_argument("--tokenizer", required=True,
                          help="Path to the tokenizer file (.json for byte-level/standard, .model for "
                               "sentencepiece), or a Hugging Face Hub model id / local directory when "
                               "--model-type huggingface.")
    p_score.add_argument("--model-type", choices=["sentencepiece", "byte-level", "standard", "huggingface"],
                          default=None,
                          help="Tokenizer family ('standard' covers WordPiece/BPE/Unigram via the "
                               "tokenizers library, 'huggingface' loads via "
                               "transformers.AutoTokenizer.from_pretrained). If omitted, you'll be prompted.")
    p_score.add_argument("--do-lower-case", action="store_true",
                          help="Passed through to AutoTokenizer.from_pretrained when --model-type huggingface.")
    special_group = p_score.add_mutually_exclusive_group()
    special_group.add_argument("--special-token", default=None,
                                help="Literal special-token marker, e.g. '##', '▁', 'Ġ'.")
    special_group.add_argument("--no-special-token", action="store_true",
                                help="Declare that this tokenizer has no special token.")
    p_score.add_argument("--output-dir", default="results",
                          help="Folder (created in the current directory by default) to write results to.")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
