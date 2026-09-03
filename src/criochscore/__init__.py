"""CríochScore: measure how well subword tokenizers respect Irish morpheme
boundaries (urú, prefixes, suffixes).

Two independent stages:
  1. MoirfEolas creation (optional) - build a word -> (urú, prefix, suffix)
     reference table from a corpus.
  2. CríochScore calculation - score a trained tokenizer against a
     MoirfEolas table (yours, or one the user built themselves).
"""

__version__ = "0.1.0"
