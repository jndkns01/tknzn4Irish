# MoirfEolas

This directory contains the data needed to build a MoirfEolas table (see
`Affixes/`), plus the CSV itself (`MoirfEolas.csv`).

The MoirfEolas here can be reproduced by running:

```bash
criochscore build-moirfeolas \
    --corpus <your_corpus.txt> \
    --output MoirfEolas.csv
```

**Note:** this MoirfEolas is not identical to the one used in the
accompanying paper. Corpus filtering here (`--filter-english`) uses a
different English-language identifier than the one used for the paper's
dataset, since that original tool isn't available for use in this repo.
Results should be broadly similar, but don't expect an exact match to
the paper's published MoirfEolas.
