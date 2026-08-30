# Reproducing the Kaggle Runs

## Required Kaggle inputs

Attach the competition data and the model asset selected in each notebook. The symmetry notebooks also require the 48-row `cube_symmetries.npy` asset. The notebooks discover files below `/kaggle/input` and fail on ambiguous matches rather than silently selecting an unrelated file.

The controlled run uses:

- competition: `cayleypy-ihes-cube`;
- model asset: `arabidopsisthalian/ihes-model-1778521793`;
- canonical model ID: `1778521793`;
- checkpoint epoch: `32692`;
- puzzle ID: `106`;
- global beam width: `1_000_000`;
- all 18 generators;
- Kaggle GPU T4.

## Notebook 1: base MLP beam

Run `notebooks/01_base_mlp_beam.ipynb`. The notebook clones this repository, resolves model `1778521793`, runs the configured global beam, requires a replay-valid solution of length 22 through 30 for puzzle 106, replaces that row in the official sample submission, and replay-validates all 1,003 rows.

## Notebook 2: symmetry and reverse

Run `notebooks/02_symmetry_reverse.ipynb`. Each enabled symmetry is algebraically validated before search. Direct and reverse candidates are converted to original coordinates and accepted only after exact replay. The best replay-valid candidate replaces puzzle 106 in `submission.csv`.

## Notebook 3: bidirectional symmetry and reverse

Run `notebooks/03_bidirectional_symmetry_reverse.ipynb`. Puzzle ID, model ID, beam width, symmetry, and retained forward/reverse depths are configuration values. For the controlled case, both directions use beam `1_000_000` and retain depths 12 through 16.

The notebook first verifies the known 24-move path and the projection equation at all complementary path depths. The ordinary run evaluates every child in its primary and exact paired projection, minimizes the primary-minus-paired MLP score, keeps protection disabled, builds frontiers, and performs a blind full intersection. If the known audit trajectory drops from natural top-K, an optional separate diagnostic run enables last-slot protection and logs generation, natural retention, first drop, and protection. Protected frontiers are never used as blind-search evidence.

## Required output checks

Before publishing a version, run:

```bash
python tools/validate_repository.py
```

Then inspect the executed notebook output and require:

- a completed GPU run without hidden exceptions;
- exact replay in original coordinates;
- a two-column, 1,003-row `submission.csv`;
- successful replay of every submission row;
- a model ID in the run path and JSON log;
- no credentials or private data in notebook source or output.
