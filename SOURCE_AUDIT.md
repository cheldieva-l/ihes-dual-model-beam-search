# Source and Asset Audit

This document records the provenance-sensitive observations used by the implementation. It is a technical audit, not a redistribution of third-party source or data.

## Official competition representation

The competition `puzzle_info.json` contains one 72-entry central state and 18 named generators in this order:

```text
f0, -f0, f1, -f1, f2, -f2,
r0, -r0, r1, -r1, r2, -r2,
d0, -d0, d1, -d1, d2, -d2
```

Every generator is a permutation and every named inverse composes to identity. The implementation loads this file at runtime and refuses an invalid or incomplete generator set.

## IHES inference assets

The inspected `p888-t000.pt` target is a 72-entry permutation equal to the official central state. The inspected `p888-t000-test_reverse_18tisach.pt` tensor has shape `(18054, 72)`, equal to `1003 * 18` rows. Each 18-row puzzle block is the complete set of official neighbours of the inverse puzzle state, but the rows are not in official generator order. For puzzle 106, the observed-to-official order was:

```text
12, 3, 2, 5, 4, 13, 15, 14, 7, 6, 9, 8, 1, 0, 10, 17, 11, 16
```

`ihes_dual.validation` therefore checks blocks as exact unordered sets and never assigns move names by tensor row position.

The current and historical IHES `searcher` files were inspected to understand batching, checkpoint use, and historical path-retention experiments. Their implementation code is not copied. The new beam defines its own deterministic global top-K rule, exact deduplication, backpointers, and diagnostic schema.

The inspected `test (3).py.txt` contained an artificial generator exclusion mechanism. This repository does not carry that mechanism. All published configurations set `prune_immediate_inverse=False`, so every retained parent expands through all 18 allowed generators.

The inspected trainer and model configuration establish the checkpoint-compatible `MLP2RB` layout. Model `1778521793` has two dense widths `2556` and `218`, 16 residual blocks, and 15,357,749 parameters. The controlled checkpoint is epoch 32692. Model `1780290207` uses the same registered architecture, with the requested epoch-40960 checkpoint.

## Checkpoint fingerprints

Fingerprints are recorded for identity and audit; weight files are not committed.

| Model ID | Asset checkpoint | SHA-256 |
|---|---|---|
| `1778521793` | `p888-t000_1778521793_e32692.pth` | `b19eb25bcb03eea831d6679ae2f532f5786583e4ac0a793d69ba8cadb259fa5b` |
| `1780290207` | `p888-t000_1780290207_e40960.pth` | `0e29248fa03ba3f522e015fa5ab8634171d152c4e2a4fc2d9f0b787982cc34d9` |
| `1763232740` | `p888-t000_1763232740_e08192.pt` | `50a3bd98f6b9739da3ce6efa3664590d5ccedbe90634760f930ebbcb99cfe820` |

The four requested Kaggle asset pages are recorded in the registry even when two pages expose the same canonical numeric model ID. `ihes-e08192` contains model ID `1763232740`, epoch 8192, with the same `2556/218/16` MLP2RB configuration. The `1780290207-ihes` dataset contains the epoch-40960 checkpoint as 11 Apple Archive (`Aar!`) `.txt` shards, not raw slices; raw concatenation is explicitly rejected. Portable runs use the equivalent direct checkpoint asset `model-ihes-1780290207-e40960`. All four inspected asset pages report Apache 2.0. A solution/run path uses the canonical model ID, never an ambiguous display label.

## Puzzle 106 controlled path

The 24-move path supplied in `submission (12).csv` was parsed using official move names and replayed successfully from test puzzle 106. Its inverse path also replays correctly in the reverse projection. It appears only in the bidirectional notebook's audit configuration; it is not supplied to the blind frontier intersection.
