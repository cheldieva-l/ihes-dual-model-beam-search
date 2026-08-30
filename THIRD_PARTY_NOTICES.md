# Third-Party Notices and Provenance

## CayleyPy Cube

Repository: <https://github.com/khoruzhii/cayleypy-cube>

The public project is licensed under the MIT License. This repository adapts the high-level residual-MLP checkpoint layout and global beam-search architecture. It does not redistribute CayleyPy model weights or competition data. The audited reference revision was `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`.

## Tetraminx dual-model beam-search method

Repository: <https://github.com/cheldieva-l/tetraminx-cpp-dual-model-beam-search>

Method description: <https://github.com/cheldieva-l/tetraminx-cpp-dual-model-beam-search/blob/main/METHOD.md>

The inspected revision was `7d107f50d847fc022646525dd454dcedb1fc0147`. No license file was present in that checkout, so no source code was copied. The reverse-projection equation, paired primary-minus-opposite scoring objective, blind-meeting requirement, path-protection diagnostic, and replay requirements were independently implemented from the published mathematical description.

## Artgor IHES TPU beam notebook

Notebook: <https://www.kaggle.com/code/artgor/cayleypy-ihes-cube-tpu-beam>

The IHES rotation convention, generator conjugation equation, and reverse-path conversion were independently reimplemented and checked against the public notebook's formulas. No notebook source is redistributed. Users should consult the notebook page for its current Kaggle license metadata.

## Kaggle assets

The competition data, the four referenced IHES model assets, the API JSON dataset, model checkpoints, symmetry arrays, and participant submissions remain under their respective Kaggle terms. They are referenced and loaded at runtime but are not included in this repository.
