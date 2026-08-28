from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model import ModelSpec


@dataclass(frozen=True)
class AssetRecord:
    model_id: str
    source_slug: str
    config_names: tuple[str, ...]
    checkpoint_names: tuple[str, ...]
    architecture: str
    epoch: int | None


ASSET_REGISTRY: tuple[AssetRecord, ...] = (
    AssetRecord(
        model_id="1778521793",
        source_slug="arabidopsisthalian/ihes-model-1778521793",
        config_names=("model_p888-t000_1778521793.json",),
        checkpoint_names=("p888-t000_1778521793_e32692.pth",),
        architecture="MLP2RB",
        epoch=32692,
    ),
    AssetRecord(
        model_id="1780290207",
        source_slug="arabidopsisthalian/model-ihes-1780290207-e40960",
        config_names=("model_p888-t000_1780290207.json",),
        checkpoint_names=("p888-t000_1780290207_e40960.pth",),
        architecture="MLP2RB",
        epoch=40960,
    ),
    AssetRecord(
        model_id="1780290207",
        source_slug="arabidopsisthalian/1780290207-ihes",
        config_names=("model_p888-t000_1780290207.json",),
        checkpoint_names=("p888-t000_1780290207_e40960.pth",),
        architecture="MLP2RB",
        epoch=40960,
    ),
    AssetRecord(
        model_id="e08192",
        source_slug="arabidopsisthalian/ihes-e08192",
        config_names=(),
        checkpoint_names=(),
        architecture="asset-defined",
        epoch=8192,
    ),
)


def _unique_match(root: Path, names: Iterable[str], suffix: str) -> Path:
    names_set = set(names)
    candidates = [path for path in root.rglob(f"*{suffix}") if not names_set or path.name in names_set]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one {suffix} asset, found {len(candidates)}")
    return candidates[0]


def resolve_model(root: str | Path, model_id: str) -> ModelSpec:
    matches = [record for record in ASSET_REGISTRY if record.model_id == str(model_id)]
    if not matches:
        raise KeyError(f"model id {model_id} is not registered")
    root_path = Path(root)
    config_names = tuple(name for record in matches for name in record.config_names)
    checkpoint_names = tuple(name for record in matches for name in record.checkpoint_names)
    return ModelSpec(
        model_id=str(model_id),
        checkpoint=_unique_match(root_path, checkpoint_names, ".pth"),
        config=_unique_match(root_path, config_names, ".json"),
        epoch=matches[0].epoch,
    )
