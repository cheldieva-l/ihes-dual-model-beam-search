from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .model import ModelSpec


@dataclass(frozen=True)
class AssetRecord:
    model_id: str
    aliases: tuple[str, ...]
    source_slug: str
    config_names: tuple[str, ...]
    checkpoint_names: tuple[str, ...]
    split_pattern: str | None
    architecture: str
    epoch: int | None
    sha256: str | None = None


ASSET_REGISTRY: tuple[AssetRecord, ...] = (
    AssetRecord(
        model_id="1778521793",
        aliases=(),
        source_slug="arabidopsisthalian/ihes-model-1778521793",
        config_names=("model_p888-t000_1778521793.json",),
        checkpoint_names=("p888-t000_1778521793_e32692.pth",),
        split_pattern=None,
        architecture="MLP2RB",
        epoch=32692,
        sha256="b19eb25bcb03eea831d6679ae2f532f5786583e4ac0a793d69ba8cadb259fa5b",
    ),
    AssetRecord(
        model_id="1780290207",
        aliases=(),
        source_slug="arabidopsisthalian/model-ihes-1780290207-e40960",
        config_names=("model_p888-t000_1780290207.json",),
        checkpoint_names=("p888-t000_1780290207_e40960.pth",),
        split_pattern=None,
        architecture="MLP2RB",
        epoch=40960,
        sha256="0e29248fa03ba3f522e015fa5ab8634171d152c4e2a4fc2d9f0b787982cc34d9",
    ),
    AssetRecord(
        model_id="1780290207",
        aliases=(),
        source_slug="arabidopsisthalian/1780290207-ihes",
        config_names=("model_p888-t000_1780290207.json",),
        checkpoint_names=(),
        split_pattern="p888-t000_1780290207_e40960_[0-9][0-9].txt",
        architecture="MLP2RB",
        epoch=40960,
        sha256="0e29248fa03ba3f522e015fa5ab8634171d152c4e2a4fc2d9f0b787982cc34d9",
    ),
    AssetRecord(
        model_id="1763232740",
        aliases=("e08192", "ihes-e08192"),
        source_slug="arabidopsisthalian/ihes-e08192/pyTorch/default",
        config_names=("model_p888-t000_1763232740.json",),
        checkpoint_names=("p888-t000_1763232740_e08192.pt",),
        split_pattern=None,
        architecture="MLP2RB",
        epoch=8192,
        sha256="50a3bd98f6b9739da3ce6efa3664590d5ccedbe90634760f930ebbcb99cfe820",
    ),
)


def _unique_match(root: Path, names: Iterable[str], suffixes: tuple[str, ...]) -> Path:
    names_set = set(names)
    candidates = [
        path
        for suffix in suffixes
        for path in root.rglob(f"*{suffix}")
        if path.name in names_set
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one asset with suffix in {suffixes}, found {len(candidates)}"
        )
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _magic(path: Path, length: int = 4) -> bytes:
    with path.open("rb") as stream:
        return stream.read(length)


def materialize_split_checkpoint(
    shards: Iterable[Path], output: Path, expected_sha256: str | None = None
) -> Path:
    """Concatenate numbered raw-binary shards and optionally verify the result."""

    ordered = sorted(Path(path) for path in shards)
    if not ordered:
        raise FileNotFoundError("no checkpoint shards were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as destination:
        for shard in ordered:
            with shard.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
    if expected_sha256 is not None:
        digest = _sha256(temporary)
        if digest != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"reassembled checkpoint fingerprint is {digest}; expected {expected_sha256}")
    temporary.replace(output)
    return output


def resolve_model(
    root: str | Path, model_id: str, cache_dir: str | Path | None = None
) -> ModelSpec:
    requested = str(model_id)
    matches = [
        record
        for record in ASSET_REGISTRY
        if requested == record.model_id or requested in record.aliases
    ]
    if not matches:
        raise KeyError(f"model id {model_id} is not registered")
    canonical_ids = {record.model_id for record in matches}
    if len(canonical_ids) != 1:
        raise AssertionError(f"model alias {model_id} maps to multiple canonical ids")
    canonical_id = next(iter(canonical_ids))
    root_path = Path(root)
    config_names = tuple(name for record in matches for name in record.config_names)
    checkpoint_names = {name for record in matches for name in record.checkpoint_names}
    config = _unique_match(root_path, config_names, (".json",))
    direct_candidates = [
        path
        for suffix in (".pth", ".pt")
        for path in root_path.rglob(f"*{suffix}")
        if path.name in checkpoint_names
    ]
    expected_hashes = {record.sha256 for record in matches if record.sha256 is not None}
    if len(expected_hashes) > 1:
        raise AssertionError(f"conflicting fingerprints for model {canonical_id}")
    expected_hash = next(iter(expected_hashes)) if expected_hashes else None
    if len(direct_candidates) == 1:
        checkpoint = direct_candidates[0]
    elif len(direct_candidates) > 1:
        raise FileNotFoundError(
            f"multiple direct checkpoints resolve model {canonical_id}: {direct_candidates}"
        )
    else:
        shard_patterns = [record.split_pattern for record in matches if record.split_pattern]
        shards = [path for pattern in shard_patterns for path in root_path.rglob(pattern)]
        if not shards:
            raise FileNotFoundError(f"no direct or split checkpoint resolves model {canonical_id}")
        if any(_magic(path) == b"Aar!" for path in shards):
            raise RuntimeError(
                f"model {canonical_id} was found as Apple Archive shards. "
                "Attach arabidopsisthalian/model-ihes-1780290207-e40960, "
                "or extract every shard with the Apple Archive 'aa'/'neoaa' tool "
                "before resolving the checkpoint. Raw byte concatenation is invalid."
            )
        if cache_dir is None:
            kaggle_working = Path("/kaggle/working")
            cache_root = (
                kaggle_working / "ihes-model-cache"
                if kaggle_working.exists()
                else Path.cwd() / ".ihes_model_cache"
            )
        else:
            cache_root = Path(cache_dir)
        checkpoint = materialize_split_checkpoint(
            shards,
            cache_root / f"p888-t000_{canonical_id}_reassembled.pth",
            expected_hash,
        )
    if expected_hash is not None:
        digest = _sha256(checkpoint)
        if digest != expected_hash:
            raise ValueError(
                f"checkpoint fingerprint for model {canonical_id} is {digest}; expected {expected_hash}"
            )
    return ModelSpec(
        model_id=canonical_id,
        checkpoint=checkpoint,
        config=config,
        epoch=matches[0].epoch,
    )
