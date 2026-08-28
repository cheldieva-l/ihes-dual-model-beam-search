from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionAssets:
    puzzle_info: Path
    test_csv: Path
    sample_submission: Path


def _find_unique(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {filename} below {root}, found {len(matches)}")
    return matches[0]


def find_competition_assets(root: str | Path = "/kaggle/input") -> CompetitionAssets:
    root_path = Path(root)
    return CompetitionAssets(
        puzzle_info=_find_unique(root_path, "puzzle_info.json"),
        test_csv=_find_unique(root_path, "test.csv"),
        sample_submission=_find_unique(root_path, "sample_submission.csv"),
    )


def find_symmetry_file(root: str | Path = "/kaggle/input") -> Path:
    return _find_unique(Path(root), "cube_symmetries.npy")

