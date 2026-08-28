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
    required = ("puzzle_info.json", "test.csv", "sample_submission.csv")
    candidates = [
        directory
        for directory in {path.parent for path in root_path.rglob("sample_submission.csv")}
        if all((directory / filename).is_file() for filename in required)
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "expected exactly one directory containing puzzle_info.json, test.csv, "
            f"and sample_submission.csv below {root_path}, found {len(candidates)}"
        )
    competition_root = candidates[0]
    return CompetitionAssets(
        puzzle_info=competition_root / "puzzle_info.json",
        test_csv=competition_root / "test.csv",
        sample_submission=competition_root / "sample_submission.csv",
    )


def find_symmetry_file(root: str | Path = "/kaggle/input") -> Path:
    return _find_unique(Path(root), "cube_symmetries.npy")
