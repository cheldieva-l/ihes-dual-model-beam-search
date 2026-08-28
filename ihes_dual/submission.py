from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .puzzle import IHESPuzzle


def build_submission(
    sample_submission: str | Path,
    output_path: str | Path,
    puzzle: IHESPuzzle,
    replacements: Mapping[int, Sequence[int]],
) -> Path:
    frame = pd.read_csv(sample_submission, usecols=["initial_state_id", "path"])
    if frame["initial_state_id"].duplicated().any():
        raise ValueError("sample submission has duplicate puzzle ids")
    known_ids = set(int(value) for value in frame["initial_state_id"])
    missing = set(int(value) for value in replacements) - known_ids
    if missing:
        raise ValueError(f"replacement ids are not present in the sample: {sorted(missing)}")
    for puzzle_id, path in replacements.items():
        frame.loc[frame["initial_state_id"] == int(puzzle_id), "path"] = puzzle.encode_path(path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def validate_submission(
    submission_path: str | Path,
    test_csv: str | Path,
    puzzle: IHESPuzzle,
) -> dict[str, int]:
    submission = pd.read_csv(submission_path)
    tests = pd.read_csv(test_csv)
    if list(submission.columns) != ["initial_state_id", "path"]:
        raise ValueError(f"submission columns are invalid: {list(submission.columns)}")
    if len(submission) != len(tests):
        raise ValueError(f"submission has {len(submission)} rows; expected {len(tests)}")
    if submission["initial_state_id"].duplicated().any():
        raise ValueError("submission contains duplicate puzzle ids")
    merged = tests.merge(submission, on="initial_state_id", validate="one_to_one")
    if len(merged) != len(tests):
        raise ValueError("submission and test puzzle ids differ")
    invalid: list[int] = []
    path_lengths: list[int] = []
    for row in merged.itertuples(index=False):
        start = np.fromstring(row.initial_state, sep=",", dtype=np.uint8)
        path = puzzle.decode_path(row.path)
        path_lengths.append(len(path))
        if not puzzle.verify_solution(start, path):
            invalid.append(int(row.initial_state_id))
    if invalid:
        preview = invalid[:20]
        raise ValueError(f"submission replay failed for {len(invalid)} ids: {preview}")
    return {
        "rows": len(merged),
        "minimum_length": min(path_lengths),
        "maximum_length": max(path_lengths),
        "total_moves": sum(path_lengths),
    }
