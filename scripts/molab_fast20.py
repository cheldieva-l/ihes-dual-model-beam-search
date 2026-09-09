"""Run the fixed IHES fast-20 comparison in a Molab GPU session.

Input: the public MLP checkpoint, two embedded audited symmetry frames and the
fixed puzzle ids 983..1002 from the replay-valid 21870 incumbent.
Output: one row per puzzle, one row per search, selected paths and JSON summary.

Every search is independent.  A failed frame/direction is recorded and the
incumbent path remains selected; no failed result stops the remaining work.
"""

from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ihes_dual.beam import BeamConfig, beam_search
from ihes_dual.model import load_mlp2rb
from ihes_dual.symmetry import load_symmetry_frames
from ihes_dual.registry import resolve_model
from scripts.molab_smoke import MODEL_ID, download_model_root, load_puzzle


EXPERIMENT = "E002"
DEFAULT_BEAM_POWER = 14
DEFAULT_FRAME_INDICES = (0, 40)
DATA_BLOB = (
    "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"
)
SYMMETRY_BLOB = (
    "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"
)


def decode_fast20() -> list[dict[str, object]]:
    """Return the fixed states and incumbent paths embedded in this script."""

    return json.loads(gzip.decompress(base64.b64decode(DATA_BLOB)))


def parse_frame_indices(frame_count: int) -> list[int]:
    """Read E002_FRAME_INDICES; return valid unique absolute frame indices."""

    text = os.environ.get("E002_FRAME_INDICES", "0,40").strip().lower()
    requested = list(range(frame_count)) if text == "all" else [int(x) for x in text.split(",") if x.strip()]
    return list(dict.fromkeys(index for index in requested if 0 <= index < frame_count))


def load_embedded_frames(puzzle):
    """Decode all 48 audited rotations and return frames plus selected indices."""

    raw = gzip.decompress(base64.b64decode(SYMMETRY_BLOB))
    rotations = np.load(io.BytesIO(raw), allow_pickle=False)
    temporary = Path("e002_symmetries.npy")
    np.save(temporary, rotations, allow_pickle=False)
    frames = load_symmetry_frames(str(temporary), puzzle)
    indices = parse_frame_indices(len(frames))
    return frames, indices


def search_one_direction(puzzle, start, model, config, frame, frame_index, direction):
    """Search one frame/direction; return a non-throwing run record and path."""

    started = time.time()
    record = {"frame_index": frame_index, "direction": direction}
    candidate = None
    try:
        if direction == "direct":
            transformed = frame.rotate_state(start)
        else:
            transformed = frame.reverse_start(start)
        trace = beam_search(puzzle, transformed, model, config)
        if trace.solution is not None:
            candidate = (
                frame.to_original_path(trace.solution)
                if direction == "direct"
                else frame.reverse_path_to_original(trace.solution, puzzle)
            )
        valid = candidate is not None and puzzle.verify_solution(start, candidate)
        record.update(
            {
                "status": "found" if trace.solution is not None else "not_found",
                "transformed_length": None if trace.solution is None else len(trace.solution),
                "original_replay_valid": bool(valid),
                "candidate_length": len(candidate) if valid else None,
            }
        )
        if not valid:
            candidate = None
    except Exception as error:
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "original_replay_valid": False,
                "candidate_length": None,
            }
        )
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record, candidate


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    """Write dictionaries with the union of keys in stable first-seen order."""

    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_fast20() -> dict[str, object]:
    """Search all selected fast-20 rows and save replay-valid incumbent fallback."""

    started = time.time()
    beam_power = int(os.environ.get("E002_BEAM_POWER", DEFAULT_BEAM_POWER))
    beam_width = 2**beam_power
    puzzle = load_puzzle()
    entries = decode_fast20()
    frames, frame_indices = load_embedded_frames(puzzle)
    summary: dict[str, object] = {
        "experiment": EXPERIMENT,
        "model_id": MODEL_ID,
        "beam_power": beam_power,
        "beam_width": beam_width,
        "frame_indices": frame_indices,
        "directions": ["direct", "reverse"],
        "puzzle_ids": [int(entry["puzzle_id"]) for entry in entries],
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        summary["status"] = "no_cuda"
        return summary

    model_root = download_model_root()
    model = load_mlp2rb(resolve_model(model_root, MODEL_ID), "cuda")
    puzzle_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for entry in entries:
        puzzle_started = time.time()
        puzzle_id = int(entry["puzzle_id"])
        start = np.asarray(entry["start"], dtype=np.uint8)
        incumbent_text = str(entry["incumbent_path"])
        incumbent = puzzle.decode_path(incumbent_text)
        incumbent_valid = puzzle.verify_solution(start, incumbent)
        best = list(incumbent)
        valid_search_candidates = 0
        for frame_index in frame_indices:
            for direction in ("direct", "reverse"):
                config = BeamConfig(
                    beam_width=beam_width,
                    max_depth=max(0, len(incumbent) - 1),
                    parent_chunk=4096,
                    inference_batch=8192,
                    device="cuda",
                    autocast=True,
                    smaller_is_better=True,
                    prune_immediate_inverse=True,
                )
                run, candidate = search_one_direction(
                    puzzle, start, model, config, frames[frame_index], frame_index, direction
                )
                run_rows.append({"puzzle_id": puzzle_id, **run})
                if candidate is not None:
                    valid_search_candidates += 1
                    if len(candidate) < len(best):
                        best = list(candidate)

        selected_valid = incumbent_valid and puzzle.verify_solution(start, best)
        selected_text = puzzle.encode_path(best) if selected_valid else incumbent_text
        selected_length = len(best) if selected_valid else len(incumbent)
        row = {
            "puzzle_id": puzzle_id,
            "incumbent_length": len(incumbent),
            "incumbent_valid": bool(incumbent_valid),
            "valid_search_candidates": valid_search_candidates,
            "selected_length": selected_length,
            "delta": selected_length - len(incumbent),
            "selected_valid": bool(selected_valid),
            "elapsed_seconds": round(time.time() - puzzle_started, 3),
        }
        puzzle_rows.append(row)
        selected_rows.append({"puzzle_id": puzzle_id, "path": selected_text, "length": selected_length})
        print("E002_ROW", json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    write_csv("e002_fast20_results.csv", puzzle_rows)
    write_csv("e002_search_runs.csv", run_rows)
    write_csv("e002_selected_paths.csv", selected_rows)
    summary.update(
        {
            "status": "complete",
            "rows": len(puzzle_rows),
            "selected_valid": sum(bool(row["selected_valid"]) for row in puzzle_rows),
            "searches": len(run_rows),
            "search_errors": sum(row["status"] == "error" for row in run_rows),
            "search_found_valid": sum(bool(row["original_replay_valid"]) for row in run_rows),
            "wins": sum(int(row["delta"]) < 0 for row in puzzle_rows),
            "ties": sum(int(row["delta"]) == 0 for row in puzzle_rows),
            "baseline_sum": sum(int(row["incumbent_length"]) for row in puzzle_rows),
            "selected_sum": sum(int(row["selected_length"]) for row in puzzle_rows),
            "sum_delta": sum(int(row["delta"]) for row in puzzle_rows),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    return summary


def main() -> None:
    """Run E002 and always persist a compact terminal status record."""

    try:
        summary = run_fast20()
    except Exception as error:
        summary = {
            "experiment": EXPERIMENT,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    Path("e002_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("E002_RESULT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
