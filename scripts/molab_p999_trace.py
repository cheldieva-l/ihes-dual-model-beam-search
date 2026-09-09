"""Measure how the incumbent puzzle-999 path competes inside an MLP beam.

Input: the embedded replay-valid p999 incumbent and public model 1778521793.
Output: per-depth beam distributions, all 18 child scores along the incumbent,
candidate paths and a compact JSON summary for beam powers 6, 10 and 14.

Runs are independent and fail soft.  Diagnostic protection is disabled, so the
known path never changes the natural beam selected by the model.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ihes_dual.beam import BeamConfig, _score_numpy, beam_search
from ihes_dual.model import load_mlp2rb
from ihes_dual.registry import resolve_model
from scripts.molab_fast20 import decode_fast20
from scripts.molab_smoke import MODEL_ID, download_model_root, load_puzzle


EXPERIMENT = "E005_P999_TRACE"
PUZZLE_ID = 999


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    """Write dictionaries with the union of keys in stable first-seen order."""

    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def p999_entry() -> dict[str, object]:
    """Return puzzle 999 from the immutable fast-20 fixture."""

    return next(entry for entry in decode_fast20() if int(entry["puzzle_id"]) == PUZZLE_ID)


def score_incumbent_children(puzzle, start, incumbent, model, config):
    """Score every generator at every incumbent vertex; return 23 x 18 rows."""

    rows: list[dict[str, object]] = []
    state = np.asarray(start, dtype=np.uint8)
    for depth, path_move in enumerate(incumbent, start=1):
        children = state[puzzle.moves]
        scores = _score_numpy(children, model, config)
        order = np.lexsort((np.arange(len(scores)), scores))
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        for move_index, score in enumerate(scores):
            rows.append(
                {
                    "depth": depth,
                    "move_index": move_index,
                    "move": puzzle.move_names[move_index],
                    "score": float(score),
                    "rank_among_18": int(ranks[move_index]),
                    "is_incumbent_move": move_index == int(path_move),
                }
            )
        state = children[int(path_move)]
    return rows


def run_trace() -> dict[str, object]:
    """Run natural direct beams and persist enough evidence to change selection."""

    started = time.time()
    powers = [
        int(value)
        for value in os.environ.get("P999_BEAM_POWERS", "6,10,14").split(",")
        if value.strip()
    ]
    puzzle = load_puzzle()
    entry = p999_entry()
    start = np.asarray(entry["start"], dtype=np.uint8)
    incumbent_text = str(entry["incumbent_path"])
    incumbent = puzzle.decode_path(incumbent_text)
    incumbent_valid = puzzle.verify_solution(start, incumbent)
    summary: dict[str, object] = {
        "experiment": EXPERIMENT,
        "model_id": MODEL_ID,
        "puzzle_id": PUZZLE_ID,
        "beam_powers": powers,
        "incumbent_length": len(incumbent),
        "incumbent_valid": bool(incumbent_valid),
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        summary["status"] = "no_cuda"
        return summary

    model = load_mlp2rb(resolve_model(download_model_root(), MODEL_ID), "cuda")
    depth_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []

    child_config = BeamConfig(
        beam_width=2 ** max(powers),
        max_depth=len(incumbent),
        parent_chunk=4096,
        inference_batch=8192,
        device="cuda",
        autocast=True,
        smaller_is_better=True,
        prune_immediate_inverse=True,
    )
    child_rows = score_incumbent_children(puzzle, start, incumbent, model, child_config)

    for power in powers:
        run_started = time.time()
        record: dict[str, object] = {"beam_power": power, "beam_width": 2**power}
        try:
            config = BeamConfig(
                beam_width=2**power,
                max_depth=len(incumbent),
                parent_chunk=min(4096, 2**power),
                inference_batch=8192,
                device="cuda",
                autocast=True,
                smaller_is_better=True,
                prune_immediate_inverse=True,
                diagnostic_protect=False,
            )
            trace = beam_search(
                puzzle,
                start,
                model,
                config,
                diagnostic_path=incumbent,
            )
            solution = trace.solution
            valid = solution is not None and puzzle.verify_solution(start, solution)
            for depth in trace.diagnostics:
                depth_rows.append({"beam_power": power, **asdict(depth)})
            if valid:
                candidate_rows.append(
                    {
                        "beam_power": power,
                        "length": len(solution),
                        "valid": True,
                        "path": puzzle.encode_path(solution),
                    }
                )
            record.update(
                {
                    "status": "solved" if solution is not None else "not_found",
                    "valid": bool(valid),
                    "length": len(solution) if valid else None,
                    "first_natural_drop": trace.first_natural_drop,
                    "completed_depth": len(trace.diagnostics),
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "valid": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        record["elapsed_seconds"] = round(time.time() - run_started, 3)
        run_summaries.append(record)
        print("P999_RUN", json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    write_csv("p999_depth_trace.csv", depth_rows)
    write_csv("p999_incumbent_children.csv", child_rows)
    write_csv("p999_candidate_paths.csv", candidate_rows)
    summary.update(
        {
            "status": "complete",
            "runs": run_summaries,
            "depth_rows": len(depth_rows),
            "child_rows": len(child_rows),
            "valid_candidates": len(candidate_rows),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    return summary


def main() -> None:
    """Run the diagnostic and always save its terminal result."""

    try:
        summary = run_trace()
    except Exception as error:
        summary = {
            "experiment": EXPERIMENT,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    Path("p999_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P999_RESULT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
