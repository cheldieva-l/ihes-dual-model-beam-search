"""Compare equal-budget root-stratified beams on IHES puzzle 999.

Input: the replay-valid embedded p999 incumbent, public MLP 1778521793 and
beam powers from P999_ROOT_BEAM_POWERS (default 10,14).
Output: one compact result per beam, per-depth root diagnostics, any valid
candidate paths, and a JSON summary.  Runs fail independently.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ihes_dual.beam import BeamConfig, beam_search
from ihes_dual.model import load_mlp2rb
from ihes_dual.registry import resolve_model
from scripts.molab_p999_trace import p999_entry, write_csv
from scripts.molab_smoke import MODEL_ID, download_model_root, load_puzzle


EXPERIMENT = "E006_P999_ROOT"
PUZZLE_ID = 999


def run_root_test() -> dict[str, object]:
    """Run equal-total-width root quotas; return and persist replay evidence."""

    started = time.time()
    powers = [
        int(value)
        for value in os.environ.get("P999_ROOT_BEAM_POWERS", "10,14").split(",")
        if value.strip()
    ]
    puzzle = load_puzzle()
    entry = p999_entry()
    start = np.asarray(entry["start"], dtype=np.uint8)
    incumbent = puzzle.decode_path(str(entry["incumbent_path"]))
    summary: dict[str, object] = {
        "experiment": EXPERIMENT,
        "model_id": MODEL_ID,
        "puzzle_id": PUZZLE_ID,
        "beam_powers": powers,
        "incumbent_length": len(incumbent),
        "incumbent_valid": bool(puzzle.verify_solution(start, incumbent)),
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        summary["status"] = "no_cuda"
        return summary

    model = load_mlp2rb(resolve_model(download_model_root(), MODEL_ID), "cuda")
    depth_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for power in powers:
        run_started = time.time()
        record: dict[str, object] = {
            "beam_power": power,
            "beam_width": 2**power,
            "selector": "equal_first_move_root_quota",
        }
        try:
            trace = beam_search(
                puzzle,
                start,
                model,
                BeamConfig(
                    beam_width=2**power,
                    max_depth=len(incumbent),
                    parent_chunk=min(4096, 2**power),
                    inference_batch=8192,
                    device="cuda",
                    autocast=True,
                    smaller_is_better=True,
                    prune_immediate_inverse=True,
                    diagnostic_protect=False,
                    root_stratified=True,
                ),
                diagnostic_path=incumbent,
            )
            for row in trace.diagnostics:
                depth_rows.append({"beam_power": power, **asdict(row)})
            solution = trace.solution
            valid = solution is not None and puzzle.verify_solution(start, solution)
            if valid:
                candidate_rows.append(
                    {
                        "beam_power": power,
                        "length": len(solution),
                        "valid": True,
                        "path": puzzle.encode_path(solution),
                    }
                )
            last = trace.diagnostics[-1] if trace.diagnostics else None
            record.update(
                {
                    "status": "solved" if solution is not None else "not_found",
                    "valid": bool(valid),
                    "length": len(solution) if valid else None,
                    "first_natural_drop": trace.first_natural_drop,
                    "completed_depth": len(trace.diagnostics),
                    "last_frontier_size": None if last is None else last.frontier_size,
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
        run_rows.append(record)
        print("P999_ROOT_RUN", json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    write_csv("p999_root_depth_trace.csv", depth_rows)
    write_csv("p999_root_candidate_paths.csv", candidate_rows)
    summary.update(
        {
            "status": "complete",
            "runs": run_rows,
            "depth_rows": len(depth_rows),
            "valid_candidates": len(candidate_rows),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    return summary


def main() -> None:
    """Run E006 and save terminal status even when one independent run fails."""

    try:
        summary = run_root_test()
    except Exception as error:
        summary = {
            "experiment": EXPERIMENT,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    Path("p999_root_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P999_ROOT_RESULT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
