"""Test a limited one-step Bellman rerank on IHES puzzle 999.

Input: public MLP 1778521793, the replay-valid length-23 p999 incumbent,
beam/pool/blend settings from environment variables.
Output: depth diagnostics, any replay-valid path and a compact JSON summary.
The known path is observed only; it is never protected or injected.
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


EXPERIMENT = "E007_P999_LOOKAHEAD"
PUZZLE_ID = 999


def run_lookahead() -> dict[str, object]:
    """Run one p999 lookahead search; return and persist full replay evidence."""

    started = time.time()
    power = int(os.environ.get("P999_LOOKAHEAD_BEAM_POWER", "14"))
    pool_multiplier = int(os.environ.get("P999_LOOKAHEAD_POOL_MULTIPLIER", "8"))
    blend = float(os.environ.get("P999_LOOKAHEAD_BLEND", "0"))
    start_depth = int(os.environ.get("P999_LOOKAHEAD_START_DEPTH", "5"))
    puzzle = load_puzzle()
    entry = p999_entry()
    start = np.asarray(entry["start"], dtype=np.uint8)
    incumbent = puzzle.decode_path(str(entry["incumbent_path"]))
    summary: dict[str, object] = {
        "experiment": EXPERIMENT,
        "model_id": MODEL_ID,
        "puzzle_id": PUZZLE_ID,
        "beam_power": power,
        "beam_width": 2**power,
        "pool_multiplier": pool_multiplier,
        "lookahead_blend_current_h": blend,
        "lookahead_start_depth": start_depth,
        "incumbent_length": len(incumbent),
        "incumbent_valid": bool(puzzle.verify_solution(start, incumbent)),
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        summary["status"] = "no_cuda"
        return summary

    model = load_mlp2rb(resolve_model(download_model_root(), MODEL_ID), "cuda")
    try:
        trace = beam_search(
            puzzle,
            start,
            model,
            BeamConfig(
                beam_width=2**power,
                max_depth=len(incumbent),
                parent_chunk=4096,
                inference_batch=8192,
                device="cuda",
                autocast=True,
                smaller_is_better=True,
                prune_immediate_inverse=True,
                diagnostic_protect=False,
                lookahead_pool_multiplier=pool_multiplier,
                lookahead_blend=blend,
                lookahead_start_depth=start_depth,
            ),
            diagnostic_path=incumbent,
        )
        write_csv(
            "p999_lookahead_depth_trace.csv",
            [asdict(row) for row in trace.diagnostics],
        )
        solution = trace.solution
        valid = solution is not None and puzzle.verify_solution(start, solution)
        candidate = []
        if valid:
            candidate.append(
                {
                    "puzzle_id": PUZZLE_ID,
                    "length": len(solution),
                    "valid": True,
                    "path": puzzle.encode_path(solution),
                }
            )
        write_csv("p999_lookahead_candidate.csv", candidate)
        summary.update(
            {
                "status": "solved" if solution is not None else "not_found",
                "valid": bool(valid),
                "length": len(solution) if valid else None,
                "first_natural_drop": trace.first_natural_drop,
                "completed_depth": len(trace.diagnostics),
                "generated_total": int(sum(row.generated_count for row in trace.diagnostics)),
                "lookahead_evaluated_total": int(
                    sum(row.lookahead_evaluated_count for row in trace.diagnostics)
                ),
            }
        )
    except Exception as error:
        summary.update(
            {
                "status": "error",
                "valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    return summary


def main() -> None:
    """Run E007 and always save a terminal JSON result."""

    try:
        summary = run_lookahead()
    except Exception as error:
        summary = {
            "experiment": EXPERIMENT,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    Path("p999_lookahead_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P999_LOOKAHEAD_RESULT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
