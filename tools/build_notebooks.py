from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REPOSITORY_URL = "https://github.com/cheldieva-l/ihes-dual-model-beam-search"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "dataSources": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = f"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

REPOSITORY_URL = {REPOSITORY_URL!r}
REPOSITORY_REF = "main"
CHECKOUT = Path("/kaggle/working/ihes-dual-model-beam-search")
if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", REPOSITORY_REF, REPOSITORY_URL, str(CHECKOUT)],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(CHECKOUT), "--no-deps", "-q"], check=True)
sys.path.insert(0, str(CHECKOUT))
print(subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip())
"""


COMMON_ASSETS = """
import numpy as np
import torch

from ihes_dual.assets import find_competition_assets
from ihes_dual.model import load_mlp2rb
from ihes_dual.puzzle import IHESPuzzle, load_test_state
from ihes_dual.registry import resolve_model

assets = find_competition_assets("/kaggle/input")
puzzle = IHESPuzzle.from_puzzle_info(assets.puzzle_info)
start = load_test_state(assets.test_csv, PUZZLE_ID)
model_spec = resolve_model("/kaggle/input", MODEL_ID)
if model_spec.model_id != MODEL_ID:
    raise AssertionError("the resolved checkpoint is not tied to the configured model ID")
model = load_mlp2rb(model_spec, DEVICE)
print({
    "puzzle_id": PUZZLE_ID,
    "model_id": model_spec.model_id,
    "checkpoint": model_spec.checkpoint.name,
    "epoch": model_spec.epoch,
    "beam_width": BEAM_WIDTH,
    "device": DEVICE,
    "generator_count": puzzle.generator_count,
})
assert puzzle.generator_count == 18
"""


BASE = notebook(
    [
        markdown(
            """
            # IHES Cube 106 — Base MLP Global Beam Search

            This notebook runs the clean-room implementation from the public repository with model `1778521793`, a global beam of 1,000,000 states, and all 18 official generators. A result is accepted only after exact replay from the original puzzle state. The final two-column submission is then replay-validated for all 1,003 puzzles.
            """
        ),
        code(
            """
            from pathlib import Path
            PUZZLE_ID = 106
            MODEL_ID = "1778521793"
            BEAM_WIDTH = 1_000_000
            MAX_DEPTH = 30
            PARENT_CHUNK = 250_000
            INFERENCE_BATCH = 8_192
            DEVICE = "cuda"
            OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
            """
        ),
        code(COMMON_SETUP),
        code(COMMON_ASSETS),
        code(
            """
            from dataclasses import asdict
            from ihes_dual.beam import BeamConfig, beam_search
            from ihes_dual.solve import write_run_log
            from ihes_dual.submission import build_submission, validate_submission

            config = BeamConfig(
                beam_width=BEAM_WIDTH,
                max_depth=MAX_DEPTH,
                parent_chunk=PARENT_CHUNK,
                inference_batch=INFERENCE_BATCH,
                device=DEVICE,
                autocast=True,
                prune_immediate_inverse=False,
                diagnostic_protect=False,
            )
            trace = beam_search(puzzle, start, model, config)
            if trace.solution is None:
                raise RuntimeError("base beam did not find a solution")
            if not 22 <= len(trace.solution) <= 30:
                raise AssertionError(f"controlled solution length is {len(trace.solution)}, expected 22..30")
            if not puzzle.verify_solution(start, trace.solution):
                raise AssertionError("base solution failed exact replay")
            print("solution length:", len(trace.solution))
            print("solution:", puzzle.encode_path(trace.solution))

            run_dir = OUTPUT_DIR / "runs" / f"model-{MODEL_ID}" / f"puzzle-{PUZZLE_ID}" / "base"
            write_run_log(
                run_dir / "run.json",
                {
                    "mode": "base",
                    "model_id": MODEL_ID,
                    "puzzle_id": PUZZLE_ID,
                    "solution": puzzle.encode_path(trace.solution),
                    "solution_length": len(trace.solution),
                    "replay_valid": True,
                    "config": asdict(config),
                    "depths": trace.diagnostic_report()["depths"],
                },
            )
            submission_path = build_submission(
                assets.sample_submission,
                OUTPUT_DIR / "submission.csv",
                puzzle,
                {PUZZLE_ID: trace.solution},
            )
            validation = validate_submission(submission_path, assets.test_csv, puzzle)
            print("submission:", submission_path)
            print("validation:", validation)
            """
        ),
    ]
)


SYMMETRY_REVERSE = notebook(
    [
        markdown(
            """
            # IHES Cube — Symmetry + Reverse MLP Beam Search

            This notebook validates the 48 IHES relabellings, searches selected symmetry frames in both direct and reverse projections, converts every candidate back to original coordinates, and accepts only exact replay-valid paths. It uses every official generator and never relies on the row order of the reverse-neighbour tensor.
            """
        ),
        code(
            """
            from pathlib import Path
            PUZZLE_ID = 106
            MODEL_ID = "1778521793"
            BEAM_WIDTH = 1_000_000
            MAX_DEPTH = 30
            PARENT_CHUNK = 250_000
            INFERENCE_BATCH = 8_192
            DEVICE = "cuda"
            K_SYM = 2
            SYMMETRY_SEED = 0
            OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
            """
        ),
        code(COMMON_SETUP),
        code(COMMON_ASSETS),
        code(
            """
            from ihes_dual.assets import find_symmetry_file
            from ihes_dual.symmetry import load_symmetry_frames

            symmetry_path = find_symmetry_file("/kaggle/input")
            frames_all = load_symmetry_frames(str(symmetry_path), puzzle)
            identity_index = next(
                index for index, frame in enumerate(frames_all)
                if np.array_equal(frame.rotation, np.arange(puzzle.state_size))
            )
            rng = np.random.default_rng(SYMMETRY_SEED)
            other_indices = [index for index in range(len(frames_all)) if index != identity_index]
            chosen_indices = [identity_index]
            if K_SYM > 1:
                chosen_indices.extend(rng.choice(other_indices, size=K_SYM - 1, replace=False).tolist())
            frames = [frames_all[index] for index in chosen_indices]
            print("validated symmetries:", len(frames_all), "chosen:", chosen_indices)
            """
        ),
        code(
            """
            from dataclasses import asdict
            from ihes_dual.beam import BeamConfig
            from ihes_dual.solve import solve_symmetry_reverse, write_run_log
            from ihes_dual.submission import build_submission, validate_submission

            config = BeamConfig(
                beam_width=BEAM_WIDTH,
                max_depth=MAX_DEPTH,
                parent_chunk=PARENT_CHUNK,
                inference_batch=INFERENCE_BATCH,
                device=DEVICE,
                autocast=True,
                prune_immediate_inverse=False,
                diagnostic_protect=False,
            )
            best, runs = solve_symmetry_reverse(
                puzzle,
                start,
                model,
                config,
                frames,
                model_id=MODEL_ID,
                include_direct=True,
                include_reverse=True,
            )
            if best is None:
                raise RuntimeError("no replay-valid direct or reverse candidate was found")
            if not puzzle.verify_solution(start, best.path):
                raise AssertionError("best transformed path failed exact original-coordinate replay")
            print("best:", best)
            print("solution:", puzzle.encode_path(best.path))

            run_dir = OUTPUT_DIR / "runs" / f"model-{MODEL_ID}" / f"puzzle-{PUZZLE_ID}" / "symmetry-reverse"
            write_run_log(
                run_dir / "run.json",
                {
                    "mode": "symmetry-reverse",
                    "model_id": MODEL_ID,
                    "puzzle_id": PUZZLE_ID,
                    "chosen_absolute_symmetry_indices": chosen_indices,
                    "best": asdict(best),
                    "best_path": puzzle.encode_path(best.path),
                    "runs": runs,
                    "config": asdict(config),
                },
            )
            submission_path = build_submission(
                assets.sample_submission,
                OUTPUT_DIR / "submission.csv",
                puzzle,
                {PUZZLE_ID: best.path},
            )
            print("validation:", validate_submission(submission_path, assets.test_csv, puzzle))
            """
        ),
    ]
)


BIDIRECTIONAL = notebook(
    [
        markdown(
            """
            # IHES Cube — Bidirectional Symmetry + Reverse Beam Search

            This notebook builds independent direct and reverse beams. The forward beam uses the controlled model's primary score, while the reverse beam uses the direction-aware `direct score - reverse score` paired-projection objective. It first maps every reverse-frontier row into the direct projection and performs the required blind exact full-frontier intersection without a supplied midpoint or midpoint hash. If that exact set intersection is empty, an explicitly labelled blind extension scans every one-move child shell against the opposite complete frontier. Shell children are generated candidates, not claimed top-K states. The known cube-106 path is used only to audit true top-K retention and the optional, separate last-slot protection mode. Protected frontiers are never used for either blind join.
            """
        ),
        code(
            """
            from pathlib import Path
            PUZZLE_ID = 106
            MODEL_ID = "1778521793"
            BEAM_WIDTH = 1_000_000
            FORWARD_DEPTHS = tuple(range(12, 17))
            REVERSE_DEPTHS = tuple(range(12, 17))
            SYMMETRY_ABSOLUTE_INDEX = None  # None selects the identity frame.
            PARENT_CHUNK = 250_000
            INFERENCE_BATCH = 8_192
            DEVICE = "cuda"
            SCORING_MODE = "forward-primary_reverse-direct-minus-reverse"
            ALLOW_ONE_MOVE_SHELL = True
            SHELL_PARENT_CHUNK = 10_000
            RUN_PROTECTED_DIAGNOSTIC = True
            KNOWN_PATH_106 = "-r2.-d2.-f2.r1.r1.d0.r2.-d0.-r0.-f0.d0.r0.f1.-d0.f1.r2.r1.-d0.-r2.-f1.-f2.d1.r0.d0"
            OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
            """
        ),
        code(COMMON_SETUP),
        code(COMMON_ASSETS),
        code(
            """
            from ihes_dual.assets import find_symmetry_file
            from ihes_dual.bidirectional import known_path_mapping_report
            from ihes_dual.symmetry import load_symmetry_frames

            frames = load_symmetry_frames(str(find_symmetry_file("/kaggle/input")), puzzle)
            if SYMMETRY_ABSOLUTE_INDEX is None:
                frame_index = next(
                    index for index, item in enumerate(frames)
                    if np.array_equal(item.rotation, np.arange(puzzle.state_size))
                )
            else:
                frame_index = int(SYMMETRY_ABSOLUTE_INDEX)
            frame = frames[frame_index]
            known_path = puzzle.decode_path(KNOWN_PATH_106)
            if PUZZLE_ID != 106:
                known_path = None
            else:
                assert len(known_path) == 24
                assert puzzle.verify_solution(start, known_path)
                mapping_report = known_path_mapping_report(puzzle, start, frame, known_path)
                assert all(row["exact_mapping"] for row in mapping_report)
                print("known-path mapping checks:", len(mapping_report))
            print("symmetry absolute index:", frame_index)
            """
        ),
        code(
            """
            from dataclasses import asdict
            from ihes_dual.beam import BeamConfig, beam_search
            from ihes_dual.bidirectional import blind_join_one_move
            from ihes_dual.puzzle import invert_path
            from ihes_dual.solve import (
                bidirectional_scorers,
                solve_bidirectional,
                write_run_log,
            )
            from ihes_dual.submission import build_submission, validate_submission

            blind_config = BeamConfig(
                beam_width=BEAM_WIDTH,
                max_depth=max(max(FORWARD_DEPTHS), max(REVERSE_DEPTHS)),
                parent_chunk=PARENT_CHUNK,
                inference_batch=INFERENCE_BATCH,
                device=DEVICE,
                autocast=True,
                prune_immediate_inverse=False,
                diagnostic_protect=False,
            )
            blind = solve_bidirectional(
                puzzle,
                start,
                model,
                blind_config,
                frame,
                forward_depths=FORWARD_DEPTHS,
                reverse_depths=REVERSE_DEPTHS,
                known_original_path=known_path,
                scoring_mode=SCORING_MODE,
            )
            exact_intersection_found = blind.joined is not None
            if blind.joined is None and ALLOW_ONE_MOVE_SHELL:
                blind.joined = blind_join_one_move(
                    puzzle,
                    start,
                    frame,
                    blind.forward,
                    blind.reverse,
                    forward_depths=FORWARD_DEPTHS,
                    reverse_depths=REVERSE_DEPTHS,
                    parent_chunk=SHELL_PARENT_CHUNK,
                )
            ordinary_reports = {
                "forward": blind.forward.diagnostic_report(),
                "reverse": blind.reverse.diagnostic_report(),
            }
            for direction, report in ordinary_reports.items():
                print(direction, "ordinary first natural drop:", report["first_natural_drop"])
                for row in report["depths"]:
                    print(direction, "ordinary", row)
            print("blind exact intersection found:", exact_intersection_found)
            print("blind exact-or-one-move join found:", blind.joined is not None)
            if blind.joined is not None:
                print("blind join kind:", blind.joined.meeting.join_kind)
            """
        ),
        code(
            """
            protected_report = None
            needs_protected_run = any(
                report["first_natural_drop"] is not None for report in ordinary_reports.values()
            )
            if RUN_PROTECTED_DIAGNOSTIC and known_path is not None and needs_protected_run:
                protected_config = BeamConfig(**{**asdict(blind_config), "diagnostic_protect": True})
                frame_start = frame.rotate_state(start)
                frame_known = frame.to_frame_path(known_path)
                reverse_start = frame.reverse_start(start)
                reverse_known = invert_path(frame_known, puzzle.inverse_move)
                diagnostic_forward_scorer, diagnostic_reverse_scorer = (
                    bidirectional_scorers(model, frame_start, SCORING_MODE)
                )
                diagnostic_forward = beam_search(
                    puzzle,
                    frame_start,
                    diagnostic_forward_scorer,
                    protected_config,
                    diagnostic_path=frame_known,
                )
                diagnostic_reverse = beam_search(
                    puzzle,
                    reverse_start,
                    diagnostic_reverse_scorer,
                    protected_config,
                    diagnostic_path=reverse_known,
                )
                protected_report = {
                    "forward": diagnostic_forward.diagnostic_report(),
                    "reverse": diagnostic_reverse.diagnostic_report(),
                }
                for direction, report in protected_report.items():
                    print(direction, "first natural drop:", report["first_natural_drop"])
                    for row in report["depths"]:
                        print(direction, row)
                        assert not row["known_state_protected"] or row["known_state_generated"]
            elif RUN_PROTECTED_DIAGNOSTIC and known_path is not None:
                protected_report = {
                    "skipped": "ordinary true top-K retained every audited path point"
                }

            run_dir = OUTPUT_DIR / "runs" / f"model-{MODEL_ID}" / f"puzzle-{PUZZLE_ID}" / "bidirectional"
            run_payload = {
                "mode": "bidirectional-symmetry-reverse",
                "model_id": MODEL_ID,
                "puzzle_id": PUZZLE_ID,
                "symmetry_absolute_index": frame_index,
                "blind_config": asdict(blind_config),
                "scoring": SCORING_MODE,
                "forward_depths": FORWARD_DEPTHS,
                "reverse_depths": REVERSE_DEPTHS,
                "mapped_reverse_frontier_rows": {
                    str(depth): len(blind.reverse.frontiers[depth].states)
                    for depth in REVERSE_DEPTHS
                },
                "reverse_mapping_formula": "mapped[q] = direct_start[reverse_state[q]]",
                "mapping_applied_before_hashing": True,
                "blind_intersector_used_known_midpoint_or_hash": False,
                "protected_frontiers_used_for_blind_join": False,
                "exact_intersection_found": exact_intersection_found,
                "one_move_shell_enabled": ALLOW_ONE_MOVE_SHELL,
                "one_move_shell_parent_chunk": SHELL_PARENT_CHUNK,
                "shell_children_are_retained_top_k": False,
                "join_kind": (
                    None if blind.joined is None else blind.joined.meeting.join_kind
                ),
                "meeting": None if blind.joined is None else asdict(blind.joined.meeting),
                "solution": (
                    None if blind.joined is None
                    else puzzle.encode_path(blind.joined.original_path)
                ),
                "solution_length": (
                    None if blind.joined is None else len(blind.joined.original_path)
                ),
                "replay_valid": (
                    False if blind.joined is None
                    else puzzle.verify_solution(start, blind.joined.original_path)
                ),
                "known_path_mapping": blind.mapping_report,
                "ordinary_forward_diagnostic": ordinary_reports["forward"],
                "ordinary_reverse_diagnostic": ordinary_reports["reverse"],
                "protected_diagnostic": protected_report,
            }
            write_run_log(run_dir / "run.json", run_payload)
            if blind.joined is None:
                raise RuntimeError("blind exact and one-move complete-frontier joins found no meeting")
            if not run_payload["replay_valid"]:
                raise AssertionError("blind joined path failed original-coordinate replay")
            print("blind meeting:", blind.joined.meeting)
            print("blind solution length:", len(blind.joined.original_path))
            print("blind solution:", puzzle.encode_path(blind.joined.original_path))
            """
        ),
        code(
            """
            submission_path = build_submission(
                assets.sample_submission,
                OUTPUT_DIR / "submission.csv",
                puzzle,
                {PUZZLE_ID: blind.joined.original_path},
            )
            submission_validation = validate_submission(submission_path, assets.test_csv, puzzle)
            run_payload["submission_validation"] = submission_validation
            write_run_log(run_dir / "run.json", run_payload)
            print("submission:", submission_path)
            print("validation:", submission_validation)
            """
        ),
    ]
)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    payloads = {
        "01_base_mlp_beam.ipynb": BASE,
        "02_symmetry_reverse.ipynb": SYMMETRY_REVERSE,
        "03_bidirectional_symmetry_reverse.ipynb": BIDIRECTIONAL,
    }
    for name, payload in payloads.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
