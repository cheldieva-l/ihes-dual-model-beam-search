from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .beam import BeamConfig, BeamTrace, beam_search
from .bidirectional import JoinedSolution, blind_join, known_path_mapping_report
from .model import PairedContrastScorer
from .puzzle import IHESPuzzle, invert_path, invert_permutation
from .symmetry import SymmetryFrame


@dataclass(frozen=True)
class CandidateSolution:
    path: tuple[int, ...]
    symmetry_index: int
    direction: str
    beam_width: int
    model_id: str


def solve_symmetry_reverse(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    scorer: Callable[[torch.Tensor], torch.Tensor],
    config: BeamConfig,
    frames: Sequence[SymmetryFrame],
    *,
    model_id: str,
    include_direct: bool = True,
    include_reverse: bool = True,
) -> tuple[CandidateSolution | None, list[dict[str, object]]]:
    candidates: list[CandidateSolution] = []
    run_log: list[dict[str, object]] = []
    for symmetry_index, frame in enumerate(frames):
        frame_start = frame.rotate_state(original_start)
        if include_direct:
            direct = beam_search(puzzle, frame_start, scorer, config)
            valid = False
            if direct.solution is not None:
                original_path = frame.to_original_path(direct.solution)
                valid = puzzle.verify_solution(original_start, original_path)
                if valid:
                    candidates.append(
                        CandidateSolution(
                            tuple(original_path), symmetry_index, "direct", config.beam_width, model_id
                        )
                    )
            run_log.append(
                {
                    "symmetry_index": symmetry_index,
                    "direction": "direct",
                    "found": direct.solution is not None,
                    "replay_valid": valid,
                    "length": None if direct.solution is None else len(direct.solution),
                }
            )
        if include_reverse:
            reverse_start = frame.reverse_start(original_start)
            reverse = beam_search(puzzle, reverse_start, scorer, config)
            valid = False
            if reverse.solution is not None:
                original_path = frame.reverse_path_to_original(reverse.solution, puzzle)
                valid = puzzle.verify_solution(original_start, original_path)
                if valid:
                    candidates.append(
                        CandidateSolution(
                            tuple(original_path), symmetry_index, "reverse", config.beam_width, model_id
                        )
                    )
            run_log.append(
                {
                    "symmetry_index": symmetry_index,
                    "direction": "reverse",
                    "found": reverse.solution is not None,
                    "replay_valid": valid,
                    "length": None if reverse.solution is None else len(reverse.solution),
                }
            )
    best = min(candidates, key=lambda item: (len(item.path), item.symmetry_index, item.direction)) if candidates else None
    return best, run_log


@dataclass
class BidirectionalRun:
    forward: BeamTrace
    reverse: BeamTrace
    joined: JoinedSolution | None
    mapping_report: list[dict[str, object]] | None


def bidirectional_scorers(
    scorer: Callable[[torch.Tensor], torch.Tensor],
    direct_start: Sequence[int] | np.ndarray,
    scoring_mode: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Build explicit forward/reverse scorers for a documented ranking mode."""

    direct = np.asarray(direct_start, dtype=np.uint8)
    reverse = invert_permutation(direct).astype(np.uint8)
    forward_contrast = PairedContrastScorer(scorer, reverse, primary_minus_paired=True)
    reverse_direct_minus_reverse = PairedContrastScorer(
        scorer, direct, primary_minus_paired=False
    )
    if scoring_mode == "primary-only":
        return scorer, scorer
    if scoring_mode == "symmetric-primary-minus-paired":
        return forward_contrast, PairedContrastScorer(
            scorer, direct, primary_minus_paired=True
        )
    if scoring_mode == "direct-minus-reverse":
        return forward_contrast, reverse_direct_minus_reverse
    if scoring_mode == "forward-primary_reverse-direct-minus-reverse":
        return scorer, reverse_direct_minus_reverse
    raise ValueError(f"unknown bidirectional scoring mode: {scoring_mode}")


def solve_bidirectional(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    scorer: Callable[[torch.Tensor], torch.Tensor],
    config: BeamConfig,
    frame: SymmetryFrame,
    *,
    forward_depths: Sequence[int],
    reverse_depths: Sequence[int],
    known_original_path: Sequence[int] | None = None,
    paired_contrast: bool = False,
    scoring_mode: str | None = None,
) -> BidirectionalRun:
    frame_start = frame.rotate_state(original_start)
    reverse_start = frame.reverse_start(original_start)
    frame_known = None if known_original_path is None else frame.to_frame_path(known_original_path)
    reverse_known = None if frame_known is None else invert_path(frame_known, puzzle.inverse_move)
    maximum_depth = max(max(forward_depths), max(reverse_depths))
    run_config = BeamConfig(**{**asdict(config), "max_depth": maximum_depth})
    if scoring_mode is None:
        scoring_mode = "symmetric-primary-minus-paired" if paired_contrast else "primary-only"
    forward_scorer, reverse_scorer = bidirectional_scorers(scorer, frame_start, scoring_mode)
    forward = beam_search(
        puzzle,
        frame_start,
        forward_scorer,
        run_config,
        retain_depths=forward_depths,
        diagnostic_path=frame_known,
    )
    reverse = beam_search(
        puzzle,
        reverse_start,
        reverse_scorer,
        run_config,
        retain_depths=reverse_depths,
        diagnostic_path=reverse_known,
    )
    joined = blind_join(
        puzzle,
        original_start,
        frame,
        forward,
        reverse,
        forward_depths=forward_depths,
        reverse_depths=reverse_depths,
    )
    mapping_report = (
        None
        if known_original_path is None
        else known_path_mapping_report(puzzle, original_start, frame, known_original_path)
    )
    return BidirectionalRun(forward, reverse, joined, mapping_report)


def write_run_log(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
