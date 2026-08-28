from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .beam import BeamTrace, StoredFrontier, Zobrist128
from .puzzle import IHESPuzzle, invert_path
from .symmetry import SymmetryFrame, map_reverse_frontier_to_direct


@dataclass(frozen=True)
class Meeting:
    forward_depth: int
    reverse_depth: int
    forward_index: int
    reverse_index: int
    combined_score: float


@dataclass(frozen=True)
class JoinedSolution:
    meeting: Meeting
    frame_path: tuple[int, ...]
    original_path: tuple[int, ...]


def _state_keys(states: np.ndarray, hasher: Zobrist128) -> np.ndarray:
    hash_1, hash_2 = hasher(states)
    pair_dtype = np.dtype([("a", "<u8"), ("b", "<u8")])
    keys = np.empty(len(states), dtype=pair_dtype)
    keys["a"], keys["b"] = hash_1, hash_2
    return keys


def _hash_intersection(
    forward: StoredFrontier,
    forward_keys: np.ndarray,
    mapped_reverse_states: np.ndarray,
    reverse_scores: np.ndarray,
    reverse_keys: np.ndarray,
) -> tuple[int, int, float] | None:
    reverse_order = np.argsort(reverse_keys, kind="stable")
    sorted_reverse_keys = reverse_keys[reverse_order]
    left = np.searchsorted(sorted_reverse_keys, forward_keys, side="left")
    right = np.searchsorted(sorted_reverse_keys, forward_keys, side="right")
    best: tuple[int, int, float] | None = None
    for forward_index in np.flatnonzero(left < right):
        # Hashes only index candidates. Every row in a collision range is
        # compared exactly, so even a real 128-bit collision cannot hide a meeting.
        for ordered_position in range(int(left[forward_index]), int(right[forward_index])):
            reverse_index = int(reverse_order[ordered_position])
            if not np.array_equal(forward.states[forward_index], mapped_reverse_states[reverse_index]):
                continue
            combined = float(forward.scores[forward_index] + reverse_scores[reverse_index])
            if best is None or combined < best[2]:
                best = (int(forward_index), reverse_index, combined)
    return best


def blind_join(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    frame: SymmetryFrame,
    forward_trace: BeamTrace,
    reverse_trace: BeamTrace,
    *,
    forward_depths: Sequence[int],
    reverse_depths: Sequence[int],
) -> JoinedSolution | None:
    """Intersect complete stored frontiers without a supplied midpoint or hash."""

    frame_start = frame.rotate_state(original_start)
    hasher = Zobrist128(puzzle.state_size, puzzle.state_size, forward_trace.config.hash_seed)
    forward_keys = {
        int(depth): _state_keys(forward_trace.frontiers[int(depth)].states, hasher)
        for depth in forward_depths
    }
    mapped_reverse = {
        int(depth): map_reverse_frontier_to_direct(
            reverse_trace.frontiers[int(depth)].states, frame_start
        )
        for depth in reverse_depths
    }
    reverse_keys = {
        int(depth): _state_keys(mapped_reverse[int(depth)], hasher) for depth in reverse_depths
    }
    best_meeting: Meeting | None = None
    for forward_depth in forward_depths:
        forward = forward_trace.frontiers[int(forward_depth)]
        for reverse_depth in reverse_depths:
            reverse = reverse_trace.frontiers[int(reverse_depth)]
            mapped = mapped_reverse[int(reverse_depth)]
            match = _hash_intersection(
                forward,
                forward_keys[int(forward_depth)],
                mapped,
                reverse.scores,
                reverse_keys[int(reverse_depth)],
            )
            if match is None:
                continue
            forward_index, reverse_index, score = match
            meeting = Meeting(
                int(forward_depth), int(reverse_depth), forward_index, reverse_index, score
            )
            if best_meeting is None or (
                meeting.forward_depth + meeting.reverse_depth,
                meeting.combined_score,
            ) < (
                best_meeting.forward_depth + best_meeting.reverse_depth,
                best_meeting.combined_score,
            ):
                best_meeting = meeting
    if best_meeting is None:
        return None
    prefix = forward_trace.reconstruct(best_meeting.forward_depth, best_meeting.forward_index)
    reverse_prefix = reverse_trace.reconstruct(best_meeting.reverse_depth, best_meeting.reverse_index)
    frame_path = prefix + invert_path(reverse_prefix, puzzle.inverse_move)
    original_path = frame.to_original_path(frame_path)
    if not puzzle.verify_solution(original_start, original_path):
        raise AssertionError("joined path failed exact replay in original coordinates")
    return JoinedSolution(best_meeting, tuple(frame_path), tuple(original_path))


def known_path_mapping_report(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    frame: SymmetryFrame,
    original_known_path: Sequence[int],
) -> list[dict[str, object]]:
    """Prove the direct/reverse projection relation along a known solution."""

    frame_start = frame.rotate_state(original_start)
    frame_path = frame.to_frame_path(original_known_path)
    reverse_path = invert_path(frame_path, puzzle.inverse_move)
    direct_states = [frame_start]
    for move in frame_path:
        direct_states.append(puzzle.apply(direct_states[-1], move))
    reverse_start = np.argsort(frame_start).astype(np.uint8)
    reverse_states = [reverse_start]
    for move in reverse_path:
        reverse_states.append(puzzle.apply(reverse_states[-1], move))
    report: list[dict[str, object]] = []
    length = len(frame_path)
    for forward_depth in range(length + 1):
        reverse_depth = length - forward_depth
        mapped = map_reverse_frontier_to_direct(reverse_states[reverse_depth], frame_start)
        exact = bool(np.array_equal(direct_states[forward_depth], mapped))
        report.append(
            {
                "forward_depth": forward_depth,
                "reverse_depth": reverse_depth,
                "exact_mapping": exact,
            }
        )
    if not all(row["exact_mapping"] for row in report):
        raise AssertionError("known-path projection mapping failed")
    return report
