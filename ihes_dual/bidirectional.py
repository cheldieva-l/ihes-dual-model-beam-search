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
    forward_bridge: tuple[int, ...] = ()
    reverse_bridge: tuple[int, ...] = ()
    join_kind: str = "exact"


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


def _expanded_match(
    puzzle: IHESPuzzle,
    source: StoredFrontier,
    target: StoredFrontier,
    target_states: np.ndarray,
    target_order: np.ndarray,
    sorted_target_keys: np.ndarray,
    hasher: Zobrist128,
    *,
    source_is_reverse: bool,
    frame_start: np.ndarray,
    parent_chunk: int,
) -> tuple[int, int, int, float] | None:
    """Scan every one-move child of one complete frontier against another."""

    best: tuple[int, int, int, float] | None = None
    for offset in range(0, len(source.states), parent_chunk):
        end = min(offset + parent_chunk, len(source.states))
        raw_children = source.states[offset:end, puzzle.moves].reshape(-1, puzzle.state_size)
        compared_children = (
            map_reverse_frontier_to_direct(raw_children, frame_start)
            if source_is_reverse
            else raw_children
        )
        child_keys = _state_keys(compared_children, hasher)
        left = np.searchsorted(sorted_target_keys, child_keys, side="left")
        right = np.searchsorted(sorted_target_keys, child_keys, side="right")
        for child_index in np.flatnonzero(left < right):
            source_parent = offset + int(child_index // puzzle.generator_count)
            move = int(child_index % puzzle.generator_count)
            for ordered_position in range(int(left[child_index]), int(right[child_index])):
                target_index = int(target_order[ordered_position])
                if not np.array_equal(compared_children[child_index], target_states[target_index]):
                    continue
                combined = float(source.scores[source_parent] + target.scores[target_index])
                candidate = (source_parent, move, target_index, combined)
                if best is None or combined < best[3]:
                    best = candidate
    return best


def blind_join_one_move(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    frame: SymmetryFrame,
    forward_trace: BeamTrace,
    reverse_trace: BeamTrace,
    *,
    forward_depths: Sequence[int],
    reverse_depths: Sequence[int],
    parent_chunk: int = 10_000,
) -> JoinedSolution | None:
    """Blindly join complete frontiers with zero or one generated bridge move.

    No target state or target hash is accepted.  If exact set intersection is
    empty, every allowed child of each complete frontier is scanned against the
    opposite complete frontier.  Generated bridge children are never described
    as retained top-K states.
    """

    exact = blind_join(
        puzzle,
        original_start,
        frame,
        forward_trace,
        reverse_trace,
        forward_depths=forward_depths,
        reverse_depths=reverse_depths,
    )
    if exact is not None:
        return exact

    frame_start = frame.rotate_state(original_start).astype(np.uint8, copy=False)
    hasher = Zobrist128(puzzle.state_size, puzzle.state_size, forward_trace.config.hash_seed)
    mapped_reverse = {
        int(depth): map_reverse_frontier_to_direct(
            reverse_trace.frontiers[int(depth)].states, frame_start
        )
        for depth in reverse_depths
    }
    forward_keys = {
        int(depth): _state_keys(forward_trace.frontiers[int(depth)].states, hasher)
        for depth in forward_depths
    }
    reverse_keys = {
        int(depth): _state_keys(mapped_reverse[int(depth)], hasher)
        for depth in reverse_depths
    }
    forward_orders = {
        depth: np.argsort(keys, kind="stable") for depth, keys in forward_keys.items()
    }
    reverse_orders = {
        depth: np.argsort(keys, kind="stable") for depth, keys in reverse_keys.items()
    }
    sorted_forward_keys = {
        depth: forward_keys[depth][order] for depth, order in forward_orders.items()
    }
    sorted_reverse_keys = {
        depth: reverse_keys[depth][order] for depth, order in reverse_orders.items()
    }

    pairs = sorted(
        ((int(fd), int(rd)) for fd in forward_depths for rd in reverse_depths),
        key=lambda item: (item[0] + item[1], item[0], item[1]),
    )
    for total_depth in sorted({fd + rd for fd, rd in pairs}):
        best: Meeting | None = None
        for forward_depth, reverse_depth in pairs:
            if forward_depth + reverse_depth != total_depth:
                continue
            forward = forward_trace.frontiers[forward_depth]
            reverse = reverse_trace.frontiers[reverse_depth]

            forward_expanded = _expanded_match(
                puzzle,
                forward,
                reverse,
                mapped_reverse[reverse_depth],
                reverse_orders[reverse_depth],
                sorted_reverse_keys[reverse_depth],
                hasher,
                source_is_reverse=False,
                frame_start=frame_start,
                parent_chunk=parent_chunk,
            )
            if forward_expanded is not None:
                forward_index, move, reverse_index, score = forward_expanded
                candidate = Meeting(
                    forward_depth,
                    reverse_depth,
                    forward_index,
                    reverse_index,
                    score,
                    forward_bridge=(move,),
                    join_kind="forward-one-move-shell",
                )
                if best is None or candidate.combined_score < best.combined_score:
                    best = candidate

            reverse_expanded = _expanded_match(
                puzzle,
                reverse,
                forward,
                forward.states,
                forward_orders[forward_depth],
                sorted_forward_keys[forward_depth],
                hasher,
                source_is_reverse=True,
                frame_start=frame_start,
                parent_chunk=parent_chunk,
            )
            if reverse_expanded is not None:
                reverse_index, move, forward_index, score = reverse_expanded
                candidate = Meeting(
                    forward_depth,
                    reverse_depth,
                    forward_index,
                    reverse_index,
                    score,
                    reverse_bridge=(move,),
                    join_kind="reverse-one-move-shell",
                )
                if best is None or candidate.combined_score < best.combined_score:
                    best = candidate

        if best is not None:
            prefix = forward_trace.reconstruct(best.forward_depth, best.forward_index)
            reverse_prefix = reverse_trace.reconstruct(best.reverse_depth, best.reverse_index)
            frame_path = (
                prefix
                + list(best.forward_bridge)
                + invert_path(
                    reverse_prefix + list(best.reverse_bridge), puzzle.inverse_move
                )
            )
            original_path = frame.to_original_path(frame_path)
            if not puzzle.verify_solution(original_start, original_path):
                raise AssertionError("one-move frontier join failed exact original replay")
            return JoinedSolution(best, tuple(frame_path), tuple(original_path))
    return None


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
