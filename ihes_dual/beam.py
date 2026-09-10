from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .puzzle import IHESPuzzle


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 1_000_000
    max_depth: int = 30
    parent_chunk: int = 25_000
    inference_batch: int = 8_192
    device: str = "cuda"
    autocast: bool = True
    smaller_is_better: bool = True
    prune_immediate_inverse: bool = False
    diagnostic_protect: bool = False
    root_stratified: bool = False
    lookahead_pool_multiplier: int = 1
    lookahead_blend: float = 1.0
    hash_seed: int = 0x1A2B3C4D

    def __post_init__(self) -> None:
        for name in ("beam_width", "max_depth", "parent_chunk", "inference_batch"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lookahead_pool_multiplier <= 0:
            raise ValueError("lookahead_pool_multiplier must be positive")
        if not 0.0 <= self.lookahead_blend <= 1.0:
            raise ValueError("lookahead_blend must be between 0 and 1")


@dataclass
class DepthDiagnostic:
    depth: int
    frontier_size: int
    generated_count: int
    frontier_score_min: float | None = None
    frontier_score_p01: float | None = None
    frontier_score_p10: float | None = None
    frontier_score_p50: float | None = None
    frontier_score_p90: float | None = None
    frontier_score_cutoff: float | None = None
    known_state_generated: bool | None = None
    known_state_natural_top_k: bool | None = None
    known_state_protected: bool = False
    known_state_score: float | None = None
    known_state_frontier_rank: int | None = None
    known_state_raw_rank_min: int | None = None
    known_state_raw_rank_max: int | None = None
    known_state_raw_percentile: float | None = None
    known_state_root: int | None = None
    known_state_root_quota: int | None = None
    known_state_root_generated_count: int | None = None
    known_state_root_raw_rank_min: int | None = None
    known_state_root_raw_rank_max: int | None = None
    known_state_root_frontier_rank: int | None = None
    lookahead_evaluated_count: int = 0
    selection_score_cutoff: float | None = None
    known_state_in_lookahead_pool: bool | None = None
    known_state_lookahead_score: float | None = None
    known_state_lookahead_rank: int | None = None


@dataclass
class StoredFrontier:
    states: np.ndarray
    scores: np.ndarray
    last_moves: np.ndarray


@dataclass
class BeamTrace:
    start: np.ndarray
    config: BeamConfig
    parent_history: list[np.ndarray] = field(default_factory=list)
    move_history: list[np.ndarray] = field(default_factory=list)
    frontiers: dict[int, StoredFrontier] = field(default_factory=dict)
    diagnostics: list[DepthDiagnostic] = field(default_factory=list)
    solution: list[int] | None = None
    first_natural_drop: int | None = None

    def reconstruct(self, depth: int, node_index: int) -> list[int]:
        if depth < 0 or depth > len(self.move_history):
            raise ValueError(f"depth {depth} is not available")
        if depth == 0:
            if node_index != 0:
                raise IndexError("the depth-zero frontier has one node")
            return []
        path: list[int] = []
        index = int(node_index)
        for layer in range(depth - 1, -1, -1):
            path.append(int(self.move_history[layer][index]))
            index = int(self.parent_history[layer][index])
        path.reverse()
        return path

    def diagnostic_report(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "first_natural_drop": self.first_natural_drop,
            "depths": [asdict(item) for item in self.diagnostics],
        }


class Zobrist128:
    def __init__(self, state_size: int, class_count: int, seed: int) -> None:
        generator = np.random.default_rng(seed)
        shape = (state_size, class_count)
        self.table_1 = generator.integers(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)
        self.table_2 = generator.integers(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)

    def __call__(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(states, dtype=np.int64)
        if array.ndim == 1:
            array = array[None, :]
        hash_1 = np.zeros(len(array), dtype=np.uint64)
        hash_2 = np.zeros(len(array), dtype=np.uint64)
        for position in range(array.shape[1]):
            values = array[:, position]
            hash_1 ^= self.table_1[position, values]
            hash_2 ^= self.table_2[position, values]
        return hash_1, hash_2


def _deduplicate_best(
    states: np.ndarray,
    scores: np.ndarray,
    parents: np.ndarray,
    moves: np.ndarray,
    hash_1: np.ndarray,
    hash_2: np.ndarray,
) -> np.ndarray:
    """Select the best row for each exact state, using 128-bit hashes as an index."""

    order = np.lexsort((scores, hash_2, hash_1))
    ordered_h1 = hash_1[order]
    ordered_h2 = hash_2[order]
    new_hash = np.r_[
        True, (ordered_h1[1:] != ordered_h1[:-1]) | (ordered_h2[1:] != ordered_h2[:-1])
    ]
    boundaries = np.flatnonzero(new_hash)
    representative_positions = np.maximum.accumulate(
        np.where(new_hash, np.arange(len(order), dtype=np.int64), 0)
    )
    representative_rows = order[representative_positions]
    exact_match = np.ones(len(order), dtype=bool)
    # Column-wise comparison avoids allocating an N x 72 temporary array.
    for column in range(states.shape[1]):
        exact_match &= states[order, column] == states[representative_rows, column]
    if np.all(exact_match):
        return order[boundaries]

    # An actual 128-bit collision is extremely unlikely, but correctness does
    # not depend on that probability. Re-split only collided groups by exact
    # state bytes; rows within a hash group are already ordered by score.
    collided_starts = set(int(value) for value in representative_positions[~exact_match])
    selected = [int(order[position]) for position in boundaries if int(position) not in collided_starts]
    ends_by_start = dict(zip(boundaries, np.r_[boundaries[1:], len(order)]))
    for begin in sorted(collided_starts):
        exact_seen: set[bytes] = set()
        for index in order[begin : int(ends_by_start[begin])]:
            key = np.ascontiguousarray(states[index]).tobytes()
            if key not in exact_seen:
                exact_seen.add(key)
                selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def _keep_top_k(
    states: np.ndarray,
    scores: np.ndarray,
    parents: np.ndarray,
    moves: np.ndarray,
    beam_width: int,
    hasher: Zobrist128,
) -> tuple[np.ndarray, ...]:
    if len(states) == 0:
        return states, scores, parents, moves
    # It is sufficient to deduplicate the best raw prefix until it contains K
    # unique states. Every unseen row then scores no better than at least K
    # already-seen unique states. This keeps exact global top-K semantics while
    # avoiding a full sort/hash of all 18*K generated rows.
    raw_limit = min(len(states), beam_width)
    while True:
        if raw_limit < len(states):
            threshold = np.partition(scores, raw_limit - 1)[raw_limit - 1]
            # Include the complete boundary-score tie group. The final
            # deterministic (score, hash1, hash2) ordering therefore does not
            # depend on NumPy's arbitrary argpartition choices.
            subset = np.flatnonzero(scores <= threshold)
        else:
            subset = np.arange(len(states), dtype=np.int64)
        subset_h1, subset_h2 = hasher(states[subset])
        local_unique = _deduplicate_best(
            states[subset],
            scores[subset],
            parents[subset],
            moves[subset],
            subset_h1,
            subset_h2,
        )
        unique = subset[local_unique]
        if len(unique) >= beam_width or raw_limit == len(states):
            break
        deficit = beam_width - len(unique)
        raw_limit = min(len(states), max(raw_limit * 2, raw_limit + 2 * deficit))
    if len(unique) > beam_width:
        unique = unique[np.argpartition(scores[unique], beam_width - 1)[:beam_width]]
    final_h1, final_h2 = hasher(states[unique])
    rank = np.lexsort((final_h2, final_h1, scores[unique]))
    keep = unique[rank]
    return states[keep], scores[keep], parents[keep], moves[keep]


def _root_quotas(beam_width: int, root_count: int) -> np.ndarray:
    """Split the total beam as evenly as possible across first-move roots."""

    quotas = np.full(root_count, beam_width // root_count, dtype=np.int64)
    quotas[: beam_width % root_count] += 1
    return quotas


def _keep_root_stratified(
    states: np.ndarray,
    scores: np.ndarray,
    parents: np.ndarray,
    moves: np.ndarray,
    roots: np.ndarray,
    beam_width: int,
    root_count: int,
    hasher: Zobrist128,
) -> tuple[np.ndarray, ...]:
    """Keep an equal exact-state-deduplicated quota for every first move.

    Inputs are generated candidates plus their first-move root.  Outputs keep
    the same five arrays aligned.  States are deduplicated within each root;
    retaining the same state through different roots is intentional diversity.
    """

    if len(states) == 0:
        return states, scores, parents, moves, roots
    kept_parts: list[tuple[np.ndarray, ...]] = []
    for root, quota in enumerate(_root_quotas(beam_width, root_count)):
        rows = np.flatnonzero(roots == root)
        if len(rows) == 0 or quota == 0:
            continue
        kept = _keep_top_k(
            states[rows], scores[rows], parents[rows], moves[rows], int(quota), hasher
        )
        kept_parts.append((*kept, np.full(len(kept[0]), root, dtype=np.int16)))
    if not kept_parts:
        empty = np.empty(0, dtype=np.int16)
        return states[:0], scores[:0], parents[:0], moves[:0], empty
    return tuple(np.concatenate([part[column] for part in kept_parts]) for column in range(5))


def _find_exact_rows(states: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find exact rows without allocating an N x state_size Boolean array."""

    if len(states) == 0:
        return np.empty(0, dtype=np.int64)
    candidate = np.flatnonzero(states[:, 0] == target[0])
    for column in range(1, states.shape[1]):
        if len(candidate) == 0:
            break
        candidate = candidate[states[candidate, column] == target[column]]
    return candidate.astype(np.int64, copy=False)


def _score_numpy(
    states: np.ndarray,
    scorer: Callable[[torch.Tensor], torch.Tensor],
    config: BeamConfig,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    device_type = torch.device(config.device).type
    for offset in range(0, len(states), config.inference_batch):
        batch = torch.from_numpy(states[offset : offset + config.inference_batch]).to(config.device)
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=config.autocast and device_type == "cuda",
        ):
            values = scorer(batch)
        score = values.detach().float().cpu().numpy().reshape(-1)
        parts.append(score if config.smaller_is_better else -score)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _one_step_backup_scores(
    puzzle: IHESPuzzle,
    states: np.ndarray,
    last_moves: np.ndarray,
    scorer: Callable[[torch.Tensor], torch.Tensor],
    config: BeamConfig,
) -> tuple[np.ndarray, int]:
    """Return `1 + min_a h(a(state))` and the number of model evaluations."""

    backup = np.full(len(states), np.inf, dtype=np.float32)
    evaluated = 0
    for offset in range(0, len(states), config.parent_chunk):
        end = min(offset + config.parent_chunk, len(states))
        parents = states[offset:end]
        count = len(parents)
        children = parents[:, puzzle.moves].reshape(-1, puzzle.state_size)
        moves = np.tile(np.arange(puzzle.generator_count, dtype=np.int16), count)
        child_parents = np.repeat(np.arange(count, dtype=np.int32), puzzle.generator_count)
        if config.prune_immediate_inverse:
            previous = np.repeat(last_moves[offset:end], puzzle.generator_count)
            allowed = (previous < 0) | (moves != puzzle.inverse_move[np.maximum(previous, 0)])
            children = children[allowed]
            child_parents = child_parents[allowed]
        scores = _score_numpy(children, scorer, config)
        local = np.full(count, np.inf, dtype=np.float32)
        np.minimum.at(local, child_parents, scores)
        backup[offset:end] = 1.0 + local
        evaluated += len(children)
    return backup, evaluated


def _candidate_path(
    parent_index: int,
    move_index: int,
    parent_history: Sequence[np.ndarray],
    move_history: Sequence[np.ndarray],
) -> list[int]:
    path = [int(move_index)]
    index = int(parent_index)
    for layer in range(len(move_history) - 1, -1, -1):
        path.append(int(move_history[layer][index]))
        index = int(parent_history[layer][index])
    path.reverse()
    return path


def beam_search(
    puzzle: IHESPuzzle,
    start: Sequence[int] | np.ndarray,
    scorer: Callable[[torch.Tensor], torch.Tensor],
    config: BeamConfig,
    *,
    goal: Sequence[int] | np.ndarray | None = None,
    retain_depths: Sequence[int] = (),
    diagnostic_path: Sequence[int] | None = None,
) -> BeamTrace:
    """Run a global beam with all configured IHES generators.

    Diagnostic protection is deliberately separate from ordinary search. A
    known state can occupy the final slot only after an exact candidate-state
    comparison proves it was generated at that depth.
    """

    start_array = np.asarray(start, dtype=np.uint8)
    goal_array = puzzle.solved if goal is None else np.asarray(goal, dtype=np.uint8)
    if start_array.shape != (puzzle.state_size,) or goal_array.shape != start_array.shape:
        raise ValueError("start or goal has the wrong shape")
    trace = BeamTrace(start=start_array.copy(), config=config)
    retain = set(int(depth) for depth in retain_depths)
    current_states = start_array[None, :]
    current_scores = np.asarray([0.0], dtype=np.float32)
    current_last_moves = np.asarray([-1], dtype=np.int16)
    current_roots = np.asarray([-1], dtype=np.int16)
    if 0 in retain:
        trace.frontiers[0] = StoredFrontier(current_states.copy(), current_scores.copy(), current_last_moves.copy())
    if np.array_equal(start_array, goal_array):
        trace.solution = []
        return trace

    known_states: list[np.ndarray] | None = None
    if diagnostic_path is not None:
        state = start_array.copy()
        known_states = [state.copy()]
        for move in diagnostic_path:
            state = puzzle.apply(state, int(move)).astype(np.uint8, copy=False)
            known_states.append(state.copy())

    hasher = Zobrist128(puzzle.state_size, puzzle.state_size, config.hash_seed)
    for depth in range(1, config.max_depth + 1):
        reservoir_states = np.empty((0, puzzle.state_size), dtype=np.uint8)
        reservoir_scores = np.empty(0, dtype=np.float32)
        reservoir_parents = np.empty(0, dtype=np.int32)
        reservoir_moves = np.empty(0, dtype=np.int16)
        reservoir_roots = np.empty(0, dtype=np.int16)
        generated_count = 0
        found_goal: tuple[float, int, int] | None = None
        known_candidate: tuple[float, int, int, np.ndarray] | None = None
        known_target = known_states[depth] if known_states is not None and depth < len(known_states) else None
        known_target_score = (
            None
            if known_target is None
            else float(_score_numpy(known_target[None, :], scorer, config)[0])
        )
        known_raw_better = 0
        known_raw_equal = 0
        known_root = (
            None
            if (
                not config.root_stratified
                or known_target is None
                or diagnostic_path is None
                or len(diagnostic_path) == 0
            )
            else int(diagnostic_path[0])
        )
        known_root_generated_count = 0
        known_root_raw_better = 0
        known_root_raw_equal = 0
        known_root_candidate_found = False
        lookahead_evaluated_count = 0
        selection_score_cutoff = None
        known_in_lookahead_pool = None
        known_lookahead_score = None
        known_lookahead_rank = None
        pool_width = (
            config.beam_width
            if config.root_stratified
            else config.beam_width * config.lookahead_pool_multiplier
        )

        for parent_offset in range(0, len(current_states), config.parent_chunk):
            parent_end = min(parent_offset + config.parent_chunk, len(current_states))
            parents_chunk = current_states[parent_offset:parent_end]
            parent_count = len(parents_chunk)
            child_states = parents_chunk[:, puzzle.moves].reshape(-1, puzzle.state_size)
            child_moves = np.tile(np.arange(puzzle.generator_count, dtype=np.int16), parent_count)
            child_parents = np.repeat(
                np.arange(parent_offset, parent_end, dtype=np.int32), puzzle.generator_count
            )
            child_roots = None
            if config.root_stratified:
                child_roots = (
                    child_moves.copy()
                    if depth == 1
                    else np.repeat(current_roots[parent_offset:parent_end], puzzle.generator_count)
                )
            if config.prune_immediate_inverse:
                previous = np.repeat(current_last_moves[parent_offset:parent_end], puzzle.generator_count)
                allowed = (previous < 0) | (child_moves != puzzle.inverse_move[np.maximum(previous, 0)])
                child_states = child_states[allowed]
                child_moves = child_moves[allowed]
                child_parents = child_parents[allowed]
                if child_roots is not None:
                    child_roots = child_roots[allowed]
            generated_count += len(child_states)
            child_scores = _score_numpy(child_states, scorer, config)
            if known_target_score is not None:
                known_raw_better += int(np.count_nonzero(child_scores < known_target_score))
                known_raw_equal += int(np.count_nonzero(child_scores == known_target_score))
                if known_root is not None:
                    in_root = child_roots == known_root
                    root_scores = child_scores[in_root]
                    known_root_generated_count += len(root_scores)
                    known_root_raw_better += int(np.count_nonzero(root_scores < known_target_score))
                    known_root_raw_equal += int(np.count_nonzero(root_scores == known_target_score))
            for candidate in _find_exact_rows(child_states, goal_array):
                record = (float(child_scores[candidate]), int(child_parents[candidate]), int(child_moves[candidate]))
                if found_goal is None or record[0] < found_goal[0]:
                    found_goal = record

            if known_target is not None:
                for candidate in _find_exact_rows(child_states, known_target):
                    if known_root is not None and child_roots[candidate] == known_root:
                        known_root_candidate_found = True
                    record = (
                        float(child_scores[candidate]),
                        int(child_parents[candidate]),
                        int(child_moves[candidate]),
                        child_states[candidate].copy(),
                    )
                    if known_candidate is None or record[0] < known_candidate[0]:
                        known_candidate = record

            combined_states = np.concatenate((reservoir_states, child_states), axis=0)
            combined_scores = np.concatenate((reservoir_scores, child_scores))
            combined_parents = np.concatenate((reservoir_parents, child_parents))
            combined_moves = np.concatenate((reservoir_moves, child_moves))
            if config.root_stratified:
                combined_roots = np.concatenate((reservoir_roots, child_roots))
                (
                    reservoir_states,
                    reservoir_scores,
                    reservoir_parents,
                    reservoir_moves,
                    reservoir_roots,
                ) = _keep_root_stratified(
                    combined_states,
                    combined_scores,
                    combined_parents,
                    combined_moves,
                    combined_roots,
                    config.beam_width,
                    puzzle.generator_count,
                    hasher,
                )
            else:
                (
                    reservoir_states,
                    reservoir_scores,
                    reservoir_parents,
                    reservoir_moves,
                ) = _keep_top_k(
                    combined_states,
                    combined_scores,
                    combined_parents,
                    combined_moves,
                    pool_width,
                    hasher,
                )

        if (
            not config.root_stratified
            and config.lookahead_pool_multiplier > 1
            and len(reservoir_states) > config.beam_width
            and found_goal is None
        ):
            known_pool_rows = (
                np.empty(0, dtype=np.int64)
                if known_target is None
                else _find_exact_rows(reservoir_states, known_target)
            )
            known_in_lookahead_pool = None if known_target is None else len(known_pool_rows) > 0
            backup_scores, lookahead_evaluated_count = _one_step_backup_scores(
                puzzle,
                reservoir_states,
                reservoir_moves,
                scorer,
                config,
            )
            selection_scores = (
                config.lookahead_blend * reservoir_scores
                + (1.0 - config.lookahead_blend) * backup_scores
            )
            if len(known_pool_rows):
                known_lookahead_score = float(np.min(selection_scores[known_pool_rows]))
                known_lookahead_rank = int(
                    np.count_nonzero(selection_scores < known_lookahead_score) + 1
                )
            (
                reservoir_states,
                kept_selection_scores,
                reservoir_parents,
                reservoir_moves,
            ) = _keep_top_k(
                reservoir_states,
                selection_scores,
                reservoir_parents,
                reservoir_moves,
                config.beam_width,
                hasher,
            )
            selection_score_cutoff = (
                None if len(kept_selection_scores) == 0 else float(np.max(kept_selection_scores))
            )
            reservoir_scores = _score_numpy(reservoir_states, scorer, config)

        if found_goal is not None:
            _, parent_index, move_index = found_goal
            trace.solution = _candidate_path(
                parent_index, move_index, trace.parent_history, trace.move_history
            )

        natural = None
        protected = False
        known_frontier_rank = None
        known_root_frontier_rank = None
        if known_target is not None:
            known_frontier_rows = _find_exact_rows(reservoir_states, known_target)
            natural = len(known_frontier_rows) > 0
            if natural:
                known_frontier_rank = int(known_frontier_rows[0]) + 1
                if config.root_stratified and known_root is not None:
                    root_rows = np.flatnonzero(reservoir_roots == known_root)
                    matches = _find_exact_rows(reservoir_states[root_rows], known_target)
                    if len(matches):
                        known_root_frontier_rank = int(matches[0]) + 1
            if known_candidate is not None and not natural and trace.first_natural_drop is None:
                trace.first_natural_drop = depth
            if config.diagnostic_protect and known_candidate is not None and not natural:
                score, parent, move, state = known_candidate
                if len(reservoir_states) < config.beam_width:
                    reservoir_states = np.concatenate((reservoir_states, state[None, :]))
                    reservoir_scores = np.append(reservoir_scores, np.float32(score))
                    reservoir_parents = np.append(reservoir_parents, np.int32(parent))
                    reservoir_moves = np.append(reservoir_moves, np.int16(move))
                else:
                    last = len(reservoir_states) - 1
                    reservoir_states[last] = state
                    reservoir_scores[last] = score
                    reservoir_parents[last] = parent
                    reservoir_moves[last] = move
                protected = True
                if config.root_stratified:
                    if len(reservoir_roots) < len(reservoir_states):
                        reservoir_roots = np.append(reservoir_roots, np.int16(known_root))
                    else:
                        reservoir_roots[-1] = np.int16(known_root)

        trace.parent_history.append(reservoir_parents.copy())
        trace.move_history.append(reservoir_moves.copy())
        current_states = reservoir_states
        current_scores = reservoir_scores
        current_last_moves = reservoir_moves
        current_roots = reservoir_roots
        if depth in retain:
            trace.frontiers[depth] = StoredFrontier(
                current_states.copy(), current_scores.copy(), current_last_moves.copy()
            )
        score_quantiles = (
            [None] * 5
            if len(current_scores) == 0
            else [float(value) for value in np.quantile(current_scores, [0.0, 0.01, 0.1, 0.5, 0.9])]
        )
        trace.diagnostics.append(
            DepthDiagnostic(
                depth=depth,
                frontier_size=len(current_states),
                generated_count=generated_count,
                frontier_score_min=score_quantiles[0],
                frontier_score_p01=score_quantiles[1],
                frontier_score_p10=score_quantiles[2],
                frontier_score_p50=score_quantiles[3],
                frontier_score_p90=score_quantiles[4],
                frontier_score_cutoff=None if len(current_scores) == 0 else float(np.max(current_scores)),
                known_state_generated=None if known_target is None else known_candidate is not None,
                known_state_natural_top_k=natural,
                known_state_protected=protected,
                known_state_score=known_target_score,
                known_state_frontier_rank=known_frontier_rank,
                known_state_raw_rank_min=(
                    None if known_target is None else known_raw_better + 1
                ),
                known_state_raw_rank_max=(
                    None
                    if known_target is None
                    else max(
                        known_raw_better + 1,
                        known_raw_better
                        + known_raw_equal
                        + (0 if known_candidate is not None else 1),
                    )
                ),
                known_state_raw_percentile=(
                    None
                    if known_target is None or generated_count == 0
                    else round(known_raw_better / generated_count, 9)
                ),
                known_state_root=known_root,
                known_state_root_quota=(
                    None
                    if known_root is None or not config.root_stratified
                    else int(_root_quotas(config.beam_width, puzzle.generator_count)[known_root])
                ),
                known_state_root_generated_count=(
                    None if known_root is None else known_root_generated_count
                ),
                known_state_root_raw_rank_min=(
                    None if known_root is None else known_root_raw_better + 1
                ),
                known_state_root_raw_rank_max=(
                    None
                    if known_root is None
                    else max(
                        known_root_raw_better + 1,
                        known_root_raw_better
                        + known_root_raw_equal
                        + (0 if known_root_candidate_found else 1),
                    )
                ),
                known_state_root_frontier_rank=known_root_frontier_rank,
                lookahead_evaluated_count=lookahead_evaluated_count,
                selection_score_cutoff=selection_score_cutoff,
                known_state_in_lookahead_pool=known_in_lookahead_pool,
                known_state_lookahead_score=known_lookahead_score,
                known_state_lookahead_rank=known_lookahead_rank,
            )
        )
        if trace.solution is not None:
            break
        if len(current_states) == 0:
            break
    return trace
