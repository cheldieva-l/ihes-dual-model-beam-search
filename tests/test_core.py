from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ihes_dual.assets import find_competition_assets
from ihes_dual.beam import (
    BeamConfig,
    BeamTrace,
    StoredFrontier,
    Zobrist128,
    _keep_root_stratified,
    _keep_top_k,
    _one_step_backup_scores,
    beam_search,
)
from ihes_dual.bidirectional import blind_join, blind_join_one_move, known_path_mapping_report
from ihes_dual.model import PairedContrastScorer
from ihes_dual.puzzle import IHESPuzzle, invert_path
from ihes_dual.registry import materialize_split_checkpoint, resolve_model
from ihes_dual.solve import bidirectional_scorers
from ihes_dual.symmetry import SymmetryFrame, map_reverse_frontier_to_direct


def tiny_puzzle() -> IHESPuzzle:
    moves = np.asarray(
        [
            [1, 0, 2],
            [1, 0, 2],
            [0, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.int16,
    )
    return IHESPuzzle(
        solved=np.arange(3, dtype=np.int16),
        moves=moves,
        move_names=("a", "-a", "b", "-b"),
        inverse_move=np.asarray([1, 0, 3, 2], dtype=np.int16),
    )


def test_invert_path_and_mapping() -> None:
    puzzle = tiny_puzzle()
    puzzle.validate()
    start = np.asarray([2, 1, 0], dtype=np.int16)
    known = [0, 2, 0]
    assert puzzle.verify_solution(start, known)
    assert puzzle.verify_solution(puzzle.solved, invert_path(known, puzzle.inverse_move), start)
    frame = SymmetryFrame.identity(puzzle)
    report = known_path_mapping_report(puzzle, start, frame, known)
    assert all(row["exact_mapping"] for row in report)
    reverse = np.argsort(start)
    assert np.array_equal(map_reverse_frontier_to_direct(reverse, start), puzzle.solved)


def test_paired_contrast_scorer_uses_exact_value_map() -> None:
    def positional_score(states: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(1, states.shape[1] + 1, device=states.device)
        return (states.float() * weights).sum(dim=1)

    states = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.uint8)
    value_map = torch.tensor([2, 0, 1])
    scorer = PairedContrastScorer(positional_score, value_map)
    expected = positional_score(states) - positional_score(value_map[states.long()])
    assert torch.equal(scorer(states), expected)
    opposite = PairedContrastScorer(
        positional_score, value_map, primary_minus_paired=False
    )
    assert torch.equal(opposite(states), -expected)


def test_direction_aware_direct_minus_reverse_scoring() -> None:
    def positional_score(states: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(1, states.shape[1] + 1, device=states.device)
        return (states.float() * weights).sum(dim=1)

    direct_start = np.asarray([2, 0, 1], dtype=np.uint8)
    forward, reverse = bidirectional_scorers(
        positional_score, direct_start, "direct-minus-reverse"
    )
    states = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.uint8)
    inverse_start = torch.tensor(np.argsort(direct_start), dtype=torch.long)
    start_map = torch.tensor(direct_start, dtype=torch.long)
    assert torch.equal(
        forward(states),
        positional_score(states) - positional_score(inverse_start[states.long()]),
    )
    assert torch.equal(
        reverse(states),
        positional_score(start_map[states.long()]) - positional_score(states),
    )


def test_blind_join_intersects_exact_mapped_frontiers() -> None:
    puzzle = tiny_puzzle()
    start = np.asarray([2, 1, 0], dtype=np.uint8)
    frame = SymmetryFrame.identity(puzzle)
    forward_state = puzzle.apply(start, 0).astype(np.uint8)
    reverse_start = np.argsort(start).astype(np.uint8)
    reverse_state = puzzle.apply(puzzle.apply(reverse_start, 1), 3).astype(np.uint8)
    assert np.array_equal(forward_state, map_reverse_frontier_to_direct(reverse_state, start))

    config = BeamConfig(beam_width=1, max_depth=1, device="cpu", autocast=False)
    forward = BeamTrace(
        start=start,
        config=config,
        parent_history=[np.asarray([0], dtype=np.int32)],
        move_history=[np.asarray([0], dtype=np.int16)],
        frontiers={
            1: StoredFrontier(
                forward_state[None, :], np.asarray([1.0], dtype=np.float32),
                np.asarray([0], dtype=np.int16),
            )
        },
    )
    reverse = BeamTrace(
        start=reverse_start,
        config=config,
        parent_history=[np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32)],
        move_history=[np.asarray([1], dtype=np.int16), np.asarray([3], dtype=np.int16)],
        frontiers={
            2: StoredFrontier(
                reverse_state[None, :], np.asarray([2.0], dtype=np.float32),
                np.asarray([3], dtype=np.int16),
            )
        },
    )
    joined = blind_join(
        puzzle, start, frame, forward, reverse, forward_depths=(1,), reverse_depths=(2,)
    )
    assert joined is not None
    assert puzzle.verify_solution(start, joined.original_path)


def test_blind_one_move_shell_join_replays() -> None:
    puzzle = tiny_puzzle()
    start = np.asarray([2, 1, 0], dtype=np.uint8)
    frame = SymmetryFrame.identity(puzzle)
    forward_state = puzzle.apply(start, 0).astype(np.uint8)
    reverse_start = np.argsort(start).astype(np.uint8)
    reverse_state = puzzle.apply(reverse_start, 1).astype(np.uint8)
    assert not np.array_equal(
        forward_state, map_reverse_frontier_to_direct(reverse_state, start)
    )

    config = BeamConfig(beam_width=1, max_depth=1, device="cpu", autocast=False)
    forward = BeamTrace(
        start=start,
        config=config,
        parent_history=[np.asarray([0], dtype=np.int32)],
        move_history=[np.asarray([0], dtype=np.int16)],
        frontiers={
            1: StoredFrontier(
                forward_state[None, :], np.asarray([1.0], dtype=np.float32),
                np.asarray([0], dtype=np.int16),
            )
        },
    )
    reverse = BeamTrace(
        start=reverse_start,
        config=config,
        parent_history=[np.asarray([0], dtype=np.int32)],
        move_history=[np.asarray([1], dtype=np.int16)],
        frontiers={
            1: StoredFrontier(
                reverse_state[None, :], np.asarray([2.0], dtype=np.float32),
                np.asarray([1], dtype=np.int16),
            )
        },
    )
    joined = blind_join_one_move(
        puzzle,
        start,
        frame,
        forward,
        reverse,
        forward_depths=(1,),
        reverse_depths=(1,),
        parent_chunk=1,
    )
    assert joined is not None
    assert joined.meeting.join_kind in {
        "forward-one-move-shell",
        "reverse-one-move-shell",
    }
    assert len(joined.original_path) == 3
    assert puzzle.verify_solution(start, joined.original_path)


def test_beam_expands_every_generator_and_replays() -> None:
    puzzle = tiny_puzzle()
    start = np.asarray([2, 1, 0], dtype=np.int16)

    def exact_distance(states: torch.Tensor) -> torch.Tensor:
        target = torch.arange(3, device=states.device)
        return (states != target).sum(dim=1).float()

    trace = beam_search(
        puzzle,
        start,
        exact_distance,
        BeamConfig(
            beam_width=12,
            max_depth=3,
            parent_chunk=12,
            inference_batch=32,
            device="cpu",
            autocast=False,
            prune_immediate_inverse=False,
        ),
    )
    assert trace.diagnostics[0].generated_count == puzzle.generator_count
    assert trace.solution is not None
    assert puzzle.verify_solution(start, trace.solution)


def test_protection_requires_generation() -> None:
    puzzle = tiny_puzzle()
    start = np.asarray([2, 1, 0], dtype=np.int16)

    def flat_score(states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(states), device=states.device)

    trace = beam_search(
        puzzle,
        start,
        flat_score,
        BeamConfig(
            beam_width=1,
            max_depth=2,
            parent_chunk=1,
            inference_batch=8,
            device="cpu",
            autocast=False,
            diagnostic_protect=True,
        ),
        diagnostic_path=[2, 2],
    )
    for row in trace.diagnostics:
        assert not row.known_state_protected or row.known_state_generated


def test_top_k_is_exact_even_under_hash_collision() -> None:
    states = np.asarray(
        [[0, 1, 2], [2, 1, 0], [0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.uint8
    )
    scores = np.asarray([5.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)
    parents = np.arange(len(states), dtype=np.int32)
    moves = np.arange(len(states), dtype=np.int16)

    class ConstantHasher:
        def __call__(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(len(values), dtype=np.uint64), np.zeros(len(values), dtype=np.uint64)

    kept_states, kept_scores, kept_parents, _ = _keep_top_k(
        states, scores, parents, moves, beam_width=3, hasher=ConstantHasher()
    )
    assert kept_scores.tolist() == [1.0, 2.0, 3.0]
    assert kept_parents.tolist() == [2, 1, 3]
    assert len({row.tobytes() for row in kept_states}) == 3


def test_root_stratified_keeps_equal_first_move_quotas() -> None:
    states = np.asarray([[index, 9] for index in range(8)], dtype=np.uint8)
    scores = np.asarray([0, 1, 2, 3, 100, 101, 102, 103], dtype=np.float32)
    parents = np.arange(8, dtype=np.int32)
    moves = np.arange(8, dtype=np.int16)
    roots = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
    kept = _keep_root_stratified(
        states,
        scores,
        parents,
        moves,
        roots,
        beam_width=4,
        root_count=2,
        hasher=Zobrist128(2, 10, 123),
    )
    assert kept[4].tolist() == [0, 0, 1, 1]
    assert kept[1].tolist() == [0.0, 1.0, 100.0, 101.0]


def test_one_step_backup_uses_best_allowed_child() -> None:
    puzzle = tiny_puzzle()
    states = np.asarray([[2, 1, 0]], dtype=np.uint8)

    def exact_distance(batch: torch.Tensor) -> torch.Tensor:
        target = torch.arange(3, device=batch.device)
        return (batch != target).sum(dim=1).float()

    config = BeamConfig(
        beam_width=4,
        max_depth=2,
        parent_chunk=4,
        inference_batch=8,
        device="cpu",
        autocast=False,
        prune_immediate_inverse=False,
    )
    backup, evaluated = _one_step_backup_scores(
        puzzle,
        states,
        np.asarray([-1], dtype=np.int16),
        exact_distance,
        config,
    )
    children = states[:, puzzle.moves].reshape(-1, puzzle.state_size)
    expected = 1.0 + float(exact_distance(torch.from_numpy(children)).min())
    assert evaluated == puzzle.generator_count
    assert backup.tolist() == [expected]


def test_split_checkpoint_materialization(tmp_path: Path) -> None:
    shards = []
    for index, payload in enumerate((b"abc", b"def", b"ghi"), start=1):
        shard = tmp_path / f"checkpoint_{index:02d}.txt"
        shard.write_bytes(payload)
        shards.append(shard)
    output = materialize_split_checkpoint(shards, tmp_path / "model.pth")
    assert output.read_bytes() == b"abcdefghi"


def test_apple_archive_shards_are_not_raw_concatenated(tmp_path: Path) -> None:
    (tmp_path / "model_p888-t000_1780290207.json").write_text("{}", encoding="utf-8")
    (tmp_path / "p888-t000_1780290207_e40960_01.txt").write_bytes(b"Aar!payload")
    try:
        resolve_model(tmp_path, "1780290207")
    except RuntimeError as error:
        assert "Apple Archive shards" in str(error)
    else:
        raise AssertionError("Apple Archive shards must not be concatenated as raw bytes")


def test_competition_assets_ignore_auxiliary_puzzle_info(tmp_path: Path) -> None:
    competition = tmp_path / "competition"
    competition.mkdir()
    for filename in ("puzzle_info.json", "test.csv", "sample_submission.csv"):
        (competition / filename).write_text("", encoding="utf-8")
    artifacts = tmp_path / "symmetry-artifacts"
    artifacts.mkdir()
    (artifacts / "puzzle_info.json").write_text("", encoding="utf-8")

    assets = find_competition_assets(tmp_path)

    assert assets.puzzle_info == competition / "puzzle_info.json"
    assert assets.test_csv == competition / "test.csv"
    assert assets.sample_submission == competition / "sample_submission.csv"
