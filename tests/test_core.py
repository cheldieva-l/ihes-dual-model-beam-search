from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ihes_dual.assets import find_competition_assets
from ihes_dual.beam import BeamConfig, _keep_top_k, beam_search
from ihes_dual.bidirectional import known_path_mapping_report
from ihes_dual.puzzle import IHESPuzzle, invert_path
from ihes_dual.registry import materialize_split_checkpoint, resolve_model
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
