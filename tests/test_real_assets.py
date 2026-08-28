from __future__ import annotations

import os
from pathlib import Path

import pytest
import numpy as np
import torch

from ihes_dual.bidirectional import known_path_mapping_report
from ihes_dual.model import load_mlp2rb
from ihes_dual.puzzle import IHESPuzzle, load_test_state
from ihes_dual.registry import resolve_model
from ihes_dual.symmetry import load_symmetry_frames
from ihes_dual.validation import validate_reverse_neighbour_block


REQUIRED = ("IHES_PUZZLE_INFO", "IHES_TEST_CSV", "IHES_SYMMETRIES", "IHES_MODEL_ROOT")
pytestmark = pytest.mark.skipif(
    any(not os.environ.get(name) for name in REQUIRED), reason="real IHES assets are not configured"
)


KNOWN_106 = "-r2.-d2.-f2.r1.r1.d0.r2.-d0.-r0.-f0.d0.r0.f1.-d0.f1.r2.r1.-d0.-r2.-f1.-f2.d1.r0.d0"


def test_pid_106_assets_and_projection() -> None:
    puzzle = IHESPuzzle.from_puzzle_info(os.environ["IHES_PUZZLE_INFO"])
    state = load_test_state(os.environ["IHES_TEST_CSV"], 106)
    path = puzzle.decode_path(KNOWN_106)
    assert len(path) == 24
    assert puzzle.verify_solution(state, path)
    frames = load_symmetry_frames(os.environ["IHES_SYMMETRIES"], puzzle)
    assert len(frames) == 48
    for frame in frames:
        assert puzzle.verify_solution(state, frame.to_original_path(frame.to_frame_path(path)))
    assert all(row["exact_mapping"] for row in known_path_mapping_report(puzzle, state, frames[0], path))


def test_model_1778521793_loads_strictly() -> None:
    spec = resolve_model(Path(os.environ["IHES_MODEL_ROOT"]), "1778521793")
    model = load_mlp2rb(spec, "cpu")
    assert sum(parameter.numel() for parameter in model.parameters()) == 15_357_749


def test_model_1763232740_when_configured() -> None:
    model_root = os.environ.get("IHES_MODEL_176_ROOT")
    if not model_root:
        pytest.skip("model 1763232740 is not configured")
    spec = resolve_model(model_root, "ihes-e08192")
    assert spec.model_id == "1763232740"
    model = load_mlp2rb(spec, "cpu")
    assert sum(parameter.numel() for parameter in model.parameters()) == 15_357_749


def test_model_1780290207_when_configured() -> None:
    model_root = os.environ.get("IHES_MODEL_178_ROOT")
    if not model_root:
        pytest.skip("model 1780290207 is not configured")
    spec = resolve_model(model_root, "1780290207")
    assert spec.model_id == "1780290207"
    model = load_mlp2rb(spec, "cpu")
    assert sum(parameter.numel() for parameter in model.parameters()) == 15_357_749


def test_reverse_neighbour_block_when_configured() -> None:
    reverse_path = os.environ.get("IHES_REVERSE_NEIGHBOURS")
    if not reverse_path:
        pytest.skip("reverse-neighbour tensor is not configured")
    puzzle = IHESPuzzle.from_puzzle_info(os.environ["IHES_PUZZLE_INFO"])
    state = load_test_state(os.environ["IHES_TEST_CSV"], 106)
    tensor = torch.load(reverse_path, map_location="cpu", weights_only=True)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    block = np.asarray(tensor)[106 * puzzle.generator_count : 107 * puzzle.generator_count]
    report = validate_reverse_neighbour_block(puzzle, state, block)
    assert report["set_equal"] is True
