from __future__ import annotations

from typing import Sequence

import numpy as np

from .puzzle import IHESPuzzle, invert_permutation


def expected_reverse_neighbours(
    puzzle: IHESPuzzle, original_start: Sequence[int] | np.ndarray
) -> np.ndarray:
    reverse_start = invert_permutation(original_start).astype(np.uint8)
    return reverse_start[puzzle.moves]


def validate_reverse_neighbour_block(
    puzzle: IHESPuzzle,
    original_start: Sequence[int] | np.ndarray,
    observed_block: np.ndarray,
) -> dict[str, object]:
    """Validate one 18-row reverse-neighbour block as an unordered set."""

    observed = np.asarray(observed_block, dtype=np.uint8)
    expected = expected_reverse_neighbours(puzzle, original_start)
    wanted_shape = (puzzle.generator_count, puzzle.state_size)
    if observed.shape != wanted_shape:
        raise ValueError(f"reverse-neighbour block shape is {observed.shape}; expected {wanted_shape}")
    expected_keys = {row.tobytes() for row in expected}
    observed_keys = {row.tobytes() for row in observed}
    if len(expected_keys) != puzzle.generator_count:
        raise AssertionError("official reverse neighbours are unexpectedly duplicated")
    if len(observed_keys) != puzzle.generator_count:
        raise AssertionError("observed reverse-neighbour block contains duplicate rows")
    if expected_keys != observed_keys:
        raise AssertionError("reverse-neighbour block differs from all 18 official neighbours")
    official_index = {row.tobytes(): index for index, row in enumerate(expected)}
    observed_to_official = [official_index[row.tobytes()] for row in observed]
    return {
        "shape": list(observed.shape),
        "set_equal": True,
        "observed_to_official_generator_index": observed_to_official,
    }


def validate_reverse_neighbour_tensor(
    puzzle: IHESPuzzle,
    original_states: np.ndarray,
    observed: np.ndarray,
) -> list[dict[str, object]]:
    states = np.asarray(original_states, dtype=np.uint8)
    tensor = np.asarray(observed, dtype=np.uint8)
    expected_rows = len(states) * puzzle.generator_count
    if tensor.shape != (expected_rows, puzzle.state_size):
        raise ValueError(
            f"reverse-neighbour tensor shape is {tensor.shape}; "
            f"expected {(expected_rows, puzzle.state_size)}"
        )
    return [
        validate_reverse_neighbour_block(
            puzzle,
            states[puzzle_id],
            tensor[
                puzzle_id * puzzle.generator_count : (puzzle_id + 1) * puzzle.generator_count
            ],
        )
        for puzzle_id in range(len(states))
    ]

