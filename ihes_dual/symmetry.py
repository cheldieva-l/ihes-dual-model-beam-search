from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .puzzle import IHESPuzzle, invert_path, invert_permutation


@dataclass(frozen=True)
class SymmetryFrame:
    """A relabelling frame using the IHES conjugation convention.

    If ``R`` is a cube relabelling, a state is rotated by
    ``R[state[R_inverse]]``. ``frame_to_original_move`` maps a move found in
    that rotated frame back to the original puzzle coordinates.
    """

    rotation: np.ndarray
    rotation_inverse: np.ndarray
    frame_to_original_move: np.ndarray
    original_to_frame_move: np.ndarray

    @classmethod
    def identity(cls, puzzle: IHESPuzzle) -> "SymmetryFrame":
        identity = np.arange(puzzle.state_size, dtype=np.uint8)
        move_identity = np.arange(puzzle.generator_count, dtype=np.int16)
        return cls(identity, identity.copy(), move_identity, move_identity.copy())

    @classmethod
    def from_rotation(cls, puzzle: IHESPuzzle, rotation: Sequence[int] | np.ndarray) -> "SymmetryFrame":
        rotation_array = np.asarray(rotation, dtype=np.uint8)
        if rotation_array.shape != (puzzle.state_size,):
            raise ValueError("rotation has the wrong shape")
        if not np.array_equal(np.sort(rotation_array), np.arange(puzzle.state_size)):
            raise ValueError("rotation is not a permutation")
        rotation_inverse = invert_permutation(rotation_array).astype(np.uint8)
        generator_lookup = {tuple(map(int, generator)): index for index, generator in enumerate(puzzle.moves)}
        frame_to_original = np.empty(puzzle.generator_count, dtype=np.int16)
        for index, generator in enumerate(puzzle.moves):
            conjugated = tuple(
                int(rotation_inverse[generator[rotation_array[position]]])
                for position in range(puzzle.state_size)
            )
            if conjugated not in generator_lookup:
                raise ValueError(f"rotation does not preserve generator {puzzle.move_names[index]}")
            frame_to_original[index] = generator_lookup[conjugated]
        original_to_frame = invert_permutation(frame_to_original).astype(np.int16)
        return cls(rotation_array, rotation_inverse, frame_to_original, original_to_frame)

    def rotate_state(self, state: Sequence[int] | np.ndarray) -> np.ndarray:
        state_array = np.asarray(state)
        return self.rotation[state_array[self.rotation_inverse]]

    def to_original_path(self, frame_path: Sequence[int]) -> list[int]:
        return [int(self.frame_to_original_move[index]) for index in frame_path]

    def to_frame_path(self, original_path: Sequence[int]) -> list[int]:
        return [int(self.original_to_frame_move[index]) for index in original_path]

    def reverse_start(self, original_start: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return the inverse of the start state in this symmetry frame."""

        return invert_permutation(self.rotate_state(original_start)).astype(np.uint8)

    def reverse_path_to_original(self, reverse_frame_path: Sequence[int], puzzle: IHESPuzzle) -> list[int]:
        direct_frame_path = invert_path(reverse_frame_path, puzzle.inverse_move)
        return self.to_original_path(direct_frame_path)


def load_symmetry_frames(path: str, puzzle: IHESPuzzle) -> list[SymmetryFrame]:
    rotations = np.load(path)
    if rotations.ndim != 2 or rotations.shape[1] != puzzle.state_size:
        raise ValueError(f"unexpected symmetry array shape: {rotations.shape}")
    return [SymmetryFrame.from_rotation(puzzle, rotation) for rotation in rotations]


def map_reverse_frontier_to_direct(
    reverse_states: np.ndarray, direct_start: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Map every reverse projection state into direct coordinates.

    The dual-model invariant is ``M_s(r)[q] = s[r[q]]``. The mapping is
    deliberately applied to the full reverse frontier before hashing.
    """

    reverse_array = np.asarray(reverse_states, dtype=np.int64)
    start_array = np.asarray(direct_start)
    if reverse_array.ndim == 1:
        return start_array[reverse_array]
    if reverse_array.ndim != 2:
        raise ValueError("reverse frontier must be one- or two-dimensional")
    return start_array[reverse_array]
