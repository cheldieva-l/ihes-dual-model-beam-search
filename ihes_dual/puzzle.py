from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def invert_permutation(state: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return the inverse of a permutation encoded as ``output[i] = input[p[i]]``."""

    state_array = np.asarray(state, dtype=np.int64)
    if state_array.ndim != 1:
        raise ValueError("a permutation must be one-dimensional")
    inverse = np.empty_like(state_array)
    inverse[state_array] = np.arange(state_array.size, dtype=np.int64)
    return inverse


def inverse_move_name(name: str) -> str:
    return name[1:] if name.startswith("-") else f"-{name}"


def invert_path(move_indices: Sequence[int], inverse_move: np.ndarray) -> list[int]:
    """Reverse a path and replace every generator with its inverse."""

    return [int(inverse_move[index]) for index in reversed(move_indices)]


@dataclass(frozen=True)
class IHESPuzzle:
    solved: np.ndarray
    moves: np.ndarray
    move_names: tuple[str, ...]
    inverse_move: np.ndarray

    @classmethod
    def from_puzzle_info(cls, path: str | Path) -> "IHESPuzzle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not {"central_state", "generators"}.issubset(payload):
            raise ValueError("expected IHES puzzle_info.json with central_state and generators")
        names = tuple(payload["generators"].keys())
        moves = np.asarray([payload["generators"][name] for name in names], dtype=np.uint8)
        solved = np.asarray(payload["central_state"], dtype=np.uint8)
        name_to_index = {name: index for index, name in enumerate(names)}
        missing = [inverse_move_name(name) for name in names if inverse_move_name(name) not in name_to_index]
        if missing:
            raise ValueError(f"missing inverse generators: {missing}")
        inverse_move = np.asarray(
            [name_to_index[inverse_move_name(name)] for name in names], dtype=np.int16
        )
        puzzle = cls(solved=solved, moves=moves, move_names=names, inverse_move=inverse_move)
        puzzle.validate()
        return puzzle

    @property
    def state_size(self) -> int:
        return int(self.solved.size)

    @property
    def generator_count(self) -> int:
        return int(self.moves.shape[0])

    def validate(self) -> None:
        expected = np.arange(self.state_size)
        if not np.array_equal(np.sort(self.solved), expected):
            raise ValueError("central_state is not a permutation")
        if self.moves.ndim != 2 or self.moves.shape[1] != self.state_size:
            raise ValueError("generator matrix has the wrong shape")
        for index, move in enumerate(self.moves):
            if not np.array_equal(np.sort(move), expected):
                raise ValueError(f"generator {self.move_names[index]} is not a permutation")
            inverse_index = int(self.inverse_move[index])
            if not np.array_equal(move[self.moves[inverse_index]], expected):
                raise ValueError(f"generator {self.move_names[index]} has a bad inverse")

    def apply(self, state: Sequence[int] | np.ndarray, move_index: int) -> np.ndarray:
        return np.asarray(state)[self.moves[int(move_index)]]

    def apply_many(self, states: np.ndarray, move_indices: np.ndarray) -> np.ndarray:
        states_array = np.asarray(states)
        indices = np.asarray(move_indices, dtype=np.int64)
        if states_array.ndim != 2 or indices.ndim != 1 or len(states_array) != len(indices):
            raise ValueError("states and move_indices have incompatible shapes")
        return np.take_along_axis(states_array, self.moves[indices], axis=1)

    def replay(self, start: Sequence[int] | np.ndarray, path: Iterable[int]) -> np.ndarray:
        state = np.asarray(start, dtype=np.uint8).copy()
        for move_index in path:
            state = state[self.moves[int(move_index)]]
        return state

    def verify_solution(
        self,
        start: Sequence[int] | np.ndarray,
        path: Sequence[int],
        target: Sequence[int] | np.ndarray | None = None,
    ) -> bool:
        expected = self.solved if target is None else np.asarray(target)
        return bool(np.array_equal(self.replay(start, path), expected))

    def encode_path(self, path: Sequence[int]) -> str:
        return ".".join(self.move_names[int(index)] for index in path)

    def decode_path(self, text: str) -> list[int]:
        if not text or text.strip() == "":
            return []
        lookup = {name: index for index, name in enumerate(self.move_names)}
        tokens = text.strip().split(".")
        unknown = [token for token in tokens if token not in lookup]
        if unknown:
            raise ValueError(f"unknown move names: {unknown}")
        return [lookup[token] for token in tokens]


def load_test_state(test_csv: str | Path, puzzle_id: int) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(test_csv)
    row = frame.loc[frame["initial_state_id"] == puzzle_id]
    if len(row) != 1:
        raise ValueError(f"puzzle id {puzzle_id} occurs {len(row)} times")
    return np.fromstring(row.iloc[0]["initial_state"], sep=",", dtype=np.uint8)
