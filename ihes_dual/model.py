from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.fc1(inputs)))
        return F.relu(inputs + self.bn2(self.fc2(hidden)))


class MLP2RB(nn.Module):
    """Checkpoint-compatible CayleyPy/Pilgrim residual MLP."""

    def __init__(
        self,
        state_size: int = 72,
        class_count: int = 72,
        hidden_1: int = 2556,
        hidden_2: int = 218,
        residual_blocks: int = 16,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.class_count = class_count
        self.input_layer = nn.Linear(state_size * class_count, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.hidden_layer = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_2) for _ in range(residual_blocks)])
        self.output_layer = nn.Linear(hidden_2, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = F.one_hot(states.long(), num_classes=self.class_count).float()
        hidden = F.relu(self.bn1(self.input_layer(encoded.flatten(1))))
        hidden = F.relu(self.bn2(self.hidden_layer(hidden)))
        for block in self.residual_blocks:
            hidden = block(hidden)
        return self.output_layer(hidden).squeeze(-1)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint: Path
    config: Path
    epoch: int | None = None


def _normalise_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint does not contain a state dict")
    result: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        result[key.removeprefix("module.")] = value
    if not result:
        raise ValueError("checkpoint state dict is empty")
    return result


def load_mlp2rb(spec: ModelSpec, device: str | torch.device = "cuda") -> MLP2RB:
    config = json.loads(spec.config.read_text(encoding="utf-8"))
    model = MLP2RB(
        state_size=int(config.get("state_size", 72)),
        class_count=int(config.get("num_classes", config.get("classes", 72))),
        hidden_1=int(config.get("hd1", config.get("hidden_1", 2556))),
        hidden_2=int(config.get("hd2", config.get("hidden_2", 218))),
        residual_blocks=int(config.get("nrd", config.get("residual_blocks", 16))),
    )
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(_normalise_state_dict(payload), strict=True)
    model.eval().to(device)
    return model


class EnsembleScorer:
    """Score states with one or more compatible MLPs and a weighted mean."""

    def __init__(self, models: Iterable[nn.Module], weights: Iterable[float] | None = None) -> None:
        self.models = tuple(models)
        if not self.models:
            raise ValueError("at least one model is required")
        raw_weights = tuple(weights) if weights is not None else tuple(1.0 for _ in self.models)
        if len(raw_weights) != len(self.models) or sum(raw_weights) <= 0:
            raise ValueError("model weights are invalid")
        total = float(sum(raw_weights))
        self.weights = tuple(float(weight) / total for weight in raw_weights)

    @torch.inference_mode()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        score = None
        for model, weight in zip(self.models, self.weights):
            contribution = model(states) * weight
            score = contribution if score is None else score + contribution
        assert score is not None
        return score


class PairedContrastScorer:
    """Rank a state by its primary score minus its paired-projection score.

    The value map is a permutation of state labels.  It therefore maps a full
    tensor batch without leaving the GPU and without changing the checkpoint's
    feature ordering.  The same underlying MLP is evaluated in both exact
    projections; this is the IHES adaptation of the dual-model contrast
    objective used by the source method.
    """

    def __init__(
        self,
        scorer: Callable[[torch.Tensor], torch.Tensor],
        paired_value_map: Sequence[int] | torch.Tensor,
    ) -> None:
        self.scorer = scorer
        value_map = torch.as_tensor(paired_value_map, dtype=torch.long)
        if value_map.ndim != 1 or not torch.equal(
            torch.sort(value_map).values, torch.arange(len(value_map), dtype=torch.long)
        ):
            raise ValueError("paired value map must be a one-dimensional permutation")
        self._value_map = value_map
        self._device_maps: dict[torch.device, torch.Tensor] = {}

    def _map_for(self, device: torch.device) -> torch.Tensor:
        mapped = self._device_maps.get(device)
        if mapped is None:
            mapped = self._value_map.to(device)
            self._device_maps[device] = mapped
        return mapped

    @torch.inference_mode()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        value_map = self._map_for(states.device)
        paired_states = value_map[states.long()]
        return self.scorer(states) - self.scorer(paired_states)
