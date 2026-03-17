from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


def _clone_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            cloned[key] = _clone_mapping(value)
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


@dataclass(slots=True)
class AnnDataLite:
    X: Any
    obs_names: list[str] = field(default_factory=list)
    var_names: list[str] = field(default_factory=list)
    obs: dict[str, np.ndarray] = field(default_factory=dict)
    var: dict[str, np.ndarray] = field(default_factory=dict)
    obsm: dict[str, Any] = field(default_factory=dict)
    varm: dict[str, Any] = field(default_factory=dict)
    obsp: dict[str, Any] = field(default_factory=dict)
    varp: dict[str, Any] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = mx.array(self.X, dtype=mx.float32)
        n_obs = int(self.X.shape[0])
        n_vars = int(self.X.shape[1])

        if not self.obs_names:
            self.obs_names = [f"cell_{index}" for index in range(n_obs)]
        if not self.var_names:
            self.var_names = [f"gene_{index}" for index in range(n_vars)]

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.X.shape[1])

    def copy(self) -> "AnnDataLite":
        return AnnDataLite(
            X=np.asarray(self.X).copy(),
            obs_names=list(self.obs_names),
            var_names=list(self.var_names),
            obs={key: np.asarray(value).copy() for key, value in self.obs.items()},
            var={key: np.asarray(value).copy() for key, value in self.var.items()},
            obsm=_clone_mapping(self.obsm),
            varm=_clone_mapping(self.varm),
            obsp=_clone_mapping(self.obsp),
            varp=_clone_mapping(self.varp),
            uns=_clone_mapping(self.uns),
            layers=_clone_mapping(self.layers),
        )
