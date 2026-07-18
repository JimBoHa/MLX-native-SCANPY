from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._mlx import get_mx


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


def _resolve_indices(selector: Any, names: list[str], length: int) -> list[int]:
    """Turn an AnnData-style selector into a list of positive integer positions."""
    if isinstance(selector, slice):
        return list(range(*selector.indices(length)))
    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        return [index + length if index < 0 else index]
    array = np.asarray(selector)
    if array.dtype == bool:
        if array.shape[0] != length:
            raise IndexError(f"Boolean index length {array.shape[0]} does not match axis length {length}")
        return np.flatnonzero(array).tolist()
    if array.dtype.kind in ("U", "S", "O"):
        lookup = {str(name): position for position, name in enumerate(names)}
        return [lookup[str(item)] for item in array.tolist()]
    return [int(item) + length if int(item) < 0 else int(item) for item in array.tolist()]


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
        mx = get_mx()
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

    def to_df(self, layer: str | None = None) -> Any:
        """Return X (or a named layer) as a pandas DataFrame indexed by obs/var names."""
        import pandas as pd

        matrix = self.layers[layer] if layer is not None else self.X
        return pd.DataFrame(
            np.asarray(matrix),
            index=list(self.obs_names),
            columns=list(self.var_names),
        )

    def __getitem__(self, key: Any) -> "AnnDataLite":
        if isinstance(key, tuple):
            obs_selector, var_selector = key
        else:
            obs_selector, var_selector = key, slice(None)

        row_idx = _resolve_indices(obs_selector, self.obs_names, self.n_obs)
        col_idx = _resolve_indices(var_selector, self.var_names, self.n_vars)

        matrix = np.asarray(self.X)[np.ix_(row_idx, col_idx)]
        return AnnDataLite(
            X=matrix,
            obs_names=[self.obs_names[i] for i in row_idx],
            var_names=[self.var_names[j] for j in col_idx],
            obs={key: np.asarray(value)[row_idx] for key, value in self.obs.items()},
            var={key: np.asarray(value)[col_idx] for key, value in self.var.items()},
            obsm={key: np.asarray(value)[row_idx] for key, value in self.obsm.items()},
            varm={key: np.asarray(value)[col_idx] for key, value in self.varm.items()},
            obsp={key: np.asarray(value)[np.ix_(row_idx, row_idx)] for key, value in self.obsp.items()},
            varp={key: np.asarray(value)[np.ix_(col_idx, col_idx)] for key, value in self.varp.items()},
            uns=_clone_mapping(self.uns),
            layers={key: np.asarray(value)[np.ix_(row_idx, col_idx)] for key, value in self.layers.items()},
        )
