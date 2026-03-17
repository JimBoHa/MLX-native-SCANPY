from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

EPSILON = 1e-8


def _to_mx_array(data: Any, dtype: Any = mx.float32) -> Any:
    return mx.array(data, dtype=dtype)


def _to_numpy(data: Any) -> np.ndarray:
    return np.asarray(data)


def normalize_total(data: Any, target_sum: float = 1e4) -> Any:
    x = _to_mx_array(data)
    totals = mx.sum(x, axis=1, keepdims=True)
    safe_totals = mx.where(totals > 0, totals, mx.ones_like(totals))
    scale_factors = target_sum / safe_totals
    return x * scale_factors


def log1p(data: Any) -> Any:
    x = _to_mx_array(data)
    return mx.log1p(x)


def scale(
    data: Any,
    zero_center: bool = True,
    max_value: float | None = 10.0,
) -> Any:
    x = _to_mx_array(data)
    centered = x - mx.mean(x, axis=0, keepdims=True) if zero_center else x
    var = mx.mean(centered * centered, axis=0, keepdims=True)
    std = mx.sqrt(mx.maximum(var, EPSILON))
    scaled = centered / std
    if max_value is not None:
        scaled = mx.clip(scaled, -max_value, max_value)
    return scaled


def highly_variable_genes(
    data: Any,
    n_top_genes: int = 2000,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = _to_mx_array(data)
    means = mx.mean(x, axis=0)
    centered = x - mx.mean(x, axis=0, keepdims=True)
    variances = mx.mean(centered * centered, axis=0)
    dispersion = variances / mx.maximum(means, EPSILON)

    dispersion_np = _to_numpy(dispersion)
    top_n = max(1, min(int(n_top_genes), dispersion_np.shape[0]))
    order = np.argsort(-dispersion_np)[:top_n]

    stats = {
        "means": _to_numpy(means),
        "variances": _to_numpy(variances),
        "dispersion": dispersion_np,
    }
    return order.astype(np.int64), stats


def pca(data: Any, n_comps: int = 50) -> dict[str, Any]:
    x = _to_mx_array(data)
    n_obs = int(x.shape[0])
    n_vars = int(x.shape[1])
    n_comps = max(1, min(int(n_comps), n_obs, n_vars))

    centered = x - mx.mean(x, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(n_obs - 1, 1)
    eigenvalues, eigenvectors = mx.linalg.eigh(covariance, stream=mx.cpu)

    eigenvalues_np = _to_numpy(eigenvalues)
    order = np.argsort(eigenvalues_np)[::-1][:n_comps]

    components = _to_mx_array(_to_numpy(eigenvectors)[:, order])
    explained_variance = np.maximum(eigenvalues_np[order], 0.0)
    total_variance = float(np.maximum(eigenvalues_np, 0.0).sum())
    explained_ratio = (
        explained_variance / total_variance
        if total_variance > 0
        else np.zeros_like(explained_variance)
    )
    scores = centered @ components

    return {
        "scores": scores,
        "components": components,
        "explained_variance": explained_variance.astype(np.float32),
        "explained_variance_ratio": explained_ratio.astype(np.float32),
    }


def neighbors(data: Any, n_neighbors: int = 15) -> dict[str, Any]:
    x = _to_mx_array(data)
    n_obs = int(x.shape[0])
    if n_obs < 2:
        raise ValueError("neighbors requires at least two observations")

    k = max(1, min(int(n_neighbors), n_obs - 1))
    squared_norms = mx.sum(x * x, axis=1, keepdims=True)
    distances_sq = squared_norms + squared_norms.T - (2.0 * (x @ x.T))
    distances_sq = mx.maximum(distances_sq, 0.0)

    distances_np = np.sqrt(np.maximum(_to_numpy(distances_sq), 0.0)).astype(np.float32)
    np.fill_diagonal(distances_np, np.inf)

    indices = np.argsort(distances_np, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(distances_np, indices, axis=1)

    connectivities = np.zeros((n_obs, n_obs), dtype=np.float32)
    for row_index, row_neighbors in enumerate(indices):
        connectivities[row_index, row_neighbors] = 1.0
    connectivities = np.maximum(connectivities, connectivities.T)

    return {
        "indices": indices.astype(np.int64),
        "distances": neighbor_distances,
        "connectivities": mx.array(connectivities),
    }


@dataclass(slots=True)
class AnalysisResult:
    raw_counts: Any
    normalized: Any
    logged: Any
    hvg_matrix: Any
    scaled: Any
    hvg_indices: np.ndarray
    hvg_stats: dict[str, np.ndarray]
    pca_scores: Any
    pca_components: Any
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    connectivities: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "hvg_indices": self.hvg_indices.tolist(),
            "explained_variance": self.explained_variance.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "neighbor_indices": self.neighbor_indices.tolist(),
            "neighbor_distances": self.neighbor_distances.tolist(),
            "pca_scores": _to_numpy(self.pca_scores).tolist(),
            "connectivities": _to_numpy(self.connectivities).tolist(),
        }


class MLXScanpyAnalyzer:
    def analyze(
        self,
        counts: Any,
        target_sum: float = 1e4,
        n_top_genes: int = 2000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        scale_max_value: float | None = 10.0,
    ) -> AnalysisResult:
        raw = _to_mx_array(counts)
        normalized = normalize_total(raw, target_sum=target_sum)
        logged = log1p(normalized)
        hvg_indices, hvg_stats = highly_variable_genes(logged, n_top_genes=n_top_genes)
        hvg_matrix = _to_mx_array(_to_numpy(logged)[:, hvg_indices])
        scaled_matrix = scale(hvg_matrix, max_value=scale_max_value)
        pca_result = pca(scaled_matrix, n_comps=n_pcs)
        neighbor_result = neighbors(pca_result["scores"], n_neighbors=n_neighbors)

        return AnalysisResult(
            raw_counts=raw,
            normalized=normalized,
            logged=logged,
            hvg_matrix=hvg_matrix,
            scaled=scaled_matrix,
            hvg_indices=hvg_indices,
            hvg_stats=hvg_stats,
            pca_scores=pca_result["scores"],
            pca_components=pca_result["components"],
            explained_variance=pca_result["explained_variance"],
            explained_variance_ratio=pca_result["explained_variance_ratio"],
            neighbor_indices=neighbor_result["indices"],
            neighbor_distances=neighbor_result["distances"],
            connectivities=neighbor_result["connectivities"],
        )


def run_analysis(
    counts: Any,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    scale_max_value: float | None = 10.0,
) -> AnalysisResult:
    analyzer = MLXScanpyAnalyzer()
    return analyzer.analyze(
        counts=counts,
        target_sum=target_sum,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        scale_max_value=scale_max_value,
    )
