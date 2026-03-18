from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._mlx import get_mx

EPSILON = 1e-8


def _to_mx_array(data: Any, dtype: Any = None) -> Any:
    mx = get_mx()
    if dtype is None:
        dtype = mx.float32
    return mx.array(data, dtype=dtype)


def _to_numpy(data: Any) -> np.ndarray:
    return np.asarray(data)


def _topk_descending(values: Any, k: int) -> Any:
    mx = get_mx()
    n_values = int(values.shape[0])
    k = max(1, min(int(k), n_values))
    if k == n_values:
        order = mx.argsort(values, axis=0)
        positions = mx.arange(n_values - 1, -1, -1)
        return mx.take(order, positions, axis=0)

    partitioned = mx.argpartition(-values, kth=k - 1, axis=0)
    candidate_positions = mx.arange(k)
    candidates = mx.take(partitioned, candidate_positions, axis=0)
    candidate_values = mx.take(values, candidates, axis=0)
    candidate_order = mx.argsort(candidate_values, axis=0)
    reverse_positions = mx.arange(k - 1, -1, -1)
    candidate_order = mx.take(candidate_order, reverse_positions, axis=0)
    return mx.take(candidates, candidate_order, axis=0)


def normalize_total(data: Any, target_sum: float = 1e4) -> Any:
    mx = get_mx()
    x = _to_mx_array(data)
    totals = mx.sum(x, axis=1, keepdims=True)
    safe_totals = mx.where(totals > 0, totals, mx.ones_like(totals))
    scale_factors = target_sum / safe_totals
    return x * scale_factors


def log1p(data: Any) -> Any:
    mx = get_mx()
    x = _to_mx_array(data)
    return mx.log1p(x)


def scale(
    data: Any,
    zero_center: bool = True,
    max_value: float | None = 10.0,
) -> Any:
    mx = get_mx()
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
    mx = get_mx()
    x = _to_mx_array(data)
    means = mx.mean(x, axis=0)
    centered = x - mx.mean(x, axis=0, keepdims=True)
    variances = mx.mean(centered * centered, axis=0)
    dispersion = variances / mx.maximum(means, EPSILON)

    top_n = max(1, min(int(n_top_genes), int(dispersion.shape[0])))
    order_mx = _topk_descending(dispersion, top_n)

    stats = {
        "means": _to_numpy(means),
        "variances": _to_numpy(variances),
        "dispersion": _to_numpy(dispersion),
    }
    return _to_numpy(order_mx).astype(np.int64), stats


def pca(data: Any, n_comps: int = 50) -> dict[str, Any]:
    mx = get_mx()
    x = _to_mx_array(data)
    n_obs = int(x.shape[0])
    n_vars = int(x.shape[1])
    n_comps = max(1, min(int(n_comps), n_obs, n_vars))

    centered = x - mx.mean(x, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(n_obs - 1, 1)
    eigenvalues, eigenvectors = mx.linalg.eigh(covariance, stream=mx.cpu)
    order = _topk_descending(eigenvalues, n_comps)
    components = mx.take(eigenvectors, order, axis=1)
    explained_variance_mx = mx.maximum(mx.take(eigenvalues, order, axis=0), 0.0)
    total_variance_mx = mx.sum(mx.maximum(eigenvalues, 0.0))
    explained_ratio_mx = explained_variance_mx / mx.maximum(total_variance_mx, EPSILON)
    scores = centered @ components
    explained_variance = _to_numpy(explained_variance_mx).astype(np.float32)
    explained_ratio = _to_numpy(explained_ratio_mx).astype(np.float32)

    return {
        "scores": scores,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
    }


def neighbors(data: Any, n_neighbors: int = 15) -> dict[str, Any]:
    mx = get_mx()
    x = _to_mx_array(data)
    n_obs = int(x.shape[0])
    if n_obs < 2:
        raise ValueError("neighbors requires at least two observations")

    k = max(1, min(int(n_neighbors), n_obs - 1))
    squared_norms = mx.sum(x * x, axis=1, keepdims=True)
    distances_sq = squared_norms + squared_norms.T - (2.0 * (x @ x.T))
    distances_sq = mx.maximum(distances_sq, 0.0)
    distances = mx.sqrt(distances_sq)
    diagonal_mask = mx.eye(n_obs, dtype=mx.bool_)
    distances = mx.where(diagonal_mask, mx.array(np.inf, dtype=distances.dtype), distances)
    sorted_indices = mx.argsort(distances, axis=1)
    neighbor_positions = mx.arange(k)
    indices_mx = mx.take(sorted_indices, neighbor_positions, axis=1)
    neighbor_distances_mx = mx.take_along_axis(distances, indices_mx, axis=1)
    connectivities = mx.put_along_axis(
        mx.zeros((n_obs, n_obs), dtype=mx.float32),
        indices_mx,
        mx.ones((n_obs, k), dtype=mx.float32),
        axis=1,
    )
    connectivities = mx.maximum(connectivities, connectivities.T)
    distance_matrix = mx.put_along_axis(
        mx.zeros((n_obs, n_obs), dtype=mx.float32),
        indices_mx,
        neighbor_distances_mx,
        axis=1,
    )
    distance_matrix = mx.maximum(distance_matrix, distance_matrix.T)

    return {
        "indices": _to_numpy(indices_mx).astype(np.int64),
        "distances": _to_numpy(neighbor_distances_mx).astype(np.float32),
        "distance_matrix": distance_matrix,
        "connectivities": connectivities,
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
        mx = get_mx()
        raw = _to_mx_array(counts)
        normalized = normalize_total(raw, target_sum=target_sum)
        logged = log1p(normalized)
        hvg_indices, hvg_stats = highly_variable_genes(logged, n_top_genes=n_top_genes)
        hvg_matrix = mx.take(logged, mx.array(hvg_indices.astype(np.int32)), axis=1)
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
