from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .anndata import AnnDataLite
from .analysis import (
    EPSILON,
    highly_variable_genes as _highly_variable_genes,
    log1p as _log1p,
    neighbors as _neighbors,
    normalize_total as _normalize_total,
    scale as _scale,
)


def _as_mx(data: Any) -> Any:
    return mx.array(data, dtype=mx.float32)


def _matrix_from(data: Any) -> Any:
    return data.X if isinstance(data, AnnDataLite) else _as_mx(data)


def _maybe_copy_adata(data: AnnDataLite, inplace: bool) -> AnnDataLite:
    return data if inplace else data.copy()


def calculate_qc_metrics(data: Any) -> dict[str, np.ndarray]:
    matrix = np.asarray(_matrix_from(data))
    total_counts = matrix.sum(axis=1).astype(np.float32)
    n_genes_by_counts = (matrix > 0).sum(axis=1).astype(np.int64)
    total_counts_per_gene = matrix.sum(axis=0).astype(np.float32)
    n_cells_by_counts = (matrix > 0).sum(axis=0).astype(np.int64)
    pct_counts_top_50 = (
        np.sort(matrix, axis=1)[:, ::-1][:, : min(50, matrix.shape[1])].sum(axis=1)
        / np.maximum(total_counts, EPSILON)
        * 100.0
    ).astype(np.float32)

    metrics = {
        "total_counts": total_counts,
        "n_genes_by_counts": n_genes_by_counts,
        "total_counts_per_gene": total_counts_per_gene,
        "n_cells_by_counts": n_cells_by_counts,
        "pct_counts_top_50_genes": pct_counts_top_50,
    }

    if isinstance(data, AnnDataLite):
        data.obs["total_counts"] = total_counts
        data.obs["n_genes_by_counts"] = n_genes_by_counts
        data.obs["pct_counts_top_50_genes"] = pct_counts_top_50
        data.var["total_counts"] = total_counts_per_gene
        data.var["n_cells_by_counts"] = n_cells_by_counts

    return metrics


def filter_cells(
    data: Any,
    min_counts: float | None = None,
    min_genes: int | None = None,
    max_counts: float | None = None,
    max_genes: int | None = None,
    inplace: bool = False,
) -> tuple[Any, np.ndarray]:
    matrix = np.asarray(_matrix_from(data))
    total_counts = matrix.sum(axis=1)
    n_genes = (matrix > 0).sum(axis=1)

    mask = np.ones(matrix.shape[0], dtype=bool)
    if min_counts is not None:
        mask &= total_counts >= min_counts
    if min_genes is not None:
        mask &= n_genes >= min_genes
    if max_counts is not None:
        mask &= total_counts <= max_counts
    if max_genes is not None:
        mask &= n_genes <= max_genes

    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = _as_mx(matrix[mask])
        target.obs_names = [name for name, keep in zip(target.obs_names, mask) if keep]
        for key, value in list(target.obs.items()):
            target.obs[key] = np.asarray(value)[mask]
        for key, value in list(target.obsm.items()):
            target.obsm[key] = np.asarray(value)[mask]
        return target, mask

    return _as_mx(matrix[mask]), mask


def filter_genes(
    data: Any,
    min_counts: float | None = None,
    min_cells: int | None = None,
    max_counts: float | None = None,
    max_cells: int | None = None,
    inplace: bool = False,
) -> tuple[Any, np.ndarray]:
    matrix = np.asarray(_matrix_from(data))
    total_counts = matrix.sum(axis=0)
    n_cells = (matrix > 0).sum(axis=0)

    mask = np.ones(matrix.shape[1], dtype=bool)
    if min_counts is not None:
        mask &= total_counts >= min_counts
    if min_cells is not None:
        mask &= n_cells >= min_cells
    if max_counts is not None:
        mask &= total_counts <= max_counts
    if max_cells is not None:
        mask &= n_cells <= max_cells

    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = _as_mx(matrix[:, mask])
        target.var_names = [name for name, keep in zip(target.var_names, mask) if keep]
        for key, value in list(target.var.items()):
            target.var[key] = np.asarray(value)[mask]
        for key, value in list(target.varm.items()):
            target.varm[key] = np.asarray(value)[mask]
        return target, mask

    return _as_mx(matrix[:, mask]), mask


def subsample(
    data: Any,
    n_obs: int | None = None,
    fraction: float | None = None,
    random_state: int = 0,
    inplace: bool = False,
) -> tuple[Any, np.ndarray]:
    matrix = np.asarray(_matrix_from(data))
    if (n_obs is None) == (fraction is None):
        raise ValueError("Specify exactly one of n_obs or fraction")

    sample_size = n_obs if n_obs is not None else int(round(matrix.shape[0] * fraction))
    sample_size = max(1, min(int(sample_size), matrix.shape[0]))
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(matrix.shape[0], size=sample_size, replace=False))
    mask = np.zeros(matrix.shape[0], dtype=bool)
    mask[selected] = True

    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = _as_mx(matrix[mask])
        target.obs_names = [name for name, keep in zip(target.obs_names, mask) if keep]
        for key, value in list(target.obs.items()):
            target.obs[key] = np.asarray(value)[mask]
        for key, value in list(target.obsm.items()):
            target.obsm[key] = np.asarray(value)[mask]
        return target, mask

    return _as_mx(matrix[mask]), mask


def normalize_total(data: Any, target_sum: float = 1e4, inplace: bool = False) -> Any:
    normalized = _normalize_total(_matrix_from(data), target_sum=target_sum)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.layers["counts"] = target.layers.get("counts", target.X)
        target.X = normalized
        return target
    return normalized


def log1p(data: Any, inplace: bool = False) -> Any:
    logged = _log1p(_matrix_from(data))
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = logged
        return target
    return logged


def highly_variable_genes(
    data: Any,
    n_top_genes: int = 2000,
    inplace: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | AnnDataLite:
    indices, stats = _highly_variable_genes(_matrix_from(data), n_top_genes=n_top_genes)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        mask = np.zeros(target.n_vars, dtype=bool)
        mask[indices] = True
        target.var["highly_variable"] = mask
        target.var["means"] = stats["means"]
        target.var["variances"] = stats["variances"]
        target.var["dispersions"] = stats["dispersion"]
        target.uns["hvg"] = {"indices": indices}
        return target
    return indices, stats


def scale(
    data: Any,
    zero_center: bool = True,
    max_value: float | None = 10.0,
    inplace: bool = False,
) -> Any:
    scaled = _scale(_matrix_from(data), zero_center=zero_center, max_value=max_value)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = scaled
        return target
    return scaled


def neighbors(
    data: Any,
    n_neighbors: int = 15,
    use_rep: str | None = None,
    inplace: bool = False,
) -> Any:
    matrix = data.obsm[use_rep] if isinstance(data, AnnDataLite) and use_rep else _matrix_from(data)
    result = _neighbors(matrix, n_neighbors=n_neighbors)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.obsp["connectivities"] = result["connectivities"]
        target.obsp["distances"] = result["distances"]
        target.uns["neighbors"] = {
            "indices": result["indices"],
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
        }
        return target
    return result
