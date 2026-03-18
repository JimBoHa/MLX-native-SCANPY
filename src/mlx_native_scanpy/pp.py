from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData as ScanpyAnnData
import scanpy as sc
from scipy import sparse

from .anndata import AnnDataLite
from ._mlx import get_mx
from .analysis import (
    EPSILON,
    highly_variable_genes as _highly_variable_genes,
    log1p as _log1p,
    neighbors as _neighbors,
    normalize_total as _normalize_total,
    pca as _pca,
    scale as _scale,
)


def _as_mx(data: Any) -> Any:
    mx = get_mx()
    return mx.array(data, dtype=mx.float32)


def _as_numpy(data: Any) -> np.ndarray:
    return np.asarray(data)


def _matrix_from(data: Any) -> Any:
    if isinstance(data, AnnDataLite):
        return data.X
    if isinstance(data, ScanpyAnnData):
        return _as_mx(_as_numpy(data.X))
    return _as_mx(data)


def _maybe_copy_adata(data: Any, inplace: bool) -> Any:
    return data if inplace else data.copy()


def _use_custom_path(data: Any) -> bool:
    if isinstance(data, AnnDataLite):
        return True
    if isinstance(data, ScanpyAnnData):
        return not sparse.issparse(data.X)
    return True


def _is_scanpy_adata(data: Any) -> bool:
    return isinstance(data, ScanpyAnnData)


def _take_rows(data: Any, row_indices: np.ndarray, inplace: bool) -> tuple[Any, np.ndarray]:
    mx = get_mx()
    matrix = _matrix_from(data)
    mx_indices = mx.array(row_indices.astype(np.int32))
    subset = mx.take(matrix, mx_indices, axis=0)
    row_mask = np.zeros(int(matrix.shape[0]), dtype=bool)
    row_mask[row_indices] = True

    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = subset
        target.obs_names = [name for name, keep in zip(target.obs_names, row_mask) if keep]
        for key, value in list(target.obs.items()):
            target.obs[key] = np.asarray(value)[row_mask]
        for key, value in list(target.obsm.items()):
            target.obsm[key] = _as_numpy(value)[row_mask]
        return target, row_mask

    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target._inplace_subset_obs(row_mask)
        target.X = _as_numpy(subset)
        return target, row_mask

    return subset, row_mask


def _take_cols(data: Any, col_indices: np.ndarray, inplace: bool) -> tuple[Any, np.ndarray]:
    mx = get_mx()
    matrix = _matrix_from(data)
    mx_indices = mx.array(col_indices.astype(np.int32))
    subset = mx.take(matrix, mx_indices, axis=1)
    col_mask = np.zeros(int(matrix.shape[1]), dtype=bool)
    col_mask[col_indices] = True

    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = subset
        target.var_names = [name for name, keep in zip(target.var_names, col_mask) if keep]
        for key, value in list(target.var.items()):
            target.var[key] = np.asarray(value)[col_mask]
        for key, value in list(target.varm.items()):
            target.varm[key] = _as_numpy(value)[col_mask]
        return target, col_mask

    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target._inplace_subset_var(col_mask)
        target.X = _as_numpy(subset)
        return target, col_mask

    return subset, col_mask


def calculate_qc_metrics(data: Any) -> dict[str, np.ndarray]:
    if not _use_custom_path(data):
        return sc.pp.calculate_qc_metrics(data)
    mx = get_mx()
    matrix = _matrix_from(data)
    total_counts = _as_numpy(mx.sum(matrix, axis=1)).astype(np.float32)
    n_genes_by_counts = _as_numpy(mx.sum(matrix > 0, axis=1)).astype(np.int64)
    total_counts_per_gene = _as_numpy(mx.sum(matrix, axis=0)).astype(np.float32)
    n_cells_by_counts = _as_numpy(mx.sum(matrix > 0, axis=0)).astype(np.int64)

    sorted_counts = mx.sort(matrix, axis=1)
    top_k = min(50, int(matrix.shape[1]))
    top_positions = mx.arange(int(matrix.shape[1]) - 1, int(matrix.shape[1]) - top_k - 1, -1)
    top_counts = mx.take(sorted_counts, top_positions, axis=1)
    pct_counts_top_50 = (
        _as_numpy(mx.sum(top_counts, axis=1)).astype(np.float32)
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

    if isinstance(data, AnnDataLite) or _is_scanpy_adata(data):
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
    if not _use_custom_path(data):
        return sc.pp.filter_cells(
            data,
            min_counts=min_counts,
            min_genes=min_genes,
            max_counts=max_counts,
            max_genes=max_genes,
            inplace=inplace,
        )
    matrix = _matrix_from(data)
    total_counts = _as_numpy(get_mx().sum(matrix, axis=1))
    n_genes = _as_numpy(get_mx().sum(matrix > 0, axis=1))

    mask = np.ones(matrix.shape[0], dtype=bool)
    if min_counts is not None:
        mask &= total_counts >= min_counts
    if min_genes is not None:
        mask &= n_genes >= min_genes
    if max_counts is not None:
        mask &= total_counts <= max_counts
    if max_genes is not None:
        mask &= n_genes <= max_genes

    return _take_rows(data, np.flatnonzero(mask), inplace=inplace)


def filter_genes(
    data: Any,
    min_counts: float | None = None,
    min_cells: int | None = None,
    max_counts: float | None = None,
    max_cells: int | None = None,
    inplace: bool = False,
) -> tuple[Any, np.ndarray]:
    if not _use_custom_path(data):
        return sc.pp.filter_genes(
            data,
            min_counts=min_counts,
            min_cells=min_cells,
            max_counts=max_counts,
            max_cells=max_cells,
            inplace=inplace,
        )
    matrix = _matrix_from(data)
    total_counts = _as_numpy(get_mx().sum(matrix, axis=0))
    n_cells = _as_numpy(get_mx().sum(matrix > 0, axis=0))

    mask = np.ones(matrix.shape[1], dtype=bool)
    if min_counts is not None:
        mask &= total_counts >= min_counts
    if min_cells is not None:
        mask &= n_cells >= min_cells
    if max_counts is not None:
        mask &= total_counts <= max_counts
    if max_cells is not None:
        mask &= n_cells <= max_cells

    return _take_cols(data, np.flatnonzero(mask), inplace=inplace)


def subsample(
    data: Any,
    n_obs: int | None = None,
    fraction: float | None = None,
    random_state: int = 0,
    inplace: bool = False,
) -> tuple[Any, np.ndarray]:
    if not _use_custom_path(data):
        return sc.pp.subsample(
            data,
            n_obs=n_obs,
            fraction=fraction,
            random_state=random_state,
            copy=not inplace,
        )
    matrix = _matrix_from(data)
    if (n_obs is None) == (fraction is None):
        raise ValueError("Specify exactly one of n_obs or fraction")

    sample_size = n_obs if n_obs is not None else int(round(int(matrix.shape[0]) * fraction))
    sample_size = max(1, min(int(sample_size), int(matrix.shape[0])))
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(int(matrix.shape[0]), size=sample_size, replace=False))
    return _take_rows(data, selected, inplace=inplace)


def normalize_total(data: Any, target_sum: float = 1e4, inplace: bool = False) -> Any:
    if not _use_custom_path(data):
        return sc.pp.normalize_total(data, target_sum=target_sum, inplace=inplace)
    normalized = _normalize_total(_matrix_from(data), target_sum=target_sum)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.layers["counts"] = target.layers.get("counts", target.X)
        target.X = normalized
        return target
    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.layers["counts"] = target.layers.get("counts", _as_numpy(target.X))
        target.X = _as_numpy(normalized)
        return target
    return normalized


def log1p(data: Any, inplace: bool = False) -> Any:
    if not _use_custom_path(data):
        return sc.pp.log1p(data, copy=not inplace)
    logged = _log1p(_matrix_from(data))
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = logged
        return target
    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = _as_numpy(logged)
        return target
    return logged


def highly_variable_genes(
    data: Any,
    n_top_genes: int = 2000,
    inplace: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | AnnDataLite:
    if not _use_custom_path(data):
        return sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, inplace=inplace)
    indices, stats = _highly_variable_genes(_matrix_from(data), n_top_genes=n_top_genes)
    if isinstance(data, AnnDataLite) or _is_scanpy_adata(data):
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
    if not _use_custom_path(data):
        return sc.pp.scale(data, zero_center=zero_center, max_value=max_value, copy=not inplace)
    scaled = _scale(_matrix_from(data), zero_center=zero_center, max_value=max_value)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = scaled
        return target
    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.X = _as_numpy(scaled)
        return target
    return scaled


def neighbors(
    data: Any,
    n_neighbors: int = 15,
    use_rep: str | None = None,
    inplace: bool = False,
) -> Any:
    if not _use_custom_path(data):
        return sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep=use_rep, copy=not inplace)
    if use_rep and isinstance(data, (AnnDataLite, ScanpyAnnData)):
        matrix = data.obsm[use_rep]
    else:
        matrix = _matrix_from(data)
    result = _neighbors(matrix, n_neighbors=n_neighbors)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.obsp["connectivities"] = result["connectivities"]
        target.obsp["distances"] = result["distance_matrix"]
        target.uns["neighbors"] = {
            "indices": result["indices"],
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
        }
        return target
    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.obsp["connectivities"] = _as_numpy(result["connectivities"])
        target.obsp["distances"] = _as_numpy(result["distance_matrix"])
        target.uns["neighbors"] = {
            "indices": result["indices"],
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
        }
        return target
    return result


def pca(data: Any, n_comps: int = 50, inplace: bool = False, **kwargs: Any) -> Any:
    if not _use_custom_path(data):
        return sc.pp.pca(data, n_comps=n_comps, **kwargs)
    result = _pca(_matrix_from(data), n_comps=n_comps)
    if isinstance(data, AnnDataLite):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.obsm["X_pca"] = result["scores"]
        target.varm["PCs"] = result["components"]
        target.uns["pca"] = {
            "variance": result["explained_variance"],
            "variance_ratio": result["explained_variance_ratio"],
        }
        return target
    if _is_scanpy_adata(data):
        target = _maybe_copy_adata(data, inplace=inplace)
        target.obsm["X_pca"] = _as_numpy(result["scores"])
        target.varm["PCs"] = _as_numpy(result["components"])
        target.uns["pca"] = {
            "variance": result["explained_variance"],
            "variance_ratio": result["explained_variance_ratio"],
        }
        return target
    return result


def __getattr__(name: str) -> Any:
    return getattr(sc.pp, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(sc.pp)))
