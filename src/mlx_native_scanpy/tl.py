from __future__ import annotations

from typing import Any

import numpy as np
import scanpy as sc
from anndata import AnnData as ScanpyAnnData
from scipy import sparse

from .anndata import AnnDataLite
from ._mlx import get_mx
from .analysis import EPSILON, _topk_descending


def _use_custom_path(data: Any) -> bool:
    if isinstance(data, AnnDataLite):
        return True
    if isinstance(data, ScanpyAnnData):
        return not sparse.issparse(data.X)
    return True


def rank_genes_groups(
    adata: Any,
    groupby: str,
    groups: list[str] | None = None,
    reference: str = "rest",
    n_genes: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    if not _use_custom_path(adata):
        return sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            groups=groups,
            reference=reference,
            n_genes=n_genes,
        )
    if groupby not in adata.obs:
        raise KeyError(f"{groupby} not found in adata.obs")
    if reference != "rest":
        raise ValueError("Only reference='rest' is currently supported")

    labels = np.asarray(adata.obs[groupby])
    mx = get_mx()
    matrix = adata.X if isinstance(adata, AnnDataLite) else mx.array(np.asarray(adata.X), dtype=mx.float32)
    unique_groups = [str(group) for group in np.unique(labels)] if groups is None else groups
    top_n = int(matrix.shape[1]) if n_genes is None else max(1, min(int(n_genes), int(matrix.shape[1])))

    names: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    logfoldchanges: dict[str, np.ndarray] = {}

    gene_names = np.asarray(adata.var_names)

    for group in unique_groups:
        target_mask = labels == group
        reference_mask = ~target_mask
        if target_mask.sum() == 0 or reference_mask.sum() == 0:
            raise ValueError(f"Group {group!r} does not have enough observations")

        target_indices = mx.array(np.flatnonzero(target_mask).astype(np.int32))
        reference_indices = mx.array(np.flatnonzero(reference_mask).astype(np.int32))
        target = mx.take(matrix, target_indices, axis=0)
        background = mx.take(matrix, reference_indices, axis=0)
        target_mean = mx.mean(target, axis=0)
        background_mean = mx.mean(background, axis=0)

        if int(target.shape[0]) > 1:
            target_centered = target - target_mean
            target_var = mx.sum(target_centered * target_centered, axis=0) / max(int(target.shape[0]) - 1, 1)
        else:
            target_var = mx.zeros((int(matrix.shape[1]),), dtype=mx.float32)

        if int(background.shape[0]) > 1:
            background_centered = background - background_mean
            background_var = mx.sum(background_centered * background_centered, axis=0) / max(int(background.shape[0]) - 1, 1)
        else:
            background_var = mx.zeros((int(matrix.shape[1]),), dtype=mx.float32)

        denom = mx.sqrt(
            (target_var / max(int(target.shape[0]), 1))
            + (background_var / max(int(background.shape[0]), 1))
            + EPSILON
        )
        t_scores = (target_mean - background_mean) / denom
        target_mean_safe = mx.maximum(target_mean, 0.0) + EPSILON
        background_mean_safe = mx.maximum(background_mean, 0.0) + EPSILON
        lfc = mx.log2(target_mean_safe / background_mean_safe)

        order = _topk_descending(t_scores, top_n)
        order_np = np.asarray(order).astype(np.int64)
        names[group] = gene_names[order_np]
        scores[group] = np.asarray(mx.take(t_scores, order, axis=0)).astype(np.float32)
        logfoldchanges[group] = np.asarray(mx.take(lfc, order, axis=0)).astype(np.float32)

    result = {
        "names": names,
        "scores": scores,
        "logfoldchanges": logfoldchanges,
    }
    adata.uns["rank_genes_groups"] = result
    return result


def __getattr__(name: str) -> Any:
    return getattr(sc.tl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(sc.tl)))
