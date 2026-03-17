from __future__ import annotations

from typing import Any

import numpy as np
import scanpy as sc
from anndata import AnnData as ScanpyAnnData

from .anndata import AnnDataLite


def _use_custom_path(data: Any) -> bool:
    return isinstance(data, AnnDataLite) or not isinstance(data, ScanpyAnnData)


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
    matrix = np.asarray(adata.X)
    unique_groups = [str(group) for group in np.unique(labels)] if groups is None else groups
    top_n = matrix.shape[1] if n_genes is None else max(1, min(int(n_genes), matrix.shape[1]))

    names: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    logfoldchanges: dict[str, np.ndarray] = {}

    gene_names = np.asarray(adata.var_names)

    for group in unique_groups:
        target_mask = labels == group
        reference_mask = ~target_mask
        if target_mask.sum() == 0 or reference_mask.sum() == 0:
            raise ValueError(f"Group {group!r} does not have enough observations")

        target = matrix[target_mask]
        background = matrix[reference_mask]
        target_mean = target.mean(axis=0)
        background_mean = background.mean(axis=0)
        target_var = target.var(axis=0, ddof=1) if target.shape[0] > 1 else np.zeros(matrix.shape[1])
        background_var = (
            background.var(axis=0, ddof=1) if background.shape[0] > 1 else np.zeros(matrix.shape[1])
        )
        denom = np.sqrt((target_var / max(target.shape[0], 1)) + (background_var / max(background.shape[0], 1)) + 1e-8)
        t_scores = (target_mean - background_mean) / denom
        target_mean_safe = np.maximum(target_mean, 0.0) + 1e-8
        background_mean_safe = np.maximum(background_mean, 0.0) + 1e-8
        lfc = np.log2(target_mean_safe / background_mean_safe)

        order = np.argsort(-t_scores)[:top_n]
        names[group] = gene_names[order]
        scores[group] = t_scores[order].astype(np.float32)
        logfoldchanges[group] = lfc[order].astype(np.float32)

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
