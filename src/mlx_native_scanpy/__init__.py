from __future__ import annotations

from importlib.metadata import version as package_version
from typing import Any

import scanpy as _sc

from . import pp, tl
from .anndata import AnnDataLite
from .analysis import (
    AnalysisResult,
    MLXScanpyAnalyzer,
    highly_variable_genes,
    log1p,
    neighbors,
    normalize_total,
    pca,
    run_analysis,
    scale,
)

AnnData = _sc.AnnData
Neighbors = _sc.Neighbors
Verbosity = _sc.Verbosity
Version = getattr(_sc, "Version", None)
datasets = _sc.datasets
experimental = _sc.experimental
external = _sc.external
get = _sc.get
logging = _sc.logging
metrics = _sc.metrics
neighbors = _sc.neighbors
pl = _sc.pl
plotting = _sc.plotting
preprocessing = pp
queries = _sc.queries
read = _sc.read
read_10x_h5 = _sc.read_10x_h5
read_10x_mtx = _sc.read_10x_mtx
read_csv = _sc.read_csv
read_excel = _sc.read_excel
read_h5ad = _sc.read_h5ad
read_hdf = _sc.read_hdf
read_loom = _sc.read_loom
read_mtx = _sc.read_mtx
read_text = _sc.read_text
read_umi_tools = _sc.read_umi_tools
read_visium = _sc.read_visium
readwrite = _sc.readwrite
settings = _sc.settings
set_figure_params = _sc.set_figure_params
sys = _sc.sys
tools = tl
version = package_version("scanpy")
write = _sc.write


def __getattr__(name: str) -> Any:
    return getattr(_sc, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_sc)))


__all__ = sorted(set(dir(_sc)) | {
    "AnnDataLite",
    "AnalysisResult",
    "MLXScanpyAnalyzer",
    "highly_variable_genes",
    "log1p",
    "normalize_total",
    "pca",
    "run_analysis",
    "scale",
})
