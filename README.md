# MLX-native-SCANPY

`MLX-native-SCANPY` is a Scanpy-compatible package surface with MLX-backed fast paths for selected numerical kernels on Apple Silicon.

The package mirrors the upstream Scanpy API surface by delegating to `scanpy` for broad compatibility and using MLX-native implementations for selected array and `AnnDataLite` workflows.

Current custom MLX-backed kernels include:

- `AnnDataLite`
- `pp.calculate_qc_metrics`
- `pp.filter_cells`
- `pp.filter_genes`
- `pp.subsample`
- total-count normalization
- `log1p` transform
- feature scaling
- highly variable gene selection
- PCA
- k-nearest-neighbor graph construction
- `tl.rank_genes_groups`
- a single end-to-end analysis entry point

Everything else in the public API resolves to the corresponding upstream Scanpy function or module, so other apps can import against a Scanpy-compatible surface from this package.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Usage

```python
from mlx_native_scanpy import MLXScanpyAnalyzer

counts = [
    [3.0, 1.0, 0.0, 5.0],
    [4.0, 0.0, 1.0, 2.0],
    [0.0, 6.0, 2.0, 1.0],
    [2.0, 2.0, 4.0, 3.0],
]

analyzer = MLXScanpyAnalyzer()
result = analyzer.analyze(counts, n_top_genes=3, n_pcs=2, n_neighbors=2)

print(result.hvg_indices)
print(result.pca_scores.shape)
print(result.connectivities.shape)
```

## Development

Run tests with:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs on `macos-14` because MLX requires Apple Silicon.
