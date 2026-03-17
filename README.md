# MLX-native-SCANPY

`MLX-native-SCANPY` is a small Python library that ports the core numerical pieces of a Scanpy-style preprocessing pipeline to Apple MLX so other macOS apps can call them directly.

This repository does not attempt full feature parity with upstream Scanpy. The first release focuses on MLX-backed kernels that are commonly used as the base of a single-cell workflow:

- total-count normalization
- `log1p` transform
- feature scaling
- highly variable gene selection
- PCA
- k-nearest-neighbor graph construction
- a single end-to-end analysis entry point

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
