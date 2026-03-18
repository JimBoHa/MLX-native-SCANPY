import unittest
from unittest import mock

import numpy as np
import scanpy as sc
import mlx_native_scanpy

from mlx_native_scanpy.analysis import neighbors as analysis_neighbors
from mlx_native_scanpy import (
    AnnDataLite,
    AnnData,
    MLXScanpyAnalyzer,
    highly_variable_genes,
    normalize_total,
    pca,
    pp,
    scale,
    tl,
)


def as_numpy(value):
    return np.asarray(value)


class NormalizeTotalTests(unittest.TestCase):
    def test_normalize_total_matches_target_sum(self):
        counts = [[1.0, 1.0, 2.0], [0.0, 0.0, 5.0]]
        normalized = as_numpy(normalize_total(counts, target_sum=10.0))
        row_sums = normalized.sum(axis=1)

        np.testing.assert_allclose(row_sums, np.array([10.0, 10.0]), rtol=1e-5)
        np.testing.assert_allclose(normalized[1], np.array([0.0, 0.0, 10.0]), rtol=1e-5)


class HighlyVariableGenesTests(unittest.TestCase):
    def test_hvg_prefers_most_variable_feature(self):
        matrix = [
            [1.0, 2.0, 3.0],
            [1.0, 9.0, 3.0],
            [1.0, 0.0, 3.0],
            [1.0, 12.0, 3.0],
        ]

        indices, stats = highly_variable_genes(matrix, n_top_genes=1)

        self.assertEqual(indices.tolist(), [1])
        self.assertEqual(stats["dispersion"].shape[0], 3)


class PCATests(unittest.TestCase):
    def test_pca_finds_single_dominant_axis(self):
        matrix = [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]

        result = pca(matrix, n_comps=1)
        components = as_numpy(result["components"])

        self.assertEqual(components.shape, (2, 1))
        self.assertGreater(float(result["explained_variance_ratio"][0]), 0.999)
        self.assertLess(abs(float(components[1, 0])), 1e-5)


class NeighborTests(unittest.TestCase):
    def test_neighbors_returns_expected_pairs(self):
        embedding = [[0.0], [1.0], [5.0], [6.0]]
        result = analysis_neighbors(embedding, n_neighbors=1)
        connectivities = as_numpy(result["connectivities"])

        self.assertEqual(result["indices"][0].tolist(), [1])
        self.assertEqual(result["indices"][3].tolist(), [2])
        self.assertEqual(connectivities[0, 1], 1.0)
        self.assertEqual(connectivities[2, 3], 1.0)
        self.assertEqual(connectivities[0, 2], 0.0)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline_shapes_and_symmetry(self):
        counts = [
            [3.0, 1.0, 0.0, 5.0],
            [4.0, 0.0, 1.0, 2.0],
            [0.0, 6.0, 2.0, 1.0],
            [2.0, 2.0, 4.0, 3.0],
            [1.0, 3.0, 5.0, 2.0],
        ]

        analyzer = MLXScanpyAnalyzer()
        result = analyzer.analyze(counts, n_top_genes=3, n_pcs=2, n_neighbors=2)
        connectivities = as_numpy(result.connectivities)
        scaled = as_numpy(result.scaled)

        self.assertEqual(result.hvg_indices.shape, (3,))
        self.assertEqual(as_numpy(result.pca_scores).shape, (5, 2))
        self.assertEqual(result.neighbor_indices.shape, (5, 2))
        self.assertEqual(connectivities.shape, (5, 5))
        np.testing.assert_allclose(connectivities, connectivities.T, rtol=1e-5)
        self.assertTrue(np.isfinite(scaled).all())

    def test_scale_produces_zero_centered_features(self):
        matrix = [[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]]
        scaled = as_numpy(scale(matrix, max_value=None))
        means = scaled.mean(axis=0)

        np.testing.assert_allclose(means, np.zeros(2), atol=1e-6)


class AnnDataLiteTests(unittest.TestCase):
    def test_qc_and_filter_functions_update_anndata(self):
        adata = AnnDataLite(
            X=[
                [0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 1.0],
            ],
            obs_names=["a", "b", "c"],
            var_names=["g1", "g2", "g3"],
        )

        metrics = pp.calculate_qc_metrics(adata)
        self.assertEqual(metrics["total_counts"].tolist(), [4.0, 0.0, 6.0])
        filtered, mask = pp.filter_cells(adata, min_counts=1)
        self.assertEqual(mask.tolist(), [True, False, True])
        self.assertEqual(filtered.obs_names, ["a", "c"])

        gene_filtered, gene_mask = pp.filter_genes(adata, min_cells=1)
        self.assertEqual(gene_mask.tolist(), [True, True, True])
        self.assertEqual(gene_filtered.var_names, ["g1", "g2", "g3"])

    def test_pp_and_tl_modules_store_scanpy_like_annotations(self):
        adata = AnnDataLite(
            X=[
                [10.0, 0.0, 1.0, 0.0],
                [11.0, 0.0, 2.0, 0.0],
                [0.0, 7.0, 0.0, 1.0],
                [0.0, 8.0, 0.0, 2.0],
            ],
            obs_names=["c1", "c2", "c3", "c4"],
            var_names=["gene_a", "gene_b", "gene_c", "gene_d"],
            obs={"cluster": np.array(["A", "A", "B", "B"])},
        )

        adata = pp.normalize_total(adata)
        adata = pp.log1p(adata)
        adata = pp.highly_variable_genes(adata, n_top_genes=2)
        adata = pp.scale(adata)
        adata = pp.pca(adata, n_comps=2)
        adata = pp.neighbors(adata, n_neighbors=1, use_rep="X_pca")
        ranks = tl.rank_genes_groups(adata, groupby="cluster", n_genes=2)

        self.assertIn("highly_variable", adata.var)
        self.assertEqual(np.asarray(adata.obsm["X_pca"]).shape, (4, 2))
        self.assertEqual(np.asarray(adata.varm["PCs"]).shape, (4, 2))
        self.assertIn("connectivities", adata.obsp)
        self.assertEqual(ranks["names"]["A"].shape[0], 2)
        self.assertEqual(ranks["names"]["B"].shape[0], 2)


class ScanpyParityTests(unittest.TestCase):
    def test_top_level_api_covers_scanpy(self):
        expected = {name for name in dir(sc) if not name.startswith("_")}
        actual = {name for name in dir(mlx_native_scanpy) if not name.startswith("_")}
        missing = expected - actual
        self.assertEqual(missing, set(), f"Missing top-level names: {sorted(missing)}")

    def test_namespace_api_covers_scanpy(self):
        for module_name in ["pp", "tl", "pl", "get", "datasets", "queries", "metrics"]:
            expected = {name for name in dir(getattr(sc, module_name)) if not name.startswith("_")}
            actual = {name for name in dir(getattr(mlx_native_scanpy, module_name)) if not name.startswith("_")}
            missing = expected - actual
            self.assertEqual(missing, set(), f"{module_name} missing names: {sorted(missing)}")

    def test_real_anndata_uses_scanpy_fallback(self):
        adata = AnnData(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        pp.normalize_total(adata, target_sum=10.0, inplace=True)
        np.testing.assert_allclose(np.asarray(adata.X).sum(axis=1), np.array([10.0, 10.0]), rtol=1e-5)

    def test_dense_anndata_normalize_total_uses_mlx_path(self):
        adata = AnnData(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        with mock.patch("mlx_native_scanpy.pp.sc.pp.normalize_total", side_effect=AssertionError("fallback used")):
            result = pp.normalize_total(adata, target_sum=10.0, inplace=False)
        np.testing.assert_allclose(np.asarray(result.X).sum(axis=1), np.array([10.0, 10.0]), rtol=1e-5)

    def test_dense_anndata_pca_and_neighbors_use_mlx_path(self):
        adata = AnnData(
            np.array(
                [
                    [3.0, 1.0, 0.0, 5.0],
                    [4.0, 0.0, 1.0, 2.0],
                    [0.0, 6.0, 2.0, 1.0],
                    [2.0, 2.0, 4.0, 3.0],
                ],
                dtype=np.float32,
            )
        )
        with mock.patch("mlx_native_scanpy.pp.sc.pp.pca", side_effect=AssertionError("fallback used")):
            adata = pp.pca(adata, n_comps=2, inplace=False)
        with mock.patch("mlx_native_scanpy.pp.sc.pp.neighbors", side_effect=AssertionError("fallback used")):
            adata = pp.neighbors(adata, n_neighbors=1, use_rep="X_pca", inplace=False)
        self.assertEqual(np.asarray(adata.obsm["X_pca"]).shape, (4, 2))
        self.assertIn("connectivities", adata.obsp)

    def test_dense_anndata_rank_genes_groups_uses_mlx_path(self):
        adata = AnnData(
            np.array(
                [
                    [10.0, 0.0, 1.0, 0.0],
                    [11.0, 0.0, 2.0, 0.0],
                    [0.0, 7.0, 0.0, 1.0],
                    [0.0, 8.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            )
        )
        adata.obs["cluster"] = np.array(["A", "A", "B", "B"])
        with mock.patch("mlx_native_scanpy.tl.sc.tl.rank_genes_groups", side_effect=AssertionError("fallback used")):
            result = tl.rank_genes_groups(adata, groupby="cluster", n_genes=2)
        self.assertEqual(result["names"]["A"].shape[0], 2)
        self.assertEqual(result["names"]["B"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
