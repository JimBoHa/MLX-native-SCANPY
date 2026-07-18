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

    def test_normalize_total_none_uses_median_of_counts(self):
        # Row totals are 2, 4, 10 -> median 4. Each row is rescaled to 4.
        counts = [[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]]
        normalized = as_numpy(normalize_total(counts, target_sum=None))
        np.testing.assert_allclose(normalized.sum(axis=1), np.array([4.0, 4.0, 4.0]), rtol=1e-5)

    def test_normalize_total_none_matches_scanpy(self):
        counts = np.array([[3.0, 1.0, 0.0], [4.0, 0.0, 1.0], [0.0, 6.0, 2.0]], dtype=np.float32)
        ours = as_numpy(normalize_total(counts, target_sum=None))
        adata = AnnData(counts.copy())
        sc.pp.normalize_total(adata, target_sum=None)
        np.testing.assert_allclose(ours, np.asarray(adata.X), rtol=1e-5)


class Log1pTests(unittest.TestCase):
    def test_log1p_default_is_natural_log(self):
        from mlx_native_scanpy import log1p as log1p_fn

        matrix = [[0.0, 1.0], [3.0, 7.0]]
        out = as_numpy(log1p_fn(matrix))
        np.testing.assert_allclose(out, np.log1p(np.array(matrix)), rtol=1e-5)

    def test_log1p_base_rescales(self):
        from mlx_native_scanpy import log1p as log1p_fn

        matrix = [[0.0, 1.0], [3.0, 7.0]]
        out = as_numpy(log1p_fn(matrix, base=2.0))
        np.testing.assert_allclose(out, np.log2(1.0 + np.array(matrix)), rtol=1e-5)

    def test_pp_log1p_base_matches_scanpy(self):
        counts = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        result = pp.log1p(AnnData(counts.copy()), base=10.0, inplace=False)
        adata = AnnData(counts.copy())
        sc.pp.log1p(adata, base=10.0)
        np.testing.assert_allclose(np.asarray(result.X), np.asarray(adata.X), rtol=1e-5)


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

    def test_filter_cells_subsets_obsp_and_layers(self):
        adata = AnnDataLite(
            X=[
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 2.0, 2.0],
                [9.0, 0.0, 0.0, 0.0],
            ]
        )
        adata.layers["counts"] = np.asarray(adata.X).copy()
        adata = pp.neighbors(adata, n_neighbors=2)
        self.assertEqual(np.asarray(adata.obsp["connectivities"]).shape, (4, 4))

        filtered, mask = pp.filter_cells(adata, min_counts=1)
        n = filtered.n_obs
        self.assertEqual(n, int(mask.sum()))
        # obsp is a square cell-by-cell matrix and must track the new cell count.
        self.assertEqual(np.asarray(filtered.obsp["connectivities"]).shape, (n, n))
        # layers are cell-by-gene and must track the kept rows.
        self.assertEqual(np.asarray(filtered.layers["counts"]).shape, (n, 4))

    def test_filter_genes_subsets_varp_and_layers(self):
        adata = AnnDataLite(
            X=[
                [5.0, 0.0, 1.0, 2.0],
                [4.0, 0.0, 1.0, 3.0],
                [6.0, 0.0, 2.0, 1.0],
            ]
        )
        adata.layers["counts"] = np.asarray(adata.X).copy()
        adata.varp["corr"] = np.eye(4, dtype=np.float32)

        filtered, mask = pp.filter_genes(adata, min_cells=1)
        n_vars = filtered.n_vars
        self.assertEqual(n_vars, int(mask.sum()))
        self.assertEqual(np.asarray(filtered.varp["corr"]).shape, (n_vars, n_vars))
        self.assertEqual(np.asarray(filtered.layers["counts"]).shape, (3, n_vars))


class RankGenesGroupsPValueTests(unittest.TestCase):
    def _make_adata(self):
        return AnnDataLite(
            X=[
                [10.0, 0.0, 1.0, 0.0],
                [11.0, 0.0, 2.0, 0.0],
                [9.0, 1.0, 0.0, 1.0],
                [0.0, 7.0, 0.0, 1.0],
                [0.0, 8.0, 0.0, 2.0],
                [1.0, 9.0, 1.0, 0.0],
            ],
            obs_names=[f"c{i}" for i in range(6)],
            var_names=["gene_a", "gene_b", "gene_c", "gene_d"],
            obs={"cluster": np.array(["A", "A", "A", "B", "B", "B"])},
        )

    def test_rank_genes_groups_reports_pvalues(self):
        adata = self._make_adata()
        ranks = tl.rank_genes_groups(adata, groupby="cluster")

        for group in ("A", "B"):
            self.assertIn(group, ranks["pvals"])
            self.assertIn(group, ranks["pvals_adj"])
            p = ranks["pvals"][group]
            padj = ranks["pvals_adj"][group]
            self.assertEqual(p.shape, ranks["names"][group].shape)
            self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
            self.assertTrue(np.all((padj >= 0.0) & (padj <= 1.0)))
            # Adjusted p-values are never smaller than the raw ones.
            self.assertTrue(np.all(padj + 1e-9 >= p))

    def test_rank_genes_groups_pvalues_stored_in_uns(self):
        adata = self._make_adata()
        tl.rank_genes_groups(adata, groupby="cluster")
        self.assertIn("pvals", adata.uns["rank_genes_groups"])
        self.assertIn("pvals_adj", adata.uns["rank_genes_groups"])


class QCMetricsTests(unittest.TestCase):
    def test_qc_vars_and_percent_top(self):
        adata = AnnDataLite(
            X=[
                [8.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 5.0],
            ],
            var_names=["mt-1", "mt-2", "gene_a", "gene_b"],
            var={"mt": np.array([True, True, False, False])},
        )
        metrics = pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=[2])

        # Row 0 is entirely mitochondrial, row 1 has none.
        np.testing.assert_allclose(metrics["pct_counts_mt"], np.array([100.0, 0.0]), rtol=1e-5)
        np.testing.assert_allclose(metrics["total_counts_mt"], np.array([10.0, 0.0]), rtol=1e-5)
        self.assertIn("pct_counts_in_top_2_genes", metrics)
        # obs is populated by default (inplace=True).
        self.assertIn("pct_counts_mt", adata.obs)

    def test_inplace_false_leaves_obs_untouched(self):
        adata = AnnDataLite(X=[[1.0, 2.0], [3.0, 4.0]])
        pp.calculate_qc_metrics(adata, inplace=False)
        self.assertNotIn("total_counts", adata.obs)

    def test_qc_vars_matches_scanpy(self):
        counts = np.array(
            [[8.0, 2.0, 1.0, 0.0], [0.0, 1.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]],
            dtype=np.float32,
        )
        var_names = ["mt-1", "mt-2", "gene_a", "gene_b"]
        adata = AnnData(counts.copy())
        adata.var_names = var_names
        adata.var["mt"] = np.array([True, True, False, False])
        obs_df, _ = sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=False)

        lite = AnnDataLite(
            X=counts.copy(),
            var_names=var_names,
            var={"mt": np.array([True, True, False, False])},
        )
        ours = pp.calculate_qc_metrics(lite, qc_vars=["mt"], percent_top=None, inplace=False)
        np.testing.assert_allclose(
            ours["pct_counts_mt"], np.asarray(obs_df["pct_counts_mt"]), rtol=1e-4
        )


class AnnDataLiteErgonomicsTests(unittest.TestCase):
    def _adata(self):
        return AnnDataLite(
            X=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            obs_names=["c0", "c1", "c2"],
            var_names=["g0", "g1", "g2"],
            obs={"score": np.array([10.0, 20.0, 30.0])},
            var={"flag": np.array([True, False, True])},
        )

    def test_to_df_shape_and_labels(self):
        df = self._adata().to_df()
        self.assertEqual(list(df.index), ["c0", "c1", "c2"])
        self.assertEqual(list(df.columns), ["g0", "g1", "g2"])
        np.testing.assert_allclose(df.loc["c1"].to_numpy(), np.array([4.0, 5.0, 6.0]))

    def test_getitem_row_slice(self):
        sub = self._adata()[1:]
        self.assertEqual(sub.obs_names, ["c1", "c2"])
        self.assertEqual(sub.n_obs, 2)
        self.assertEqual(sub.obs["score"].tolist(), [20.0, 30.0])

    def test_getitem_boolean_and_names(self):
        adata = self._adata()
        sub = adata[np.array([True, False, True]), ["g0", "g2"]]
        self.assertEqual(sub.obs_names, ["c0", "c2"])
        self.assertEqual(sub.var_names, ["g0", "g2"])
        np.testing.assert_allclose(np.asarray(sub.X), np.array([[1.0, 3.0], [7.0, 9.0]]))
        self.assertEqual(sub.var["flag"].tolist(), [True, True])

    def test_getitem_subsets_square_obsp(self):
        adata = self._adata()
        adata.obsp["conn"] = np.arange(9, dtype=np.float32).reshape(3, 3)
        sub = adata[[0, 2]]
        np.testing.assert_allclose(
            np.asarray(sub.obsp["conn"]), np.array([[0.0, 2.0], [6.0, 8.0]])
        )


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


class PCAFallbackInplaceTests(unittest.TestCase):
    def test_sparse_pca_respects_inplace_false(self):
        from scipy import sparse

        X = sparse.csr_matrix(
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
        adata = AnnData(X)
        result = pp.pca(adata, n_comps=2, inplace=False)

        # inplace=False must return a new object and leave the original untouched.
        self.assertIsNotNone(result)
        self.assertNotIn("X_pca", adata.obsm)
        self.assertIn("X_pca", result.obsm)

    def test_sparse_pca_respects_inplace_true(self):
        from scipy import sparse

        X = sparse.csr_matrix(
            np.array([[3.0, 1.0, 0.0, 5.0], [4.0, 0.0, 1.0, 2.0], [0.0, 6.0, 2.0, 1.0]], dtype=np.float32)
        )
        adata = AnnData(X)
        returned = pp.pca(adata, n_comps=2, inplace=True)
        self.assertIsNone(returned)
        self.assertIn("X_pca", adata.obsm)


if __name__ == "__main__":
    unittest.main()
