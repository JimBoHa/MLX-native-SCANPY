import unittest

import numpy as np

from mlx_native_scanpy import (
    MLXScanpyAnalyzer,
    highly_variable_genes,
    neighbors,
    normalize_total,
    pca,
    scale,
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
        result = neighbors(embedding, n_neighbors=1)
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


if __name__ == "__main__":
    unittest.main()
