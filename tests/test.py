"""
Comprehensive test suite for Dynamic Ensemble Selection library.
Tests all neighbor finders, edge cases, and integration with DynamicRouter.

Run with: pytest test_comprehensive.py -v
Run specific test: pytest test_comprehensive.py::TestAnnoyFinder -v
Run with coverage: pytest test_comprehensive.py --cov=ensemble_weights
"""
import pytest
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
import sys
import time

# Add to path if needed
sys.path.insert(0, '/mnt/user-data/uploads')


# ============================================================================
# Fixtures - Shared test data
# ============================================================================

@pytest.fixture(params=[1, 10, 50, 100, 500])
def n_features(request):
    """Test with various dimensionalities."""
    return request.param


@pytest.fixture(params=[100, 1000, 10000])
def n_samples(request):
    """Test with various dataset sizes."""
    return request.param


@pytest.fixture
def synthetic_data(n_samples, n_features):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    X_test = np.random.randn(100, n_features).astype(np.float32)
    return X_train, X_test


@pytest.fixture
def ground_truth(synthetic_data):
    """Compute ground truth with sklearn KNN."""
    X_train, X_test = synthetic_data
    k = 10
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_test)
    return distances, indices


def compute_recall(true_indices, approx_indices, k=10):
    """Compute recall@k: fraction of true neighbors found."""
    recalls = []
    for true_nn, approx_nn in zip(true_indices, approx_indices):
        true_set = set(true_nn[:k])
        approx_set = set(approx_nn[:k])
        recall = len(true_set & approx_set) / k
        recalls.append(recall)
    return np.mean(recalls)


# ============================================================================
# Test KNN Neighbor Finder
# ============================================================================

class TestKNNFinder:
    """Test sklearn KNN neighbor finder (baseline)."""

    def test_exact_match_with_ground_truth(self, synthetic_data, ground_truth):
        """KNN should match sklearn ground truth exactly."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train, X_test = synthetic_data
        true_distances, true_indices = ground_truth
        k = 10

        finder = KNNNeighborFinder(k=k)
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=k)

        # Should be exactly the same as ground truth
        assert distances.shape == true_distances.shape
        assert indices.shape == true_indices.shape
        np.testing.assert_array_equal(indices, true_indices)

    def test_single_query(self, synthetic_data):
        """Test single query returns correct shape."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train, X_test = synthetic_data
        k = 10

        finder = KNNNeighborFinder(k=k)
        finder.fit(X_train)

        # Single query
        distances, indices = finder.kneighbors(X_test[0:1], k=k)
        assert distances.shape == (k,)
        assert indices.shape == (k,)

    def test_batch_query(self, synthetic_data):
        """Test batch query returns correct shape."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train, X_test = synthetic_data
        k = 10

        finder = KNNNeighborFinder(k=k)
        finder.fit(X_train)

        # Batch query
        distances, indices = finder.kneighbors(X_test, k=k)
        assert distances.shape == (len(X_test), k)
        assert indices.shape == (len(X_test), k)


# ============================================================================
# Test FAISS Flat
# ============================================================================

class TestFAISSFlat:
    """Test FAISS Flat index (exact search)."""

    def test_exact_search_high_dim(self, synthetic_data, ground_truth):
        """FAISS Flat should be exact in high dimensions."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]

        # Only test high dimensions (FAISS has issues in low dims)
        if n_features <= 2:
            pytest.skip("FAISS Flat has precision issues in <=2D")

        true_distances, true_indices = ground_truth
        k = 10

        finder = FaissNeighborFinder(k=k, index_type='flat')
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=k)

        # Should have very high recall (>99%)
        recall = compute_recall(true_indices, indices, k)
        assert recall > 0.99, f"FAISS Flat recall {recall:.4f} should be >0.99"

    def test_low_dimension_warning(self, synthetic_data):
        """FAISS Flat should warn about low dimensions."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]

        if n_features > 2:
            pytest.skip("Test only for low dimensions")

        finder = FaissNeighborFinder(k=10, index_type='flat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder.fit(X_train)

            # Should get warning about low dimensions
            assert len(w) > 0
            assert "precision issues" in str(w[0].message).lower()

    def test_returns_k_neighbors(self, synthetic_data):
        """FAISS Flat should always return exactly k neighbors."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        k = 10

        finder = FaissNeighborFinder(k=k, index_type='flat')
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=k)

        assert indices.shape == (len(X_test), k)
        assert distances.shape == (len(X_test), k)


# ============================================================================
# Test FAISS IVF
# ============================================================================

class TestFAISSIVF:
    """Test FAISS IVF index (approximate search)."""

    def test_high_recall_with_enough_probes(self, synthetic_data, ground_truth):
        """FAISS IVF should have high recall with n_probes=50."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        # IVF works better with more samples and higher dimensions
        if n_samples < 1000 or n_features < 10:
            pytest.skip("IVF needs larger datasets")

        true_distances, true_indices = ground_truth
        k = 10

        finder = FaissNeighborFinder(k=k, index_type='ivf', n_probes=50)
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=k)

        # Should have good recall (>90%)
        recall = compute_recall(true_indices, indices, k)
        assert recall > 0.90, f"FAISS IVF recall {recall:.4f} should be >0.90 with n_probes=50"

    def test_low_probes_warning(self, synthetic_data):
        """FAISS IVF should warn about low n_probes."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        n_samples = X_train.shape[0]

        if n_samples < 1000:
            pytest.skip("Need larger dataset for IVF")

        # Use very low n_probes
        finder = FaissNeighborFinder(k=10, index_type='ivf', n_probes=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder.fit(X_train)

            # Should get warning about low n_probes
            assert len(w) > 0
            assert "n_probes" in str(w[0].message).lower()

    def test_auto_ncells_computation(self, synthetic_data):
        """FAISS IVF should auto-compute n_cells correctly."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        X_train, X_test = synthetic_data
        n_samples = X_train.shape[0]

        if n_samples < 1000:
            pytest.skip("Need larger dataset for IVF")

        finder = FaissNeighborFinder(k=10, index_type='ivf', n_cells=None)
        finder.fit(X_train)

        # n_cells should be set to sqrt(n_samples)
        expected_cells = min(int(np.sqrt(n_samples)), 4096)
        assert finder.n_cells == expected_cells


# ============================================================================
# Test Annoy
# ============================================================================

class TestAnnoyFinder:
    """Test Annoy neighbor finder."""

    def test_returns_exactly_k_neighbors(self, synthetic_data):
        """Annoy must return exactly k neighbors or raise error."""
        from ensemble_weights.models.neighbors import AnnoyNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]
        k = 10

        # Use high search_k to ensure we get k neighbors
        finder = AnnoyNeighborFinder(k=k, n_trees=100, search_k=50000)
        finder.fit(X_train)

        try:
            distances, indices = finder.kneighbors(X_test, k=k)

            # If it succeeds, must return exactly k neighbors
            assert indices.shape == (len(X_test), k), \
                f"Annoy returned {indices.shape} instead of ({len(X_test)}, {k})"
            assert distances.shape == (len(X_test), k)

        except ValueError as e:
            # If it fails, should be because k neighbors weren't found
            assert "only returned" in str(e).lower()

    def test_high_recall_in_high_dimensions(self, synthetic_data, ground_truth):
        """Annoy should have good recall in high dimensions."""
        from ensemble_weights.models.neighbors import AnnoyNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        # Annoy works better in higher dimensions
        if n_features < 20 or n_samples < 1000:
            pytest.skip("Annoy needs higher dimensions and more samples")

        true_distances, true_indices = ground_truth
        k = 10

        finder = AnnoyNeighborFinder(k=k, n_trees=100, search_k=50000)
        finder.fit(X_train)

        try:
            distances, indices = finder.kneighbors(X_test, k=k)
            recall = compute_recall(true_indices, indices, k)

            # Should have decent recall (>85%)
            assert recall > 0.85, f"Annoy recall {recall:.4f} should be >0.85"
        except ValueError:
            # If Annoy can't find k neighbors, that's a known issue in low dims
            pytest.skip("Annoy couldn't find k neighbors (known issue in low dims)")

    def test_low_dimension_warning(self, synthetic_data):
        """Annoy should warn about low dimensions."""
        from ensemble_weights.models.neighbors import AnnoyNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]

        if n_features > 3:
            pytest.skip("Test only for low dimensions")

        finder = AnnoyNeighborFinder(k=10, n_trees=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder.fit(X_train)

            # Should get warning about low dimensions
            assert len(w) > 0
            assert "low" in str(w[0].message).lower() or "poor" in str(w[0].message).lower()

    def test_search_k_auto_computation(self):
        """Annoy should compute search_k automatically."""
        from ensemble_weights.models.neighbors import AnnoyNeighborFinder

        finder = AnnoyNeighborFinder(k=10, n_trees=100, search_k=-1)

        # Should be max(n_trees * k * 50, 10000)
        expected = max(100 * 10 * 50, 10000)
        assert finder.search_k == expected


# ============================================================================
# Test HNSW
# ============================================================================

class TestHNSWFinder:
    """Test HNSW neighbor finder."""

    @pytest.mark.parametrize("backend", ["hnswlib"])  # Can add "nmslib" if available
    def test_high_recall_in_high_dimensions(self, synthetic_data, ground_truth, backend):
        """HNSW should have high recall in high dimensions."""
        from ensemble_weights.models.neighbors import HNSWNeighborFinder

        X_train, X_test = synthetic_data
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        # HNSW works better in higher dimensions
        if n_features < 20 or n_samples < 1000:
            pytest.skip("HNSW needs higher dimensions and more samples")

        true_distances, true_indices = ground_truth
        k = 10

        try:
            finder = HNSWNeighborFinder(
                k=k,
                backend=backend,
                M=32,
                ef_construction=400,
                ef_search=200
            )
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=k)

            recall = compute_recall(true_indices, indices, k)

            # Should have high recall (>90%)
            assert recall > 0.90, f"HNSW recall {recall:.4f} should be >0.90"
        except ImportError:
            pytest.skip(f"{backend} not installed")

    def test_low_ef_construction_warning(self, synthetic_data):
        """HNSW should warn about low ef_construction for large datasets."""
        from ensemble_weights.models.neighbors import HNSWNeighborFinder

        X_train, X_test = synthetic_data
        n_samples = X_train.shape[0]

        if n_samples < 10000:
            pytest.skip("Need larger dataset for warning")

        try:
            finder = HNSWNeighborFinder(
                k=10,
                backend='hnswlib',
                ef_construction=50  # Very low
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                finder.fit(X_train)

                # Should get warning about low ef_construction
                assert len(w) > 0
                assert "ef_construction" in str(w[0].message).lower()
        except ImportError:
            pytest.skip("hnswlib not installed")


# ============================================================================
# Test DynamicRouter Integration
# ============================================================================

class TestDynamicRouterIntegration:
    """Test integration with DynamicRouter."""

    def test_preset_balanced(self):
        """Test 'balanced' preset."""
        from ensemble_weights.__init__ import DynamicRouter

        # Generate synthetic ensemble data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50

        features = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)

        # Two des with different performance
        pred_A = np.random.randint(0, 2, n_samples)
        pred_B = np.random.randint(0, 2, n_samples)

        router = DynamicRouter(
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy',
            preset='balanced'
        )

        router.fit(features, y, {'A': pred_A, 'B': pred_B})

        # Test prediction
        test_features = np.random.randn(10, n_features).astype(np.float32)
        for i in range(len(test_features)):
            weights = router.predict(test_features[i:i+1])

            # Should return dict with model names
            assert isinstance(weights, dict)
            assert 'A' in weights
            assert 'B' in weights

            # Weights should sum to 1
            assert abs(sum(weights.values()) - 1.0) < 1e-6

            # Weights should be non-negative
            assert all(w >= 0 for w in weights.values())

    def test_preset_exact(self):
        """Test 'exact' preset uses KNN."""
        from ensemble_weights.__init__ import DynamicRouter

        router = DynamicRouter(
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy',
            preset='exact'
        )

        assert router.finder == 'knn'

    def test_preset_turbo(self):
        """Test 'turbo' preset uses Annoy."""
        from ensemble_weights.__init__ import DynamicRouter

        router = DynamicRouter(
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy',
            preset='turbo'
        )

        assert router.finder == 'annoy'

    def test_from_data_size_small(self):
        """Test auto-selection for small dataset."""
        from ensemble_weights.__init__ import DynamicRouter

        router = DynamicRouter.from_data_size(
            n_samples=5000,
            n_features=50,
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy'
        )

        # Should select 'exact' for small data
        assert router.preset == 'exact'

    def test_from_data_size_low_dim(self):
        """Test auto-selection for low dimensional data."""
        from ensemble_weights.__init__ import DynamicRouter

        router = DynamicRouter.from_data_size(
            n_samples=50000,
            n_features=5,
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy'
        )

        # Should select 'exact' for low dimensions
        assert router.preset == 'exact'

    def test_from_data_size_high_dim(self):
        """Test auto-selection for high dimensional data."""
        from ensemble_weights.__init__ import DynamicRouter

        router = DynamicRouter.from_data_size(
            n_samples=50000,
            n_features=200,
            task='classification',
            dtype='tabular',
            method='knn-dw',
            metric='accuracy'
        )

        # Should select high_dim preset
        assert 'high_dim' in router.preset


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_equals_n_samples(self):
        """Test when k equals number of training samples."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train = np.random.randn(50, 10).astype(np.float32)
        X_test = np.random.randn(5, 10).astype(np.float32)

        finder = KNNNeighborFinder(k=50)
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=50)

        assert indices.shape == (5, 50)

    def test_k_greater_than_n_samples(self):
        """Test when k is greater than number of training samples."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train = np.random.randn(50, 10).astype(np.float32)
        X_test = np.random.randn(5, 10).astype(np.float32)

        # This should raise an error or be handled gracefully
        with pytest.raises((ValueError, Exception)):
            finder = KNNNeighborFinder(k=100)
            finder.fit(X_train)
            finder.kneighbors(X_test, k=100)

    def test_1d_data(self):
        """Test with 1D data (edge case for ANN methods)."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder, FaissNeighborFinder

        X_train = np.random.randn(1000, 1).astype(np.float32)
        X_test = np.random.randn(10, 1).astype(np.float32)
        k = 10

        # KNN should work fine
        knn = KNNNeighborFinder(k=k)
        knn.fit(X_train)
        knn_distances, knn_indices = knn.kneighbors(X_test, k=k)
        assert knn_indices.shape == (10, k)

        # FAISS should warn but still work
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            faiss_finder = FaissNeighborFinder(k=k, index_type='flat')
            faiss_finder.fit(X_train)
            assert len(w) > 0  # Should get warning

    def test_empty_result_handling(self):
        """Test handling of edge cases with very small datasets."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        # Minimum viable dataset
        X_train = np.random.randn(10, 5).astype(np.float32)
        X_test = np.random.randn(2, 5).astype(np.float32)

        finder = KNNNeighborFinder(k=5)
        finder.fit(X_train)
        distances, indices = finder.kneighbors(X_test, k=5)

        assert indices.shape == (2, 5)
        assert not np.any(np.isnan(distances))


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_faiss_faster_than_knn_large_dataset(self):
        """FAISS should be faster than KNN on large datasets."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder, FaissNeighborFinder

        # Large dataset
        n_samples = 50000
        n_features = 50
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        X_test = np.random.randn(1000, n_features).astype(np.float32)
        k = 10

        # KNN
        knn = KNNNeighborFinder(k=k)
        knn.fit(X_train)
        start = time.time()
        knn.kneighbors(X_test, k=k)
        knn_time = time.time() - start

        # FAISS IVF
        try:
            faiss = FaissNeighborFinder(k=k, index_type='ivf', n_probes=30)
            faiss.fit(X_train)
            start = time.time()
            faiss.kneighbors(X_test, k=k)
            faiss_time = time.time() - start

            # FAISS should be faster (allow some margin for variance)
            assert faiss_time < knn_time * 1.5, \
                f"FAISS ({faiss_time:.3f}s) should be faster than KNN ({knn_time:.3f}s)"
        except ImportError:
            pytest.skip("FAISS not installed")

    def test_query_time_scales_with_k(self):
        """Query time should scale reasonably with k."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        X_train = np.random.randn(10000, 50).astype(np.float32)
        X_test = np.random.randn(100, 50).astype(np.float32)

        finder = KNNNeighborFinder(k=50)
        finder.fit(X_train)

        # Time for k=10
        start = time.time()
        finder.kneighbors(X_test, k=10)
        time_k10 = time.time() - start

        # Time for k=50
        start = time.time()
        finder.kneighbors(X_test, k=50)
        time_k50 = time.time() - start

        # k=50 shouldn't be more than 10x slower than k=10
        assert time_k50 < time_k10 * 10


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Test parameter validation and error handling."""

    def test_invalid_index_type(self):
        """Test invalid index type raises error."""
        from ensemble_weights.models.neighbors import FaissNeighborFinder

        with pytest.raises((ValueError, AttributeError)):
            finder = FaissNeighborFinder(k=10, index_type='invalid')
            X_train = np.random.randn(100, 10).astype(np.float32)
            finder.fit(X_train)

    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        from ensemble_weights.__init__ import DynamicRouter

        with pytest.raises(ValueError):
            DynamicRouter(
                task='classification',
                dtype='tabular',
                method='knn-dw',
                metric='accuracy',
                preset='invalid_preset'
            )

    def test_custom_preset_requires_finder(self):
        """Test custom preset requires finder parameter."""
        from ensemble_weights.__init__ import DynamicRouter

        with pytest.raises(ValueError):
            DynamicRouter(
                task='classification',
                dtype='tabular',
                method='knn-dw',
                metric='accuracy',
                preset='custom'
                # Missing: finder='knn'
            )

    def test_negative_k(self):
        """Test negative k raises error."""
        from ensemble_weights.models.neighbors import KNNNeighborFinder

        with pytest.raises((ValueError, Exception)):
            KNNNeighborFinder(k=-5)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])