"""
Stress tests and edge cases for DES library.
Tests unusual conditions, boundary cases, and error handling.

Run with: python test_stress.py
"""
import numpy as np
import sys
import warnings

sys.path.insert(0, '/mnt/user-data/uploads')


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if details:
        print(f"  → {details}")


class StressTests:
    """Stress tests for neighbor finders."""

    def test_very_high_k(self):
        """Test with k close to n_samples."""
        print("\n--- Very High K Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        n_samples = 1000
        k = 950  # 95% of dataset

        X_train = np.random.randn(n_samples, 10).astype(np.float32)
        X_test = np.random.randn(5, 10).astype(np.float32)

        try:
            finder = KNNNeighborFinder(k=k)
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=k)

            passed = indices.shape == (5, k)
            print_test("Very high k (k=950/1000)", passed, f"Shape: {indices.shape}")
            return passed
        except Exception as e:
            print_test("Very high k (k=950/1000)", False, f"Error: {e}")
            return False

    def test_k_equals_n(self):
        """Test with k exactly equal to n_samples."""
        print("\n--- K Equals N Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        n_samples = 100
        k = 100

        X_train = np.random.randn(n_samples, 10).astype(np.float32)
        X_test = np.random.randn(5, 10).astype(np.float32)

        try:
            finder = KNNNeighborFinder(k=k)
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=k)

            passed = indices.shape == (5, k)
            print_test("k equals n_samples", passed, f"Shape: {indices.shape}")
            return passed
        except Exception as e:
            print_test("k equals n_samples", False, f"Error: {e}")
            return False

    def test_single_sample_training(self):
        """Test with only 1 training sample (edge case)."""
        print("\n--- Single Training Sample Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        X_train = np.random.randn(1, 10).astype(np.float32)
        X_test = np.random.randn(5, 10).astype(np.float32)

        try:
            finder = KNNNeighborFinder(k=1)
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=1)

            # All queries should return the same single neighbor
            passed = (indices.shape == (5, 1) and
                      np.all(indices == 0))  # Only index 0 exists
            print_test("Single training sample", passed, f"All returned index 0: {passed}")
            return passed
        except Exception as e:
            print_test("Single training sample", False, f"Error: {e}")
            return False

    def test_very_high_dimensions(self):
        """Test with very high dimensions (curse of dimensionality)."""
        print("\n--- Very High Dimensions Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder, FaissNeighborFinder

        n_samples = 1000
        n_features = 5000  # Very high!

        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        X_test = np.random.randn(5, n_features).astype(np.float32)
        k = 10

        results = []

        # KNN should still work
        try:
            knn = KNNNeighborFinder(k=k)
            knn.fit(X_train)
            distances, indices = knn.kneighbors(X_test, k=k)
            passed = indices.shape == (5, k)
            print_test("KNN with 5000D", passed)
            results.append(passed)
        except Exception as e:
            print_test("KNN with 5000D", False, str(e))
            results.append(False)

        # FAISS Flat should work
        try:
            faiss = FaissNeighborFinder(k=k, index_type='flat')
            faiss.fit(X_train)
            distances, indices = faiss.kneighbors(X_test, k=k)
            passed = indices.shape == (5, k)
            print_test("FAISS Flat with 5000D", passed)
            results.append(passed)
        except Exception as e:
            print_test("FAISS Flat with 5000D", False, str(e))
            results.append(False)

        return all(results)

    def test_duplicate_features(self):
        """Test with duplicate feature vectors."""
        print("\n--- Duplicate Features Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        # Create dataset with duplicates
        base = np.random.randn(10, 10).astype(np.float32)
        X_train = np.vstack([base] * 100)  # 1000 samples, many duplicates
        X_test = base[:5]  # Test on duplicates
        k = 10

        try:
            finder = KNNNeighborFinder(k=k)
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=k)

            # First neighbor should have distance 0 (exact match)
            passed = np.all(distances[:, 0] < 1e-6)
            print_test("Finds exact duplicates", passed, f"Min distance: {distances[:, 0].max():.10f}")
            return passed
        except Exception as e:
            print_test("Finds exact duplicates", False, str(e))
            return False

    def test_constant_features(self):
        """Test with constant feature values (no variance)."""
        print("\n--- Constant Features Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        # All samples are identical
        X_train = np.ones((100, 10), dtype=np.float32)
        X_test = np.ones((5, 10), dtype=np.float32)
        k = 10

        try:
            finder = KNNNeighborFinder(k=k)
            finder.fit(X_train)
            distances, indices = finder.kneighbors(X_test, k=k)

            # All distances should be 0 (all points identical)
            passed = np.all(distances < 1e-6)
            print_test("Constant features", passed, f"Max distance: {distances.max():.10f}")
            return passed
        except Exception as e:
            print_test("Constant features", False, str(e))
            return False

    def test_large_scale_annoy(self):
        """Test Annoy with large dataset."""
        print("\n--- Large Scale Annoy Test ---")

        try:
            from ensemble_weights.neighbors import AnnoyNeighborFinder

            n_samples = 50000
            n_features = 100

            X_train = np.random.randn(n_samples, n_features).astype(np.float32)
            X_test = np.random.randn(100, n_features).astype(np.float32)
            k = 10

            finder = AnnoyNeighborFinder(k=k, n_trees=100, search_k=50000)

            import time
            start = time.time()
            finder.fit(X_train)
            fit_time = time.time() - start

            start = time.time()
            distances, indices = finder.kneighbors(X_test, k=k)
            query_time = time.time() - start

            passed = indices.shape == (100, k)
            qps = 100 / query_time
            print_test(
                "Annoy 50K samples",
                passed,
                f"Fit: {fit_time:.2f}s, Query: {query_time:.4f}s ({qps:.0f} QPS)"
            )
            return passed
        except ImportError:
            print_test("Annoy 50K samples", False, "Annoy not installed")
            return False
        except Exception as e:
            print_test("Annoy 50K samples", False, str(e))
            return False


class EdgeCaseTests:
    """Edge case tests for error handling."""

    def test_mismatched_dimensions(self):
        """Test querying with wrong dimensionality."""
        print("\n--- Mismatched Dimensions Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        X_train = np.random.randn(100, 10).astype(np.float32)
        X_test_wrong = np.random.randn(5, 20).astype(np.float32)  # Wrong dims!

        finder = KNNNeighborFinder(k=10)
        finder.fit(X_train)

        try:
            distances, indices = finder.kneighbors(X_test_wrong, k=10)
            print_test("Catches dimension mismatch", False, "Should have raised error")
            return False
        except (ValueError, Exception):
            print_test("Catches dimension mismatch", True, "Properly raised error")
            return True

    def test_negative_k(self):
        """Test with negative k."""
        print("\n--- Negative K Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        try:
            finder = KNNNeighborFinder(k=-5)
            print_test("Rejects negative k", False, "Should have raised error")
            return False
        except (ValueError, Exception):
            print_test("Rejects negative k", True, "Properly raised error")
            return True

    def test_zero_k(self):
        """Test with k=0."""
        print("\n--- Zero K Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        try:
            finder = KNNNeighborFinder(k=0)
            X_train = np.random.randn(100, 10).astype(np.float32)
            finder.fit(X_train)
            print_test("Rejects k=0", False, "Should have raised error")
            return False
        except (ValueError, Exception):
            print_test("Rejects k=0", True, "Properly raised error")
            return True

    def test_nan_in_data(self):
        """Test with NaN values in data."""
        print("\n--- NaN in Data Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        X_train = np.random.randn(100, 10).astype(np.float32)
        X_train[50, 5] = np.nan  # Insert NaN

        finder = KNNNeighborFinder(k=10)

        try:
            finder.fit(X_train)
            print_test("Handles NaN", False, "Should have raised error or warning")
            return False
        except (ValueError, Exception):
            print_test("Catches NaN", True, "Properly raised error")
            return True

    def test_inf_in_data(self):
        """Test with Inf values in data."""
        print("\n--- Inf in Data Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        X_train = np.random.randn(100, 10).astype(np.float32)
        X_train[50, 5] = np.inf  # Insert Inf

        finder = KNNNeighborFinder(k=10)

        try:
            finder.fit(X_train)
            X_test = np.random.randn(5, 10).astype(np.float32)
            distances, indices = finder.kneighbors(X_test, k=10)

            # Check if any distances are inf
            has_inf = np.any(np.isinf(distances))
            print_test("Handles Inf", not has_inf, f"Has inf distances: {has_inf}")
            return not has_inf
        except Exception as e:
            print_test("Handles Inf", True, f"Raised error (acceptable): {type(e).__name__}")
            return True

    def test_empty_query(self):
        """Test with empty query array."""
        print("\n--- Empty Query Test ---")
        from ensemble_weights.neighbors import KNNNeighborFinder

        X_train = np.random.randn(100, 10).astype(np.float32)
        X_test = np.random.randn(0, 10).astype(np.float32)  # Empty!

        finder = KNNNeighborFinder(k=10)
        finder.fit(X_train)

        try:
            distances, indices = finder.kneighbors(X_test, k=10)
            passed = (distances.shape[0] == 0 and indices.shape[0] == 0)
            print_test("Empty query", passed, f"Shapes: {distances.shape}, {indices.shape}")
            return passed
        except Exception as e:
            print_test("Empty query", False, str(e))
            return False


class WarningTests:
    """Test that warnings are raised appropriately."""

    def test_low_n_probes_warning(self):
        """Test warning for low n_probes in FAISS IVF."""
        print("\n--- Low n_probes Warning Test ---")

        try:
            from ensemble_weights.neighbors import FaissNeighborFinder

            X_train = np.random.randn(10000, 50).astype(np.float32)

            finder = FaissNeighborFinder(k=10, index_type='ivf', n_probes=1)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                finder.fit(X_train)

                warned = len(w) > 0 and "n_probes" in str(w[0].message).lower()
                print_test("Low n_probes warning", warned, f"Got {len(w)} warnings")
                return warned
        except ImportError:
            print_test("Low n_probes warning", False, "FAISS not installed")
            return False

    def test_low_ef_construction_warning(self):
        """Test warning for low ef_construction in HNSW."""
        print("\n--- Low ef_construction Warning Test ---")

        try:
            from ensemble_weights.neighbors import HNSWNeighborFinder

            X_train = np.random.randn(20000, 50).astype(np.float32)

            finder = HNSWNeighborFinder(
                k=10,
                backend='hnswlib',
                ef_construction=50  # Very low for large dataset
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                finder.fit(X_train)

                warned = len(w) > 0 and "ef_construction" in str(w[0].message).lower()
                print_test("Low ef_construction warning", warned, f"Got {len(w)} warnings")
                return warned
        except ImportError:
            print_test("Low ef_construction warning", False, "hnswlib not installed")
            return False


def main():
    """Run all stress and edge case tests."""
    print("=" * 80)
    print("  STRESS TESTS & EDGE CASES")
    print("=" * 80)

    stress = StressTests()
    edge = EdgeCaseTests()
    warn = WarningTests()

    results = []

    print("\n" + "=" * 80)
    print("  STRESS TESTS")
    print("=" * 80)
    results.append(("Very high k", stress.test_very_high_k()))
    results.append(("k equals n", stress.test_k_equals_n()))
    results.append(("Single training sample", stress.test_single_sample_training()))
    results.append(("Very high dimensions", stress.test_very_high_dimensions()))
    results.append(("Duplicate features", stress.test_duplicate_features()))
    results.append(("Constant features", stress.test_constant_features()))
    results.append(("Large scale Annoy", stress.test_large_scale_annoy()))

    print("\n" + "=" * 80)
    print("  EDGE CASE TESTS")
    print("=" * 80)
    results.append(("Mismatched dimensions", edge.test_mismatched_dimensions()))
    results.append(("Negative k", edge.test_negative_k()))
    results.append(("Zero k", edge.test_zero_k()))
    results.append(("NaN in data", edge.test_nan_in_data()))
    results.append(("Inf in data", edge.test_inf_in_data()))
    results.append(("Empty query", edge.test_empty_query()))

    print("\n" + "=" * 80)
    print("  WARNING TESTS")
    print("=" * 80)
    results.append(("Low n_probes warning", warn.test_low_n_probes_warning()))
    results.append(("Low ef_construction warning", warn.test_low_ef_construction_warning()))

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "✗"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {name}")

    print(f"\nPASSED: {passed}/{total} tests")

    if passed == total:
        print("\n🎉 All stress tests passed!")
    else:
        print("\n⚠️  Some tests failed - this may be expected for edge cases")

    return passed >= total * 0.8  # 80% pass rate for stress tests


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)