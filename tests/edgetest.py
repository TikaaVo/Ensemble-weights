"""
Test demonstrating the shape mismatch bug and its fix.
Run with: python test_shape_bug.py
"""
import numpy as np
import sys

print("="*80)
print("SHAPE MISMATCH BUG DEMONSTRATION")
print("="*80)

# ============================================================================
# Setup: Create mock data
# ============================================================================

np.random.seed(42)
X_train = np.random.randn(100, 10).astype(np.float32)
X_test_single = np.random.randn(1, 10).astype(np.float32)  # Single query
X_test_batch = np.random.randn(5, 10).astype(np.float32)   # Batch query

# ============================================================================
# BROKEN VERSION: Current neighbors.py
# ============================================================================

print("\n" + "="*80)
print("CURRENT (BROKEN) BEHAVIOR")
print("="*80)

class BrokenKNNFinder:
    """Simulates current (broken) neighbors.py behavior."""

    def __init__(self, k=10):
        self.k = k
        from sklearn.neighbors import NearestNeighbors
        self.model = NearestNeighbors(n_neighbors=k)

    def fit(self, X):
        self.model.fit(X)
        return self

    def kneighbors(self, X):
        """BROKEN: Collapses dimension for single queries."""
        X = np.atleast_2d(X)
        single = (X.shape[0] == 1)  # Check if single query
        distances, indices = self.model.kneighbors(X, n_neighbors=self.k)

        if single:
            # THIS IS THE BUG: Collapses (1, k) to (k,)
            return distances[0], indices[0]
        return distances, indices

# Test with single query
print("\nSingle query (1 sample):")
print(f"  Input shape: {X_test_single.shape}")

finder = BrokenKNNFinder(k=10)
finder.fit(X_train)
distances, indices = finder.kneighbors(X_test_single)

print(f"  Output distances shape: {distances.shape}")
print(f"  Output indices shape: {indices.shape}")
print(f"  ❌ PROBLEM: Shape is (10,) instead of (1, 10)")

# Simulate what happens in knn.py
print("\nWhat happens in knn.py predict():")
try:
    # Simulate the matrix indexing
    n_models = 3
    performance_matrix = np.random.rand(len(X_train), n_models)

    print(f"  performance_matrix shape: {performance_matrix.shape}")
    print(f"  indices shape: {indices.shape}")

    neighbor_scores = performance_matrix[indices]
    print(f"  neighbor_scores shape: {neighbor_scores.shape}")
    print(f"    Expected: (1, 10, 3)")
    print(f"    Got: {neighbor_scores.shape} ❌")

    avg_scores = neighbor_scores.mean(axis=1)
    print(f"  avg_scores shape: {avg_scores.shape}")
    print(f"    Expected: (1, 3)")
    print(f"    Got: {avg_scores.shape} ❌")

    # This will crash!
    max_scores = np.max(avg_scores, axis=1, keepdims=True)
    print(f"  max_scores shape: {max_scores.shape}")

except Exception as e:
    print(f"  💥 CRASH: {type(e).__name__}: {e}")

# Test with batch query
print("\nBatch query (5 samples):")
print(f"  Input shape: {X_test_batch.shape}")

distances, indices = finder.kneighbors(X_test_batch)
print(f"  Output distances shape: {distances.shape}")
print(f"  Output indices shape: {indices.shape}")
print(f"  ✓ Works fine: Shape is (5, 10)")

# ============================================================================
# FIXED VERSION: Fixed neighbors.py
# ============================================================================

print("\n\n" + "="*80)
print("FIXED BEHAVIOR")
print("="*80)

class FixedKNNFinder:
    """Simulates fixed neighbors.py behavior."""

    def __init__(self, k=10):
        self.k = k
        from sklearn.neighbors import NearestNeighbors
        self.model = NearestNeighbors(n_neighbors=k)

    def fit(self, X):
        self.model.fit(X)
        return self

    def kneighbors(self, X):
        """FIXED: Always returns 2D arrays."""
        X = np.atleast_2d(X)
        distances, indices = self.model.kneighbors(X, n_neighbors=self.k)

        # FIXED: Always return 2D, never collapse
        return distances, indices

# Test with single query
print("\nSingle query (1 sample):")
print(f"  Input shape: {X_test_single.shape}")

finder = FixedKNNFinder(k=10)
finder.fit(X_train)
distances, indices = finder.kneighbors(X_test_single)

print(f"  Output distances shape: {distances.shape}")
print(f"  Output indices shape: {indices.shape}")
print(f"  ✓ FIXED: Shape is (1, 10)")

# Simulate what happens in knn.py
print("\nWhat happens in knn.py predict():")
try:
    n_models = 3
    performance_matrix = np.random.rand(len(X_train), n_models)

    print(f"  performance_matrix shape: {performance_matrix.shape}")
    print(f"  indices shape: {indices.shape}")

    neighbor_scores = performance_matrix[indices]
    print(f"  neighbor_scores shape: {neighbor_scores.shape}")
    print(f"    Expected: (1, 10, 3)")
    print(f"    Got: {neighbor_scores.shape} ✓")

    avg_scores = neighbor_scores.mean(axis=1)
    print(f"  avg_scores shape: {avg_scores.shape}")
    print(f"    Expected: (1, 3)")
    print(f"    Got: {avg_scores.shape} ✓")

    max_scores = np.max(avg_scores, axis=1, keepdims=True)
    print(f"  max_scores shape: {max_scores.shape}")
    print(f"    Expected: (1, 1)")
    print(f"    Got: {max_scores.shape} ✓")

    print("\n  ✓ SUCCESS: No crash!")

except Exception as e:
    print(f"  💥 CRASH: {type(e).__name__}: {e}")

# Test with batch query
print("\nBatch query (5 samples):")
print(f"  Input shape: {X_test_batch.shape}")

distances, indices = finder.kneighbors(X_test_batch)
print(f"  Output distances shape: {distances.shape}")
print(f"  Output indices shape: {indices.shape}")
print(f"  ✓ Still works: Shape is (5, 10)")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
PROBLEM:
  Current neighbors.py collapses batch dimension for single queries
  - Single query: returns (k,) instead of (1, k)
  - This breaks knn.py predict() which expects 2D arrays
  
THE BUG:
  In neighbors.py kneighbors():
  
    if single:
        return distances[0], indices[0]  # ❌ Collapses (1, k) to (k,)
  
FIX:
  Remove the dimension collapse:
  
    # Always return 2D
    return distances, indices  # ✓ Always (batch_size, k)
  
IMPACT:
  - Affects: KNNNeighborFinder, FaissNeighborFinder, AnnoyNeighborFinder, HNSWNeighborFinder
  - Breaking: Code that expects (k,) for single queries will break
  - Benefit: Consistent behavior, no crashes, simpler code
  
FILES PROVIDED:
  - SHAPE_MISMATCH_BUG.md - Detailed analysis
  - neighbors_fixed_shapes.py - Fixed implementation
  - This test file - Demonstrates the bug and fix
  
NEXT STEPS:
  1. Replace neighbors.py with neighbors_fixed_shapes.py
  2. Test with: pytest test_comprehensive.py::TestDynamicRouterIntegration
  3. Verify single-query predictions work
""")

print("="*80)