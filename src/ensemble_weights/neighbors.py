import numpy as np
import warnings

# FAISS IVF k-means needs at least this many training samples per cell to converge.
# Below this ratio the training loop can hang indefinitely.
_FAISS_MIN_SAMPLES_PER_CELL = 40


class NeighborFinder:
    """Base class for neighbor search backends."""

    def fit(self, X):
        raise NotImplementedError

    def kneighbors(self, X, k=None):
        raise NotImplementedError


class KNNNeighborFinder(NeighborFinder):
    """Exact nearest neighbors via sklearn NearestNeighbors."""

    def __init__(self, k=10, **kwargs):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.n_neighbors = k
        self.kwargs = kwargs
        self.model = None

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        X = np.atleast_2d(X)
        if X.shape[0] < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{X.shape[0]} samples. Reduce k to at most {X.shape[0]}."
            )
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, **self.kwargs)
        self.model.fit(X)
        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)
        return self.model.kneighbors(X, n_neighbors=k)


class FaissNeighborFinder(NeighborFinder):
    """Approximate nearest neighbors via FAISS (flat, IVF, or HNSW index)."""

    def __init__(self, k=10, index_type='flat', n_cells=None, n_probes=50,
                 hnsw_M=32, hnsw_efConstruction=400, hnsw_efSearch=200):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.n_neighbors = k
        self.index_type = index_type.lower()
        self.n_cells = n_cells
        self.n_probes = n_probes
        self.hnsw_M = hnsw_M
        self.hnsw_efConstruction = hnsw_efConstruction
        self.hnsw_efSearch = hnsw_efSearch
        self.index_ = None
        self._check_availability()

    def _check_availability(self):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not found. Install with: pip install faiss-cpu")

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        n_samples, dim = X.shape

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )

        if self.index_type == 'flat':
            if dim <= 2:
                warnings.warn(
                    f"FAISS Flat may have floating-point precision issues in {dim}D. "
                    f"Consider KNNNeighborFinder for low-dimensional data.",
                    UserWarning
                )
            self.index_ = self.faiss.IndexFlatL2(dim)
            self.index_.add(X)

        elif self.index_type == 'ivf':
            if self.n_cells is None:
                self.n_cells = min(int(np.sqrt(n_samples)), 4096)

            # Auto-reduce n_cells if dataset is too small; otherwise training hangs.
            min_required = self.n_cells * _FAISS_MIN_SAMPLES_PER_CELL
            if n_samples < min_required:
                safe_cells = max(1, n_samples // _FAISS_MIN_SAMPLES_PER_CELL)
                warnings.warn(
                    f"n_cells={self.n_cells} requires {min_required} samples but only "
                    f"{n_samples} provided. Reducing to {safe_cells} to prevent hanging. "
                    f"Consider index_type='flat' or KNNNeighborFinder for small datasets.",
                    UserWarning
                )
                self.n_cells = safe_cells

            effective_probes = min(self.n_probes, self.n_cells)
            if effective_probes < self.n_probes:
                warnings.warn(
                    f"n_probes={self.n_probes} exceeds n_cells={self.n_cells}. "
                    f"Clamping to {self.n_cells}.",
                    UserWarning
                )
            if effective_probes < self.n_cells * 0.1:
                warnings.warn(
                    f"n_probes={effective_probes} is below 10% of n_cells={self.n_cells}. "
                    f"Recall may be poor. Consider n_probes >= {max(1, int(self.n_cells * 0.1))}.",
                    UserWarning
                )

            quantizer = self.faiss.IndexFlatL2(dim)
            self.index_ = self.faiss.IndexIVFFlat(quantizer, dim, self.n_cells)
            self.index_.train(X)
            self.index_.add(X)
            self.index_.nprobe = effective_probes

        elif self.index_type == 'hnsw':
            if n_samples >= 10000 and self.hnsw_efConstruction < 300:
                warnings.warn(
                    f"ef_construction={self.hnsw_efConstruction} may be too low for "
                    f"{n_samples} samples. Consider ef_construction >= 400.",
                    UserWarning
                )
            self.index_ = self.faiss.IndexHNSWFlat(dim, self.hnsw_M)
            self.index_.hnsw.efConstruction = self.hnsw_efConstruction
            self.index_.hnsw.efSearch = self.hnsw_efSearch
            self.index_.add(X)

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X).astype(np.float32)
        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
        distances, indices = self.index_.search(X, k)
        # FAISS returns squared L2; clamp to 0 before sqrt to handle floating-point noise.
        return np.sqrt(np.maximum(distances, 0)), indices


class AnnoyNeighborFinder(NeighborFinder):
    """Approximate nearest neighbors via Annoy."""

    def __init__(self, k=10, n_trees=100, metric='euclidean', search_k=-1):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.k = k
        self.n_trees = n_trees
        self.metric = metric
        # Annoy's recommended default; the previous value (n_trees * k * 50)
        # caused near-freezes on large batches.
        self.search_k = n_trees * k if search_k == -1 else search_k
        self.index_ = None
        self.n_samples_ = None
        self._check_availability()

    def _check_availability(self):
        try:
            from annoy import AnnoyIndex
            self.AnnoyIndex = AnnoyIndex
        except ImportError:
            raise ImportError("Annoy not found. Install with: pip install annoy")

    def fit(self, X):
        X = np.atleast_2d(X)
        n_samples, dim = X.shape
        self.n_samples_ = n_samples

        if n_samples < self.k:
            raise ValueError(
                f"Cannot find {self.k} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )
        if dim <= 3:
            warnings.warn(
                f"Annoy tree structure can degenerate in {dim}D. "
                f"Consider KNNNeighborFinder for low-dimensional data.",
                UserWarning
            )

        metric_map = {
            'euclidean': 'euclidean', 'l2': 'euclidean',
            'angular': 'angular',     'cosine': 'angular',
            'manhattan': 'manhattan', 'hamming': 'hamming', 'dot': 'dot',
        }
        self.index_ = self.AnnoyIndex(dim, metric_map.get(self.metric.lower(), 'euclidean'))
        for i, vec in enumerate(X):
            self.index_.add_item(i, vec.tolist())
        self.index_.build(self.n_trees)

        # Sanity check: verify the index returns the expected number of neighbors.
        # Annoy silently returns 1 result on Apple Silicon (M1/M2/M3) due to a
        # known ARM64 compilation bug, regardless of n_trees or search_k.
        test_vec = X[0].tolist()
        test_result = self.index_.get_nns_by_vector(test_vec, self.k, search_k=self.search_k)
        if len(test_result) < self.k:
            raise RuntimeError(
                f"Annoy index returned {len(test_result)} neighbors but {self.k} were "
                f"requested. This is a known Annoy bug on Apple Silicon (M1/M2/M3) — "
                f"the package does not work correctly on ARM64. "
                f"Use preset='fast' (FAISS IVF) or preset='exact' (sklearn KNN) instead."
            )

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.k
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)

        all_indices, all_distances = [], []
        for vec in X:
            idx, dist = self.index_.get_nns_by_vector(
                vec.tolist(), k, search_k=self.search_k, include_distances=True
            )
            if len(idx) != k:
                raise ValueError(
                    f"Annoy returned {len(idx)} neighbors but {k} were requested. "
                    f"Try increasing n_trees (current: {self.n_trees}) or search_k "
                    f"(current: {self.search_k}), or use KNNNeighborFinder."
                )
            all_indices.append(idx)
            all_distances.append(dist)

        return np.array(all_distances), np.array(all_indices)


class HNSWNeighborFinder(NeighborFinder):
    """Approximate nearest neighbors via HNSW (hnswlib or nmslib backend)."""

    def __init__(self, k=10, space='l2', M=32, ef_construction=400,
                 ef_search=200, backend='hnswlib'):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.n_neighbors = k
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.backend = backend.lower()
        self.index_ = None
        self._check_availability()

    def _check_availability(self):
        if self.backend == 'hnswlib':
            try:
                import hnswlib
                self.hnswlib = hnswlib
            except ImportError:
                raise ImportError("hnswlib not found. Install with: pip install hnswlib")
        elif self.backend == 'nmslib':
            try:
                import nmslib
                self.nmslib = nmslib
            except ImportError:
                raise ImportError("nmslib not found. Install with: pip install nmslib")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        n_samples, dim = X.shape

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )
        if n_samples >= 10000 and self.ef_construction < 300:
            warnings.warn(
                f"ef_construction={self.ef_construction} may be too low for "
                f"{n_samples} samples. Consider ef_construction >= 400.",
                UserWarning
            )

        if self.backend == 'hnswlib':
            self.index_ = self.hnswlib.Index(space=self.space, dim=dim)
            self.index_.init_index(
                max_elements=n_samples, M=self.M, ef_construction=self.ef_construction
            )
            self.index_.set_ef(self.ef_search)
            self.index_.add_items(X, np.arange(n_samples))

        else:  # nmslib
            space_map = {'l2': 'l2', 'cosine': 'cosinesimil', 'ip': 'negdotprod'}
            self.index_ = self.nmslib.init(
                method='hnsw',
                space=space_map.get(self.space, 'l2'),
                data_type=self.nmslib.DataType.DENSE_VECTOR
            )
            self.index_.addDataPointBatch(X)
            self.index_.createIndex(
                {'M': self.M, 'efConstruction': self.ef_construction, 'post': 0},
                print_progress=False
            )
            self.index_.setQueryTimeParams({'efSearch': self.ef_search})

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X).astype(np.float32)
        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if self.backend == 'nmslib':
            results = self.index_.knnQueryBatch(X, k=k)
            return (
                np.array([dist for _, dist in results]),
                np.array([idx  for idx, _ in results]),
            )

        # hnswlib returns (indices, distances) — reversed from convention.
        indices, distances = self.index_.knn_query(X, k=k)
        return distances, indices