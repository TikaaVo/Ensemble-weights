import numpy as np
import warnings


# Minimum ratio of samples-to-cells required for FAISS IVF k-means to converge.
# FAISS documentation recommends 39x; we use a slightly larger value for safety.
_FAISS_MIN_SAMPLES_PER_CELL = 40


class NeighborFinder:
    """Base class for neighbor finders."""

    def fit(self, X):
        raise NotImplementedError

    def kneighbors(self, X, k=None):
        raise NotImplementedError


class KNNNeighborFinder(NeighborFinder):
    """Exact nearest neighbors using sklearn."""

    def __init__(self, k=10, **kwargs):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.n_neighbors = k
        self.kwargs = kwargs
        self.model = None

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        X = np.atleast_2d(X)
        n_samples = X.shape[0]

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )

        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, **self.kwargs)
        self.model.fit(X)
        return self

    def kneighbors(self, X, k=None):
        """
        Find k nearest neighbors.
        Always returns (batch_size, k) shaped arrays.
        """
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X)

        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)

        distances, indices = self.model.kneighbors(X, n_neighbors=k)
        return distances, indices


class FaissNeighborFinder(NeighborFinder):
    """FAISS-based approximate nearest neighbor search."""

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
            raise ImportError(
                "FAISS not found. Install with: pip install faiss-cpu"
            )

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        n_samples, dim = X.shape

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )

        if dim <= 2 and self.index_type == 'flat':
            warnings.warn(
                f"FAISS Flat may have precision issues in {dim}D due to "
                f"floating-point arithmetic. Consider using KNNNeighborFinder "
                f"for low-dimensional data (<=2D).",
                UserWarning
            )

        if self.index_type == 'flat':
            self.index_ = self.faiss.IndexFlatL2(dim)
            self.index_.add(X)

        elif self.index_type == 'ivf':
            if self.n_cells is None:
                self.n_cells = min(int(np.sqrt(n_samples)), 4096)

            # FIX: FAISS IVF k-means requires ~40x samples per cell or training
            # will fail to converge and hang indefinitely.
            min_required = self.n_cells * _FAISS_MIN_SAMPLES_PER_CELL
            if n_samples < min_required:
                safe_cells = max(1, n_samples // _FAISS_MIN_SAMPLES_PER_CELL)
                warnings.warn(
                    f"n_cells={self.n_cells} requires at least {min_required} training "
                    f"samples for FAISS IVF k-means to converge, but only {n_samples} "
                    f"were provided. Reducing n_cells from {self.n_cells} to {safe_cells} "
                    f"to prevent hanging. Consider using index_type='flat' or switching "
                    f"to KNNNeighborFinder for small datasets.",
                    UserWarning
                )
                self.n_cells = safe_cells

            # Clamp n_probes to n_cells so FAISS doesn't silently degrade
            effective_probes = min(self.n_probes, self.n_cells)
            if effective_probes < self.n_probes:
                warnings.warn(
                    f"n_probes={self.n_probes} exceeds n_cells={self.n_cells}. "
                    f"Clamping n_probes to {self.n_cells}.",
                    UserWarning
                )

            if effective_probes < self.n_cells * 0.1:
                warnings.warn(
                    f"n_probes={effective_probes} is less than 10% of n_cells={self.n_cells}. "
                    f"This may result in poor recall. Consider increasing n_probes to at least "
                    f"{max(1, int(self.n_cells * 0.1))}.",
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
                    f"For a dataset with {n_samples} samples, "
                    f"ef_construction={self.hnsw_efConstruction} may be too low. "
                    f"Consider ef_construction >= 400 for better recall.",
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
        """
        Find k nearest neighbors.
        Always returns (batch_size, k) shaped arrays.
        """
        if k is None:
            k = self.n_neighbors

        X = np.atleast_2d(X).astype(np.float32)

        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        distances, indices = self.index_.search(X, k)
        distances = np.sqrt(np.maximum(distances, 0))  # FIX: clamp negatives before sqrt

        return distances, indices


class AnnoyNeighborFinder(NeighborFinder):
    """Annoy-based approximate nearest neighbor search."""

    def __init__(self, k=10, n_trees=100, metric='euclidean', search_k=-1):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")

        self.k = k
        self.n_trees = n_trees
        self.metric = metric

        # FIX: The previous formula (n_trees * k * 50) produced search_k values
        # up to 50,000+, causing near-freezes on large batch predictions.
        # Annoy's own recommended default is n_trees * k, which gives good
        # recall without the excessive cost. Users can still override explicitly.
        if search_k == -1:
            self.search_k = n_trees * k
        else:
            self.search_k = search_k

        self.index_ = None
        self.n_samples_ = None
        self._check_availability()

    def _check_availability(self):
        try:
            from annoy import AnnoyIndex
            self.AnnoyIndex = AnnoyIndex
        except ImportError:
            raise ImportError(
                "Annoy not found. Install with: pip install annoy"
            )

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
                f"Annoy may have poor performance in {dim}D. Tree structure can "
                f"degenerate in low dimensions. Consider using KNNNeighborFinder "
                f"for low-dimensional data (<=3D).",
                UserWarning
            )

        metric_map = {
            'euclidean': 'euclidean',
            'l2': 'euclidean',
            'angular': 'angular',
            'cosine': 'angular',
            'manhattan': 'manhattan',
            'hamming': 'hamming',
            'dot': 'dot'
        }
        annoy_metric = metric_map.get(self.metric.lower(), 'euclidean')
        self.index_ = self.AnnoyIndex(dim, annoy_metric)

        for i, vec in enumerate(X):
            self.index_.add_item(i, vec.tolist())

        self.index_.build(self.n_trees)
        return self

    def kneighbors(self, X, k=None):
        """
        Find k nearest neighbors.
        Always returns (batch_size, k) shaped arrays.
        """
        if k is None:
            k = self.k

        X = np.atleast_2d(X)

        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)

        all_indices = []
        all_distances = []

        for vec in X:
            idx, dist = self.index_.get_nns_by_vector(
                vec.tolist(), k,
                search_k=self.search_k,
                include_distances=True
            )

            if len(idx) != k:
                raise ValueError(
                    f"Annoy returned {len(idx)} neighbors but {k} were requested. "
                    f"This typically means the index has fewer items than k, or there "
                    f"are connectivity issues. "
                    f"Try: (1) Increasing n_trees (current: {self.n_trees}), "
                    f"(2) Increasing search_k (current: {self.search_k}), or "
                    f"(3) Using KNNNeighborFinder for this dataset."
                )

            all_indices.append(idx)
            all_distances.append(dist)

        return np.array(all_distances), np.array(all_indices)


class HNSWNeighborFinder(NeighborFinder):
    """HNSW-based approximate nearest neighbor search."""

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
                raise ImportError(
                    "hnswlib not found. Install with: pip install hnswlib"
                )
        elif self.backend == 'nmslib':
            try:
                import nmslib
                self.nmslib = nmslib
            except ImportError:
                raise ImportError(
                    "nmslib not found. Install with: pip install nmslib"
                )
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
                f"For a dataset with {n_samples} samples, "
                f"ef_construction={self.ef_construction} may be too low. "
                f"Consider ef_construction >= 400 for better recall.",
                UserWarning
            )

        if self.backend == 'hnswlib':
            self.index_ = self.hnswlib.Index(space=self.space, dim=dim)
            self.index_.init_index(
                max_elements=n_samples,
                M=self.M,
                ef_construction=self.ef_construction
            )
            self.index_.set_ef(self.ef_search)
            self.index_.add_items(X, np.arange(n_samples))

        elif self.backend == 'nmslib':
            space_map = {'l2': 'l2', 'cosine': 'cosinesimil', 'ip': 'negdotprod'}
            nms_space = space_map.get(self.space, 'l2')

            self.index_ = self.nmslib.init(
                method='hnsw',
                space=nms_space,
                data_type=self.nmslib.DataType.DENSE_VECTOR
            )
            self.index_.addDataPointBatch(X)
            self.index_.createIndex(
                {
                    'M': self.M,
                    'efConstruction': self.ef_construction,
                    'post': 0
                },
                print_progress=False
            )
            self.index_.setQueryTimeParams({'efSearch': self.ef_search})

        return self

    def kneighbors(self, X, k=None):
        """
        Find k nearest neighbors.
        Always returns (batch_size, k) shaped arrays.
        """
        if k is None:
            k = self.n_neighbors

        X = np.atleast_2d(X).astype(np.float32)

        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if self.backend == 'nmslib':
            results = self.index_.knnQueryBatch(X, k=k)
            all_indices = []
            all_distances = []
            for idx, dist in results:
                all_indices.append(np.array(idx))
                all_distances.append(np.array(dist))
            return np.array(all_distances), np.array(all_indices)

        else:  # hnswlib
            indices, distances = self.index_.knn_query(X, k=k)
            return distances, indices