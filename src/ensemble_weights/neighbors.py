from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import NearestNeighbors


class NeighborFinder(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def kneighbors(self, X, k):
        pass


class KNNNeighborFinder(NeighborFinder):
    def __init__(self, k=10, **kwargs):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.n_neighbors = k
        self.kwargs = kwargs
        self.model = None

    def fit(self, X):
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, **self.kwargs)
        self.model.fit(X)

    def kneighbors(self, X, k=None):
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X)
        single = (X.shape[0] == 1)
        distances, indices = self.model.kneighbors(X, n_neighbors=k)
        if single:
            return distances[0], indices[0]
        return distances, indices


class FaissNeighborFinder(NeighborFinder):
    """FAISS-based nearest neighbor search.

    Parameters
    ----------
    n_neighbors : int
        Default number of neighbors.
    index_type : str, default='flat'
        Type of FAISS index: 'flat' (exact), 'ivf' (inverted file), or 'hnsw'.
    n_cells : int, default=None
        Number of Voronoi cells for IVF index. If None, auto-computed as sqrt(n_samples).
    n_probes : int, default=50
        Number of cells to probe during search (IVF only). Higher = more accurate.
        Increased default from 10 to 50 for better recall.
    hnsw_M : int, default=32
        Number of neighbors per node in HNSW graph.
    hnsw_efConstruction : int, default=400
        Construction time accuracy/speed tradeoff for HNSW.
        Increased default from 200 to 400 for better recall on large datasets.
    hnsw_efSearch : int, default=200
        Query time search depth for HNSW.
    """

    def __init__(self, k=10, index_type='flat', n_cells=None, n_probes=50,
                 hnsw_M=32, hnsw_efConstruction=400, hnsw_efSearch=200):
        self.n_neighbors = k
        self.index_type = index_type
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
            raise ImportError("FAISS not installed. Please install with: pip install faiss-cpu")

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        dim = X.shape[1]
        n_samples = X.shape[0]

        # BUG FIX #2: Warn about low-dimensional data with FAISS Flat
        if dim <= 2 and self.index_type == 'flat':
            import warnings
            warnings.warn(
                f"FAISS Flat may have precision issues in {dim}D due to floating-point "
                f"arithmetic. For 1D or 2D data, consider using KNN (sklearn) instead.",
                UserWarning
            )

        if self.index_type == 'flat':
            self.index_ = self.faiss.IndexFlatL2(dim)
        elif self.index_type == 'ivf':
            # Auto-compute n_cells if not provided
            if self.n_cells is None:
                self.n_cells = min(int(np.sqrt(n_samples)), 4096)

            # BUG FIX #3: Validate n_probes is reasonable for good recall
            min_probes = max(int(0.1 * self.n_cells), 10)
            if self.n_probes < min_probes:
                import warnings
                warnings.warn(
                    f"n_probes={self.n_probes} is low for n_cells={self.n_cells}. "
                    f"Expected recall may be poor (<80%). "
                    f"For good recall (>95%), use n_probes >= {min_probes}",
                    UserWarning
                )

            quantizer = self.faiss.IndexFlatL2(dim)
            self.index_ = self.faiss.IndexIVFFlat(quantizer, dim, self.n_cells)
            # IVF requires training
            self.index_.train(X)
            self.index_.nprobe = self.n_probes
        elif self.index_type == 'hnsw':
            # BUG FIX #4: Validate efConstruction for large datasets
            min_ef_construction = max(self.hnsw_M * 8, 400) if n_samples > 10000 else 200
            if self.hnsw_efConstruction < min_ef_construction:
                import warnings
                warnings.warn(
                    f"hnsw_efConstruction={self.hnsw_efConstruction} may be too low "
                    f"for {n_samples} samples. Expected recall may be poor. "
                    f"For good recall (>95%), use hnsw_efConstruction >= {min_ef_construction}",
                    UserWarning
                )

            self.index_ = self.faiss.IndexHNSWFlat(dim, self.hnsw_M)
            self.index_.hnsw.efConstruction = self.hnsw_efConstruction
            # BUG FIX #4: Also set efSearch for queries
            self.index_.hnsw.efSearch = self.hnsw_efSearch
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index_.add(X)
        return self

    def kneighbors(self, X, k=None):
        if k is None:
            k = self.n_neighbors

        X = np.atleast_2d(X).astype(np.float32)
        single = (X.shape[0] == 1)
        distances, indices = self.index_.search(X, k)

        # FAISS returns squared L2 distances; convert to Euclidean if desired
        distances = np.sqrt(distances)

        if single:
            return distances[0], indices[0]
        return distances, indices


class AnnoyNeighborFinder(NeighborFinder):
    """Annoy-based approximate nearest neighbor search.

    Parameters
    ----------
    n_neighbors : int
        Default number of neighbors.
    n_trees : int, default=100
        Number of trees in the forest. More trees = higher accuracy, larger index.
        For large datasets (>1M), use 100-200 trees.
    metric : str, default='euclidean'
        Distance metric. Options: 'euclidean', 'angular', 'manhattan', 'hamming', 'dot'.
    search_k : int, default=-1
        Number of nodes to inspect during search.
        -1 means auto: max(n_trees * n_neighbors * 50, 10000).
        Higher = more accurate but slower.
        Increased multiplier from 10 to 50 for better recall.
    """

    def __init__(self, k=10, n_trees=100, metric='euclidean', search_k=-1):
        self.k = k
        self.n_trees = n_trees
        self.metric = metric
        # BUG FIX #1: Much higher default for search_k
        # Increased multiplier from 10 to 50, and added minimum of 10000
        if search_k == -1:
            self.search_k = max(n_trees * k * 50, 10000)
        else:
            self.search_k = search_k
        self.index_ = None
        self._check_availability()

    def _check_availability(self):
        try:
            from annoy import AnnoyIndex
            self.AnnoyIndex = AnnoyIndex
        except ImportError:
            raise ImportError("Annoy not installed. Install with: pip install annoy")

    def fit(self, X):
        dim = X.shape[1]
        n_samples = X.shape[0]

        # BUG FIX #1: Warn about low-dimensional data
        if dim <= 3:
            import warnings
            warnings.warn(
                f"Annoy may have poor performance in {dim}D. "
                f"Tree structure can degenerate in low dimensions. "
                f"Consider using KNN or FAISS Flat for low-dimensional data.",
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
        if k is None:
            k = self.k

        X = np.atleast_2d(X)
        single = (X.shape[0] == 1)
        all_indices = []
        all_distances = []

        for vec in X:
            idx, dist = self.index_.get_nns_by_vector(
                vec.tolist(), k,
                search_k=self.search_k,
                include_distances=True
            )

            # BUG FIX #1: Validate that Annoy returned k neighbors
            if len(idx) != k:
                raise ValueError(
                    f"Annoy only returned {len(idx)} neighbors out of {k} requested. "
                    f"This usually means the index has connectivity issues. "
                    f"Try: (1) Increasing n_trees (current: {self.n_trees}), "
                    f"(2) Increasing search_k (current: {self.search_k}), or "
                    f"(3) Using a different neighbor finder for this data."
                )

            all_indices.append(idx)
            all_distances.append(dist)

        if single:
            return np.array(all_distances[0]), np.array(all_indices[0])
        return np.array(all_distances), np.array(all_indices)


class HNSWNeighborFinder(NeighborFinder):
    """HNSW-based approximate nearest neighbor search.

    Parameters
    ----------
    n_neighbors : int
        Default number of neighbors.
    space : str, default='l2'
        Distance space. Common options: 'l2', 'cosinesimil', 'ip' (inner product).
    M : int, default=32
        Number of bi-directional links per element.
        Higher = more accurate but slower and more memory.
        Recommended: 16-64 depending on dataset size.
    ef_construction : int, default=400
        Size of the dynamic list for the nearest neighbors during index construction.
        Higher = more accurate index but slower to build.
        Increased default from 200 to 400 for better recall on large datasets.
    ef_search : int, default=200
        Size of the dynamic list during search.
        Higher = more accurate but slower queries.
    backend : str, default='hnswlib'
        Which library to use: 'nmslib' or 'hnswlib'.
        hnswlib is generally faster for large datasets.
    """

    def __init__(self, k=10, space='l2', M=32, ef_construction=400,
                 ef_search=200, backend='hnswlib'):
        self.n_neighbors = k
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.backend = backend.lower()
        self.index_ = None
        self._check_availability()

    def _check_availability(self):
        if self.backend == 'nmslib':
            try:
                import nmslib
                self.nmslib = nmslib
            except ImportError:
                raise ImportError("nmslib not installed. Install with: pip install nmslib")
        elif self.backend == 'hnswlib':
            try:
                import hnswlib
                self.hnswlib = hnswlib
            except ImportError:
                raise ImportError("hnswlib not installed. Install with: pip install hnswlib")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        # BUG FIX #5: Validate ef_construction for large datasets
        min_ef_construction = max(self.M * 8, 400) if n_samples > 10000 else 200
        if self.ef_construction < min_ef_construction:
            import warnings
            warnings.warn(
                f"ef_construction={self.ef_construction} may be too low "
                f"for {n_samples} samples. Expected recall may be poor. "
                f"For good recall (>95%), use ef_construction >= {min_ef_construction}",
                UserWarning
            )

        if self.backend == 'nmslib':
            # nmslib method space strings: 'l2', 'cosinesimil', 'ip', etc.
            self.index_ = self.nmslib.init(method='hnsw', space=self.space)
            self.index_.addDataPointBatch(X)
            self.index_.createIndex({
                'M': self.M,
                'efConstruction': self.ef_construction,
                'post': 0  # BUG FIX #5: Disable post-processing for consistent behavior
            }, print_progress=False)
            # BUG FIX #5: Set query-time ef parameter
            self.index_.setQueryTimeParams({'ef': self.ef_search})
        else:  # hnswlib
            dim = X.shape[1]
            self.index_ = self.hnswlib.Index(space=self.space, dim=dim)
            self.index_.init_index(max_elements=X.shape[0],
                                   ef_construction=self.ef_construction,
                                   M=self.M)
            self.index_.add_items(X)
            # BUG FIX #5: Make sure ef is set for queries
            self.index_.set_ef(self.ef_search)
        return self

    def kneighbors(self, X, k=None):
        if k is None:
            k = self.n_neighbors

        X = np.atleast_2d(X).astype(np.float32)
        single = (X.shape[0] == 1)

        if self.backend == 'nmslib':
            results = self.index_.knnQueryBatch(X, k=k)
            # results is a list of (indices, distances) tuples
            all_indices = []
            all_distances = []
            for idx, dist in results:
                all_indices.append(np.array(idx))
                all_distances.append(np.array(dist))
            if single:
                return all_distances[0], all_indices[0]
            else:
                return np.array(all_distances), np.array(all_indices)
        else:  # hnswlib
            indices, distances = self.index_.knn_query(X, k=k)
            if single:
                return distances[0], indices[0]
            return distances, indices