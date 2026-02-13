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
    n_cells : int, default=100
        Number of Voronoi cells for IVF index.
    n_probes : int, default=1
        Number of cells to probe during search (IVF only).
    hnsw_M : int, default=32
        Number of neighbors per node in HNSW graph.
    hnsw_efConstruction : int, default=40
        Construction time accuracy/speed tradeoff for HNSW.
    """
    def __init__(self, k=10, index_type='flat', n_cells=100, n_probes=1,
                 hnsw_M=32, hnsw_efConstruction=40):
        self.n_neighbors = k
        self.index_type = index_type
        self.n_cells = n_cells
        self.n_probes = n_probes
        self.hnsw_M = hnsw_M
        self.hnsw_efConstruction = hnsw_efConstruction
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

        if self.index_type == 'flat':
            self.index_ = self.faiss.IndexFlatL2(dim)
        elif self.index_type == 'ivf':
            quantizer = self.faiss.IndexFlatL2(dim)
            self.index_ = self.faiss.IndexIVFFlat(quantizer, dim, self.n_cells)
            # IVF requires training
            self.index_.train(X)
            self.index_.nprobe = self.n_probes
        elif self.index_type == 'hnsw':
            self.index_ = self.faiss.IndexHNSWFlat(dim, self.hnsw_M)
            self.index_.hnsw.efConstruction = self.hnsw_efConstruction
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
    n_trees : int, default=10
        Number of trees in the forest. More trees = higher accuracy, larger index.
    metric : str, default='euclidean'
        Distance metric. Options: 'euclidean', 'angular', 'manhattan', 'hamming', 'dot'.
    search_k : int, default=-1
        Number of nodes to inspect during search. -1 means use default (n_trees * n_neighbors).
    """
    def __init__(self, k=10, n_trees=10, metric='euclidean', search_k=-1):
        self.k = k
        self.n_trees = n_trees
        self.metric = metric
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
        print(f"Annoy index built with {self.index_.get_n_items()} items")  # DEBUG
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
            print(f"Query: {vec}, indices: {idx} (len={len(idx)}), distances: {dist}")  # DEBUG
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
    M : int, default=16
        Number of bi-directional links per element.
    ef_construction : int, default=200
        Size of the dynamic list for the nearest neighbors during index construction.
    ef_search : int, default=50
        Size of the dynamic list during search (higher = more accurate, slower).
    backend : str, default='nmslib'
        Which library to use: 'nmslib' or 'hnswlib'.
    """
    def __init__(self, k=10, space='l2', M=16, ef_construction=200,
                 ef_search=50, backend='nmslib'):
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
        if self.backend == 'nmslib':
            # nmslib method space strings: 'l2', 'cosinesimil', 'ip', etc.
            self.index_ = self.nmslib.init(method='hnsw', space=self.space)
            self.index_.addDataPointBatch(X)
            self.index_.createIndex({
                'M': self.M,
                'efConstruction': self.ef_construction
            }, print_progress=False)
        else:  # hnswlib
            dim = X.shape[1]
            self.index_ = self.hnswlib.Index(space=self.space, dim=dim)
            self.index_.init_index(max_elements=X.shape[0],
                                   ef_construction=self.ef_construction,
                                   M=self.M)
            self.index_.add_items(X)
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