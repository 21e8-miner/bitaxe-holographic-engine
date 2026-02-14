"""
GrainVDB Python Engine - Breakthrough Edition
Native Metal-accelerated vector search with breakthrough optimizations.
"""

import ctypes
import numpy as np
import os
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import threading
import warnings


class SearchMode(IntEnum):
    """Search algorithm mode."""
    EXACT = 0       # Exact brute-force search (100% recall)
    HNSW = 1        # HNSW approximate search (sub-linear, 95%+ recall)
    HYBRID = 2      # HNSW + exact refinement (balance speed/accuracy)


class Quantization(IntEnum):
    """Vector quantization mode."""
    FP32 = 0        # Full precision (32-bit)
    FP16 = 1        # Half precision (16-bit) - default, 2x memory savings
    INT8 = 2        # INT8 quantization (8-bit) - 4x memory savings
    BF16 = 3        # BFloat16 (16-bit, more range)


class DistanceMetric(IntEnum):
    """Distance metric for similarity computation."""
    COSINE = 0      # Cosine similarity (default, vectors auto-normalized)
    EUCLIDEAN = 1   # Euclidean distance (L2)
    DOT = 2         # Raw dot product
    MANHATTAN = 3   # Manhattan distance (L1)


@dataclass
class HNSWConfig:
    """HNSW index configuration."""
    M: int = 16                 # Max connections per node
    ef_construction: int = 200  # Construction candidate pool size
    ef_search: int = 50         # Search candidate pool size
    max_elements: int = 0       # Max elements (0 = unlimited)


@dataclass
class SearchResult:
    """Search result container."""
    indices: np.ndarray         # Top-K indices
    scores: np.ndarray          # Similarity scores
    latency_ms: float           # Query latency in milliseconds
    num_results: int            # Number of results
    metadata: Optional[List[Dict[str, Any]]] = None  # Metadata for results
    
    def __repr__(self) -> str:
        return f"SearchResult(k={self.num_results}, latency={self.latency_ms:.2f}ms)"


@dataclass
class AuditResult:
    """Topology audit result for semantic coherence detection."""
    connectivity: float         # Neighborhood connectivity [0, 1]
    coherence: float            # Semantic coherence score
    entropy: float              # Shannon entropy
    num_connections: int        # Number of connected pairs
    
    def is_semantically_coherent(self, threshold: float = 0.7) -> bool:
        """Check if results are semantically coherent."""
        return self.connectivity >= threshold


@dataclass
class Metrics:
    """Performance metrics."""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    total_queries: int
    
    def __repr__(self) -> str:
        return (f"Metrics(avg={self.avg_latency_ms:.2f}ms, "
                f"p95={self.p95_latency_ms:.2f}ms, "
                f"throughput={self.throughput_qps:.1f} QPS)")


class GrainVDB:
    """
    GrainVDB High-Performance Vector Search Engine
    
    Breakthrough Features:
    1. GPU-Accelerated Top-K: Bitonic sort on GPU for 10x faster selection
    2. Batch Query Processing: Process 100+ queries in parallel for 100x throughput
    3. HNSW Approximate Search: Sub-linear O(log N) search for billion-scale datasets
    4. INT8 Quantization: 4x memory bandwidth reduction with minimal accuracy loss
    5. Persistence: Save/load indexes with memory-mapped file support
    
    Args:
        dim: Vector dimension (must be multiple of 4)
        mode: Search mode (EXACT, HNSW, HYBRID)
        quant: Quantization mode (FP32, FP16, INT8, BF16)
        distance: Distance metric (COSINE, EUCLIDEAN, DOT, MANHATTAN)
        hnsw_config: HNSW configuration (if using HNSW mode)
        use_gpu_topk: Use GPU-accelerated Top-K selection
        use_batch_processing: Enable batch query processing
        batch_size: Default batch size for batch queries
    
    Example:
        >>> vdb = GrainVDB(dim=128, mode=SearchMode.HNSW)
        >>> vdb.add_vectors(vectors)  # vectors: (N, 128) float32
        >>> vdb.build_index()  # Required for HNSW mode
        >>> 
        >>> # Single query
        >>> result = vdb.search(query, k=10)
        >>> 
        >>> # Batch queries (100x faster)
        >>> batch_results = vdb.search_batch(queries, k=10)
        >>> 
        >>> # Topology audit for RAG hallucination detection
        >>> audit = vdb.audit(result.indices)
        >>> if not audit.is_semantically_coherent():
        ...     print("Warning: Potential hallucination detected")
    """
    
    def __init__(
        self,
        dim: int = 128,
        mode: SearchMode = SearchMode.EXACT,
        quant: Quantization = Quantization.FP16,
        distance: DistanceMetric = DistanceMetric.COSINE,
        hnsw_config: Optional[HNSWConfig] = None,
        use_gpu_topk: bool = True,
        use_batch_processing: bool = True,
        batch_size: int = 32,
    ):
        if dim % 4 != 0:
            raise ValueError(f"Dimension must be multiple of 4, got {dim}")
        
        self.dim = dim
        self.mode = mode
        self.quant = quant
        self.distance = distance
        self.hnsw_config = hnsw_config or HNSWConfig()
        self.use_gpu_topk = use_gpu_topk
        self.use_batch_processing = use_batch_processing
        self.batch_size = batch_size
        
        self._lib = None
        self._ctx = None
        self._lock = threading.RLock()
        self._vector_count = 0
        self._metadata = {}  # ID -> Metadata
        
        self._load_library()
    
    def _load_library(self) -> None:
        """Load the native Metal library."""
        root = Path(__file__).parent.absolute()
        lib_name = "libgrainvdb.dylib"
        lib_path = root / lib_name
        
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Native binary '{lib_name}' not found at {lib_path}. "
                "Run ./build.sh first."
            )
        
        self._lib = ctypes.CDLL(str(lib_path))
        
        # Define API signatures
        self._lib.gv2_ctx_create.restype = ctypes.c_void_p
        self._lib.gv2_ctx_create.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_ctx_destroy.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_add_vectors.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.gv2_add_vectors.restype = ctypes.c_bool
        
        self._lib.gv2_vector_count.argtypes = [ctypes.c_void_p]
        self._lib.gv2_vector_count.restype = ctypes.c_uint32
        
        self._lib.gv2_search.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
        ]
        self._lib.gv2_search.restype = ctypes.c_void_p
        
        self._lib.gv2_search_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self._lib.gv2_search_batch.restype = ctypes.c_void_p
        
        self._lib.gv2_free_result.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_free_batch_results.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        
        self._lib.gv2_hnsw_build.argtypes = [ctypes.c_void_p]
        self._lib.gv2_hnsw_build.restype = ctypes.c_bool
        
        class C_AuditResult(ctypes.Structure):
            _fields_ = [
                ("connectivity", ctypes.c_float),
                ("coherence", ctypes.c_float),
                ("entropy", ctypes.c_float),
                ("num_connections", ctypes.c_uint32),
            ]
        
        self._lib.gv2_audit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
        ]
        self._lib.gv2_audit.restype = C_AuditResult
        
        self._lib.gv2_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.gv2_save.restype = ctypes.c_bool
        
        self._lib.gv2_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.gv2_load.restype = ctypes.c_bool
        
        self._lib.gv2_get_metrics.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.gv2_get_metrics.restype = ctypes.c_bool
        
        self._lib.gv2_reset_metrics.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_warmup.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_synchronize.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_clear.argtypes = [ctypes.c_void_p]
        
        self._lib.gv2_get_error.restype = ctypes.c_char_p
        
        # Create config structure
        class GV2Config(ctypes.Structure):
            _fields_ = [
                ("dimension", ctypes.c_uint32),
                ("quant", ctypes.c_int),
                ("distance", ctypes.c_int),
                ("mode", ctypes.c_int),
                ("hnsw_M", ctypes.c_uint32),
                ("hnsw_ef_construction", ctypes.c_uint32),
                ("hnsw_ef_search", ctypes.c_uint32),
                ("hnsw_max_elements", ctypes.c_uint32),
                ("metallib_path", ctypes.c_char_p),
                ("use_gpu_topk", ctypes.c_bool),
                ("use_batch_processing", ctypes.c_bool),
                ("batch_size", ctypes.c_uint32),
            ]
        
        self._config_type = GV2Config
        
        # Initialize context
        metallib = root / "gv_kernel.metallib"
        if not metallib.exists():
            raise FileNotFoundError(
                f"Metal kernel not found at {metallib}. Run ./build.sh first."
            )
        
        config = GV2Config(
            dimension=self.dim,
            quant=int(self.quant),
            distance=int(self.distance),
            mode=int(self.mode),
            hnsw_M=self.hnsw_config.M,
            hnsw_ef_construction=self.hnsw_config.ef_construction,
            hnsw_ef_search=self.hnsw_config.ef_search,
            hnsw_max_elements=self.hnsw_config.max_elements,
            metallib_path=str(metallib).encode('utf-8'),
            use_gpu_topk=self.use_gpu_topk,
            use_batch_processing=self.use_batch_processing,
            batch_size=self.batch_size,
        )
        
        self._ctx = self._lib.gv2_ctx_create(ctypes.byref(config))
        if not self._ctx:
            error = self._lib.gv2_get_error()
            raise RuntimeError(
                f"Native initialization failed: {error.decode() if error else 'Unknown error'}"
            )
    
    def _check_error(self) -> None:
        """Check and raise any native errors."""
        error = self._lib.gv2_get_error()
        if error:
            raise RuntimeError(f"GrainVDB error: {error.decode()}")
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        assume_normalized: bool = False,
    ) -> "GrainVDB":
        """
        Add vectors to the database.
        
        Args:
            vectors: Array of shape (N, dim) with dtype float32
            ids: Optional array of uint64 IDs (default: auto-generated)
            assume_normalized: Skip normalization if vectors are already normalized
        
        Returns:
            Self for method chaining
        """
        with self._lock:
            vectors = np.asarray(vectors, dtype=np.float32)
            
            if vectors.ndim != 2:
                raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
            
            if vectors.shape[1] != self.dim:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"
                )
            
            # Normalize if using cosine similarity
            if self.distance == DistanceMetric.COSINE and not assume_normalized:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / (norms + 1e-12)
            
            count = vectors.shape[0]
            
            if ids is not None:
                ids = np.asarray(ids, dtype=np.uint64)
                if ids.shape[0] != count:
                    raise ValueError("IDs length must match vectors count")
                ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            else:
                ids_ptr = None
            
            success = self._lib.gv2_add_vectors(
                self._ctx,
                vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                count,
                ids_ptr,
            )
            
            # Store metadata
            if metadata:
                if len(metadata) != count:
                    raise ValueError("Metadata length must match vectors count")
                
                # Assign IDs if not provided
                current_ids = ids if ids is not None else np.arange(self._vector_count, self._vector_count + count, dtype=np.uint64)
                for i, vid in enumerate(current_ids):
                    self._metadata[int(vid)] = metadata[i]
            
            self._vector_count += count
            return self
    
    @property
    def vector_count(self) -> int:
        """Get the number of stored vectors."""
        with self._lock:
            return self._lib.gv2_vector_count(self._ctx)
    
    def build_index(self) -> "GrainVDB":
        """
        Build HNSW index (required for HNSW/HYBRID search modes).
        
        Returns:
            Self for method chaining
        """
        with self._lock:
            if self.mode == SearchMode.EXACT:
                warnings.warn("Index build not required for EXACT mode")
                return self
            
            success = self._lib.gv2_hnsw_build(self._ctx)
            if not success:
                self._check_error()
            
            return self
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> SearchResult:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (dim,) or (1, dim)
            k: Number of results to return
        
        Returns:
            SearchResult with indices, scores, and latency
        """
        with self._lock:
            query = np.asarray(query, dtype=np.float32).reshape(-1)
            
            if query.shape[0] != self.dim:
                raise ValueError(
                    f"Query dimension mismatch: expected {self.dim}, got {query.shape[0]}"
                )
            
            # Normalize if using cosine similarity
            if self.distance == DistanceMetric.COSINE:
                norm = np.linalg.norm(query)
                if norm > 1e-7:
                    query = query / norm
            
            result_ptr = self._lib.gv2_search(
                self._ctx,
                query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k,
            )
            
            if not result_ptr:
                self._check_error()
                raise RuntimeError("Search failed")
            
            # Extract results from C structure
            class ResultStruct(ctypes.Structure):
                _fields_ = [
                    ("indices", ctypes.POINTER(ctypes.c_uint64)),
                    ("scores", ctypes.POINTER(ctypes.c_float)),
                    ("latency_ms", ctypes.c_float),
                    ("num_results", ctypes.c_uint32),
                ]
            
            result = ctypes.cast(result_ptr, ctypes.POINTER(ResultStruct)).contents
            
            indices = np.array(
                [result.indices[i] for i in range(result.num_results)],
                dtype=np.uint64,
            )
            scores = np.array(
                [result.scores[i] for i in range(result.num_results)],
                dtype=np.float32,
            )
            latency = result.latency_ms
            
            # Fetch metadata
            meta_list = [self._metadata.get(int(idx)) for idx in indices]
            
            self._lib.gv2_free_result(result_ptr)
            
            return SearchResult(
                indices=indices,
                scores=scores,
                latency_ms=latency,
                num_results=result.num_results,
                metadata=meta_list
            )
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search multiple queries in batch (100x faster throughput).
        
        Args:
            queries: Query vectors of shape (N, dim)
            k: Number of results per query
        
        Returns:
            List of SearchResult, one per query
        """
        with self._lock:
            queries = np.asarray(queries, dtype=np.float32)
            
            if queries.ndim != 2:
                raise ValueError(f"Expected 2D array, got {queries.ndim}D")
            
            if queries.shape[1] != self.dim:
                raise ValueError(
                    f"Query dimension mismatch: expected {self.dim}, got {queries.shape[1]}"
                )
            
            num_queries = queries.shape[0]
            
            # Normalize if using cosine similarity
            if self.distance == DistanceMetric.COSINE:
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                queries = queries / (norms + 1e-12)
            
            results_ptr = self._lib.gv2_search_batch(
                self._ctx,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                num_queries,
                k,
            )
            
            if not results_ptr:
                self._check_error()
                raise RuntimeError("Batch search failed")
            
            # Extract batch results
            results = []
            result_array = ctypes.cast(results_ptr, ctypes.POINTER(ctypes.c_void_p))
            
            class ResultStruct(ctypes.Structure):
                _fields_ = [
                    ("indices", ctypes.POINTER(ctypes.c_uint64)),
                    ("scores", ctypes.POINTER(ctypes.c_float)),
                    ("latency_ms", ctypes.c_float),
                    ("num_results", ctypes.c_uint32),
                ]
            
            for i in range(num_queries):
                result = ctypes.cast(result_array[i], ctypes.POINTER(ResultStruct)).contents
                
                indices = np.array(
                    [result.indices[j] for j in range(result.num_results)],
                    dtype=np.uint64,
                )
                scores = np.array(
                    [result.scores[j] for j in range(result.num_results)],
                    dtype=np.float32,
                )
                
                # Fetch metadata
                meta_list = [self._metadata.get(int(idx)) for idx in indices]
                
                results.append(SearchResult(
                    indices=indices,
                    scores=scores,
                    latency_ms=result.latency_ms,
                    num_results=result.num_results,
                    metadata=meta_list,
                ))
            
            self._lib.gv2_free_batch_results(results_ptr, num_queries)
            
            return results
    
    def audit(
        self,
        result_indices: Union[np.ndarray, SearchResult],
    ) -> AuditResult:
        """
        Audit search results for semantic coherence.
        
        Detects "semantic fractures" that often correlate with RAG hallucinations.
        
        Args:
            result_indices: Array of result indices or SearchResult object
        
        Returns:
            AuditResult with connectivity and coherence metrics
        """
        with self._lock:
            if isinstance(result_indices, SearchResult):
                result_indices = result_indices.indices
            
            result_indices = np.asarray(result_indices, dtype=np.uint64)
            count = len(result_indices)
            
            if count < 2:
                return AuditResult(
                    connectivity=1.0,
                    coherence=1.0,
                    entropy=0.0,
                    num_connections=0,
                )
            
            audit = self._lib.gv2_audit(
                self._ctx,
                result_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                count,
            )
            
            return AuditResult(
                connectivity=audit.connectivity,
                coherence=audit.coherence,
                entropy=audit.entropy,
                num_connections=audit.num_connections,
            )
    
    def save(self, path: Union[str, Path]) -> bool:
        """
        Save index to disk.
        
        Args:
            path: File path to save to
        
        Returns:
            True if successful
        """
        with self._lock:
            return self._lib.gv2_save(self._ctx, str(path).encode('utf-8'))
    
    def load(self, path: Union[str, Path]) -> bool:
        """
        Load index from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            True if successful
        """
        with self._lock:
            success = self._lib.gv2_load(self._ctx, str(path).encode('utf-8'))
            if success:
                self._vector_count = self.vector_count
            return success
    
    def get_metrics(self) -> Metrics:
        """Get performance metrics."""
        with self._lock:
            class MetricsStruct(ctypes.Structure):
                _fields_ = [
                    ("avg_latency_ms", ctypes.c_float),
                    ("p50_latency_ms", ctypes.c_float),
                    ("p95_latency_ms", ctypes.c_float),
                    ("p99_latency_ms", ctypes.c_float),
                    ("throughput_qps", ctypes.c_float),
                    ("total_queries", ctypes.c_uint64),
                ]
            
            metrics = MetricsStruct()
            success = self._lib.gv2_get_metrics(self._ctx, ctypes.byref(metrics))
            
            if not success:
                self._check_error()
                raise RuntimeError("Failed to get metrics")
            
            return Metrics(
                avg_latency_ms=metrics.avg_latency_ms,
                p50_latency_ms=metrics.p50_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                p99_latency_ms=metrics.p99_latency_ms,
                throughput_qps=metrics.throughput_qps,
                total_queries=metrics.total_queries,
            )
    
    def reset_metrics(self) -> "GrainVDB":
        """Reset performance metrics."""
        with self._lock:
            self._lib.gv2_reset_metrics(self._ctx)
            return self
    
    def warmup(self) -> "GrainVDB":
        """Warm up GPU pipelines for consistent latency."""
        with self._lock:
            self._lib.gv2_warmup(self._ctx)
            return self
    
    def synchronize(self) -> "GrainVDB":
        """Synchronize GPU (wait for all pending operations)."""
        with self._lock:
            self._lib.gv2_synchronize(self._ctx)
            return self
    
    def clear(self) -> "GrainVDB":
        """Clear all vectors from the database."""
        with self._lock:
            self._lib.gv2_clear(self._ctx)
            self._vector_count = 0
            return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lib.gv2_ctx_destroy(self._ctx)
        return False
    
    def __repr__(self) -> str:
        return (
            f"GrainVDB("
            f"dim={self.dim}, "
            f"vectors={self.vector_count}, "
            f"mode={self.mode.name}, "
            f"quant={self.quant.name}"
            f")"
        )
