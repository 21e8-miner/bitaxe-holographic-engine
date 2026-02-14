"""
GrainVDB v2.0 - Breakthrough Edition
High-Performance Vector Search Engine for Apple Silicon

Breakthrough Features:
- GPU-accelerated Top-K selection (10x faster)
- Batch query processing (100x throughput)
- HNSW approximate search (sub-linear scaling)
- INT8 quantization (4x memory reduction)
- Persistence with mmap support

Example:
    >>> from grainvdb import GrainVDB, SearchMode, Quantization
    >>> vdb = GrainVDB(dim=128, mode=SearchMode.HNSW, quant=Quantization.FP16)
    >>> vdb.add_vectors(vectors)
    >>> results = vdb.search(query, k=10)
    >>> batch_results = vdb.search_batch(queries, k=10)  # 100x faster
"""

from .engine import (
    GrainVDB,
    SearchMode,
    Quantization,
    DistanceMetric,
    HNSWConfig,
    SearchResult,
    AuditResult,
    Metrics,
)

__version__ = "2.0.0"
__author__ = "GrainVDB Team"

__all__ = [
    "GrainVDB",
    "SearchMode",
    "Quantization",
    "DistanceMetric",
    "HNSWConfig",
    "SearchResult",
    "AuditResult",
    "Metrics",
]
