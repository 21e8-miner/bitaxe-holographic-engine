import os
import numpy as np
from typing import List, Dict, Optional
try:
    from .grainvdb import GrainVDB as NativeGrainVDB, SearchMode, DistanceMetric
except ImportError:
    # Handle if run as a script
    from grainvdb import GrainVDB as NativeGrainVDB, SearchMode, DistanceMetric

class GrainVDB:
    """
    Grain Vector Database for Physics Pattern Matching.
    Powered by Native Metal Acceleration (v2.0 Breakthrough Edition).
    """
    def __init__(self, embed_dim: int = 128, max_size: int = 5000, storage_dir: str = "grain_vdb_store"):
        # Note: dim must be multiple of 4 for Metal optimizations. 
        # If user passes 256 or 128, it's fine.
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.storage_dir = storage_dir
        
        self.engine = NativeGrainVDB(
            dim=embed_dim,
            mode=SearchMode.EXACT,
            distance=DistanceMetric.COSINE,
            use_gpu_topk=False # Use stable CPU PQ for now until bitonic sort is fully validated
        )
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def add(self, vector: np.ndarray, meta: Dict):
        """Add a vector and metadata to the database."""
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self.embed_dim:
            # Handle potential dimension mismatch if necessary
            # For now, assume it matches or raise error
            pass
            
        # The native engine handles normalization if COSINE is set
        self.engine.add_vectors(
            vectors=vector.reshape(1, -1),
            metadata=[meta]
        )
        
    def save(self):
        """Save database to disk."""
        path = os.path.join(self.storage_dir, "db.bin")
        self.engine.save(path)
        # We also need to save metadata as the native save/load current binary format 
        # might not include the Python metadata dict.
        import pickle
        with open(os.path.join(self.storage_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.engine._metadata, f)
            
    def load(self):
        """Load database from disk."""
        path = os.path.join(self.storage_dir, "db.bin")
        if os.path.exists(path):
            self.engine.load(path)
            meta_path = os.path.join(self.storage_dir, "metadata.pkl")
            if os.path.exists(meta_path):
                import pickle
                with open(meta_path, "rb") as f:
                    self.engine._metadata = pickle.load(f)
                
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for nearest neighbors using native Metal acceleration."""
        if self.engine.vector_count == 0:
            return []
            
        result = self.engine.search(query_vector, k=k)
        
        results = []
        for i in range(len(result.indices)):
            results.append({
                "score": float(result.scores[i]),
                "metadata": result.metadata[i]
            })
            
        return results
