import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import Dict, List, Optional, Tuple
import logging
import zlib

logger = logging.getLogger("QuantumEngine")

class QuantumCoherenceEngine:
    """
    FMTR Quantum Coherence Engine
    =============================
    Calculates market coherence using multi-resolution first-principles wave mechanics.
    Analyzes phase alignment across Fast and Slow windows to detect spectral depth.
    """

    def __init__(self, window_size: int = 20, slow_window_size: int = 60):
        self.window_size = window_size
        self.slow_window_size = slow_window_size
        self.prev_phase = None

    def extract_phase_vector(self, data: np.ndarray) -> np.ndarray:
        """
        Uses Hilbert Transform to extract the instantaneous phase of the flow.
        """
        if len(data) < 4:
            return np.zeros_like(data)
            
        # Zero-mean the data for better Hilbert performance
        detrended = data - np.mean(data)
        analytic_signal = hilbert(detrended)
        phase = np.angle(analytic_signal)
        return phase

    def compute_ncd(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Normalized Compression Distance (proxy for Kolmogorov similarity).
        Used for the 'No-Cloning Filter'.
        """
        # Quantize to improve compression stability
        q1 = np.round(v1 * 1000).astype(np.int16).tobytes()
        q2 = np.round(v2 * 1000).astype(np.int16).tobytes()
        
        c1 = len(zlib.compress(q1))
        c2 = len(zlib.compress(q2))
        c12 = len(zlib.compress(q1 + q2))
        
        return (c12 - min(c1, c2)) / max(c1, c2)

    def compute_coherence(self, flow_components: Dict[str, np.ndarray]) -> Dict:
        """
        Computes Multi-Resolution Coherence and Spectral Interference.
        Analyzes phase alignment across Fast and Slow windows.
        """
        if not flow_components:
            return {
                'q': 0, 
                'spectral_interference': 0, 
                'amplification': 1.0,
                'entropy': 1.0,
                'stability': 0.0
            }

        results_fast = self._compute_window_coherence(flow_components, self.window_size)
        results_slow = self._compute_window_coherence(flow_components, self.slow_window_size)
        
        # Calculate Spectral Interference (Phase Lock between scales)
        # Higher = more aligned across multiple timeframes (Grand Cycle)
        spectral_lock = 0.0
        if results_fast['phases'] and results_slow['phases']:
            lock_sum = 0j
            count = 0
            for k in results_fast['phases']:
                if k in results_slow['phases']:
                    p_fast = results_fast['phases'][k]
                    p_slow = results_slow['phases'][k]
                    lock_sum += np.exp(1j * (p_fast - p_slow))
                    count += 1
            if count > 0:
                spectral_lock = float(np.abs(lock_sum) / count)

        # Merge results, prioritizing the fast window for real-time response
        results = dict(results_fast)
        results['q_slow'] = results_slow['q']
        results['spectral_interference'] = spectral_lock
        
        # Multi-Resolution Q (weighted combination)
        results['multi_q'] = (results_fast['q'] * 0.7) + (results_slow['q'] * 0.3)
        
        # √N Amplification Factor based on fast Q
        num_components = len(flow_components)
        results['amplification'] = 1.0 + (results_fast['q'] * (np.sqrt(num_components) - 1))
        
        # 4. Integrate No-Cloning Integrity (Information Diversity)
        # If components are too similar (clones), penalize the manifold strength
        cloning_penalty = 1.0
        if num_components > 1:
            ncd_sum = 0
            comparisons = 0
            keys = list(flow_components.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ncd = self.compute_ncd(flow_components[keys[i]], flow_components[keys[j]])
                    ncd_sum += ncd
                    comparisons += 1
            
            avg_ncd = ncd_sum / comparisons if comparisons > 0 else 1.0
            # Low NCD = High Cloning = Bad
            cloning_penalty = min(1.0, avg_ncd * 1.2) 
            
        results['no_cloning_integrity'] = cloning_penalty
        results['q'] = results['q'] * cloning_penalty
        results['multi_q'] = results['multi_q'] * cloning_penalty
        
        return results

    def _compute_window_coherence(self, flow_components: Dict[str, np.ndarray], window: int) -> Dict:
        """Helper to calculate coherence for a specific window size"""
        complex_sum = 0j
        total_weight = 0.0
        phases = {}

        for name, vector in flow_components.items():
            if len(vector) < window:
                continue
                
            # Get phase of the recent window
            vec_slice = vector[-window:]
            phi = self.extract_phase_vector(vec_slice)[-1]
            
            # Weight based on recent energy (amplitude)
            weight = np.std(vec_slice) + 1e-9
            
            # Map to complex wave: ψ = A * e^(iφ)
            complex_sum += weight * np.exp(1j * phi)
            total_weight += weight
            phases[name] = float(phi)

        if total_weight == 0:
            return {'q': 0, 'phases': {}, 'stability': 0.0, 'entropy': 1.0}

        q_score = abs(complex_sum) / total_weight
        
        # Stability/Entropy
        stability = 1.0
        if len(phases) > 1:
            phase_list = list(phases.values())
            phase_diffs = np.diff(phase_list)
            stability = float(np.abs(np.mean(np.exp(1j * phase_diffs))))
            
        return {
            'q': float(q_score),
            'phases': phases,
            'stability': stability,
            'entropy': 1.0 - stability
        }

    def analyze_flow_state(self, df: pd.DataFrame) -> Dict:
        """
        Analyzes OHLCV dataframe using Multi-Resolution Quantum principles.
        """
        if len(df) < self.slow_window_size + 5:
            return {'q': 0, 'status': 'insufficient_data'}

        # Decompose OHLCV into virtual 'particles'
        flow_components = {
            'momentum': df['Close'].diff().fillna(0).values,
            'volatility': (df['High'] - df['Low']).values,
            'velocity': df['Volume'].values,
            'acceleration': df['Close'].diff().diff().fillna(0).values
        }

        results = self.compute_coherence(flow_components)
        
        # Enhanced Regime detection using Multi-Resolution Q and Interference
        q = results['multi_q']
        lock = results['spectral_interference']
        
        if q > 0.8 and lock > 0.8:
            results['regime'] = "GRAND_CYCLE_COHERENCE" # Solid multi-scale trend
        elif q > 0.7:
            results['regime'] = "FAST_SCALE_FLOW" if lock < 0.5 else "COHERENT_ACCUMULATION"
        elif q > 0.4:
            results['regime'] = "TRANSITION"
        else:
            results['regime'] = "TURBULENT_NOISE"

        # Map clustering_score for backward compatibility
        results['clustering_score'] = float(results['q'] * 100)
        
        return results

if __name__ == "__main__":
    # Quick Test
    engine = QuantumCoherenceEngine()
    print("Testing Multi-Resolution Spectral Analysis...")
    
    # Generate complex dual-frequency signal
    t = np.linspace(0, 10, 200)
    data1 = np.sin(t) + 0.5 * np.sin(3*t) # Fast + Slow components
    data2 = np.sin(t + 0.1) + 0.5 * np.sin(3*t + 0.2)
    
    res = engine.compute_coherence({'v1': data1, 'v2': data2})
    print(f"Spectral Lock: {res['spectral_interference']:.4f}")
    print(f"Multi-Q: {res['multi_q']:.4f}, Fast-Q: {res['q']:.4f}")
