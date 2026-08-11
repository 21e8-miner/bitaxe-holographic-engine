"""
PH-REAL-MINER: experimental work-filter stub
=============================================
Research placeholder for "Holographic Veto" experiments.

IMPORTANT: This does NOT improve mining profitability as shipped.
The evaluate_work path is a STUB (random accept). Do not use on a
production pool connection expecting better share rates.

The supported production path is:
  python -m hme doctor
  python -m hme tune --apply --yes
"""

from __future__ import annotations

import random
from typing import Any, Optional

# Optional predictor — never hard-fail import if missing
try:
    from nonce_range_predictor import NonceRangePredictor  # type: ignore
except ImportError:  # pragma: no cover
    NonceRangePredictor = None  # type: ignore


class HolographicWorker:
    """Experimental filter. Default is random 10% accept — for lab use only."""

    def __init__(self, accept_rate: float = 0.10):
        self.predictor = NonceRangePredictor() if NonceRangePredictor else None
        self.accept_rate = float(accept_rate)
        self.total_vetos = 0
        self.total_accepted = 0

    def evaluate_work(self, header_hex: str, target_hex: str) -> bool:
        """
        Return True to hash this job.

        STUB: random accept unless a real NonceRangePredictor is installed.
        """
        if self.predictor is not None:
            try:
                # expected API if you ship a real predictor later
                return bool(self.predictor.should_hash(header_hex, target_hex))  # type: ignore[attr-defined]
            except Exception:
                pass

        if random.random() < self.accept_rate:
            self.total_accepted += 1
            return True
        self.total_vetos += 1
        return False


def stratum_client() -> None:
    """Stub — real Stratum is not implemented in this package yet."""
    print("[Stratum-PH] Stub only. Use AxeOS pool settings + `python -m hme` for ops.")


holographic_worker = HolographicWorker()
