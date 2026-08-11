"""Rolling share deltas for share-truth hashrate estimates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .units import NormalizedMetrics, merge_share_delta


@dataclass
class ShareSample:
    ts: float
    accepted: int
    rejected: int
    stratum_diff: Optional[float]
    api_ghs: float


@dataclass
class ShareTruthTracker:
    """Keep short history of cumulative share counters from AxeOS."""

    max_points: int = 120
    history: List[ShareSample] = field(default_factory=list)

    def push(self, m: NormalizedMetrics, info: Optional[Dict[str, Any]] = None) -> None:
        if m.shares_accepted is None:
            return
        diff = None
        if info is not None:
            try:
                diff = float(info.get("stratumDiff")) if info.get("stratumDiff") is not None else None
            except (TypeError, ValueError):
                diff = None
        self.history.append(ShareSample(
            ts=time.time(),
            accepted=int(m.shares_accepted),
            rejected=int(m.shares_rejected or 0),
            stratum_diff=diff,
            api_ghs=m.hashrate_ghs,
        ))
        if len(self.history) > self.max_points:
            self.history = self.history[-self.max_points:]

    def window(self, seconds: float = 300.0) -> Optional[Dict[str, Any]]:
        if len(self.history) < 2:
            return None
        latest = self.history[-1]
        target = latest.ts - seconds
        older = None
        for s in self.history:
            if s.ts <= target:
                older = s
        if older is None:
            older = self.history[0]
        elapsed = latest.ts - older.ts
        if elapsed < 5:
            return None
        d_acc = max(0, latest.accepted - older.accepted)
        d_rej = max(0, latest.rejected - older.rejected)
        # counter reset detection
        if latest.accepted < older.accepted:
            return {"error": "share counter reset", "elapsed_sec": elapsed}
        out = merge_share_delta(
            NormalizedMetrics(0, 0, 0, 0, None, None, None, 0, "ghs"),
            d_acc=d_acc,
            d_rej=d_rej,
            elapsed_sec=elapsed,
            share_diff=latest.stratum_diff or older.stratum_diff,
        )
        out["api_ghs_latest"] = latest.api_ghs
        out["window_sec"] = elapsed
        # ghost hashrate: API much higher than share estimate
        est = out.get("share_hr_ghs_est")
        if est is not None and est > 0 and latest.api_ghs > 0:
            out["api_vs_share_ratio"] = latest.api_ghs / est
            out["ghost_hashrate"] = latest.api_ghs > est * 1.35
        return out
