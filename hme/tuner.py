"""
Safe V/F tuner for Bitaxe AxeOS boards.

Control loop:
  measure baseline → propose step → gate → apply (or dry-run) → dwell → remeasure
  → accept if score improves without regression → else rollback
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .client import ApplyResult, BitaxeClient, BitaxeError
from .config import HMEConfig
from .logger import TelemetryLog
from .units import NormalizedMetrics, score_efficiency

log = logging.getLogger("hme.tuner")


@dataclass
class SampleWindow:
    samples: List[NormalizedMetrics] = field(default_factory=list)

    def add(self, m: NormalizedMetrics) -> None:
        self.samples.append(m)

    def mean(self) -> Optional[NormalizedMetrics]:
        if not self.samples:
            return None
        n = len(self.samples)
        ghs = sum(s.hashrate_ghs for s in self.samples) / n
        power = sum(s.power_w for s in self.samples) / n
        temp = sum(s.temp_c for s in self.samples) / n
        jth = None
        jths = [s.j_per_th for s in self.samples if s.j_per_th is not None]
        if jths:
            jth = sum(jths) / len(jths)
        freq = next((s.frequency_mhz for s in reversed(self.samples) if s.frequency_mhz is not None), None)
        volt = next((s.voltage_mv for s in reversed(self.samples) if s.voltage_mv is not None), None)
        last = self.samples[-1]
        return NormalizedMetrics(
            hashrate_ghs=ghs,
            hashrate_ths=ghs / 1000.0,
            power_w=power,
            temp_c=temp,
            frequency_mhz=freq,
            voltage_mv=volt,
            j_per_th=jth,
            raw_hashrate=last.raw_hashrate,
            hashrate_unit_assumed=last.hashrate_unit_assumed,
        )


@dataclass
class Candidate:
    frequency_mhz: int
    voltage_mv: Optional[int] = None

    def label(self) -> str:
        if self.voltage_mv is not None:
            return f"{self.frequency_mhz}MHz@{self.voltage_mv}mV"
        return f"{self.frequency_mhz}MHz"


@dataclass
class StepResult:
    candidate: Candidate
    accepted: bool
    dry_run: bool
    reason: str
    baseline_score: float
    candidate_score: Optional[float]
    baseline: Optional[NormalizedMetrics]
    measured: Optional[NormalizedMetrics]
    applied: bool


class SafeTuner:
    def __init__(self, cfg: HMEConfig, client: Optional[BitaxeClient] = None, tlog: Optional[TelemetryLog] = None):
        self.cfg = cfg
        self.client = client or BitaxeClient(cfg)
        self.tlog = tlog or TelemetryLog(cfg)
        self._last_apply_ts = 0.0
        self.baseline_profile: Dict[str, Any] = {}

    # ── measurement ───────────────────────────────────────────

    def poll(self) -> NormalizedMetrics:
        m = self.client.metrics()
        self.tlog.sample(m, extra={"ip": self.cfg.device.ip})
        return m

    def measure_window(self, seconds: float, label: str = "measure") -> SampleWindow:
        """Poll for `seconds`, abort early on hard gate failure or zero hashrate streak."""
        win = SampleWindow()
        t0 = time.time()
        zero_since: Optional[float] = None
        poll = max(1.0, float(self.cfg.tuner.poll_sec))
        log.info("Measuring %-10s for %.0fs (poll=%.1fs)", label, seconds, poll)

        while time.time() - t0 < seconds:
            try:
                m = self.poll()
            except BitaxeError as e:
                self.tlog.event("poll_error", error=str(e), label=label)
                log.warning("poll error: %s", e)
                time.sleep(poll)
                continue

            win.add(m)
            ok, reason = self.client.gate_ok(m)
            if not ok:
                self.tlog.event("gate_fail", reason=reason, label=label, temp=m.temp_c, power=m.power_w)
                log.error("GATE FAIL during %s: %s", label, reason)
                raise BitaxeError(f"gate fail: {reason}")

            if m.hashrate_ghs <= 1.0:
                if zero_since is None:
                    zero_since = time.time()
                elif time.time() - zero_since >= self.cfg.tuner.zero_hash_abort_sec:
                    raise BitaxeError(
                        f"hashrate ~0 for ≥{self.cfg.tuner.zero_hash_abort_sec}s during {label}"
                    )
            else:
                zero_since = None

            remaining = seconds - (time.time() - t0)
            time.sleep(min(poll, max(0.2, remaining)))

        if len(win.samples) < max(1, self.cfg.tuner.good_samples):
            raise BitaxeError(f"insufficient samples during {label}: {len(win.samples)}")
        return win

    # ── candidates ────────────────────────────────────────────

    def _current_freq(self, m: NormalizedMetrics) -> int:
        if m.frequency_mhz is not None:
            return int(m.frequency_mhz)
        return int(self.cfg.bounds.base_freq_mhz)

    def generate_candidates(self, current: NormalizedMetrics) -> List[Candidate]:
        b = self.cfg.bounds
        t = self.cfg.tuner
        cur_f = self._current_freq(current)
        cur_v = int(current.voltage_mv) if current.voltage_mv is not None else None
        step = max(5, int(t.freq_step_mhz))
        cands: List[Candidate] = []

        if t.mode == "grid":
            f = b.min_freq_mhz
            while f <= b.max_freq_mhz:
                if not t.voltage_steps_mv:
                    cands.append(Candidate(f, None))
                else:
                    for v in t.voltage_steps_mv:
                        cands.append(Candidate(f, int(v)))
                f += step
        else:
            # climb: try +step, then -step, optional voltage nudges
            for df in (step, -step, 2 * step, -2 * step):
                f = max(b.min_freq_mhz, min(b.max_freq_mhz, cur_f + df))
                cands.append(Candidate(f, None))
            if t.voltage_steps_mv and cur_v is not None:
                for v in t.voltage_steps_mv:
                    vv = max(b.min_voltage_mv, min(b.max_voltage_mv, int(v)))
                    cands.append(Candidate(cur_f, vv))
                    cands.append(Candidate(max(b.min_freq_mhz, min(b.max_freq_mhz, cur_f + step)), vv))

        # unique preserve order, skip identity
        seen = set()
        out: List[Candidate] = []
        for c in cands:
            key = (c.frequency_mhz, c.voltage_mv)
            if key in seen:
                continue
            if c.frequency_mhz == cur_f and (c.voltage_mv is None or c.voltage_mv == cur_v):
                continue
            seen.add(key)
            out.append(c)
        return out[: max(1, t.max_steps * 3)]

    # ── accept / reject ───────────────────────────────────────

    def _score(self, m: NormalizedMetrics) -> float:
        return score_efficiency(m, temp_soft=self.cfg.bounds.warn_temp_c)

    def evaluate(
        self,
        baseline: NormalizedMetrics,
        candidate: NormalizedMetrics,
    ) -> Tuple[bool, str, float, float]:
        """Return (accept, reason, base_score, cand_score)."""
        t = self.cfg.tuner
        bs = self._score(baseline)
        cs = self._score(candidate)

        ok, reason = self.client.gate_ok(candidate)
        if not ok:
            return False, f"gate: {reason}", bs, cs

        if baseline.j_per_th and candidate.j_per_th:
            if candidate.j_per_th > baseline.j_per_th * (1.0 + t.max_jth_regression):
                return False, (
                    f"J/TH regression {baseline.j_per_th:.2f}→{candidate.j_per_th:.2f} "
                    f"(>{t.max_jth_regression:.0%})"
                ), bs, cs

        if baseline.hashrate_ghs > 0:
            drop = (baseline.hashrate_ghs - candidate.hashrate_ghs) / baseline.hashrate_ghs
            if drop > t.max_hashrate_drop:
                return False, (
                    f"hashrate drop {drop:.1%} > max {t.max_hashrate_drop:.0%}"
                ), bs, cs

        if cs <= bs:
            return False, f"score {cs:.2f} ≤ baseline {bs:.2f}", bs, cs

        return True, f"score {bs:.2f}→{cs:.2f} improved", bs, cs

    # ── apply + rollback ──────────────────────────────────────

    def _rate_limit_ok(self) -> Tuple[bool, str]:
        gap = time.time() - self._last_apply_ts
        need = float(self.cfg.tuner.min_change_interval_sec)
        if self._last_apply_ts > 0 and gap < need:
            return False, f"rate limit: {need - gap:.0f}s remaining"
        return True, "ok"

    def apply_candidate(self, c: Candidate, dry_run: bool) -> ApplyResult:
        if not dry_run:
            ok, reason = self._rate_limit_ok()
            if not ok:
                return ApplyResult(False, False, {}, False, reason)

        res = self.client.apply_vf(
            frequency=c.frequency_mhz,
            core_voltage=c.voltage_mv,
            dry_run=dry_run,
            force_restart=False,  # safe path never restarts
        )
        if res.ok and not dry_run and not res.dry_run:
            self._last_apply_ts = time.time()
        self.tlog.event(
            "apply",
            candidate=c.label(),
            ok=res.ok,
            dry_run=res.dry_run,
            payload=res.payload,
            message=res.message,
        )
        return res

    def rollback(self, dry_run: bool) -> ApplyResult:
        prof = self.baseline_profile or self.client.safe_profile()
        freq = prof.get("frequency")
        volt = prof.get("coreVoltage")
        log.warning("ROLLBACK → freq=%s volt=%s dry_run=%s", freq, volt, dry_run)
        self.tlog.event("rollback", frequency=freq, voltage=volt, dry_run=dry_run)
        # bypass rate limit for safety rollback
        prev = self._last_apply_ts
        self._last_apply_ts = 0.0
        res = self.client.apply_vf(
            frequency=int(freq) if freq is not None else None,
            core_voltage=int(volt) if volt is not None else None,
            dry_run=dry_run,
            force_restart=False,
        )
        if not res.ok:
            self._last_apply_ts = prev
        return res

    # ── main run ──────────────────────────────────────────────

    def run(
        self,
        *,
        dry_run: Optional[bool] = None,
        max_steps: Optional[int] = None,
        baseline_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        dry = self.cfg.tuner.dry_run if dry_run is None else dry_run
        steps = max_steps if max_steps is not None else self.cfg.tuner.max_steps
        base_sec = baseline_sec if baseline_sec is not None else max(30.0, float(self.cfg.tuner.dwell_sec) * 0.5)
        dwell = float(self.cfg.tuner.dwell_sec)

        self.tlog.event(
            "tuner_start",
            dry_run=dry,
            max_steps=steps,
            mode=self.cfg.tuner.mode,
            ip=self.cfg.device.ip,
        )
        log.info(
            "SafeTuner start dry_run=%s mode=%s max_steps=%s bounds=%s–%s MHz temp≤%.0f°C",
            dry, self.cfg.tuner.mode, steps,
            self.cfg.bounds.min_freq_mhz, self.cfg.bounds.max_freq_mhz,
            self.cfg.bounds.max_temp_c,
        )

        # Baseline
        base_win = self.measure_window(base_sec, "baseline")
        baseline = base_win.mean()
        assert baseline is not None
        self.baseline_profile = {
            "frequency": int(baseline.frequency_mhz or self.cfg.bounds.base_freq_mhz),
        }
        if baseline.voltage_mv is not None:
            self.baseline_profile["coreVoltage"] = int(baseline.voltage_mv)

        base_score = self._score(baseline)
        log.info(
            "Baseline: %.1f GH/s | %.2f W | %.1f°C | %.2f J/TH | score=%.2f | %s",
            baseline.hashrate_ghs, baseline.power_w, baseline.temp_c,
            baseline.j_per_th or -1, base_score,
            f"{baseline.frequency_mhz}MHz",
        )
        self.tlog.event("baseline", metrics=_mdict(baseline), score=base_score, profile=self.baseline_profile)

        results: List[StepResult] = []
        best = baseline
        best_score = base_score
        best_profile = dict(self.baseline_profile)
        applied_accepts = 0

        candidates = self.generate_candidates(baseline)
        log.info("Candidates (%d): %s", len(candidates), ", ".join(c.label() for c in candidates[:12]))

        for i, cand in enumerate(candidates):
            if applied_accepts >= steps:
                break
            log.info("── Step %d/%d: try %s ──", i + 1, len(candidates), cand.label())

            # propose only when dry_run; still measure path is skippable
            apply_res = self.apply_candidate(cand, dry_run=dry)
            if not apply_res.ok:
                results.append(StepResult(
                    cand, False, dry, apply_res.message, base_score, None, baseline, None, False,
                ))
                log.warning("Skip %s: %s", cand.label(), apply_res.message)
                continue

            if dry:
                # Simulate accept by score heuristic without hardware change:
                # we cannot measure — log proposal only
                results.append(StepResult(
                    cand, False, True,
                    f"proposed {apply_res.payload} (dry-run; not applied)",
                    base_score, None, baseline, None, False,
                ))
                self.tlog.event("dry_run_proposal", candidate=cand.label(), payload=apply_res.payload)
                continue

            # Live path: dwell + measure
            try:
                # short settle then full dwell window
                time.sleep(min(10.0, dwell * 0.1))
                win = self.measure_window(dwell, f"dwell:{cand.label()}")
                measured = win.mean()
                assert measured is not None
            except BitaxeError as e:
                log.error("Measure failed after apply %s: %s — rolling back", cand.label(), e)
                self.rollback(dry_run=False)
                results.append(StepResult(
                    cand, False, False, f"measure fail: {e}", base_score, None, baseline, None, True,
                ))
                continue

            accept, reason, bs, cs = self.evaluate(best, measured)
            log.info(
                "Result %s: %s | %.1f GH/s %.2f J/TH score=%.2f | %s",
                cand.label(), "ACCEPT" if accept else "REJECT",
                measured.hashrate_ghs, measured.j_per_th or -1, cs, reason,
            )
            self.tlog.event(
                "evaluate",
                candidate=cand.label(),
                accept=accept,
                reason=reason,
                baseline_score=bs,
                candidate_score=cs,
                metrics=_mdict(measured),
            )

            if accept:
                best = measured
                best_score = cs
                best_profile = {
                    "frequency": int(measured.frequency_mhz or cand.frequency_mhz),
                }
                if measured.voltage_mv is not None:
                    best_profile["coreVoltage"] = int(measured.voltage_mv)
                applied_accepts += 1
                results.append(StepResult(
                    cand, True, False, reason, bs, cs, baseline, measured, True,
                ))
            else:
                # rollback to best accepted so far
                rb = self.client.apply_vf(
                    frequency=best_profile.get("frequency"),
                    core_voltage=best_profile.get("coreVoltage"),
                    dry_run=False,
                    force_restart=False,
                )
                self.tlog.event("reject_rollback", candidate=cand.label(), ok=rb.ok, message=rb.message)
                results.append(StepResult(
                    cand, False, False, reason, bs, cs, baseline, measured, True,
                ))
                # respect rate limit after bounce
                time.sleep(2)

        summary = {
            "dry_run": dry,
            "baseline": _mdict(baseline),
            "baseline_score": base_score,
            "best": _mdict(best),
            "best_score": best_score,
            "best_profile": best_profile,
            "steps": [
                {
                    "candidate": r.candidate.label(),
                    "accepted": r.accepted,
                    "dry_run": r.dry_run,
                    "reason": r.reason,
                    "candidate_score": r.candidate_score,
                    "applied": r.applied,
                    "measured": _mdict(r.measured) if r.measured else None,
                }
                for r in results
            ],
            "accepted_count": sum(1 for r in results if r.accepted),
            "proposal_count": len(results),
        }
        self.tlog.event("tuner_end", **{k: summary[k] for k in ("dry_run", "best_score", "accepted_count", "best_profile")})
        return summary


def _mdict(m: Optional[NormalizedMetrics]) -> Optional[Dict[str, Any]]:
    if m is None:
        return None
    return {
        "hashrate_ghs": round(m.hashrate_ghs, 3),
        "hashrate_ths": round(m.hashrate_ths, 6),
        "power_w": round(m.power_w, 3),
        "temp_c": round(m.temp_c, 2),
        "frequency_mhz": m.frequency_mhz,
        "voltage_mv": m.voltage_mv,
        "j_per_th": None if m.j_per_th is None else round(m.j_per_th, 3),
        "hashrate_unit": m.hashrate_unit_assumed,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("Safe tuner summary")
    print("=" * 60)
    print(f"dry_run   : {summary.get('dry_run')}")
    b = summary.get("baseline") or {}
    best = summary.get("best") or {}
    print(f"baseline  : {b.get('hashrate_ghs')} GH/s | {b.get('j_per_th')} J/TH | "
          f"{b.get('temp_c')}°C | {b.get('frequency_mhz')} MHz | score={summary.get('baseline_score')}")
    print(f"best      : {best.get('hashrate_ghs')} GH/s | {best.get('j_per_th')} J/TH | "
          f"{best.get('temp_c')}°C | profile={summary.get('best_profile')} | score={summary.get('best_score')}")
    print(f"proposals : {summary.get('proposal_count')}  accepted: {summary.get('accepted_count')}")
    print()
    for s in summary.get("steps") or []:
        mark = "✓" if s.get("accepted") else ("·" if s.get("dry_run") else "✗")
        print(f"  [{mark}] {s.get('candidate')}: {s.get('reason')}")
    print("=" * 60)
    if summary.get("dry_run"):
        print("No hardware changes (dry-run). Re-run with: python -m hme tune --apply")
    print()
