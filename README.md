# Φ Holographic Mining Engine (HCE)

A high-performance standalone framework for Bitcoin mining optimization, specializing in **Aperiodic Scheduling** and **Telemetry-Driven Hardware Feedback**.

## Overview
The Holographic Mining Engine (HCE) is a **Sidecar** control framework for Bitaxe hardware. It provides a technical layer for telemetry aggregation, hardware-aware frequency scheduling, and aperiodic polling to minimize system resonance.

## Key Capabilities
- **Aperiodic Scheduling**: Implements irrational-offset polling (based on the Golden Ratio) to minimize beat-frequency interference with system and device loops.
- **Predictive Prioritization**: Uses spectral analysis to re-rank / prioritize work packages from a Stratum stream without discarding search coverage (maintaining 100% correctness).
- **Consolidated Observability**: Integrated 3D visualization of high-dimensional telemetry (Temp, Power, Hashrate) to assist in identifying stability drifts.
- **Dynamic Feedback Loop**: Monitors J/TH efficiency in real-time and manages frequency/voltage envelopes based on user-defined constraints.

## Operational Safety & Guardrails
HCE is built with explicit safety constraints to prevent hardware degradation:
- **Voltage/Frequency Bounds**: Hard caps on core voltage (1350mV max) and frequency (600MHz max).
- **Thermal Ceiling**: Automatic system rollback to a "Last Known Good" configuration if core temperature exceeds 80°C.
- **Damped Control**: Rate-limited adjustments to prevent "hunting" or destabilizing oscillations during autonomous tuning.

## Telemetry & Measurement Integrity
- **Measurement Methodology**: Efficiency results are derived from board-level power sensors and hashrate is averaged over a sliding 2-minute window.
- **Stale Share Accounting**: Real-time tracking of stale vs. accepted shares to ensure efficiency gains aren't offset by network latency or communication overhead.

## Quick Start (Standalone)

### 1. Prerequisites
Python 3.10+ and a connected Bitaxe (Gamma/Ultra).

### 2. Installation
```bash
cd holographic_mining_engine
pip install -r requirements.txt
```

### 3. Launch
```bash
export BITAXE_IP="192.168.0.23"
python core/bitaxe_holographic_engine.py
```
Open `http://localhost:5033` to view the diagnostic dashboard.

## Documentation
- **[ROADMAP 2026](./ROADMAP.md)**: Current strategy for fleet expansion and enterprise ASIC support.
- **[Engineering Review](./docs/ENGINEERING_REVIEW.md)**: Design audit and implementation analysis (Draft v2).
- **[Hardware Architecture](./docs/HARDWARE_ARCHITECTURE.md)**: Technical specifications for the HCE control stack.

## Strategic Vision
HCE is evolving from a single-device sidecar to a **Universal Fleet Manager**. Our core milestones now include:
- **Historical Auditing**: SQLite-backed efficiency tracking with integrated trend visualization.
- **Enterprise Support**: Native API bridges for Antminer S19/S21 (now active) and Whatsminer series.
- **Protocol Evolution**: Native Stratum V2 integration and industrial Prometheus/Grafana monitoring.
