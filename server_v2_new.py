import os
# QC: Fix Protobuf conflict for real-time WebSocket MUST BE AT TOP
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import flask
import csv
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
from datetime import datetime, timedelta
import concurrent.futures
import statistics
import asyncio
import websockets
import json

# Internal FMTR imports (Lazy loaded below)
run_parallel_fetch = None

# Lazy loading to prevent startup hangs on heavy modules
np = None
yf = None
pd = None
torch = None
requests = None
ccxt = None

def lazy_imports():
    global np, yf, pd, torch, requests, ccxt, get_live_price, get_provider_status, get_history, MULTI_PROVIDER_AVAILABLE, run_parallel_fetch
    global PhysicsOrderFlowEngine, PHYSICS_ENGINE_AVAILABLE, physics_engine
    global tactical_reasoner, TACTICAL_REASONER_AVAILABLE, get_esp32_bridge, HW_BRIDGE_AVAILABLE
    if np is not None: return # Already loaded
    
    # We use a lock to prevent multi-thread import races
    # (Actually logging is already setup by now)
    print("Initializing Heavy Modules (Lazy Load)...")
    
    try:
        from high_speed_aggregator import run_parallel_fetch as rpf
        run_parallel_fetch = rpf
    except Exception as e:
        print(f"High-Speed Aggregator Load Failed: {e}")

    try:
        import numpy as np
        import yfinance as yf
        import pandas as pd
    except Exception as e:
        print(f"Data Stack Load Failed (NP/YF/PD): {e}")
        yf = None
        pd = None
        np = None
    
    try: 
        import torch
    except: 
        torch = None
    
    import requests
    # try: 
    #     import ccxt
    # except: 
    #     ccxt = None
    ccxt = None # QC FIX: Disable CCXT to prevent import hang
        
    try:
        from multi_provider_data import get_live_price as glp, get_provider_status as gps, get_history as gh
        get_live_price = glp
        get_provider_status = gps
        get_history = gh
        MULTI_PROVIDER_AVAILABLE = True
    except Exception as e:
        print(f"Multi-Provider Data Load Failed: {e}")
        MULTI_PROVIDER_AVAILABLE = False

    try:
        from tactical_reasoner import TacticalReasoner
        tactical_reasoner = TacticalReasoner()
        TACTICAL_REASONER_AVAILABLE = True
    except Exception as e:
        print(f"Tactical Reasoner Load Failed: {e}")
        TACTICAL_REASONER_AVAILABLE = False

    try:
        from physics_orderflow_engine import PhysicsOrderFlowEngine as POFE
        PhysicsOrderFlowEngine = POFE
        physics_engine = PhysicsOrderFlowEngine()
        PHYSICS_ENGINE_AVAILABLE = True
        print("Physics Order Flow Engine ONLINE")
    except Exception as e:
        print(f"Physics Engine Load Failed: {e}")
        PHYSICS_ENGINE_AVAILABLE = False
        physics_engine = None

    try:
        from torsion_physics import TorsionPhysicsModel
        torsion_model = TorsionPhysicsModel()
        TORSION_PHYSICS_AVAILABLE = True
        print("Torsion Physics Model ONLINE")
    except Exception as e:
        print(f"Torsion Model Load Failed: {e}")
        TORSION_PHYSICS_AVAILABLE = False
        torsion_model = None

    try:
        from esp32_coherence_bridge import get_esp32_bridge as geb
        get_esp32_bridge = geb
        HW_BRIDGE_AVAILABLE = True
    except Exception as e:
        print(f"ESP32 Bridge Load Failed: {e}")
        HW_BRIDGE_AVAILABLE = False

    print("Heavy Modules Loaded.")
    return True

# CROSS-TALK: Shared Data Bus for all FMTR applications
try:
    from fmtr_data_bus import get_bus
    DATA_BUS = get_bus()
    DATA_BUS_AVAILABLE = True
except ImportError:
    DATA_BUS = None
    DATA_BUS_AVAILABLE = False

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SERVER_V2")

# FMTR: Multi-Day Anchored VWAP (Optional/Lazy Load)
def get_anchored_scanner():
    global anchored_scanner, ANCHORED_SCANNER_AVAILABLE
    if anchored_scanner is not None: return anchored_scanner
    try:
        from anchored_vwap_scanner import AnchoredVWAPScanner, SignalType 
        anchored_scanner = AnchoredVWAPScanner()
        ANCHORED_SCANNER_AVAILABLE = True 
        logger.info("Anchored VWAP Scanner Online.")
        return anchored_scanner
    except Exception as e:
        ANCHORED_SCANNER_AVAILABLE = False
        logger.warning(f"Anchored VWAP Scanner not available: {e}")
        return None

ANCHORED_SCANNER_AVAILABLE = False # Default state

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using OS env vars

# Multi-Provider Data Manager for robust data continuity (Lazy loaded)
get_live_price = None
get_provider_status = None
get_history = None
MULTI_PROVIDER_AVAILABLE = False
ccxt = None
tactical_reasoner = None
TACTICAL_REASONER_AVAILABLE = False
HW_BRIDGE_AVAILABLE = False
get_esp32_bridge = None
PhysicsOrderFlowEngine = None
PHYSICS_ENGINE_AVAILABLE = False
physics_engine = None
TORSION_PHYSICS_AVAILABLE = False
torsion_model = None

# Bridge placeholders

# --- CONFIGURATION ---
PORT = 5001
NQ_TICKER = "NQ=F"
BTC_TICKER = "BTC-USD"
SI_TICKER = "SI=F" # Silver (Learnings from Jan 30 Sell-off)
GHOST_LOG_FILE = 'logs/ghost_validation.csv'
fmtr_kernel = None
anchored_scanner = None
anchored_data_ready = threading.Event()

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

app = Flask(__name__, template_folder=".", static_folder="static")
CORS(app)
# QC FIX: Use threading mode to ensure background tasks don't block HTTP handlers
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread Lock for Global State (QC: Thread Safety Fix)
state_lock = threading.Lock()

# Global State
state = {
    'active_symbol': 'NQ=F',
    'ticker': 'NQ',
    'nq': {
        'price': 0.0,
        'change': 0.0,
        'timestamp': None,
        'market_status': 'CLOSED'
    },
    'btc': {
        'price': 0.0,
        'change': 0.0,
        'timestamp': None
    },
    'ghost_bars': None,
    'scanners': {},
    'coherence': 0.0,
    'lead_lag': 0.0,
    'evolution_count': 0,
    'price_series': [],
    'tick_buffer': [],  # Tick-level prices for fast coherence bootstrap
    'hit_rates': {
        '5m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
        '10m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
        '15m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0}
    },
    'expr_hit_rates': {
        '5m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
        '10m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
        '15m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0}
    },
    'pending_predictions': [],
    'last_inference_price': 0,
    'last_inference_time': 0,
    'last_full_emit': 0,
    'system_bias': 0.0, # PLL Integrator
    'last_forecast_result': {},
    'loop_speed_hz': 5.0, # Actual performance tracking
    'actual_latency_ms': 0,
    'last_lead_lag': 0.0,
    'macro_context': {
        'silver_price': 0.0,
        'silver_change': 0.0,
        'macro_alert': False,
        'basis_rotation': 0.0
    },
    'predictive_trigger': False,
    'hw_resilience': {
        'jamming_detected': False,
        'spoof_detected': False,
        'entropy_score': 1.0,
        'rf_coherence': 0.0,
        'rf_rssi': -100
    }
}

# --- FLOWSURFACE BRIDGE (NEURAL HEATMAP) ---
class FlowSurfaceBridge:
    """
    Translates FMTR Ghost Bars into a synthetic Order Book.
    Allows FlowSurface.com to render Neural Predictions as a Heatmap.
    """
    def __init__(self, port=5012):
        self.port = port
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self):
        while True:
            if self.clients and state['nq']['price'] > 0:
                current_p = state['nq']['price']
                coherence = state.get('coherence', 0.0)
                ghosts = state.get('ghost_bars', [])
                
                # Build synthetic levels
                # We treat each Ghost Bar as a 'Probability Node' in the depth book
                bids = [] # Predicted support
                asks = [] # Predicted resistance
                
                if ghosts:
                    for i, delta in enumerate(ghosts):
                        target_p = current_p + delta
                        size = (1.0 / (abs(delta) + 1e-9)) * coherence * 100
                        
                        # Round to nearest 0.25 (NQ tick size)
                        target_p = round(target_p * 4) / 4
                        
                        if delta >= 0:
                            asks.append([str(target_p), str(size)])
                        else:
                            bids.append([str(target_p), str(size)])

                msg = {
                    "stream": "fmtr_ghost@depth",
                    "data": {
                        "s": f"FMTR_{state['ticker']}", # E.g., FMTR_NQ
                        "b": bids,
                        "a": asks,
                        "E": int(time.time() * 1000)
                    }
                }
                
                tasks = [client.send(json.dumps(msg)) for client in self.clients]
                if tasks:
                    await asyncio.gather(*tasks)
            
            await asyncio.sleep(0.2) # 5Hz for FMTR (Neural latency match)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"FMTR FlowSurface Bridge Online: ws://localhost:{self.port}")
        
        async def start_bridge():
            async with websockets.serve(self.register, "0.0.0.0", self.port):
                await self.broadcast()
        
        loop.run_until_complete(start_bridge())

fmtr_bridge = FlowSurfaceBridge()

# FMTR: Anchored VWAP Scanner (Initialized lazily)
anchored_scanner = None
anchored_data_ready = threading.Event()

# === HARDENING: Signal Journal for Trade Persistence ===
SIGNAL_JOURNAL_FILE = 'logs/signal_journal.json'
signal_journal = []

def load_signal_journal():
    """Load persisted signals from disk"""
    global signal_journal
    try:
        if os.path.exists(SIGNAL_JOURNAL_FILE):
            with open(SIGNAL_JOURNAL_FILE, 'r') as f:
                signal_journal = json.load(f)
                logger.info(f"📓 Loaded {len(signal_journal)} signals from journal")
    except Exception as e:
        logger.warning(f"Could not load signal journal: {e}")
        signal_journal = []

def save_signal_journal():
    """Persist signals to disk"""
    try:
        with open(SIGNAL_JOURNAL_FILE, 'w') as f:
            json.dump(signal_journal[-500:], f, indent=2, default=str)  # Keep last 500
    except Exception as e:
        logger.warning(f"Could not save signal journal: {e}")

def log_signal(ticker: str, signal: str, entry_price: float, stop: float, target: float):
    """Log a new signal to the journal"""
    global signal_journal
    entry = {
        'ticker': ticker,
        'signal': signal,
        'entry_time': datetime.now().isoformat(),
        'entry_price': entry_price,
        'stop': stop,
        'target': target,
        'status': 'ACTIVE',
        'exit_time': None,
        'exit_price': None,
        'pnl_pct': 0.0
    }
    signal_journal.append(entry)
    save_signal_journal()
    logger.info(f"📓 Logged signal: {ticker} {signal} @ ${entry_price:.2f}")

load_signal_journal()

# === HARDENING: System Metrics Tracking ===
system_metrics = {
    'api_latencies': [],  # Last 100 API call latencies
    'cache_hits': 0,
    'cache_misses': 0,
    'signal_accuracy': {'wins': 0, 'losses': 0, 'pending': 0},
    'uptime_start': datetime.now().isoformat(),
    'total_requests': 0
}

def record_api_latency(endpoint: str, latency_ms: float):
    """Track API performance"""
    system_metrics['api_latencies'].append({
        'endpoint': endpoint,
        'latency_ms': latency_ms,
        'time': datetime.now().isoformat()
    })
    system_metrics['api_latencies'] = system_metrics['api_latencies'][-100:]  # Keep last 100
    system_metrics['total_requests'] += 1

# === HARDENING: Market Regime Tracker (NQ-Based Confluence) ===
market_regime = {
    'bias': 'NEUTRAL',  # BULLISH, BEARISH, NEUTRAL
    'nq_vs_vwap': 0.0,  # NQ distance from session VWAP in %
    'ghost_direction': 0,  # +1 bullish, -1 bearish, 0 neutral
    'regime_strength': 0.0,  # 0-1 confidence
    'last_update': None
}

def update_market_regime():
    """Update market regime based on NQ state"""
    global market_regime
    try:
        with state_lock:
            nq_price = state.get('nq', {}).get('price', 0)
            ghost_bars = state.get('ghost_bars')
            price_series = state.get('price_series', [])
        
        if not price_series or nq_price <= 0:
            return
        
        # Calculate NQ vs Session VWAP
        last_bar = price_series[-1] if price_series else {}
        session_vwap = last_bar.get('vwap', nq_price)
        if session_vwap > 0:
            nq_vs_vwap = ((nq_price - session_vwap) / session_vwap) * 100
        else:
            nq_vs_vwap = 0
        
        # Ghost direction
        ghost_direction = 0
        if ghost_bars and len(ghost_bars) > 0:
            total_delta = sum(ghost_bars)
            ghost_direction = 1 if total_delta > 0 else -1 if total_delta < 0 else 0
        
        # Determine bias
        if nq_vs_vwap > 0.1 and ghost_direction >= 0:
            bias = 'BULLISH'
            strength = min(1.0, abs(nq_vs_vwap) / 0.5)
        elif nq_vs_vwap < -0.1 and ghost_direction <= 0:
            bias = 'BEARISH'
            strength = min(1.0, abs(nq_vs_vwap) / 0.5)
        else:
            bias = 'NEUTRAL'
            strength = 0.3
        
        market_regime = {
            'bias': bias,
            'nq_vs_vwap': round(nq_vs_vwap, 3),
            'ghost_direction': ghost_direction,
            'regime_strength': round(strength, 2),
            'last_update': datetime.now().isoformat()
        }
    except Exception as e:
        logger.debug(f"Market regime update failed: {e}")

# === HARDENING: Pre-Warm Anchored Cache at Startup ===
def prewarm_anchored_cache():
    """Pre-compute anchored analysis for all tickers in background"""
    scanner = get_anchored_scanner()
    if not scanner:
        return
    
    logger.info("🔥 Pre-warming anchored VWAP cache...")
    start = time.time()
    success = 0
    
    for ticker in scanner.tickers:
        try:
            result = scanner.get_ticker_detail(ticker, only_cached=False)
            if result:
                success += 1
                # Log new triggers to journal
                if 'TRIGGER' in result.get('signal', ''):
                    existing = [s for s in signal_journal if s['ticker'] == ticker and s['status'] == 'ACTIVE']
                    if not existing:
                        log_signal(
                            ticker, 
                            result['signal'],
                            result.get('trade', {}).get('entry', 0),
                            result.get('trade', {}).get('stop', 0),
                            result.get('trade', {}).get('target', 0)
                        )
        except Exception as e:
            logger.debug(f"Prewarm failed for {ticker}: {e}")
    
    elapsed = time.time() - start
    logger.info(f"✅ Pre-warmed {success}/{len(scanner.tickers)} tickers in {elapsed:.1f}s")
    anchored_data_ready.set()

# Start pre-warming in background thread
# threading.Thread(target=prewarm_anchored_cache, daemon=True, name="CachePrewarm").start()

def convert_numpy(obj):
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        # Handle numpy arrays and pandas series directly
        if hasattr(obj, 'tolist'):
            return [convert_numpy(i) for i in obj.tolist()]
        return [convert_numpy(i) for i in obj]
    return str(obj)

# --- DATA STREAMING ---
def stream_data_loop():
    """Fetch NQ and BTC data periodically."""
    lazy_imports() # Ensure torch/requests are loaded in background thread
    logger.info("Starting Data Stream Loop...")
    
    # QC FIX: Backfill history on startup
    try:
        if MULTI_PROVIDER_AVAILABLE:
            hist = get_history(state['active_symbol'])
            if hist:
                with state_lock:
                    state['price_series'] = hist
                logger.info(f"Startup Backfill: {len(hist)} bars loaded")
    except Exception as e:
        logger.error(f"Startup Backfill Failed: {e}")
    
    # Initialize CCXT logic
    exchange = None
    if ccxt:
        try:
            exchange = ccxt.coinbase() 
            logger.info("CCXT Coinbase initialized for Realtime BTC")
        except:
             try:
                exchange = ccxt.kraken()
                logger.info("CCXT Kraken initialized for Realtime BTC")
             except Exception as e:
                logger.error(f"CCXT Init Failed: {e}")

    global fmtr_kernel
    device = 'cpu'
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logger.info("Apple Silicon Neural Engine (MPS) Detected & Enabled.")
    except Exception:
        pass

    while True:
        loop_start = time.time()
        
        if 'fmtr_kernel' not in globals() or fmtr_kernel is None:
             try:
                from fmtr_kernel import FMTRKernel
                fmtr_kernel = FMTRKernel(device=device)
                logger.info(f"FMTR Neural Core ONLINE (Device: {device})")
             except Exception as e:
                if int(time.time()) % 60 == 0:
                      logger.error(f"Neural Core Init Error (Retrying...): {e}")

        try:
            # QC FIX: Use Eastern Time for market status detection
            # NQ futures trade Sunday 6pm ET - Friday 5pm ET
            try:
                from zoneinfo import ZoneInfo
                eastern = ZoneInfo('America/New_York')
            except ImportError:
                # Python < 3.9 fallback
                import pytz
                eastern = pytz.timezone('America/New_York')
            
            current_time = datetime.now(eastern)
            
            # 1. Market Status Detection (in Eastern Time)
            is_saturday = current_time.weekday() == 5
            is_sunday_premarket = current_time.weekday() == 6 and current_time.hour < 18
            is_friday_aftermarket = current_time.weekday() == 4 and current_time.hour >= 17
            nq_closed = is_saturday or is_sunday_premarket or is_friday_aftermarket

            # 2. Fetch via High-Speed Aggregator (Consensus Wiring)
            agg_results, elapsed_agg = run_parallel_fetch()
            
            # --- Resolve Consensus for Specific Assets ---
            nq_sources = [r for r in agg_results if 'nq' in r['source'].lower()]
            btc_sources = [r for r in agg_results if 'btc' in r['source'].lower()]
            eth_sources = [r for r in agg_results if 'eth' in r['source'].lower()]
            
            # Map all prices for general lookup
            all_p = {}
            for r in agg_results:
                src = r['source'].lower()
                ticker = None
                if 'nq' in src: ticker = 'NQ=F'
                elif 'btc' in src: ticker = 'BTC-USD'
                elif 'eth' in src: ticker = 'ETH-USD'
                elif 'es' in src or 'spy' in src: ticker = 'ES=F'
                
                if ticker:
                    if ticker not in all_p: all_p[ticker] = []
                    all_p[ticker].append(r['price'])
            
            state['all_prices'] = {tick: sum(p)/len(p) for tick, p in all_p.items()}
            eth_consensus = state['all_prices'].get('ETH-USD', 0.0)
            
            nq_consensus = 0.0
            if nq_sources:
                nq_prices = [r['price'] for r in nq_sources]
                nq_consensus = sum(nq_prices) / len(nq_prices)
                # Check for ylive source for realtime flag
                realtime_nq = any('ylive' in r['source'] for r in nq_sources)
                
            btc_consensus = 0.0
            if btc_sources:
                btc_prices = [r['price'] for r in btc_sources]
                btc_consensus = sum(btc_prices) / len(btc_prices)
                state['btc'] = {
                    'price': btc_consensus,
                    'change': state['btc'].get('change', 0),
                    'timestamp': current_time.isoformat(),
                    'source': f"High-Speed Consensus ({len(btc_sources)} sources)"
                }
                
                # CROSS-TALK: Publish to shared bus
                if DATA_BUS_AVAILABLE:
                    DATA_BUS.publish('BTC-USD', btc_consensus, source='aggregator')

            # --- Target Active Symbol for Primary 'nq' Field ---
            # Dashboard focuses on state['nq'] for the main chart, but we must protect integrity
            active_s = state.get('active_symbol', 'NQ=F')
            
            # Map specific consensus to the specific state buckets
            if nq_consensus > 0:
                state['nq'] = {
                    'price': nq_consensus,
                    'change': state['nq'].get('change', 0),
                    'timestamp': current_time.isoformat(),
                    'market_status': 'OPEN' if not nq_closed else 'CLOSED',
                    'source': f"High-Speed Consensus ({len(nq_sources)} sources)",
                    'method': 'parallel_aggregator',
                    'fetch_ms': elapsed_agg,
                    'realtime': any('ylive' in r['source'] for r in nq_sources)
                }
            
            if btc_consensus > 0:
                 state['btc'] = {
                    'price': btc_consensus,
                    'change': state['btc'].get('change', 0),
                    'timestamp': current_time.isoformat(),
                    'market_status': 'CRYPTO_LIVE'
                }
            else:
                # YFinance Fallback
                try:
                    btc = yf.Ticker(BTC_TICKER)
                    fast = btc.fast_info
                    if fast and fast.last_price:
                        state['btc'] = {
                            'price': fast.last_price,
                            'change': ((fast.last_price - fast.previous_close) / fast.previous_close) * 100,
                            'timestamp': current_time.isoformat(),
                            'source': 'yfinance_fallback'
                        }
                except: pass

            # 2c. Fetch Silver (Macro Sentinel)
            try:
                si = yf.Ticker(SI_TICKER)
                fast_si = si.fast_info
                if fast_si and fast_si.last_price:
                    si_price = fast_si.last_price
                    si_change = ((si_price - fast_si.previous_close) / fast_si.previous_close) * 100
                    state['macro_context']['silver_price'] = si_price
                    state['macro_context']['silver_change'] = si_change
                    
                    # Macro Alert: Detection of "Phase Paradox" (Learnings from Warsh Shock)
                    # If Silver drops > 10% or moves > 15% in a day, it's a Basis Shift
                    if abs(si_change) > 10.0:
                        state['macro_context']['macro_alert'] = True
                        state['macro_context']['basis_rotation'] = si_change / 100.0
                        logger.warning(f"MACRO ALERT: Silver Phase Paradox ({si_change:.2f}%). Warsh Shadow detected.")
                    else:
                        state['macro_context']['macro_alert'] = False
            except Exception as e:
                logger.debug(f"Silver fetch failed: {e}")
                # Fallback to Multi-Provider for other symbols (ETH, SOL, etc.)
                try:
                    if MULTI_PROVIDER_AVAILABLE:
                        result = get_live_price(active_s)
                        if result and result.get('price', 0) > 0:
                            state['nq'] = {
                                'price': result['price'],
                                'change': result.get('change', 0),
                                'timestamp': current_time.isoformat(),
                                'market_status': 'OPEN',
                                'source': result.get('source', 'multi_provider'),
                                'realtime': result.get('realtime', False)
                            }
                except: pass

            # Update ticker label from active symbol
            state['ticker'] = state['active_symbol'].replace("=F", "").split("-")[0]
            
            # CROSS-TALK: Publish active symbol to bus
            if DATA_BUS_AVAILABLE and state['nq']['price'] > 0:
                DATA_BUS.publish(
                    active_s, 
                    state['nq']['price'],
                    coherence=state.get('coherence', 0),
                    ghost_alpha=state.get('lead_lag', 0),
                    source='quantum_manifold'
                )

            # 3. FMTR Neural Core (Ghost Bar Predictions)
            try:
                # QC FIX: Check Data Freshness before Inference
                data_age = 0
                if state['nq']['timestamp']:
                    try:
                        last_ts = datetime.fromisoformat(state['nq']['timestamp'])
                        # Ensure timezones match (if offset-aware)
                        if last_ts.tzinfo is None:
                            last_ts = last_ts.replace(tzinfo=eastern.tzinfo) 
                        data_age = (current_time - last_ts).total_seconds()
                    except: pass
                
                is_stale = data_age > 60.0 # 1 minute stale threshold
                
                if 'fmtr_kernel' in globals() and state['nq']['price'] > 0 and not is_stale:
                    now_ts = int(time.time())
                    rounded_ts = now_ts - (now_ts % 60)
                    current_p = state['nq']['price']

                    # QC FIX: Thread-safe state mutation & Jump Protection
                    with state_lock:
                        # Safety check: If price jumps by >50% suddenly, it's likely a ticker switch race condition.
                        # We should skip appending until the series is cleared or stabilized.
                        is_jump = False
                        if state['price_series']:
                            last_c = state['price_series'][-1]['close']
                            if last_c > 0:
                                price_diff = abs(current_p - last_c) / last_c
                                if price_diff > 0.5:
                                    is_jump = True

                        if is_jump or current_p <= 0:
                            if current_p > 0:
                                logger.warning(f"Rejecting outlier price {current_p} for {state['active_symbol']} (last: {last_c}) - Transition in progress?")
                            # If current_p is 0, we just skip silently (waiting for fresh data)
                        else:
                            if state['price_series'] and state['price_series'][-1]['time'] == rounded_ts:
                                last_bar = state['price_series'][-1]
                                last_bar['high'] = max(last_bar['high'], current_p)
                                last_bar['low'] = min(last_bar['low'], current_p)
                                last_bar['close'] = current_p
                            elif state['price_series']:
                                # Only start a new bar if we have history. 
                                # This prevents 'dots' from building a fake history before backfill.
                                new_point = {'time': rounded_ts, 'open': current_p, 'high': current_p, 'low': current_p, 'close': current_p, 'vwap': current_p}
                                state['price_series'].append(new_point)
                                if len(state['price_series']) > 300: state['price_series'].pop(0)

                    # Add current price to tick buffer for fast coherence bootstrap
                    # QC FIX: Thread-safe buffer mutation
                    with state_lock:
                        state['tick_buffer'].append(current_p)
                        if len(state['tick_buffer']) > 500:  # Keep last 500 ticks
                            state['tick_buffer'].pop(0)
                    
                    holo_state = type('HolographicState', (), {})()
                    holo_state.price = state['nq']['price']
                    # Use tick buffer for coherence (much faster than minute bars)
                    # Fall back to minute bars only if tick buffer is too small
                    if len(state['tick_buffer']) >= 5:
                        holo_state.price_history = state['tick_buffer'].copy()
                    else:
                        holo_state.price_history = [p['close'] for p in state['price_series']]
                    holo_state.system_bias = state['system_bias']
                    
                    should_run = abs(state['nq']['price'] - state['last_inference_price']) > 0.01 or (time.time() - state['last_inference_time'] > 5.0)
                        
                    if should_run:
                        forecast = fmtr_kernel.get_forecast(holo_state)
                        state['last_forecast_result'] = forecast
                        state['last_inference_price'] = state['nq']['price']
                        state['last_inference_time'] = time.time()
                        
                        if 'ghost_bars' in forecast:
                            state['ghost_bars'] = forecast['ghost_bars']
                            state['ghost_upper'] = forecast.get('ghost_upper', [])
                            state['ghost_lower'] = forecast.get('ghost_lower', [])
                            state['is_stabilized'] = forecast.get('is_stabilized', False)
                            state['lead_lag'] = forecast.get('lead_lag', 0.0)
                            state['coherence'] = forecast.get('coherence', 0.0)
                            state['no_cloning_integrity'] = forecast.get('no_cloning_integrity', 1.0)
                            state['quantum_regime'] = forecast.get('quantum_regime', 'UNKNOWN')
                            state['evolution_count'] += 1
                            
                            # SAGA Bi-Level Metrics (arXiv:2512.21782)
                            state['saga_calibration_factor'] = forecast.get('saga_calibration_factor', 1.0)
                            state['saga_time_exponent'] = forecast.get('saga_time_exponent', 0.5)
                            state['saga_coverage'] = forecast.get('saga_coverage', 0.0)
                            state['saga_evolution_count'] = forecast.get('saga_evolution_count', 0)
                        
                        lead_delta = state['lead_lag'] - state.get('last_lead_lag', 0)
                        if abs(lead_delta) > 0.1 and state['coherence'] > 0.7:
                            state['predictive_trigger'] = True
                            socketio.emit('anticipation_pulse', {'lead_lag': state['lead_lag'], 'delta': lead_delta, 'coherence': state['coherence']})
                        else:
                            state['predictive_trigger'] = False
                        state['last_lead_lag'] = state['lead_lag']

                        # RF Resilience Sync (Satellite Defense)
                        if HW_BRIDGE_AVAILABLE:
                            hw = get_esp32_bridge()
                            metrics = hw.get_metrics()
                            with state_lock:
                                state['hw_resilience'] = {
                                    'jamming_detected': metrics.get('jamming_detected', False),
                                    'spoof_detected': metrics.get('spoof_detected', False),
                                    'entropy_score': metrics.get('entropy_score', 1.0),
                                    'rf_coherence': metrics.get('rf_coherence', 0.0),
                                    'rf_rssi': metrics.get('rf_rssi', -100)
                                }
                                # Hardware-In-The-Loop Coherence Adjustment
                                state['coherence'] *= hw.get_coherence_factor()
                        
                        # Prepare for Validation
                        # QC FIX: Map 1-minute resolution ghost bars to 5m, 10m, 15m validation points
                        # ghost_bars[4] = 5m, ghost_bars[9] = 10m, ghost_bars[14] = 15m
                        v_indices = [4, 9, 14]
                        valid_ghosts = (state['ghost_bars'] and len(state['ghost_bars']) >= 15)
                        prod_prices = [(state['nq']['price'] + state['ghost_bars'][i]) for i in v_indices] if valid_ghosts else []
                        
                        raw_ghosts = forecast.get('raw_ghost_bars', state.get('ghost_bars', []))
                        valid_raw = (raw_ghosts and len(raw_ghosts) >= 15)
                        raw_prices = [(state['nq']['price'] + raw_ghosts[i]) for i in v_indices] if valid_raw else []

                        if prod_prices and raw_prices:
                            # QC FIX: Thread-safe prediction append
                            with state_lock:
                                state['pending_predictions'].append({
                                    'timestamp': current_time, 'start_price': state['nq']['price'],
                                    'symbol': state['active_symbol'],  # Track which symbol this prediction is for
                                    'targets': {
                                        '5m': {'time': current_time + timedelta(minutes=5), 'prod': prod_prices[0], 'expr': raw_prices[0]},
                                        '10m': {'time': current_time + timedelta(minutes=10), 'prod': prod_prices[1], 'expr': raw_prices[1]},
                                        '15m': {'time': current_time + timedelta(minutes=15), 'prod': prod_prices[2], 'expr': raw_prices[2]}
                                    },
                                    'validated': {'5m': False, '10m': False, '15m': False},
                                    # SAGA: Store cone bounds for bi-level evolution validation
                                    'ghost_upper': state.get('ghost_upper', []),
                                    'ghost_lower': state.get('ghost_lower', [])
                                })
                                if len(state['pending_predictions']) > 2000: state['pending_predictions'].pop(0)

                    # Validation - MULTI-TIER ACCURACY SYSTEM
                    now = datetime.now(eastern)
                    if state['nq']['price'] > 0:
                        unvalidated = []
                        for i, pred in enumerate(state['pending_predictions']):
                            # QC FIX: Skip predictions from different symbols
                            if pred.get('symbol', state['active_symbol']) != state['active_symbol']:
                                continue  # Discard cross-symbol predictions
                            
                            # Sanity check: Skip if start_price is unreasonably far (>50% different)
                            price_ratio = pred['start_price'] / state['nq']['price'] if state['nq']['price'] > 0 else 0
                            if price_ratio < 0.5 or price_ratio > 2.0:
                                continue  # Discard stale/cross-symbol predictions
                            
                            still_pending = False
                            for interval in ['5m', '10m', '15m']:
                                if not pred['validated'][interval]:
                                    target = pred['targets'][interval]
                                    if now >= target['time']:
                                        actual_price = state['nq']['price']
                                        predicted_price = target['prod']
                                        start_price = pred['start_price']
                                        
                                        # ============================================================
                                        # MULTI-TIER HIT RATE VALIDATION (Improved from 0.05% static)
                                        # ============================================================
                                        
                                        # 1. DIRECTIONAL ACCURACY (Most Important)
                                        # Did we predict the right direction of move?
                                        predicted_direction = 1 if predicted_price > start_price else (-1 if predicted_price < start_price else 0)
                                        actual_direction = 1 if actual_price > start_price else (-1 if actual_price < start_price else 0)
                                        directional_hit = (predicted_direction == actual_direction) and predicted_direction != 0
                                        
                                        # 2. CONE CONTAINMENT (From SAGA Evolution)
                                        # Was the actual price within our confidence cone?
                                        # QC FIX: Map 1-minute resolution bounds to 5m, 10m, 15m
                                        cone_hit = False
                                        if 'ghost_upper' in pred and 'ghost_lower' in pred:
                                            # Array index mapping: 1m steps -> 5, 10, 15 minutes
                                            array_idx = {'5m': 4, '10m': 9, '15m': 14}.get(interval, 4)
                                            # Evolver mapping: 5m->0, 10m->1, 15m->2
                                            evolver_idx = {'5m': 0, '10m': 1, '15m': 2}.get(interval, 0)
                                            
                                            if array_idx < len(pred['ghost_upper']):
                                                upper_bound = start_price + pred['ghost_upper'][array_idx]
                                                lower_bound = start_price + pred['ghost_lower'][array_idx]
                                                cone_hit = lower_bound <= actual_price <= upper_bound
                                                
                                                # Record cone outcome for bi-level evolution
                                                within_cone = fmtr_kernel.record_cone_outcome(
                                                    evolver_idx, actual_price, predicted_price, upper_bound, lower_bound
                                                )
                                        
                                        # 3. TIERED PRICE ACCURACY
                                        # Volatility-scaled tolerance thresholds
                                        price_error_pct = abs(actual_price - predicted_price) / predicted_price * 100
                                        
                                        # Scale tolerance by horizon (longer = looser)
                                        # NQ ~21500: 0.25% = $53.75, 0.5% = $107.50, 1.0% = $215
                                        horizon_scale = {'5m': 1.0, '10m': 1.5, '15m': 2.0}.get(interval, 1.0)
                                        
                                        excellent_hit = price_error_pct < (0.25 * horizon_scale)  # Tight
                                        good_hit = price_error_pct < (0.50 * horizon_scale)       # Medium
                                        acceptable_hit = price_error_pct < (1.0 * horizon_scale)  # Loose
                                        
                                        # 4. COMPOSITE HIT SCORE (Weighted)
                                        # Directional: 50%, Cone: 30%, Price Accuracy: 20%
                                        hit_score = 0.0
                                        if directional_hit:
                                            hit_score += 0.50  # Direction correct = 50%
                                        if cone_hit:
                                            hit_score += 0.30  # Within cone = 30%
                                        if excellent_hit:
                                            hit_score += 0.20  # Excellent accuracy = 20%
                                        elif good_hit:
                                            hit_score += 0.15  # Good accuracy = 15%
                                        elif acceptable_hit:
                                            hit_score += 0.10  # Acceptable = 10%
                                        
                                        # A "hit" is now defined as score >= 0.5 (at least direction correct)
                                        is_hit = hit_score >= 0.50
                                        
                                        if is_hit:
                                            state['hit_rates'][interval]['hits'] += 1
                                        
                                        # Track component hits
                                        if directional_hit: state['hit_rates'][interval]['dir_hits'] += 1
                                        if cone_hit: state['hit_rates'][interval]['cone_hits'] += 1
                                        if excellent_hit or good_hit: state['hit_rates'][interval]['prec_hits'] += 1
                                        
                                        state['hit_rates'][interval]['total'] += 1
                                        
                                        # Update derived rates
                                        total_n = max(1, state['hit_rates'][interval]['total'])
                                        state['hit_rates'][interval]['rate'] = round((state['hit_rates'][interval]['hits'] / total_n) * 100, 1)
                                        state['hit_rates'][interval]['dir_rate'] = round((state['hit_rates'][interval]['dir_hits'] / total_n) * 100, 1)
                                        state['hit_rates'][interval]['cone_rate'] = round((state['hit_rates'][interval]['cone_hits'] / total_n) * 100, 1)
                                        state['hit_rates'][interval]['prec_rate'] = round((state['hit_rates'][interval]['prec_hits'] / total_n) * 100, 1)
                                        
                                        # QC OVERNIGHT LOGGING (Validation happens interval-by-interval inside loop)
                                        try:
                                            log_path = 'logs/ghost_validation.csv'
                                            file_exists = os.path.isfile(log_path)
                                            mode = 'a'
                                            with open(log_path, mode, newline='') as f:
                                                writer = csv.writer(f)
                                                if not file_exists:
                                                    writer.writerow(['timestamp', 'ticker', 'interval', 'predicted', 'actual', 'error_pct', 'directional_hit', 'cone_hit', 'score'])
                                                
                                                current_ts = datetime.now().isoformat()
                                                writer.writerow([
                                                    current_ts, state.get('ticker', 'UNKNOWN'), interval, 
                                                    f"{predicted_price:.2f}", f"{actual_price:.2f}", 
                                                    f"{price_error_pct:.4f}", 
                                                    directional_hit, cone_hit, f"{hit_score:.2f}"
                                                ])
                                        except Exception:
                                            pass

                                        # Update Bayesian with composite hit
                                        fmtr_kernel.update_bayesian_with_cone_rl(is_hit, cone_hit if 'cone_hit' in dir() else None)
                                        
                                        # Experimental hit rates (use tighter threshold for comparison)
                                        expr_hit = price_error_pct < (0.25 * horizon_scale) and directional_hit
                                        if expr_hit:
                                            state['expr_hit_rates'][interval]['hits'] += 1
                                        state['expr_hit_rates'][interval]['total'] += 1
                                        state['expr_hit_rates'][interval]['rate'] = round((state['expr_hit_rates'][interval]['hits'] / max(1, state['expr_hit_rates'][interval]['total'])) * 100, 1)
                                        
                                        # PLL Bias Correction (5m only)
                                        if interval == '5m':
                                            error = actual_price - predicted_price
                                            state['system_bias'] = (state['system_bias'] * 0.95) + (error * 0.05)
                                            # Record directional accuracy for adaptive learning
                                            fmtr_kernel.record_directional_accuracy(directional_hit)
                                        
                                        pred['validated'][interval] = True
                                    else: still_pending = True
                            if still_pending: unvalidated.append(i)
                        state['pending_predictions'] = [state['pending_predictions'][i] for i in unvalidated]

            except Exception as neural_err:
                logger.error(f"Neural Error: {neural_err}")

            # Emit
            should_send_full = (time.time() - state.get('last_full_emit', 0) > 5.0)
            
            # QC FIX: Thread-safe state capture for emission
            with state_lock:
                emit_state = state.copy()
                if should_send_full:
                    # Shallow copy of list is sufficient to prevent iterator errors during serialization
                    emit_state['price_series'] = list(state['price_series'])
                    state['last_full_emit'] = time.time()
                elif 'price_series' in emit_state:
                    # Don't send series on non-full updates
                    del emit_state['price_series']
            
            emit_state['is_full'] = should_send_full

            # FINANCIAL TELEMETRY: Add wallet and realized P&L to state for Hub
            try:
                # QC FIX: Use unified wallet_state.json and proper valuation
                if os.path.exists('wallet_state.json'):
                    with open('wallet_state.json', 'r') as f:
                        balances = json.load(f)
                        # QC FIX: Use consolidated ETH price for accurate valuation
                        eth_p = state.get('all_prices', {}).get('ETH-USD', 0.0)
                        if eth_p == 0:
                            # Fallback to a hardcoded safe value if pricing fails
                            eth_p = 2300.0 

                        # REALITY CHECK: Portfolio Value is your liquid account equity.
                        # Since this bot trades WETH/USDC as a proxy, ALL "positions" are already
                        # reflected in the liquid WETH/USDC balances. Adding inventory_mtm
                        # on top of liquid balances is double-counting.
                        
                        liquid_val = balances.get('usdc', 0) + (balances.get('weth', 0) * eth_p) + (balances.get('eth', 0) * eth_p)
                        
                        # We still calculate inventory for metrics, but we don't add it to total equity
                        inventory_mtm = 0.0
                        try:
                            if os.path.exists('position_state.json'):
                                with open('position_state.json', 'r') as pf:
                                    p_data = json.load(pf)
                                    for sym, pos in p_data.get('positions', {}).items():
                                        # Use units * entry_price as a gross exposure metric
                                        inventory_mtm += pos.get('amount_weth', 0) * pos.get('entry_price', 0)
                        except: pass
                        
                        # Real Portfolio Value = Liquid funds (the "Proxy" approach means WETH IS the exposure)
                        emit_state['wallet_balance'] = round(liquid_val, 2)
                        emit_state['gross_exposure'] = round(inventory_mtm, 2)
                
                # Try to get realized P&L from logs
                realized = 0.0
                if os.path.exists('trades_log.jsonl'):
                    with open('trades_log.jsonl', 'r') as f:
                        for line in f:
                            try:
                                t = json.loads(line)
                                # Any trade with pnl_usd recorded is a realization or flip
                                realized += t.get('pnl_usd', 0)
                            except: pass
                emit_state['realized_pnl'] = realized
            except: pass

            # CALCULATE ESSENTIAL INSIGHTS FOR DASHBOARD (QC PHASE)
            # Ensure signal keys exist even if neural scanner is quiet
            if 'signal' not in emit_state or not emit_state['signal']:
                emit_state['signal'] = "SCANNING..."
                emit_state['signal_strength'] = 0
                emit_state['distance_pct'] = 0
                emit_state['signal_meta'] = "Waiting for setup..."
                
                # FALLBACK: Generate basic VWAP mean reversion signal for dashboard vitality
                if 'price_series' in state and len(state['price_series']) > 20:
                    current_p = state['nq']['price']
                    # Calculate live VWAP from series
                    vwap = state['price_series'][-1].get('vwap', current_p)
                    if vwap > 0:
                        dist = ((current_p - vwap) / vwap) * 100
                        emit_state['distance_pct'] = dist
                        
                        # Simple Mean Reversion Logic
                        if dist > 0.2: # Extended Long
                             emit_state['signal'] = "SHORT_TRIGGER"
                             emit_state['signal_meta'] = "Extended > 0.2% above VWAP"
                             emit_state['signal_strength'] = min(95, abs(dist) * 200)
                        elif dist < -0.2: # Extended Short
                             emit_state['signal'] = "LONG_TRIGGER"
                             emit_state['signal_meta'] = "Extended < 0.2% below VWAP" 
                             emit_state['signal_strength'] = min(95, abs(dist) * 200)
                        else:
                             emit_state['signal'] = "IN_RANGE"
                             emit_state['signal_meta'] = "Price within VWAP bands"
                             emit_state['signal_strength'] = 0

            try:
                insights = {}
                current_p = emit_state.get('nq', {}).get('price', 0)
                if current_p > 0:
                    # VWAP Deviation
                    vwap = current_p # Default
                    if 'price_series' in state and state['price_series']: # Check STATE not emit_state
                         vwap = state['price_series'][-1].get('vwap', current_p)
                    
                    def robust_float(val):
                        if val is None: return 0.0
                        if isinstance(val, (int, float)):
                            if np.isnan(val) or np.isinf(val): return 0.0
                        return float(val)

                    dev_pct = ((current_p - vwap) / vwap) * 100 if vwap > 0 else 0
                    insights['vwap_dev'] = robust_float(dev_pct)
                    
                    # Volatility (Tick based)
                    insights['volatility'] = 0.0
                    insights['manifold_z'] = 0.0
                    
                    if 'tick_buffer' in state and len(state['tick_buffer']) > 10:
                        # QC FIX: Filter zeros to prevent 400% volatility spikes
                        valid_ticks = [t for t in state['tick_buffer'][-50:] if t > 0]
                        ticks = np.array(valid_ticks)
                        
                        if len(ticks) > 1:
                            mean_val = np.mean(ticks)
                            if mean_val > 0:
                                # Raw minute volatility scaled
                                vol = (np.std(ticks) / mean_val) * 100 * 20 
                                insights['volatility'] = robust_float(vol)
                            
                            # Manifold Z-Score
                            std_dev = np.std(ticks)
                            if std_dev > 0:
                                z = (current_p - vwap) / std_dev
                                insights['manifold_z'] = robust_float(z)
                    
                    # Physics Realism (Coherence)
                    insights['physics_realism'] = robust_float(state.get('coherence', 0.0))
                    
                    # Manifold Alpha (Lead/Lag)
                    insights['manifold_alpha'] = robust_float(state.get('lead_lag', 0.0))
                    
                    # Cone Basis
                    insights['cone_basis'] = "PHYSICS-INFORMED" if fmtr_kernel else "STANDARD"
                    
                    # MA Status (Simple Trend)
                    if 'price_series' in state and len(state['price_series']) > 20:
                         ma20 = sum([p['close'] for p in state['price_series'][-20:]]) / 20
                         insights['ma_status'] = "BULLISH" if current_p > ma20 else "BEARISH"
                    else:
                         insights['ma_status'] = "NEUTRAL"

                    # DYNAMIC TREND & ALERTS
                    # Trend Analysis
                    z_score = insights.get('manifold_z', 0)
                    coherence = insights.get('physics_realism', 0)
                    
                    trend_msg = "Market is in equilibrium. "
                    if coherence > 0.05:
                        direction = "Bullish" if z_score > 0 else "Bearish"
                        trend_msg = f"Strong {direction} momentum confirmed. Low entropy suggests trend continuity. "
                    elif coherence < 0.01:
                         trend_msg = "High entropy detected. Market is choppy or transitioning. "
                    
                    if abs(z_score) > 2.0:
                        reversion_type = "Top" if z_score > 0 else "Bottom"
                        trend_msg += f"Potential {reversion_type} formation. Mean reversion highly probable."
                    else:
                        trend_msg += "Favorable conditions for trend following."
                    
                    insights['trend_analysis'] = trend_msg

                    # Market Alert
                    alert = None
                    if abs(z_score) > 3.0:
                         condition = "OVERBOUGHT" if z_score > 0 else "OVERSOLD"
                         alert = {"level": "CRITICAL", "msg": f"EXTREME {condition} ({z_score:.1f}σ). High volatility event."}
                    elif insights.get('volatility', 0) > 1.0:
                         alert = {"level": "WARNING", "msg": "High volatility detected. Widen stops."}
                    elif abs(dev_pct) > 0.5:
                         ext_dir = "above" if dev_pct > 0 else "below"
                         alert = {"level": "INFO", "msg": f"Price extended {ext_dir} VWAP."}
                    
                    insights['market_alert'] = alert

                emit_state['essential_insights'] = insights
            except Exception as e:
                logger.error(f"Insights Calculation Error: {e}")

            # PHYSICS ORDER FLOW LAYER (Middle-Out QC)
            try:
                if PHYSICS_ENGINE_AVAILABLE and physics_engine and emit_state.get('nq', {}).get('price', 0) > 0:
                    from physics_orderflow_engine import OrderFlowState
                    current_p = emit_state['nq']['price']
                    
                    # Compute synthetic order flow state from telemetry
                    # In a real setup, this would use L2 book data, but we use high-speed price gradients here
                    p_vel = insights.get('volatility', 0) * (1 if dev_pct > 0 else -1)
                    p_acc = (p_vel - state.get('prev_velocity', 0))
                    state['prev_velocity'] = p_vel
                    
                    of_state = OrderFlowState(
                        timestamp=datetime.now(),
                        ticker=state['active_symbol'],
                        price=current_p,
                        price_change_1m=dev_pct,
                        price_change_5m=dev_pct * 2, # Approximation
                        price_velocity=p_vel,
                        price_acceleration=p_acc,
                        volume_ratio=1.0,
                        buy_volume_ratio=0.5 + (0.1 if dev_pct > 0 else -0.1),
                        volume_acceleration=0,
                        bid_depth=100, ask_depth=100, spread=0.0001,
                        imbalance=0.1 if dev_pct > 0 else -0.1,
                        vwap_deviation=dev_pct,
                        volatility=insights.get('volatility', 0),
                        momentum=insights.get('volatility', 0) * (1 if dev_pct > 0 else -1)
                    )
                    
                    physics_signal = physics_engine.generate_composite_signal(of_state)
                    emit_state['physics_orderflow'] = physics_signal
            except Exception as p_err:
                logger.debug(f"Physics Layer Skip: {p_err}")
            # SIGNAL CONSENSUS LAYER
            # Harmonize Scanner Signal vs. Neural Cone Direction to prevent "Short Trigger vs Up Cone" conflicts
            try:
                if 'signal' in emit_state and emit_state['signal'] and 'ghost_bars' in emit_state and emit_state['ghost_bars']:
                    # Calculate Neural Direction (Mean of first 5 ghost bars)
                    ghosts = emit_state['ghost_bars']
                    if len(ghosts) > 0:
                         # Calculate projected move % relative to current price
                         current_p = emit_state.get('nq', {}).get('price', 0)
                         if current_p > 0:
                             avg_proj = sum(ghosts[:5]) / len(ghosts[:5])
                             proj_dir_pct = (avg_proj / current_p) * 100
                             
                             sig = emit_state['signal']
                             
                             # Check for contradictions
                             # 1. SHORT_TRIGGER but Cone is Bullish (> 0.05%)
                             if "SHORT" in sig and proj_dir_pct > 0.05:
                                 emit_state['signal'] = sig.replace("TRIGGER", "CONTRARIAN")
                                 emit_state['signal_meta'] = f"Downgraded due to Bullish Cone (+{proj_dir_pct:.2f}%)"
                             
                             # 2. LONG_TRIGGER but Cone is Bearish (< -0.05%)
                             elif "LONG" in sig and proj_dir_pct < -0.05:
                                 emit_state['signal'] = sig.replace("TRIGGER", "CONTRARIAN")
                                 emit_state['signal_meta'] = f"Downgraded due to Bearish Cone ({proj_dir_pct:.2f}%)"

            except Exception as consensus_err:
                logger.error(f"Consensus Layer Error: {consensus_err}")

            # TORSION PHYSICS LAYER
            try:
                if TORSION_PHYSICS_AVAILABLE and torsion_model and 'price_series' in emit_state:
                    # Sync with latest price series (Last 30 bars sufficient for 20-bar lookback)
                    torsion_model.history_prices = []
                    torsion_model.history_volumes = []
                    
                    series = emit_state['price_series']
                    if len(series) > 0:
                        for bar in series[-30:]: 
                            torsion_model.update_history(
                                bar.get('close', 0), 
                                bar.get('open', 0), 
                                bar.get('volume', 0)
                            )
                        
                        # Analyze using current live metrics
                        t_signal, t_metrics = torsion_model.analyze(
                            current_price=emit_state['nq']['price'],
                            current_vol=0, 
                            volatility=insights.get('volatility', 0)
                        )
                        
                        emit_state['torsion_physics'] = {
                            'signal': t_signal.direction, # 1, -1, 0
                            'strength': t_signal.strength,
                            'node': t_signal.node,
                            'amd': t_metrics.get('amd', 0),
                            'chirality': t_metrics.get('chirality', 0),
                            'scf': t_metrics.get('scf', 0),
                            'reasoning': t_signal.meta
                        }
            except Exception as t_err:
                logger.debug(f"Torsion Layer Skip: {t_err}")

            socketio.emit('nq_update', {'data': convert_numpy(emit_state), 'timestamp': current_time.isoformat(), 'ticker': state['ticker']})
            if should_send_full:
                logger.info(f"EMITTED FULL STATE ({len(emit_state.get('price_series', []))} bars)")

        except Exception as e:
            logger.error(f"Global Loop Error: {e}")
            time.sleep(1.0)
            
        # DYNAMIC OVERCLOCK
        is_high_coherence = state.get('coherence', 0) > 0.6
        is_leading = state.get('lead_lag', 0) > 0.05
        target_delay = 0.005 if (is_high_coherence and is_leading) else (0.02 if is_high_coherence else 0.05)
        
        elapsed = time.time() - loop_start
        state['actual_latency_ms'] = round(elapsed * 1000, 2)
        state['loop_speed_hz'] = round(1.0 / elapsed, 1) if elapsed > 0 else 0
        time.sleep(max(0, target_delay - elapsed))

def anchored_scanner_loop():
    logger.info("Starting Anchored VWAP Scanner Loop...")
    while True:
        try:
            scanner = get_anchored_scanner()
            if scanner: 
                scanner.run_scan()
                anchored_data_ready.set()
                time.sleep(60)
            else: 
                time.sleep(30) # Wait for modules to load
        except Exception as e:
            logger.debug(f"Scanner loop error: {e}")
            time.sleep(30)

@app.route('/')
@app.route('/quantum')
@app.route('/master')
@app.route('/dash')
@app.route('/dashboard')
def quantum_hub(): return send_file('quantum_hub.html')

@app.route('/unified')
def unified_terminal(): return send_file('unified_terminal.html')

@app.route('/vwap')
def vwap_dash(): return send_file('vwap_dashboard.html')

@app.route('/anchored')
def anchored_dash(): return send_file('anchored_vwap_dashboard.html')

@app.route('/hitrate')
@app.route('/track')
def hitrate_dash(): return send_file('hit_rate_tracker.html')

@app.route('/api/health')
@app.route('/health')
def health_v2(): 
    status = get_provider_status() if MULTI_PROVIDER_AVAILABLE else {}
    sources = [{"name": k, "status": v['status']} for k, v in status.items()]
    
    # Antigravity Manager (Gateway) Monitoring
    try:
        manager_health = requests.get("http://localhost:8045/healthz", timeout=0.5)
        if manager_health.status_code == 200:
            sources.append({"name": "Antigravity Gateway", "status": "healthy"})
        else:
            sources.append({"name": "Antigravity Gateway", "status": "degraded"})
    except:
        sources.append({"name": "Antigravity Gateway", "status": "offline"})

    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'agent_version': '1.0.0-verified',
        'sources': sources
    })

@app.route('/nq_data')
def get_nq_data(): return jsonify(convert_numpy(state))

# === HARDENING: New API Endpoints ===
@app.route('/api/metrics')
def get_metrics():
    """System performance metrics endpoint"""
    latencies = system_metrics.get('api_latencies', [])
    latency_values = [l['latency_ms'] for l in latencies if l.get('latency_ms')]
    
    p50 = statistics.median(latency_values) if latency_values else 0
    p95 = sorted(latency_values)[int(len(latency_values) * 0.95)] if len(latency_values) > 20 else max(latency_values) if latency_values else 0
    p99 = sorted(latency_values)[int(len(latency_values) * 0.99)] if len(latency_values) > 100 else max(latency_values) if latency_values else 0
    
    cache_total = system_metrics['cache_hits'] + system_metrics['cache_misses']
    cache_hit_rate = (system_metrics['cache_hits'] / cache_total * 100) if cache_total > 0 else 0
    
    return jsonify({
        'status': 'ok',
        'uptime_start': system_metrics['uptime_start'],
        'total_requests': system_metrics['total_requests'],
        'latency': {
            'p50_ms': round(p50, 2),
            'p95_ms': round(p95, 2),
            'p99_ms': round(p99, 2),
            'samples': len(latency_values)
        },
        'cache': {
            'hits': system_metrics['cache_hits'],
            'misses': system_metrics['cache_misses'],
            'hit_rate_pct': round(cache_hit_rate, 1)
        },
        'signal_accuracy': system_metrics['signal_accuracy'],
        'market_regime': market_regime
    })

@app.route('/api/signals')
@app.route('/api/journal')
def get_signals():
    """Trade signal journal endpoint"""
    # Filter by status if requested
    status_filter = request.args.get('status', None)
    ticker_filter = request.args.get('ticker', None)
    
    filtered = signal_journal
    if status_filter:
        filtered = [s for s in filtered if s['status'] == status_filter.upper()]
    if ticker_filter:
        filtered = [s for s in filtered if s['ticker'].upper() == ticker_filter.upper()]
    
    return jsonify({
        'status': 'ok',
        'total_signals': len(signal_journal),
        'filtered_count': len(filtered),
        'signals': filtered[-50:]  # Return last 50 matching
    })

@app.route('/api/wallet')
def get_wallet():
    """Returns the latest wallet balance state"""
    try:
        wallet_file = '/Users/adamsussman/Desktop/Active_Projects/vwap_scanner/wallet_state.json'
        if not os.path.exists(wallet_file):
            return jsonify({'status': 'no_data', 'balances': {'eth': 0, 'weth': 0, 'usdc': 0}})
            
        with open(wallet_file, 'r') as f:
            balances = json.load(f)
            
        return jsonify({
            'status': 'ok',
            'balances': balances,
            'timestamp': os.path.getmtime(wallet_file)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/signals_log')
def get_signals_log():
    """Returns the last 200 lines of signals_log.jsonl"""
    try:
        log_file = '/Users/adamsussman/Desktop/Active_Projects/vwap_scanner/signals_log.jsonl'
        logger.info(f"🔍 Reading signals_log from {log_file}")
        
        if not os.path.exists(log_file):
            logger.warning(f"❌ Log file not found at {log_file}")
            return jsonify({'status': 'error', 'message': 'Log file not found'}), 404
            
        signals = []
        with open(log_file, 'r') as f:
            # Efficiently read last 200 lines
            lines = f.readlines()
            logger.info(f"📖 Read {len(lines)} lines from signals_log")
            last_lines = lines[-200:]
            for line in last_lines:
                try:
                    signals.append(json.loads(line))
                except:
                    continue
        
        # Calculate running hit rate from these 200
        wins = [s for s in signals if s.get('win') is True]
        losses = [s for s in signals if s.get('win') is False]
        total = len(wins) + len(losses)
        hit_rate = (len(wins) / total * 100) if total > 0 else 0
        
        return jsonify({
            'status': 'ok',
            'hit_rate': round(hit_rate, 2),
            'total_completed': total,
            'signals': signals[::-1] # Newest first
        })
    except Exception as e:
        logger.error(f"Error reading signals_log: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/spectral_trades')
def get_spectral_trades():
    """Returns full session stats + last 50 actual trades (Multi-Asset M2M PnL)"""
    try:
        log_file = '/Users/adamsussman/Desktop/Active_Projects/vwap_scanner/trades_log.jsonl'
        if not os.path.exists(log_file):
            return jsonify({'status': 'ok', 'trades': [], 'stats': {}})
            
        all_trades = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    all_trades.append(json.loads(line))
                except:
                    continue
                    
        # Filter for executed trades only
        executed = [t for t in all_trades if t.get('status') == 'executed']
        
        # --- SHARPE CALCULATION ---
        returns = [t.get('pnl_usd', 0) / max(1.0, t.get('position_usd', 1.0)) for t in executed]
        sharpe = 0
        if len(returns) > 2:
            try:
                avg = statistics.mean(returns)
                std = statistics.stdev(returns)
                if std > 0:
                    sharpe = (avg / std) * (252**0.5)
            except: pass

        # --- ACTIVE POSITIONS ---
        active_positions = []
        state_path = '/Users/adamsussman/Desktop/Active_Projects/vwap_scanner/position_state.json'
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    p_state = json.load(f)
                    for sym, p in p_state.get('positions', {}).items():
                        active_positions.append({
                            'symbol': p['symbol'],
                            'side': p['side'],
                            'entry_price': p['entry_price'],
                            'position_size': p.get('amount_weth', 0),
                            'timestamp': datetime.fromtimestamp(p['timestamp']).isoformat() if p.get('timestamp') else datetime.now().isoformat()
                        })
            except: pass

        # --- Multi-Asset Stats Calculation ---
        # Track cash and inventory per symbol
        net_usdc = 0.0          # Realized cash flow
        inventory = {}          # { 'ETH-USD': 0.005, 'BTC-USD': -0.001 }
        last_prices = {}        # { 'ETH-USD': 3200, ... }
        total_vol = 0.0
        
        for t in executed:
            sym = t.get('symbol', 'UNKNOWN')
            side = t.get('side')
            units = t.get('position_weth', 0)
            usd_val = t.get('position_usd', 0)
            price = t.get('entry_price', 0)
            
            total_vol += usd_val
            last_prices[sym] = price
            
            if sym not in inventory: inventory[sym] = 0.0
            
            if side == 'BUY':
                net_usdc -= usd_val      # Cash Out
                inventory[sym] += units  # Asset In
            else:
                net_usdc += usd_val      # Cash In
                inventory[sym] -= units  # Asset Out

        # Mark-to-Market PnL
        inventory_value = 0.0
        for sym, units in inventory.items():
            price = last_prices.get(sym, 0)
            inventory_value += units * price
            
        realized_pnl = net_usdc + inventory_value
        
        # Calculate Win Rate
        winners = [t for t in executed if t.get('pnl_usd', 0) > 0]
        win_rate = (len(winners) / len(executed) * 100) if executed else 0
        eth_inv = inventory.get('ETH-USD', 0)
        
        stats = {
            'realized_pnl': round(realized_pnl, 2),
            'total_volume': round(total_vol, 2),
            'trade_count': len(executed),
            'net_usdc_flow': round(net_usdc, 2),
            'net_weth_delta': round(eth_inv, 6),
            'net_portfolio_delta_usd': round(inventory_value, 2),
            'hedge_required': abs(inventory_value) > 500,
            'win_rate_sim': round(win_rate, 1),
            'sharpe_ratio': round(sharpe, 2)
        }
        
        return jsonify({
            'status': 'ok',
            'stats': stats,
            'positions': active_positions,
            'trades': executed[-50:][::-1] 
        })
    except Exception as e:
        logger.error(f"Error in trade stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/regime')
def get_regime():
    """Market regime endpoint"""
    update_market_regime()
    return jsonify({
        'status': 'ok',
        'regime': market_regime
    })

@app.route('/api/prices')
def get_prices():
    """Returns real-time prices for major tracked symbols without cross-contamination."""
    with state_lock:
        active_s = state.get('active_symbol', 'NQ=F')
        # Resolve what price to show for active ticker
        active_price = 0
        if active_s == 'NQ=F': active_price = state['nq']['price']
        elif active_s == 'BTC-USD': active_price = state['btc']['price']
        else:
             # Try to find in scanner if it's an alt
             try:
                 active_price = scanner.get_ticker_detail(active_s).get('price', 0)
             except: pass

        prices = {
            'BTC-USD': state['btc']['price'],
            'NQ=F': state['nq']['price'],
            active_s: active_price
        }
    return jsonify(prices)

@app.route('/api/all_prices')
def get_all_prices():
    """Comprehensive price feed for all tracked tickers (Middle-Out Truth)"""
    results = {}
    for ticker in scanner.tickers:
        try:
            results[ticker] = scanner.get_ticker_detail(ticker).get('price', 0)
        except: pass
    results['BTC-USD'] = state['btc']['price']
    results['NQ=F'] = state['nq']['price']
    return jsonify(results)

@app.route('/api/graph_physics')
def get_graph_physics():
    """
    Graph Physics (Laplacian Arbitrage) endpoint.
    Returns Fiedler regime and potential-based mean-reversion signals.
    Source: GRAINVDB_MOONSHOT_IDEAS.md TIER 5
    """
    try:
        from graph_physics import GraphPhysicsEngine
        import pandas as pd
        
        # Collect price data from our crypto assets
        crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
            'ADA-USD', 'AVAX-USD', 'LINK-USD', 'LTC-USD', 
            'XRP-USD', 'DOT-USD', 'UNI-USD', 'BCH-USD'
        ]
        
        prices_data = {}
        
        # Try to get prices from fmtr data bus first (fastest)
        try:
            with open('/tmp/fmtr_data_bus.json', 'r') as f:
                bus = json.load(f)
                for sym in crypto_symbols:
                    if sym in bus and bus[sym].get('price', 0) > 0:
                        prices_data[sym] = bus[sym]['price']
        except:
            pass
        
        # If not enough data, try multi-provider
        if len(prices_data) < 5 and MULTI_PROVIDER_AVAILABLE:
            try:
                for sym in crypto_symbols:
                    p = get_price(sym)
                    if p and p > 0:
                        prices_data[sym] = p
            except:
                pass
        
        # Need at least 5 assets for meaningful graph analysis
        if len(prices_data) < 5:
            return jsonify({
                'status': 'insufficient_data',
                'fiedler_value': 0.0,
                'regime': 'UNKNOWN',
                'message': f'Only {len(prices_data)} assets available, need 5+',
                'signals': []
            })
        
        # Build a simple price DataFrame (single row for now, expand later)
        # For a proper analysis we'd need historical prices, but this gives regime info
        engine = GraphPhysicsEngine(correlation_threshold=0.4)
        
        # Create mock historical returns based on current prices
        # This is a simplification - ideally we'd store price history
        import numpy as np
        np.random.seed(int(time.time()) % 1000)
        
        n_steps = 50
        symbols = list(prices_data.keys())
        
        # Generate correlated returns based on BTC as market factor
        market_factor = np.cumsum(np.random.normal(0, 0.02, n_steps))
        
        price_df_data = {}
        for sym in symbols:
            beta = 0.8 + np.random.uniform(0, 0.4)
            noise = np.cumsum(np.random.normal(0, 0.01, n_steps))
            price_series = prices_data[sym] * np.exp(beta * market_factor + noise)
            price_df_data[sym] = price_series
        
        prices_df = pd.DataFrame(price_df_data)
        
        result = engine.analyze(prices_df)
        
        # Add to market regime info
        market_regime['fiedler_value'] = round(result['fiedler_value'], 4)
        market_regime['fiedler_regime'] = result['regime']
        
        return jsonify({
            'status': 'ok',
            'fiedler_value': round(result['fiedler_value'], 4),
            'regime': result['regime'],
            'regime_description': {
                'FRAGMENTED': 'Assets moving independently - low correlation risk',
                'NORMAL': 'Healthy correlation structure',
                'HYPER_COUPLED': '⚠️ CRASH/BUBBLE RISK - all assets moving together'
            }.get(result['regime'], 'Unknown'),
            'signals': result['signals'][:10],  # Top 10 opportunities
            'timestamp': time.time()
        })
        
    except ImportError as e:
        return jsonify({
            'status': 'error',
            'message': f'Graph Physics module not available: {e}'
        }), 500
    except Exception as e:
        logger.error(f"Graph Physics error: {e}")
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500


@app.route('/api/funding')
def get_funding_rates():
    """
    Funding Rate Arbitrage endpoint.
    Data from Loris Tools (25+ exchanges, 1500+ symbols).
    Source: https://loris.tools/
    """
    try:
        from loris_bridge import get_top_opportunities, get_funding_sentiment
        
        # Get top 15 arbitrage opportunities
        opportunities = get_top_opportunities(15, min_spread=0.0005)  # >0.05% spread
        
        # Get sentiment for major assets
        sentiment = {}
        for sym in ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP']:
            rate, crowd = get_funding_sentiment(sym)
            sentiment[sym] = {
                'rate_pct': f"{rate:.4%}",
                'crowd': crowd
            }
        
        return jsonify({
            'status': 'ok',
            'top_opportunities': opportunities,
            'sentiment': sentiment,
            'source': 'loris.tools',
            'timestamp': time.time()
        })
        
    except ImportError as e:
        return jsonify({
            'status': 'error',
            'message': f'Loris Bridge module not available: {e}'
        }), 500
    except Exception as e:
        logger.error(f"Funding rates error: {e}")
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500


@app.route('/api/quant-dashboard')
def get_quant_dashboard():
    """Support endpoint for Unified Terminal"""
    # Generate spectral summary from state
    coherence = state.get('coherence', 0.0)
    lead_lag = state.get('lead_lag', 0.0)
    
    spectral = {
        'effective_rank': 1 + (coherence * 3.5),
        'physical_coherence': coherence,
        'lead_lag_parameter': lead_lag,
        'entropy': -np.log(max(0.01, coherence))
    }
    
    # Generate anomalies from scanner results
    anomalies = []
    if market_regime.get('bias') != 'NEUTRAL':
        anomalies.append({
            'ticker': 'NQ',
            'score': abs(market_regime.get('nq_vs_vwap', 0)),
            'description': f"Session bias: {market_regime['bias']}"
        })
        
    return jsonify({
        'spectral_summary': spectral,
        'top_anomalies': anomalies,
        'timestamp': time.time()
    })

@app.route('/vwap')
def vwap_dashboard(): return send_file('vwap_dashboard.html')

@app.route('/anchored')
def anchored_dashboard(): return send_file('anchored_vwap_dashboard.html')
@app.route('/api/scanner')
@app.route('/api/anchored/scan')
def get_anchored_scan():
    if not ANCHORED_SCANNER_AVAILABLE: return jsonify({'status': 'ok', 'data': []})
    global anchored_scanner
    if not anchored_scanner: 
        from anchored_vwap_scanner import AnchoredVWAPScanner
        anchored_scanner = AnchoredVWAPScanner()
    
    # Update market regime before returning results
    update_market_regime()
    
    results = anchored_scanner.get_scan_results()
    return jsonify({
        'status': 'ok', 
        'data': convert_numpy(results), 
        'market_regime': market_regime,
        'timestamp': datetime.now().isoformat()
    })

# Earnings Watchlist for Week of Jan 12, 2026
EARNINGS_WATCHLIST = [
    {'ticker': 'JPM', 'timing': 'FRI BMO', 'sector': 'Finance'},
    {'ticker': 'BAC', 'timing': 'FRI BMO', 'sector': 'Finance'},
    {'ticker': 'C', 'timing': 'FRI BMO', 'sector': 'Finance'},
    {'ticker': 'WFC', 'timing': 'FRI BMO', 'sector': 'Finance'},
    {'ticker': 'BLK', 'timing': 'FRI BMO', 'sector': 'Finance'},
    {'ticker': 'DAL', 'timing': 'FRI BMO', 'sector': 'Airline'},
    {'ticker': 'UNH', 'timing': 'FRI BMO', 'sector': 'Healthcare'},
    {'ticker': 'GS', 'timing': 'WED BMO', 'sector': 'Finance'},
    {'ticker': 'MS', 'timing': 'THU BMO', 'sector': 'Finance'},
    {'ticker': 'DRCT', 'timing': 'MON BMO', 'sector': 'Tech'}, # Monday Jan 12 Specific
    {'ticker': 'GOVX', 'timing': 'MON BMO', 'sector': 'Bio'},  # Monday Jan 12 Specific
]

earnings_cache = {
    'last_update': 0,
    'data': []
}

@app.route('/api/earnings')
def get_earnings_scan():
    """
    Returns earnings watchlist with live gap analysis.
    Cached for 60 seconds to avoid rate limits.
    """
    global earnings_cache
    
    # Check cache
    if time.time() - earnings_cache['last_update'] < 60:
        return jsonify({'status': 'cached', 'data': convert_numpy(earnings_cache['data'])})
        
    results = []
    
    def fetch_ticker_data(item):
        try:
            ticker = item['ticker']
            stock = yf.Ticker(ticker)
            # Fetch 2 days to get prev close and current
            hist = stock.history(period="5d")
            
            if len(hist) < 2:
                # Try getting quote (maybe pre-market)
                info = stock.fast_info
                price = info.last_price
                prev_close = info.previous_close
            else:
                price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                
            # If market is open, use live price; if closed, this acts as "Last Close" vs "Prev Day Close"
            # For pre-market gap, we'd ideally want pre-market price.
            # yfinance history usually gives regular hours. 
            # We'll use this for now as a baseline.
            
            if price is None or prev_close is None: return None

            gap_pct = ((price - prev_close) / prev_close) * 100
            
            # Simple VWAP (Same Day) - approximated as (High+Low+Close)/3 or just Price if only 1 data point
            vwap_today = price 
            if len(hist) > 0:
                last_bar = hist.iloc[-1]
                vwap_today = (last_bar['High'] + last_bar['Low'] + last_bar['Close']) / 3
                
            # Target (Prior VWAP)
            vwap_prior = prev_close # fallback
            if len(hist) > 1:
                prior_bar = hist.iloc[-2]
                vwap_prior = (prior_bar['High'] + prior_bar['Low'] + prior_bar['Close']) / 3

            # Signal Logic
            signal = "NO_TRADE"
            is_gap_play = False
            
            if abs(gap_pct) > 2.0:
                is_gap_play = True
                if gap_pct > 2.0: signal = "WATCH_FOR_LONG"
                else: signal = "WATCH_FOR_SHORT"
            
            volume_ratio = 1.0 # Placeholder unless we get relative vol
            
            return {
                'ticker': ticker,
                'timing': item['timing'],
                'sector': item['sector'],
                'isGapPlay': is_gap_play,
                'signal': signal,
                'gapPercent': round(gap_pct, 2),
                'volumeRatio': round(volume_ratio, 2),
                'currentPrice': round(price, 2),
                'vwapSameDay': round(vwap_today, 2),
                'vwapPriorDay': round(vwap_prior, 2),
                'distanceToEntry': 0, # Needs live calc
                'distanceToTarget': round(((vwap_prior - price) / price) * 100, 2),
                'riskReward': 2.5 # Placeholder
            }
        except Exception as e:
            logger.error(f"Error fetching earnings for {item['ticker']}: {e}")
            return None

    # Parallel Fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_ticker_data, item): item for item in EARNINGS_WATCHLIST}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    
    # Sort: Gap Plays first, then by Ticker
    results.sort(key=lambda x: (not x['isGapPlay'], x['ticker']))
    
    # Update Cache
    earnings_cache = {
        'last_update': time.time(),
        'data': results
    }
    
    return jsonify({'status': 'ok', 'data': convert_numpy(results)})

@app.route('/api/anchored/ticker/<ticker>')
def get_anchored_ticker(ticker):
    if not ANCHORED_SCANNER_AVAILABLE: return jsonify({'status': 'error', 'message': 'Scanner unavailable'})
    global anchored_scanner
    if not anchored_scanner: 
        from anchored_vwap_scanner import AnchoredVWAPScanner
        anchored_scanner = AnchoredVWAPScanner()
    result = anchored_scanner.get_ticker_detail(ticker)
    if result:
        return jsonify({'status': 'ok', 'data': convert_numpy(result)})
    return jsonify({'status': 'error', 'message': 'Ticker not found'}), 404

@app.route('/api/refresh', methods=['POST'])
def trigger_refresh():
    # Force a re-scan in the background thread
    anchored_data_ready.clear()
    return jsonify({'status': 'ok', 'message': 'Refresh triggered'})

@app.route('/system_coherence.json')
def get_system_coherence():
    return jsonify({
        "lead_lag": state.get('lead_lag', 0.0),
        "coherence": state.get('coherence', 0.0),
        "status": "ALIGNED" if state.get('is_stabilized', False) else "SYNCING",
        "is_stabilized": state.get('is_stabilized', False),
        "hit_rates": state.get('hit_rates', {}),
        "expr_hit_rates": state.get('expr_hit_rates', {})
    })

@app.route('/api/ui_render')
def get_ui_render():
    """
    Returns a json-render dynamic UI specification based on current market state.
    This allows the AI to 'decide' how its thoughts are rendered on the dashboard.
    """
    if not TACTICAL_REASONER_AVAILABLE:
        return jsonify({'type': 'Message', 'props': {'text': 'Tactical AI Offline'}})

    # Extract relevant metrics for the reasoner
    with state_lock:
        metrics = {
            'price': state['nq'].get('price', 0),
            'vwap': state['nq'].get('vwap', 0), 
            'signal': 'LONG' if state.get('predictive_trigger') else 'NEUTRAL',
            'ticker': state.get('ticker', 'NQ'),
            'clustering': {
                'clustering_score': state.get('coherence', 0) * 100,
                'entropy': state.get('hw_resilience', {}).get('entropy_score', 0.5),
                'quantum_amplification': state.get('saga_calibration_factor', 1.0)
            }
        }
    
    # Get the spec from the reasoner
    ui_spec = tactical_reasoner.get_ui_spec(metrics)
    
    # Trigger commentary update
    tactical_reasoner.get_commentary(metrics) 
    
    return jsonify(convert_numpy(ui_spec))

@app.route('/api/price-feeds')
def get_price_feeds():
    """Return prices from all active data sources for spread analysis."""
    import requests
    import json as json_lib
    
    feeds = []
    active_sym = state.get('active_symbol', 'BTC-USD')
    is_crypto = '-USD' in active_sym or active_sym in ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # 1. Aggregator Consensus (what we display)
    agg_price = state.get('nq', {}).get('price', 0)
    if agg_price > 0:
        feeds.append({
            'source': 'Aggregator Consensus',
            'price': agg_price,
            'type': 'primary',
            'latency_ms': 0
        })
    
    # 2. Data Bus (cross-app)
    try:
        with open('/tmp/fmtr_data_bus.json', 'r') as f:
            bus = json_lib.load(f)
            bus_entry = bus.get(active_sym, {})
            if bus_entry.get('price', 0) > 0:
                import time as time_mod
                age = time_mod.time() - bus_entry.get('timestamp', 0)
                feeds.append({
                    'source': f"Data Bus ({bus_entry.get('source', 'unknown')})",
                    'price': bus_entry['price'],
                    'type': 'crosswalk',
                    'latency_ms': int(age * 1000)
                })
    except: pass
    
    # 3. Direct exchange APIs (for crypto)
    if is_crypto:
        base = active_sym.replace('-USD', '')
        
        # Coinbase
        try:
            r = requests.get(f'https://api.exchange.coinbase.com/products/{active_sym}/ticker', timeout=1)
            if r.ok:
                feeds.append({'source': 'Coinbase', 'price': float(r.json()['price']), 'type': 'exchange'})
        except: pass
        
        # Kraken
        try:
            pair = f'X{base}ZUSD' if base == 'BTC' else f'{base}USD'
            r = requests.get(f'https://api.kraken.com/0/public/Ticker?pair={pair}', timeout=1)
            if r.ok:
                result = r.json().get('result', {})
                for k, v in result.items():
                    feeds.append({'source': 'Kraken', 'price': float(v['c'][0]), 'type': 'exchange'})
                    break
        except: pass
        
        # Binance
        try:
            sym = f'{base}USDT'
            r = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={sym}', timeout=1)
            if r.ok:
                feeds.append({'source': 'Binance', 'price': float(r.json()['price']), 'type': 'exchange'})
        except: pass
    
    # Calculate spread
    prices = [f['price'] for f in feeds if f['price'] > 0]
    if len(prices) >= 2:
        min_p, max_p = min(prices), max(prices)
        spread = max_p - min_p
        spread_pct = (spread / min_p) * 100 if min_p > 0 else 0
    else:
        spread, spread_pct = 0, 0
    
    return jsonify(convert_numpy({
        'symbol': active_sym,
        'feeds': feeds,
        'spread': round(spread, 2),
        'spread_pct': round(spread_pct, 4),
        'spread_status': 'TIGHT' if spread_pct < 0.02 else ('NORMAL' if spread_pct < 0.1 else 'WIDE'),
        'timestamp': time.time()
    }))


@app.route('/api/intraday/<ticker>')
def get_api_intraday(ticker):
    try:
        # Standardize ticker for internal systems (NQ -> NQ=F)
        yf_ticker = ticker.replace("NQ", "NQ=F").replace("BTC", "BTC-USD")
        
        # 1. Fetch Intraday History (5-day context)
        if MULTI_PROVIDER_AVAILABLE:
            hist = get_history(yf_ticker)
        else:
            # Fallback: use yfinance directly if multi_provider unavailable
            import yfinance as yf
            try:
                stock = yf.Ticker(yf_ticker)
                df = stock.history(period='5d', interval='1m')
                hist = []
                if not df.empty:
                    for idx, row in df.iterrows():
                        hist.append({
                            'time': idx.strftime('%H:%M [%m/%d]') if hasattr(idx, 'strftime') else str(idx),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'price': float(row['Close']),
                            'volume': float(row['Volume'])
                        })
            except Exception as yf_err:
                logger.error(f"Fallback yfinance history failed: {yf_err}")
                hist = []
        
        # 2. Fetch Anchored Overlay (The "Strategic" View)
        anchored_meta = None
        global anchored_scanner
        # JIT Initialization of Scanning Engine
        if (not anchored_scanner) and ANCHORED_SCANNER_AVAILABLE:
            try:
                from anchored_vwap_scanner import AnchoredVWAPScanner
                anchored_scanner = AnchoredVWAPScanner()
            except: pass
            
        if anchored_scanner:
            try:
                # 1. Try fast cache lookup first (Zero Latency)
                detail = anchored_scanner.get_ticker_detail(yf_ticker, only_cached=True)
                
                if not detail:
                    # 2. Cache Miss: Trigger background update and return ONLY price data for now
                    # This ensures the chart LOADS INSTANTLY. The next poll will get the anchored data.
                    logger.info(f"Anchored cache miss for {ticker} - scheduling background analysis")
                    def bg_analyze():
                        try:
                            anchored_scanner.get_ticker_detail(yf_ticker, only_cached=False)
                        except: pass
                    threading.Thread(target=bg_analyze, daemon=True).start()
                
                if detail:
                    anchored_meta = {
                        'vwap': detail['vwap'],
                        'type': detail['anchor_type'],
                        'stdev': detail['vwap_std'],
                        'upper': detail['bands']['upper_2'], 
                        'lower': detail['bands']['lower_2'],
                        'date': detail['anchor_date'],
                        'entry': detail['trade']['entry'],
                        'stop': detail['trade']['stop'],
                        'target': detail['trade']['target']
                    }
                    # STRATEGIC FIX: If the scanner has a high-fidelity anchored series, use it as the primary history.
                    # This ensures the chart starts from the anchor/trigger event as requested.
                    if detail.get('price_series'):
                        hist = detail['price_series']
            except Exception as e:
                # Analysis failure shouldn't block the chart
                logger.error(f"Anchored overlay failed for {ticker}: {e}")

        # Final sanity check: if history is too long, trim it for performance.
        # But ensure we keep enough to see the context.
        if len(hist) > 600:
            hist = hist[-600:]

        # Return Unified Data Model
        return jsonify({'status': 'ok', 'data': convert_numpy(hist), 'anchored': convert_numpy(anchored_meta)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/switch/<symbol>')
def api_switch_symbol(symbol):
    s = symbol.upper().replace("/", "")
    if s == "NQ": s = "NQ=F"
    if s == "BTC": s = "BTC-USD"
    
    # QC FIX: Atomic state transition
    with state_lock:
        state['active_symbol'] = s
        state['ticker'] = s.split("-")[0].replace("=F", "")
        # Clear ALL prediction state on symbol switch to prevent cross-symbol contamination
        state['price_series'] = []
        state['tick_buffer'] = []
        state['ghost_bars'] = None
        state['ghost_upper'] = []
        state['ghost_lower'] = []
        state['nq']['price'] = 0.0
        state['system_bias'] = 0.0
        state['pending_predictions'] = []  # CRITICAL: Clear stale predictions
        # Reset hit rates for new symbol
        state['hit_rates'] = {
            '5m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
            '10m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
            '15m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0}
        }
        state['expr_hit_rates'] = {
            '5m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
            '10m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0},
            '15m': {'rate': 0, 'total': 0, 'hits': 0, 'dir_hits': 0, 'dir_rate': 0, 'cone_hits': 0, 'cone_rate': 0, 'prec_hits': 0, 'prec_rate': 0}
        }

    # QC FIX: Backfill history OUTSIDE of lock to prevent blocking the data loop too long
    # But we update the price_series safely afterwards
    try:
        new_hist = get_history(s)
        if new_hist:
            with state_lock:
                # Only update if the symbol hasn't switched AGAIN while we were fetching
                if state['active_symbol'] == s:
                    state['price_series'] = new_hist
                    if new_hist:
                        state['nq']['price'] = new_hist[-1]['close']
    except Exception as e:
        logger.error(f"Backfill error on switch to {s}: {e}")
    
    logger.info(f"Symbol switched to {s} - All prediction state cleared")
    return jsonify({'status': 'ok', 'symbol': s})

@socketio.on('switch_symbol')
def handle_socket_switch(data):
    api_switch_symbol(data.get('symbol', 'NQ=F'))

@app.route('/dashboard')
def dashboard():
    return send_file('nq_dashboard.html')

if __name__ == '__main__':
    # Start FlowSurface Bridge for native GPU charting
    threading.Thread(target=fmtr_bridge.run, daemon=True, name="FlowSurfaceBridge").start()
    
    # Start data streaming loop in its own thread
    threading.Thread(target=stream_data_loop, daemon=True, name="DataStreamLoop").start()
    
    # Start anchored VWAP scanner loop
    threading.Thread(target=anchored_scanner_loop, daemon=True, name="AnchoredScannerLoop").start()
    
    logger.info(f"Server starting on port {PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, allow_unsafe_werkzeug=True)
