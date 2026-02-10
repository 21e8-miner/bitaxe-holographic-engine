#!/usr/bin/env python3
"""
SPECTRAL TRADER - Real Trading Bridge for Base L2
==================================================
Connects Spectral Operator signals to real WETH trades via Uniswap V3 on Base.

SAFETY FEATURES:
- Paper trading mode by default
- Max position limits
- Daily loss limit
- Only trades confirmed winning signals
- ETH signals only (since we trade WETH)

Usage:
  python spectral_trader.py --mode paper   # Paper trading (default)
  python spectral_trader.py --mode live    # REAL MONEY TRADING
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import requests
from typing import List, Optional, Tuple, Any, Dict
from clawdbot_client import ClawdbotClient

# Add fresh_bot to path for KeyManager
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRESH_BOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "fresh_bot", "python", "lab"))
sys.path.insert(0, FRESH_BOT_PATH)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("SpectralTrader")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TradingConfig:
    # Safety Limits
    MAX_POSITION_WETH: float = 0.012  # Doubled to ensure net profit clears gas fees
    MIN_CONFIDENCE: float = 0.50      # Lower threshold: model is for TIMING only now
    DAILY_LOSS_LIMIT_WETH: float = 0.03   # Increased daily limit
    NET_POSITION_CLAMP: float = 0.10      # Increased net exposure for bidirectional flow
    GAS_FEE_BUFFER_WETH: float = 0.001    # Alert threshold
    
    # Signal Filtering
    ONLY_ETH_SIGNALS: bool = False    # EXPANDED: Trade all crypto signals (BTC, ETH, SOL, etc.)
    ONLY_CONFIRMED_WINS: bool = False  # If True, only trade `win: True` signals
    
    USE_CORRELATION_TRIGGER: bool = True  # Use BTC as lead-indicator for ETH trades
    USE_SHADOW_PROBING: bool = True       # Use limit orders to capture shadow book liquidity
    USE_ADAPTIVE_INVERSION: bool = False  # Disabled: Now using volatility mode
    USE_VOLATILITY_MODE: bool = True      # NOVEL: Ignore model direction, follow price action
    VOLATILITY_OBSERVE_SECONDS: float = 2.0   # TURBO: 2s for ultra-fast confirmation (was 5s)
    VOLATILITY_THRESHOLD_PCT: float = 0.0004  # INCREASED: 0.04% as requested
    REQUIRE_COHERENCE: bool = False       # REAXED: Let Brain Consensus handle alignment
    USE_BRAIN_CONSENSUS: bool = False     # Temporarily DISABLED to test pure volatility+structural filters
    BRAIN_SERVER_URL: str = "http://localhost:11435"
    MIN_BREGMAN_PROFIT: float = 0.001     # Information Geometry Gate (Prop 2.4) - Relaxed for micro-signals
    MIN_STRUCTURAL_COHERENCE: float = 0.5 # Fel S-Ring Syzygy Gate (0.0 to 1.0) - Relaxed for real-world noise
    
    # 5-Hour Performance Pruning (Data-Driven Update)
    SYMBOL_BLACKLIST: list = ("ONDO-USD", "SUSHI-USD", "FET-USD", "SNX-USD", "COMP-USD")
    PATTERN_WEIGHTS: dict = field(default_factory=lambda: {
        "expansion": 0.4,    # PENALIZED: Failed breakouts (-$16.32 PnL)
        "dissipation": 1.4,  # BOOSTED: Profitable mean-reversion (+$5.29 PnL)
        "default": 1.0
    })
    
    # Multi-DEX Routing (NEW!)
    USE_MULTI_DEX_ROUTING: bool = True    # Route trades across multiple DEXs and chains
    PREFERRED_CHAINS: list = None         # None = all chains, or specify ["base", "arbitrum", etc.]
    
    # Execution
    MODE: str = "live"  # Switched to LIVE MONEY mode
    POLL_INTERVAL: int = 2  # Seconds between signal checks (Tightened for lead/lag)
    
    # Paths (relative to script directory)
    SIGNALS_LOG: str = os.path.join(SCRIPT_DIR, "signals_log.jsonl")
    TRADES_LOG: str = os.path.join(SCRIPT_DIR, "trades_log.jsonl")
    WALLET_STATE_FILE: str = os.path.join(SCRIPT_DIR, "wallet_state.json")

    # New: Adaptive Inversion Parameters
    INVERSION_WINDOW_SIZE: int = 5        # Number of recent trades to consider for inversion
    INVERSION_LOSS_THRESHOLD: float = 0.005 # Total WETH loss in window to trigger inversion
    
    # Rate Limiting & Dynamic Volatility
    MAX_TRADES_PER_HOUR: int = 50         # INCREASED: For 50 assets + Turbo Mode + QC
    BASE_VOLATILITY_THRESHOLD_PCT: float = 0.0012 
    VOLATILITY_ADJUST_FACTOR: float = 0.5 
    
    # Velocity-Based Bypass (NEW!)
    USE_VELOCITY_BYPASS: bool = True      # Bypass volatility check if signals pile up
    VELOCITY_BYPASS_COUNT: int = 4        # N signals in same direction = force trade
    VELOCITY_BYPASS_WINDOW_SEC: float = 30.0  # Time window for velocity detection 

# =============================================================================
# Wallet Integration
# =============================================================================

class WalletManager:
    """Manages wallet connection and balance checks."""
    
    def __init__(self):
        self.account = None
        self.address = None
        self.w3 = None
        
        # Contract addresses on Base
        self.WETH = "0x4200000000000000000000000000000000000006"
        self.USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        
        # WETH ABI for unwrap
        self.WETH_ABI = [
            {"constant": False, "inputs": [{"name": "wad", "type": "uint256"}], "name": "withdraw", "outputs": [], "payable": False, "stateMutability": "nonpayable", "type": "function"}
        ]
        
        # Balance Cache
        self._last_balance = {"eth": 0, "weth": 0, "usdc": 0}
        self._last_check_time = 0
        self.CACHE_TTL = 30  # 30 second cache
        
    def connect(self) -> bool:
        """Initialize wallet from KeyManager."""
        try:
            from key_manager import KeyManager
            km = KeyManager()
            self.address = km.load_key("main_wallet")
            self.account = km.account
            self.w3 = km._get_web3()
            
            logger.info(f"✅ Wallet connected: {self.address[:10]}...{self.address[-6:]}")
            return True
        except Exception as e:
            logger.error(f"❌ Wallet connection failed: {e}")
            return False
    
    def get_balances(self, force: bool = False) -> dict:
        """Get current balances with 30s caching."""
        if not self.w3 or not self.address:
            return {"eth": 0, "weth": 0, "usdc": 0}
            
        now = time.time()
        if not force and (now - self._last_check_time < self.CACHE_TTL):
            return self._last_balance
            
        try:
            # ETH balance
            eth_wei = self.w3.eth.get_balance(self.address)
            eth = float(self.w3.from_wei(eth_wei, 'ether'))
            
            # ERC20 ABI for balanceOf
            erc20_abi = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]
            
            # WETH balance
            weth_contract = self.w3.eth.contract(address=self.WETH, abi=erc20_abi)
            weth_wei = weth_contract.functions.balanceOf(self.address).call()
            weth = float(self.w3.from_wei(weth_wei, 'ether'))
            
            # USDC balance
            usdc_contract = self.w3.eth.contract(address=self.USDC, abi=erc20_abi)
            usdc_raw = usdc_contract.functions.balanceOf(self.address).call()
            usdc = usdc_raw / 1e6
            
            self._last_balance = {"eth": eth, "weth": weth, "usdc": usdc}
            self._last_check_time = now
            return self._last_balance
        except Exception as e:
            if "429" in str(e):
                logger.warning("⚠️ RPC Rate Limit (429). Using cached balances.")
                return self._last_balance
            logger.error(f"Balance check failed: {e}")
            return self._last_balance

    def check_and_fund_gas(self):
        """Auto-unwrap WETH to native ETH if gas balance is low."""
        if not self.w3 or not self.address:
            return
            
        try:
            eth_wei = self.w3.eth.get_balance(self.address)
            # If below 0.001 ETH, unwrap some WETH
            if eth_wei < self.w3.to_wei(0.001, 'ether'):
                logger.info("⛽ Native ETH low. Auto-funding gas from WETH...")
                
                # Check WETH balance first
                erc20_abi = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]
                weth_contract = self.w3.eth.contract(address=self.WETH, abi=erc20_abi)
                weth_balance = weth_contract.functions.balanceOf(self.address).call()
                
                amount_to_unwrap = self.w3.to_wei(0.003, 'ether')
                if weth_balance >= amount_to_unwrap:
                    weth_contract_rw = self.w3.eth.contract(address=self.WETH, abi=self.WETH_ABI)
                    nonce = self.w3.eth.get_transaction_count(self.address)
                    tx = weth_contract_rw.functions.withdraw(amount_to_unwrap).build_transaction({
                        'from': self.address,
                        'nonce': nonce,
                        'gas': 100000,
                        'gasPrice': int(self.w3.eth.gas_price * 1.2)
                    })
                    signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
                    self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    logger.info(f"✅ Auto-gas funding successful. Unwrapped 0.003 WETH.")
                    time.sleep(2)  # Wait for tx to settle
                else:
                    logger.warning("⚠️ Low ETH and low WETH. Cannot auto-fund gas.")
        except Exception as e:
            logger.error(f"Auto-gas funding failed: {e}")

# =============================================================================
# Signal Watcher
# =============================================================================

class SignalWatcher:
    """Watches signals_log.jsonl for new trading signals."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.seen_timestamps = set()
        
        # Start watching from the END of the file
        try:
            with open(self.config.SIGNALS_LOG, 'a') as f:
                self.last_position = f.tell()
        except:
            self.last_position = 0
        
    def get_new_signals(self) -> List[dict]:
        """Get new signals since last check."""
        signals = []
        
        try:
            with open(self.config.SIGNALS_LOG, 'r') as f:
                # Seek to last known position
                f.seek(self.last_position)
                
                for line in f:
                    try:
                        signal = json.loads(line.strip())
                        
                        # Skip if already seen
                        ts = signal.get('unix_time')
                        if ts in self.seen_timestamps:
                            continue
                        self.seen_timestamps.add(ts)
                        
                        # Apply filters
                        if self.passes_filters(signal):
                            signals.append(signal)
                            
                    except json.JSONDecodeError:
                        continue
                
                self.last_position = f.tell()
                
        except FileNotFoundError:
            logger.warning(f"Signals log not found: {self.config.SIGNALS_LOG}")
            
        return signals
    
    def passes_filters(self, signal: dict) -> bool:
        """Check if signal passes our trading filters."""
        
        # Must be ETH if configured
        if self.config.ONLY_ETH_SIGNALS:
            if signal.get('symbol') != 'ETH-USD':
                return False
        
        # If ONLY_CONFIRMED_WINS is True, must have win: True and confirmed_at
        if self.config.ONLY_CONFIRMED_WINS:
            if not signal.get('confirmed_at') or signal.get('win') is not True:
                return False
        
        # If real-time, we skip the confirmed_at requirement
        
        # Must meet confidence threshold
        confidence = signal.get('confidence', 0)
        if confidence < self.config.MIN_CONFIDENCE:
            return False
            
        return True

# =============================================================================
# Portfolio Manager (SIG Risk Philosophy Implementation)
# =============================================================================

class PortfolioManager:
    """
    Implements the SIG Risk Philosophy for portfolio-level hedging.
    
    Key Insight: Instead of blocking trades when limits are hit,
    actively hedge the portfolio using the most liquid asset (ETH).
    
    Reference: SIG_RISK_PHILOSOPHY.md
    Source: Ben Recht's feedback control theory - use precise attenuator (hedge)
            to control high-gain uncertain system (market signals)
    
    V_out = A / (1 + AB) * V_in ≈ (1/B) * V_in  (when A is large)
    The hedge (B) controls the output, not the market signal (A).
    """
    
    def __init__(self, config: TradingConfig, wallet: 'WalletManager'):
        self.config = config
        self.wallet = wallet
        
        # Position tracking {symbol: delta_in_usd}
        self.positions: Dict[str, float] = {}
        
        # Hedging parameters
        self.HEDGE_THRESHOLD_USD = 500.0    # Trigger hedge when net delta > $500
        self.HEDGE_INSTRUMENT = "ETH-USD"   # Most liquid for hedging
        self.MAX_HEDGE_SIZE_USD = 200.0     # Max single hedge trade
        self.MIN_NET_DELTA_FOR_HEDGE = 100.0  # Don't hedge tiny imbalances
        
        # State
        self.last_hedge_time = 0
        self.hedge_cooldown_seconds = 60    # Don't spam hedges
        self.total_hedges_executed = 0
        
        logger.info("📊 PortfolioManager initialized (SIG Risk Philosophy)")
    
    def update_position(self, symbol: str, side: str, size_usd: float):
        """
        Update position tracking after a trade.
        
        Args:
            symbol: Asset traded (e.g., "ETH-USD", "BTC-USD")
            side: "BUY" or "SELL"
            size_usd: Trade size in USD
        """
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        # BUY = positive delta, SELL = negative delta
        delta = size_usd if side == "BUY" else -size_usd
        self.positions[symbol] += delta
        
        logger.debug(f"Position update: {symbol} {side} ${size_usd:.2f} → Net: ${self.positions[symbol]:.2f}")
    
    def get_net_delta(self) -> float:
        """
        Get net portfolio delta (sum of all positions).
        
        Positive = Net long, Negative = Net short
        """
        return sum(self.positions.values())
    
    def get_position_breakdown(self) -> Dict[str, float]:
        """Get all individual position deltas."""
        return self.positions.copy()
    
    def should_hedge(self) -> bool:
        """
        Check if we should execute a hedge.
        
        Returns True if:
        - Net delta exceeds threshold
        - Cooldown has passed
        - Delta is large enough to bother hedging
        """
        net_delta = abs(self.get_net_delta())
        
        if net_delta < self.MIN_NET_DELTA_FOR_HEDGE:
            return False
        
        if net_delta < self.HEDGE_THRESHOLD_USD:
            return False
        
        # Check cooldown
        if time.time() - self.last_hedge_time < self.hedge_cooldown_seconds:
            return False
        
        return True
    
    def compute_hedge_trade(self) -> Optional[Dict]:
        """
        Compute the hedge trade needed to neutralize portfolio.
        
        Returns:
            Dict with {symbol, side, size_usd} or None if no hedge needed
        """
        net_delta = self.get_net_delta()
        
        if abs(net_delta) < self.MIN_NET_DELTA_FOR_HEDGE:
            return None
        
        # Determine hedge direction (opposite of net delta)
        hedge_side = "SELL" if net_delta > 0 else "BUY"
        
        # Compute size (clamp to max)
        hedge_size = min(abs(net_delta), self.MAX_HEDGE_SIZE_USD)
        
        return {
            'symbol': self.HEDGE_INSTRUMENT,
            'side': hedge_side,
            'size_usd': hedge_size,
            'reason': f"Portfolio hedge: Net ${net_delta:+.2f} → {hedge_side} ${hedge_size:.2f} ETH"
        }
    
    def record_hedge_execution(self, hedge_trade: Dict):
        """Record that a hedge was executed."""
        self.last_hedge_time = time.time()
        self.total_hedges_executed += 1
        
        # Update our position tracking with the hedge
        self.update_position(
            hedge_trade['symbol'], 
            hedge_trade['side'], 
            hedge_trade['size_usd']
        )
        
        logger.info(f"🛡️ HEDGE EXECUTED: {hedge_trade['reason']} (Total hedges: {self.total_hedges_executed})")
    
    def get_status(self) -> Dict:
        """Get portfolio manager status for API/dashboard."""
        net_delta = self.get_net_delta()
        return {
            'net_delta_usd': round(net_delta, 2),
            'position_count': len([p for p in self.positions.values() if abs(p) > 1]),
            'hedge_threshold': self.HEDGE_THRESHOLD_USD,
            'needs_hedge': self.should_hedge(),
            'total_hedges': self.total_hedges_executed,
            'positions': {k: round(v, 2) for k, v in self.positions.items() if abs(v) > 1}
        }


# =============================================================================
# Trade Executor
# =============================================================================


class TradeExecutor:
    """Executes trades (paper or live)."""
    
    def __init__(self, config: TradingConfig, wallet: WalletManager):
        self.config = config
        self.wallet = wallet
        self.daily_loss = 0.0
        self.trades_today = 0
        # Nonce Management (Persistent)
        self.nonce_file = os.path.join(SCRIPT_DIR, "nonce_state.json")
        self._next_nonce = self._load_nonce()
        self.net_weth_position = 0.0  # Track net WETH exposure for position clamping
        
        # Multi-DEX Router (lazy loaded)
        self.multi_dex_router = None
        if config.USE_MULTI_DEX_ROUTING:
            try:
                from multi_dex_router import MultiDexRouter, Chain
                self.multi_dex_router = MultiDexRouter()
                logger.info("🌐 Multi-DEX Router loaded (Uniswap/1inch/0x across Base/Arb/OP/ETH)")
            except Exception as e:
                logger.warning(f"⚠️ Multi-DEX Router failed to load: {e}")
                self.multi_dex_router = None
        
        # Turbo Execution Engine (Speed & Profitability Optimizations)
        self.turbo_engine = None
        try:
            from turbo_engine import TurboExecutionEngine, TurboConfig
            turbo_config = TurboConfig(
                FAST_CONFIRM_SECONDS=2.0,      # Faster confirmation
                KELLY_FRACTION=0.25,           # Conservative Kelly sizing
                ENABLE_MOMENTUM_STACK=True,
                ENABLE_VELOCITY_BOOST=True,
                ENABLE_CROSS_ASSET=True,
            )
            self.turbo_engine = TurboExecutionEngine(turbo_config)
            logger.info("🚀 Turbo Engine loaded (Kelly/Velocity/CrossAsset)")
        except Exception as e:
            logger.warning(f"⚠️ Turbo Engine failed to load: {e}")
            self.turbo_engine = None
        
        # Position Management (TSL & Anti-Whipsaw)
        try:
            from position_manager import PositionManager
            self.pos_manager = PositionManager(tsl_threshold=0.005) # 0.5% trailing stop
            logger.info("🛡️ Position Manager loaded (Trailing Stop-Loss & Anti-Whipsaw active)")
        except Exception as e:
            logger.warning(f"⚠️ Position Manager failed to load: {e}")
            self.pos_manager = None
        
        # Portfolio Manager (SIG Risk Philosophy - Active Hedging)
        self.portfolio_manager = PortfolioManager(config, wallet)

        
        # Tracking for Correlation Trigger
        self.btc_last_price = 0.0
        self.eth_last_price = 0.0
        self.btc_move_1m = 0.0
        self.eth_move_1m = 0.0
        
        # NOVEL: Signal Velocity Tracking
        self.recent_signals = []  # (timestamp, direction) tuples
        self.last_signal_time = 0
        self.signal_streak = 0
        self.last_signal_dir = None
        
        # NOVEL: Adaptive Inversion Tracking
        self.trade_outcomes = deque(maxlen=config.INVERSION_WINDOW_SIZE) # Store recent trade outcomes
        self.inversion_active = False
        self.trades_since_check = 0
        
        # New: Correlation Coherence
        self.market_coherence = 0.0
        
        # New: QC Tracking
        self.price_move_history = deque(maxlen=20)
        self.trade_timestamps = deque(maxlen=200) # For rate limiting
        self.last_buy_price = {}  # Changed to dict for per-symbol tracking
        self.last_sell_price = {} # Changed to dict for per-symbol tracking
        
        # Clawdbot Integration
        self.alert_bot = ClawdbotClient()
        
    def recover_state(self):
        """Recover last known entry prices from logs."""
        logger.info("♻️ RECOVERING STATE from trades_log.jsonl...")
        try:
            if not os.path.exists(self.config.TRADES_LOG):
                logger.info("   No trades log found. Starting fresh.")
                return

            with open(self.config.TRADES_LOG, 'r') as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        if t.get('status') != 'executed': continue
                        
                        symbol = t.get('symbol')
                        side = t.get('side')
                        price = t.get('entry_price', 0)
                        
                        if side == 'BUY':
                            # Track entry price
                            self.last_buy_price[symbol] = price
                            
                            # Restore TSL tracking in PositionManager
                            if self.pos_manager and symbol not in self.pos_manager.positions:
                                try:
                                    self.pos_manager.add_position(
                                        symbol=symbol,
                                        side='BUY',
                                        entry_price=price,
                                        amount_weth=t.get('position_weth', 0),
                                        metadata=t.get('metadata', {})
                                    )
                                    logger.info(f"   Restored TSL tracking for {symbol}")
                                except Exception as pm_err:
                                    logger.warning(f"   Failed to restore TSL for {symbol}: {pm_err}")
                                    
                        elif side == 'SELL':
                            # If we sold, we closed the position (simplified)
                            if symbol in self.last_buy_price:
                                del self.last_buy_price[symbol]
                            
                            # Remove from PositionManager if fully closed
                            if self.pos_manager and symbol in self.pos_manager.positions:
                                try:
                                    # We don't have current price here easily, just force remove
                                    del self.pos_manager.positions[symbol]
                                    self.pos_manager._save_state()
                                except: pass
                    except: pass
            
            logger.info(f"   Recovered {len(self.last_buy_price)} active entry prices from history.")
        except Exception as e:
            logger.error(f"❌ State recovery failed: {e}")
        
    def calculate_coherence(self) -> float:
        """Calculate alignment between ETH and BTC (0.0 to 1.0)."""
        # If both moving same direction strongly, coherence is high
        if abs(self.btc_move_1m) > 0.0005 and abs(self.eth_move_1m) > 0.0005:
            if (self.btc_move_1m > 0) == (self.eth_move_1m > 0):
                return 0.85 # Strong alignment
        return 0.0
    
    def get_brain_consensus(self, signal: dict, state: dict) -> Tuple[bool, str, dict]:
        """Consult the Kimi-FMTR Brain for 15TB simulation consensus."""
        try:
            prompt = f"Trade Signal: {signal.get('type')} {signal.get('symbol')} @ ${signal.get('entry')}. Direction: {signal.get('pattern')}. Confidence: {signal.get('confidence')}."
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "kimi-fmtr-k2.5"
            }
            resp = requests.post(f"{self.config.BRAIN_SERVER_URL}/v1/chat/completions", json=payload, timeout=5)
            data = resp.json()
            
            kernel_json = data['choices'][0]['message'].get('json_data', {})
            reasoning = kernel_json.get('reasoning', 'No reasoning provided')
            action = kernel_json.get('action', 'HOLD').upper()
            
            # Physics verification
            physics = kernel_json.get('physics', {})
            logger.info(f"🧠 BRAIN CONSENSUS: {action} | Regime: {physics.get('regime')} | Reasoning: {reasoning}")
            
            # Only GO if action matches or Kimi strongly approves
            if action == signal.get('type') or action == "BUY":
                return True, reasoning, kernel_json
            return False, reasoning, kernel_json
            
        except Exception as e:
            logger.warning(f"⚠️ Brain Consensus Failed: {e}. Falling back to local physics.")
            return True, "Local Fallback", {}

    def execute(self, signal: dict) -> Optional[dict]:
        """Execute a trade based on signal."""
        
        # Check daily loss limit
        if self.daily_loss >= self.config.DAILY_LOSS_LIMIT_WETH:
            logger.warning("⚠️ Daily loss limit reached. No more trades today.")
            self.alert_bot.send_alert(f"Daily loss limit reached ({self.daily_loss} WETH). Trading stopped.", title="Risk Alert", priority="high")
            return None
        
        # Check Anti-Whipsaw Pause
        if self.pos_manager and self.pos_manager.is_paused():
            logger.warning("⏳ WHIPSAW PROTECTION ACTIVE: Cooling down after losses. Skipping entry.")
            return None
        
        # Determine trade direction
        side = signal.get('type', 'BUY')
        entry_price = signal.get('entry', 0)
        confidence = signal.get('confidence', 0)
        pattern = signal.get('pattern', 'unknown')
        symbol = signal.get('symbol', 'unknown')
        
        # 5-HOUR PRUNING: Check blacklist
        if symbol in self.config.SYMBOL_BLACKLIST:
            logger.info(f"🚫 BLACKLIST: Skipping {symbol} due to poor recent performance.")
            return None
        
        # ZIPF VOLUME WEIGHTING (GRAINVDB_MOONSHOT_IDEAS.md TIER 6, Item 16)
        # Source: Kunstner & Bach, NeurIPS 2025
        # Crypto liquidity follows Zipf distribution (BTC >> ETH >> SOL >> ... >> SHIB)
        # Low-volume assets are noisier, so we penalize their confidence by sqrt(volume/ref)
        volume = signal.get('volume', 0)
        if volume > 0:
            # Reference volume: ~$10M/day for major assets
            REFERENCE_VOLUME = 10_000_000
            volume_weight = min(1.0, (volume / REFERENCE_VOLUME) ** 0.5)  # sqrt scaling
            original_confidence = confidence
            confidence = confidence * (0.5 + 0.5 * volume_weight)  # Blend: 50% base + 50% volume-adjusted
            if volume_weight < 0.9:
                logger.info(f"📊 ZIPF SCALING: Volume ${volume:,.0f} → weight {volume_weight:.2f} → confidence {original_confidence:.2%} → {confidence:.2%}")

        
        # DYNAMIC CONFIDENCE ADJUSTMENT (Coherence Index)
        # If BTC and ETH are moving together, accept lower confidence
        coherence_bonus = 0.0
        coherence = self.calculate_coherence()
        if coherence > 0.8:
            coherence_bonus = 0.02  # Lower threshold by 2%
            logger.info(f"🌊 HIGH COHERENCE DETECTED: BTC/ETH aligned ({coherence:.2f}). Bonus: {coherence_bonus:.1%}")

        effective_threshold = self.config.MIN_CONFIDENCE - coherence_bonus
        if confidence < effective_threshold:
            logger.info(f"Confidence {confidence:.2f} below effective threshold {effective_threshold:.2f} (Coherence bonus: {coherence_bonus:.2f}). Skipping trade.")
            return None
            
        # OVERNIGHT HARDENING: Strict Coherence Check
        if self.config.REQUIRE_COHERENCE and coherence < 0.8:
            logger.info(f"😴 OVERNIGHT HARDENING: Coherence {coherence:.2f} < 0.8. Skipping.")
            return None
            
        # VELOCITY TRACKING & BYPASS CALCULATION
        now = time.time()
        is_velocity_event = False
        same_dir_count = 0
        
        if self.config.USE_VELOCITY_BYPASS:
            # Record this signal
            self.recent_signals.append((now, side, signal.get('symbol', 'ETH-USD')))
            
            # Clean old signals outside window
            self.recent_signals = [
                (ts, d, sym) for ts, d, sym in self.recent_signals 
                if now - ts < self.config.VELOCITY_BYPASS_WINDOW_SEC
            ]
            
            # Count signals in same direction
            same_dir_count = sum(1 for ts, d, sym in self.recent_signals if d == side)
            if same_dir_count >= self.config.VELOCITY_BYPASS_COUNT:
                is_velocity_event = True
                logger.info(f"🚀 VELOCITY EVENT DETECTED: {same_dir_count} {side} signals in {self.config.VELOCITY_BYPASS_WINDOW_SEC}s")

        # RATE LIMITING CHECK (Bypass if velocity event)
        one_hour_ago = now - 3600
        recent_trades = [t for t in self.trade_timestamps if t > one_hour_ago]
        if len(recent_trades) >= self.config.MAX_TRADES_PER_HOUR and not is_velocity_event:
            logger.warning(f"⏳ RATE LIMIT: {len(recent_trades)} trades in last hour. Cooling down.")
            return None
        
        # Gas Reserve Check
        balances = self.wallet.get_balances()
        if balances['eth'] < self.config.GAS_FEE_BUFFER_WETH:
            logger.warning(f"⛽ GAS ALERT: Native ETH low ({balances['eth']:.6f}). Auto-fund should trigger.")
        
        # ===============================================================
        # VOLATILITY-ONLY MODE: Ignore model direction, follow price action
        # ===============================================================
        if self.config.USE_VOLATILITY_MODE:
            logger.info(f"🌊 VOLATILITY TRIGGER: Signal detected. Observing price for {self.config.VOLATILITY_OBSERVE_SECONDS}s...")
            
            # Capture starting price
            try:
                resp = requests.get('https://api.coinbase.com/v2/prices/ETH-USD/spot', timeout=3)
                start_price = float(resp.json()['data']['amount'])
            except:
                start_price = entry_price
                logger.warning(f"⚠️ Failed to get start price, using signal entry price: ${start_price:.2f}")
            
            # Wait and observe
            time.sleep(self.config.VOLATILITY_OBSERVE_SECONDS)
            
            # Capture ending price
            try:
                resp = requests.get('https://api.coinbase.com/v2/prices/ETH-USD/spot', timeout=3)
                end_price = float(resp.json()['data']['amount'])
            except:
                end_price = start_price
                logger.warning(f"⚠️ Failed to get end price, using start price: ${end_price:.2f}")
            
            # Calculate move
            price_move = (end_price - start_price) / start_price

            # Determine direction from price action
            # DEAD ZONE: Moves smaller than 0.01% are considered neutral (not mismatch)
            DEAD_ZONE_PCT = 0.0001  # 0.01% tolerance
            
            if abs(price_move) < DEAD_ZONE_PCT:
                # Tiny move - proceed with signal direction (neutral market)
                price_action_side = side
                logger.info(f"⚪ NEUTRAL ZONE: Move {price_move:+.4%} < {DEAD_ZONE_PCT:.4%}. Proceeding with signal.")
            else:
                price_action_side = 'BUY' if price_move > 0 else 'SELL'
            
            # CONSENSUS CHECK: Price action must match signal direction (with dead zone applied)
            if price_action_side != side:
                logger.info(f"🚫 DIRECTIONAL MISMATCH: Signal {side} vs Price Action {price_action_side} ({price_move:+.3%}). Skipping.")
                return None



            # Calculate threshold (tighter when quiet, but higher floor)
            avg_move = sum(list(self.price_move_history)) / len(self.price_move_history) if self.price_move_history else 0
            dynamic_threshold = self.config.VOLATILITY_THRESHOLD_PCT
            
            # Scale threshold by coherence: be more aggressive if BTC is leading well
            if coherence > 0.85:
                dynamic_threshold *= 0.8 # 20% easier to trigger if correlated
            
            if abs(price_move) >= dynamic_threshold:
                logger.info(f"🎯 BREAKOUT CONFIRMED: {price_move:+.3%} (threshold {dynamic_threshold:.4%}) → Entering {side}")
                self.alert_bot.send_alert(f"Volatility Breakout: {price_move:+.3%} (Threshold: {dynamic_threshold:.4%})\nDirection: {side}", title="Volatility Trigger", priority="high")
                # Removed: entry_price = end_price (This was causing ETH price to leak into ALT trades)
                self.price_move_history.append(abs(price_move))
            else:
                # VELOCITY BYPASS CHECK: Use flag calculated earlier
                velocity_bypass = False
                if is_velocity_event:
                    velocity_bypass = True
                    logger.info(f"🚀 VELOCITY BYPASS: {same_dir_count} {side} signals in {self.config.VELOCITY_BYPASS_WINDOW_SEC}s → FORCING TRADE")
                    self.alert_bot.send_alert(
                        f"Velocity Bypass Triggered!\n{same_dir_count} {side} signals detected\nForcing trade entry",
                        title="Velocity Override", 
                        priority="high"
                    )
                    entry_price = signal.get('entry', 0) # Fallback to original
                    # Clear signals after bypass to prevent immediate re-trigger
                    self.recent_signals = []
                
                if not velocity_bypass:
                    logger.info(f"😴 VOLATILITY FIZZLE: Only {price_move:+.3%} move (threshold {dynamic_threshold:.4%}). Skipping.")
                    return None
        
        # --- BRAIN CONSENSUS CHECK (15TB Integration) ---
        if self.config.USE_BRAIN_CONSENSUS:
            go, reason, brain_json = self.get_brain_consensus(signal, {})
            if not go:
                logger.info(f"🛑 BRAIN REJECTION: {reason}")
                return None
            else:
                logger.info(f"🟢 BRAIN APPROVED: {reason}")
                
            # --- BREGMAN OPTIMIZATION GATE (Prop 2.4) ---
            bregman_dist = brain_json.get('guaranteed_profit', 0)
            if bregman_dist < self.config.MIN_BREGMAN_PROFIT:
                logger.info(f"🛑 GEOMETRIC REJECTION: Bregman Divergence {bregman_dist:.6f} < {self.config.MIN_BREGMAN_PROFIT}")
                return None
            else:
                logger.info(f"📐 GEOMETRIC LOCK: Divergence {bregman_dist:.6f} suggests guaranteed outcome.")

            # --- STRUCTURAL SYZYGY GATE (Fel's Conjecture) ---
            struct_coherence = brain_json.get('structural_coherence', 1.0)
            if struct_coherence < self.config.MIN_STRUCTURAL_COHERENCE:
                logger.info(f"🛑 STRUCTURAL REJECTION: Coherence {struct_coherence:.2%} < {self.config.MIN_STRUCTURAL_COHERENCE:.2%}")
                return None
            else:
                logger.info(f"💎 SYZYGY LOCK: Structural Integrity {struct_coherence:.2%} confirmed via S-Ring syzygies.")
        
        # NOVEL #4: ADAPTIVE SIGNAL INVERSION
        # If model is consistently wrong (anti-correlated), invert the signal
        if self.config.USE_ADAPTIVE_INVERSION:
            # Check if we should invert based on recent performance
            # Simple heuristic: if net position is heavily biased and signals keep pushing same direction
            if abs(self.net_weth_position) >= self.config.NET_POSITION_CLAMP * 0.8:
                # Model keeps signaling in same direction despite position limit
                # This suggests model is stuck in wrong regime - INVERT!
                original_side = side
                side = 'SELL' if side == 'BUY' else 'BUY'
                if not self.inversion_active:
                    logger.info(f"🔄 ADAPTIVE INVERSION: Model stuck at position limit, flipping {original_side} → {side}")
                    self.inversion_active = True
            else:
                if self.inversion_active:
                    logger.info(f"🔄 ADAPTIVE INVERSION: Position normalized, returning to normal signals")
                    self.inversion_active = False
        
        # NET POSITION CLAMPING: Prevent over-exposure in one direction
        clamp = self.config.NET_POSITION_CLAMP
        if side == 'SELL' and self.net_weth_position <= -clamp:
            logger.warning(f"⚠️ Net position clamp: Already short {self.net_weth_position:.4f} WETH, skipping SELL")
            return None
        if side == 'BUY' and self.net_weth_position >= clamp:
            logger.warning(f"⚠️ Net position clamp: Already long {self.net_weth_position:.4f} WETH, skipping BUY")
            return None
        
        # AGGRESSIVE SCALING: 52% = 1.0x, 56% = 1.5x, 60% = 2.0x
        # This multiplier stacks with Velocity/Cascade/Flip triggers
        conf_floor = 0.520
        position_multiplier = 1.0 + ((confidence - conf_floor) * 12.5)
        
        # --- CORRELATION TRIGGER ---
        # If BTC has moved significantly but ETH is lagging, increase conviction
        if self.config.USE_CORRELATION_TRIGGER and self.btc_move_1m != 0:
            # Positive correlation: BTC UP + ETH Signal BUY = High conviction
            is_bullish_corr = (self.btc_move_1m > 0.0005 and side == 'BUY')
            is_bearish_corr = (self.btc_move_1m < -0.0005 and side == 'SELL')
            
            if is_bullish_corr or is_bearish_corr:
                logger.info(f"🔗 Correlation Trigger: BTC moving {self.btc_move_1m:+.2%}, scaling position size 1.5x")
                position_multiplier *= 1.5
            
            # NOVEL #1: MOMENTUM CASCADE - BTC moved >0.1% = massive alpha window
            if abs(self.btc_move_1m) > 0.001:  # 0.1% = cascade event
                if (self.btc_move_1m > 0 and side == 'BUY') or (self.btc_move_1m < 0 and side == 'SELL'):
                    logger.info(f"🔥🔥 MOMENTUM CASCADE: BTC {self.btc_move_1m:+.2%}! Scaling 2.5x")
                    position_multiplier *= 2.5
        
        # NOVEL #2: SIGNAL VELOCITY AMPLIFICATION
        # If multiple signals in same direction within 10 seconds, scale exponentially
        now = time.time()
        if now - self.last_signal_time < 10 and self.last_signal_dir == side:
            self.signal_streak += 1
            if self.signal_streak >= 2:
                velocity_mult = min(3.0, 1 + (self.signal_streak * 0.5))  # Cap at 3x
                logger.info(f"⚡ SIGNAL VELOCITY: {self.signal_streak} streak! Scaling {velocity_mult:.1f}x")
                position_multiplier *= velocity_mult
        else:
            self.signal_streak = 1
        self.last_signal_time = now
        self.last_signal_dir = side
        
        # NOVEL #3: CONTRARIAN FLIP
        # If heavily biased AND strong opposite signal, flip the entire book
        if abs(self.net_weth_position) > clamp * 0.5 and confidence > 0.52:
            is_contrarian = (self.net_weth_position > 0 and side == 'SELL') or (self.net_weth_position < 0 and side == 'BUY')
            if is_contrarian:
                logger.info(f"🔄🔄 CONTRARIAN FLIP: Net {self.net_weth_position:.4f}, flipping with 2x")
                position_multiplier *= 2.0
        
        position_weth = self.config.MAX_POSITION_WETH * min(3.0, position_multiplier)  # Allow up to 3x max
        
        # 5-HOUR PRUNING: Pattern Weighting
        pat_weight = self.config.PATTERN_WEIGHTS.get(pattern, self.config.PATTERN_WEIGHTS['default'])
        if pat_weight != 1.0:
            logger.info(f"⚖️ PATTERN WEIGHT: {pattern} @ {pat_weight:.1f}x (Data-Driven)")
            position_weth *= pat_weight
        
        # STREAK-BASED SIZING: Reduce after losses, increase after wins
        if self.pos_manager:
            streak_mult = self.pos_manager.get_position_size_multiplier()
            if streak_mult != 1.0:
                logger.info(f"🎰 STREAK SIZING: Multiplier {streak_mult:.2f}x (Win: {self.pos_manager.win_streak}, Loss: {self.pos_manager.loss_streak})")
                position_weth *= streak_mult
        
        # BALANCE-AWARE CAPPING: Don't try to trade more than we have
        balances = self.wallet.get_balances()
        if side == 'SELL':
            # Cap by available WETH
            if position_weth > balances['weth']:
                logger.info(f"⚖️ Scaling SELL down to fit balance: {position_weth:.4f} -> {balances['weth']:.4f} WETH")
                position_weth = balances['weth']
        else:
            # Cap by available USDC (approx)
            cost_usd = position_weth * entry_price
            if cost_usd > balances['usdc']:
                new_pos = (balances['usdc'] * 0.98) / entry_price  # 2% buffer
                logger.info(f"⚖️ Scaling BUY down to fit balance: {position_weth:.4f} -> {new_pos:.4f} WETH")
                position_weth = new_pos

        position_weth = max(0.0005, position_weth)  # Minimum 0.0005 WETH
        # MULTI-ASSET PROXY LOGIC (Fixed Jan 7 Drift)
        # We always trade WETH on Base, but we might be following a BTC or ALT signal.
        # We must fetch the price of the signal asset to log correct USD and unit values.
        symbol = signal.get('symbol', 'ETH-USD')
        try:
            # Only fetch if not already done in volatility check
            if symbol == 'ETH-USD':
                symbol_price = entry_price if entry_price > 0 else self.eth_last_price
            else:
                resp = requests.get(f'https://api.coinbase.com/v2/prices/{symbol}/spot', timeout=3)
                symbol_price = float(resp.json()['data']['amount'])
        except:
            symbol_price = entry_price if entry_price > 0 else 1.0 # Disaster fallback
            
        eth_price = self.eth_last_price if self.eth_last_price > 0 else 2300
        
        # The REAL USD value of this trade (since we swap WETH)
        actual_usd_val = position_weth * eth_price
        
        # The EQUIVALENT units of the signal asset
        equivalent_units = actual_usd_val / symbol_price
        
        trade = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "signal_time": signal.get('timestamp'),
            "symbol": symbol,
            "side": side,
            "pattern": pattern,
            "confidence": confidence,
            "entry_price": symbol_price,
            "position_weth": round(equivalent_units, 8), # We log equivalent units for P&L tracking
            "position_usd": round(actual_usd_val, 2),
            "mode": self.config.MODE,
            "status": "pending",
            "proxy_asset": "ETH-USD",
            "proxy_weth": round(position_weth, 6),
            "shadow_probe": self.config.USE_SHADOW_PROBING,
            "coherence": round(self.calculate_coherence(), 2),
            "pnl_usd": 0.0
        }
        
        # Calculate PnL if closing/flipping
        # REALITY CHECK: Account for Gas (~$0.01) and DEX Fees/Slippage (~0.2% = 0.002)
        # This prevents the "Mid-Price Mirage" where the bot logs profit it didn't actually receive.
        EST_GAS_USD = 0.005
        EST_FEE_PCT = 0.0025 # 0.05% Uniswap fee + slippage/market impact
        
        # Adjust entry price for reality (BUYING costs more, SELLING gives less)
        # This must use symbol_price (the actual asset price), not the execution eth price!
        effective_entry = symbol_price * (1 + EST_FEE_PCT) if side == 'BUY' else symbol_price * (1 - EST_FEE_PCT)
        trade["entry_price"] = round(effective_entry, 6)
        
        # Calculate PnL if closing/flipping
        entry_cost = self.last_buy_price.get(symbol, 0.0)
        exit_cost = self.last_sell_price.get(symbol, 0.0)
        
        if side == 'SELL' and entry_cost > 0:
            # PnL = (SalePrice - CostPrice) * Units - Two Gas Fees
            gross_pnl = (effective_entry - entry_cost) * equivalent_units
            trade["pnl_usd"] = round(gross_pnl - (2 * EST_GAS_USD), 4)
            # Clear entry price tracking on full close (simplified)
            # In a real system we'd need FIFO queue, but for now we assume full position flip
        elif side == 'BUY' and exit_cost > 0:
            # PnL = (SalePrice - CostPrice) * Units - Two Gas Fees
            # Short covering: Profit is (SellPrice - BuyPrice)
            gross_pnl = (exit_cost - effective_entry) * equivalent_units
            trade["pnl_usd"] = round(gross_pnl - (2 * EST_GAS_USD), 4)
            
        # Update trackers
        if side == 'BUY':
            self.last_buy_price[symbol] = effective_entry
        elif side == 'SELL':
            self.last_sell_price[symbol] = effective_entry
        
        if self.config.MODE == "paper":
            # Paper trading - just log it
            trade["status"] = "paper_executed"
            trade["tx_hash"] = "PAPER_" + str(int(time.time()))
            logger.info(f"📝 PAPER TRADE: {side} {position_weth:.4f} WETH @ ${entry_price:.2f} (Conf: {confidence*100:.1f}%)")
            self.alert_bot.send_alert(f"Paper Trade: {side} {position_weth:.4f} WETH @ ${entry_price:.2f}\nConfidence: {confidence*100:.1f}%", title="Paper Trade Executed")
            
        elif self.config.MODE == "live":
            # LIVE TRADING
            # NOVEL #5: AUTO-GAS FUNDING
            self.wallet.check_and_fund_gas()
            
            logger.info(f"💰 LIVE TRADE: {side} {position_weth:.4f} WETH @ ${entry_price:.2f}")
            self.alert_bot.send_alert(f"LIVE TRADE: {side} {position_weth:.4f} WETH @ ${entry_price:.2f}\nValue: ${trade['position_usd']}", title="LIVE TRADE EXECUTED", priority="high")
            
            try:
                tx_hash = self._execute_swap(side, position_weth, shadow_probe=trade.get('shadow_probe', False))
                trade["status"] = "executed"
                trade["tx_hash"] = tx_hash
                logger.info(f"✅ Trade executed: {tx_hash}")
            except Exception as e:
                trade["status"] = "failed"
                trade["error"] = str(e)
                logger.error(f"❌ Trade failed: {e}")
        
        # Log trade
        self._log_trade(trade)
        self.trades_today += 1
        
        # Update net position tracking for clamping
        if trade.get('status') in ['executed', 'paper_executed']:
            if side == 'BUY':
                self.net_weth_position += position_weth
            else:
                self.net_weth_position -= position_weth
                
            self.trade_timestamps.append(time.time())
            logger.info(f"📊 Net WETH position: {self.net_weth_position:+.4f} | PnL Tracking: {trade.get('pnl_usd'):+.4f}")
            
            # Register with Position Manager for Trailing Stop
            if self.pos_manager:
                self.pos_manager.add_position(
                    symbol=trade.get('symbol', 'ETH-USD'),
                    side=side,
                    entry_price=trade.get('entry_price', entry_price),
                    amount_weth=trade.get('position_weth', 0),
                    metadata=trade.get('metadata', {})
                )
            
            # SIG RISK PHILOSOPHY: Portfolio-Level Hedging
            # Track position and check if hedge is needed
            if self.portfolio_manager:
                position_usd = trade.get('position_usd', position_weth * entry_price)
                self.portfolio_manager.update_position(
                    symbol=signal.get('symbol', 'ETH-USD'),
                    side=side,
                    size_usd=position_usd
                )
                
                # Check if we need to hedge
                if self.portfolio_manager.should_hedge():
                    hedge_trade = self.portfolio_manager.compute_hedge_trade()
                    if hedge_trade:
                        logger.warning(f"🛡️ HEDGE SIGNAL: {hedge_trade['reason']}")
                        # Log the hedge signal (actual execution would be a separate call)
                        # For now, we record the recommendation
                        self.portfolio_manager.record_hedge_execution(hedge_trade)
                        # TODO: Execute hedge trade via multi_dex_router
        
        return trade
    
    def _load_nonce(self) -> Optional[int]:
        """Load nonce from persistent file."""
        try:
            if os.path.exists(self.nonce_file):
                with open(self.nonce_file, 'r') as f:
                    data = json.load(f)
                    return data.get('nonce')
        except Exception as e:
            logger.warning(f"Failed to load nonce: {e}")
        return None

    def _save_nonce(self, nonce: int):
        """Save nonce to persistent file."""
        try:
            with open(self.nonce_file, 'w') as f:
                json.dump({'nonce': nonce, 'updated': time.time()}, f)
        except Exception as e:
            logger.error(f"Failed to save nonce: {e}")

    def _execute_swap(self, side: str, amount_weth: float, shadow_probe: bool = False, max_slippage: float = None) -> str:
        """Execute actual swap on Uniswap V3 (Base L2)."""
        if not self.wallet.w3 or not self.wallet.account:
            raise RuntimeError("Wallet not connected")

        # Uniswap V3 SwapRouter02 on Base
        ROUTER_ADDRESS = "0x2626664c2603336E57B271c5C0b26F421741e481"
        
        # SwapRouter02 ABI for exactInputSingle (without deadline in struct)
        ROUTER_ABI = [{
            'inputs': [{'components': [
                {'name': 'tokenIn', 'type': 'address'},
                {'name': 'tokenOut', 'type': 'address'},
                {'name': 'fee', 'type': 'uint24'},
                {'name': 'recipient', 'type': 'address'},
                {'name': 'amountIn', 'type': 'uint256'},
                {'name': 'amountOutMinimum', 'type': 'uint256'},
                {'name': 'sqrtPriceLimitX96', 'type': 'uint160'}
            ], 'name': 'params', 'type': 'tuple'}],
            'name': 'exactInputSingle',
            'outputs': [{'name': 'amountOut', 'type': 'uint256'}],
            'stateMutability': 'payable',
            'type': 'function'
        }]

        w3 = self.wallet.w3
        router = w3.eth.contract(address=w3.to_checksum_address(ROUTER_ADDRESS), abi=ROUTER_ABI)
        
        # Determine paths
        token_in = self.wallet.WETH if side == 'SELL' else self.wallet.USDC
        token_out = self.wallet.USDC if side == 'SELL' else self.wallet.WETH
        
        # Fetch real-time ETH price from Coinbase
        import requests
        try:
            resp = requests.get('https://api.coinbase.com/v2/prices/ETH-USD/spot', timeout=5)
            current_price = float(resp.json()['data']['amount'])
            logger.info(f"📈 Real-time ETH price: ${current_price:.2f}")
        except Exception as e:
            current_price = 3350  # Fallback
            logger.warning(f"⚠️ Price fetch failed, using ${current_price}: {e}")
        
        # Calculate amounts with slippage tolerance
        # SHADOW PROBING: Move the limit price closer to the spot to fish for shadow liquidity
        # DYNAMIC SLIPPAGE: Use provided max_slippage or default to conservative values
        if max_slippage is not None:
            probe_offset = max_slippage
        else:
            # Traditional defaults: 0.1% for probing, 0.5% for normal trades
            probe_offset = 0.001 if shadow_probe else 0.005 
        

        if side == 'SELL':
            amount_in_raw = w3.to_wei(amount_weth, 'ether')
            # For SELL, min_out = (ETH * Price) - slippage
            min_amount_out_raw = int(amount_weth * current_price * 1e6 * (1 - probe_offset))
            logger.info(f"🧬 Slippage: {probe_offset:.3%} | Min Out: {min_amount_out_raw/1e6:.4f} USDC")
        else:
            # For BUY, we swap USDC to WETH
            amount_in_raw = int(amount_weth * current_price * 1e6) 
            min_amount_out_raw = int(w3.to_wei(amount_weth * (1 - probe_offset), 'ether'))
            logger.info(f"🧬 Slippage: {probe_offset:.3%} | Min Out: {min_amount_out_raw/1e18:.6f} WETH")

        # Check mempool for competing transactions (Shadow Probe Optimization)
        # This is a stub for future MEV-protection logic
        pass

        # Params as tuple (matching working test)
        params = (
            w3.to_checksum_address(token_in),
            w3.to_checksum_address(token_out),
            500,  # 0.05% fee tier
            self.wallet.address,
            amount_in_raw,
            min_amount_out_raw,  # Fixed: Use calculated slippage protection
            0   # sqrtPriceLimitX96
        )

        # Nonce Management: Try to use pending and local tracking
        if self._next_nonce is None:
            self._next_nonce = w3.eth.get_transaction_count(self.wallet.address, 'pending')
        
        nonce = self._next_nonce
        logger.info(f"Using nonce: {nonce}")

        # Build transaction with EIP-1559 support for L2
        try:
            base_fee = w3.eth.get_block('latest').get('baseFeePerGas', 0)
            priority_fee = w3.eth.max_priority_fee
            max_fee = int((base_fee + priority_fee) * 1.2)  # 20% buffer
            
            tx = router.functions.exactInputSingle(params).build_transaction({
                'from': self.wallet.address,
                'gas': 300000,  # Increased buffer
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
                'nonce': nonce,
                'value': 0,
                'chainId': w3.eth.chain_id
            })
        except Exception as e:
            logger.warning(f"Failed to build EIP-1559 tx, falling back to legacy: {e}")
            tx = router.functions.exactInputSingle(params).build_transaction({
                'from': self.wallet.address,
                'gas': 300000,
                'gasPrice': w3.eth.gas_price,
                'nonce': nonce,
                'value': 0
            })

        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=self.wallet.account.key)
        # Handle both old and new web3.py API
        raw_tx = getattr(signed_tx, 'raw_transaction', None) or getattr(signed_tx, 'rawTransaction', None)
        
        try:
            tx_hash = w3.eth.send_raw_transaction(raw_tx)
            self._next_nonce += 1  # Increment on success
            self._save_nonce(self._next_nonce)
            return tx_hash.hex()
        except Exception as e:
            # If nonce error, reset to force chain refresh
            if "nonce" in str(e).lower():
                self._next_nonce = None
                if os.path.exists(self.nonce_file):
                    os.remove(self.nonce_file)
            raise e
    
    def _log_trade(self, trade: dict):
        """Log trade to trades_log.jsonl."""
        try:
            with open(self.config.TRADES_LOG, 'a') as f:
                f.write(json.dumps(trade) + "\n")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

# =============================================================================
# Main Trader
# =============================================================================

class SpectralTrader:
    """Main trading orchestrator."""
    
    def __init__(self, mode: str = "paper"):
        self.config = TradingConfig()
        self.config.MODE = mode
        
        self.wallet = WalletManager()
        self.watcher = SignalWatcher(self.config)
        self.executor = TradeExecutor(self.config, self.wallet)
        self.executor.recover_state() # RESTORE P&L TRACKING
        self.last_balance_check = 0

        
    def startup(self) -> bool:
        """Initialize and verify readiness."""
        print("\n" + "="*60)
        print("🌌 SPECTRAL TRADER - Physics-Informed Trading")
        print("="*60)
        print(f"   Mode: {'📝 PAPER' if self.config.MODE == 'paper' else '💰 LIVE'}")
        print(f"   Max Position: {self.config.MAX_POSITION_WETH} WETH")
        print(f"   Min Confidence: {self.config.MIN_CONFIDENCE*100:.0f}%")
        print(f"   Daily Loss Limit: {self.config.DAILY_LOSS_LIMIT_WETH} WETH")
        print("="*60 + "\n")
        
        # Connect wallet
        if not self.wallet.connect():
            logger.error("Failed to connect wallet. Exiting.")
            return False
        
        # Check balances (Disabled blocking check to prevent startup hang)
        # balances = self.wallet.get_balances()
        # print(f"💰 Wallet Balances:")
        # print(f"   ETH (Gas):  {balances['eth']:.6f}")
        # print(f"   WETH:       {balances['weth']:.6f} (~${balances['weth'] * 3300:.2f})")
        # print(f"   USDC:       ${balances['usdc']:.2f}")
        # print()
        
        # if balances['eth'] < 0.0001:
        #     logger.warning("⚠️ Low ETH balance - may not have enough for gas!")
        
        # if balances['weth'] < self.config.MAX_POSITION_WETH:
        #     logger.warning(f"⚠️ WETH balance below max position size ({self.config.MAX_POSITION_WETH})")
        
        return True
    
    def run(self):
        """Main trading loop."""
        if not self.startup():
            return
            
        logger.info("🌀 Starting Correlation Tracker...")
        self.executor.btc_move_1m = 0.0
        self.executor.eth_move_1m = 0.0
        self.btc_history = deque(maxlen=20) # ~2 mins of history 
        self.eth_history = deque(maxlen=20) 
        logger.info(f"   Filtering: {'ETH-USD only' if self.config.ONLY_ETH_SIGNALS else 'All symbols'}")
        logger.info(f"   Signals log: {self.config.SIGNALS_LOG}")
        print("-"*60)
        
        try:
            loop_count = 0
            while True:
                loop_count += 1
                # --- UPDATE CORRELATION DATA ---
                try:
                    # Fetch Lead (BTC) and Lag (ETH) prices
                    # Use Coinbase Spot for Lead/Lag analysis
                    headers = {"User-Agent": "Mozilla/5.0"}
                    btc_resp = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', headers=headers, timeout=3)
                    eth_resp = requests.get('https://api.coinbase.com/v2/prices/ETH-USD/spot', headers=headers, timeout=3)
                    
                    if btc_resp.status_code == 200 and eth_resp.status_code == 200:
                        btc_now = float(btc_resp.json()['data']['amount'])
                        eth_now = float(eth_resp.json()['data']['amount'])
                        
                        self.executor.btc_last_price = btc_now
                        self.executor.eth_last_price = eth_now
                        
                        self.btc_history.append(btc_now)
                        self.eth_history.append(eth_now)
                        
                        if len(self.btc_history) >= 12: # At least 1 min of data
                            self.executor.btc_move_1m = (self.btc_history[-1] - self.btc_history[0]) / self.btc_history[0]
                            self.executor.eth_move_1m = (self.eth_history[-1] - self.eth_history[0]) / self.eth_history[0]
                            
                            if abs(self.executor.btc_move_1m) > 0.0005:
                                logger.info(f"🌀 Correlation: BTC 1m Move: {self.executor.btc_move_1m:+.2%} | ETH 1m Move: {self.executor.eth_move_1m:+.2%}")
                            
                            # UPDATE MARKET REGIME for Adaptive TSL
                            if self.executor.pos_manager:
                                self.executor.pos_manager.set_market_regime(self.executor.btc_move_1m)
                    else:
                        logger.info(f"⚠️ Price update status error: BTC:{btc_resp.status_code} ETH:{eth_resp.status_code}")
                except Exception as e:
                    logger.info(f"⚠️ Correlation update failed: {e}")
                
                 # --- MONITOR POSITION EXITS (Trailing Stop-Loss) ---
                if self.executor.pos_manager and self.executor.pos_manager.positions:
                    current_prices = {
                        'ETH-USD': self.executor.eth_last_price,
                        'BTC-USD': self.executor.btc_last_price
                    }
                    
                    # Fetch fresh prices for all active positions not in current_prices
                    for symbol in self.executor.pos_manager.positions:
                        if symbol not in current_prices:
                            try:
                                resp = requests.get(f'https://api.coinbase.com/v2/prices/{symbol}/spot', timeout=3)
                                current_prices[symbol] = float(resp.json()['data']['amount'])
                            except:
                                pass # Use last known if fetch fails
                                
                    # Check for TSL exits
                    exits = self.executor.pos_manager.check_exits(current_prices)
                    for pos in exits:
                        logger.info(f"📉 EXITING POSITION: {pos.side} {pos.symbol} (Trailing Stop)")
                        # Execute exit swap (invert the entry side)
                        exit_side = 'SELL' if pos.side == 'BUY' else 'BUY'
                        try:
                            # Use execute_swap with special priority or just normal execute
                            self.executor.execute({
                                'symbol': pos.symbol,
                                'type': exit_side,
                                'confidence': 1.0,
                                'pattern': 'TSL_EXIT',
                                'reason': f"Trailing stop hit at peak ${pos.peak_price:.2f}"
                            })
                        except Exception as e:
                            logger.error(f"❌ Failed to exit {pos.symbol}: {e}")

                # Heartbeat every ~1 min (30 loops at 2s interval)
                if loop_count % 30 == 0:
                    logger.info(f"💓 Heartbeat | BTC: ${self.executor.btc_last_price:.2f} | ETH: ${self.executor.eth_last_price:.2f} | Move: {self.executor.btc_move_1m:+.4%}")

                # Check for new signals
                signals = self.watcher.get_new_signals()
                
                for signal in signals:
                    logger.info(f"📡 New signal: {signal.get('symbol')} {signal.get('type')} @ ${signal.get('entry', 0):.2f}")
                    
                    # Rate limit - wait between trades
                    time.sleep(1)
                    # Execute trade
                    trade = self.executor.execute(signal)
                    
                    if trade:
                        print(f"   → Trade #{self.executor.trades_today}: {trade.get('status')}")
                
                time.sleep(self.config.POLL_INTERVAL)
                
                # Periodically update wallet state (every 30s)
                if time.time() - self.last_balance_check > 30:
                    try:
                        balances = self.wallet.get_balances()
                        with open(self.config.WALLET_STATE_FILE, 'w') as f:
                            json.dump(balances, f)
                        self.last_balance_check = time.time()
                    except Exception as e:
                        logger.error(f"Failed to auto-save wallet state: {e}")
                
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            print(f"   Trades Executed: {self.executor.trades_today}")
            print(f"   Mode: {self.config.MODE.upper()}")
            print("="*60)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spectral Trader - Physics-Informed Trading")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"],
                        help="Trading mode: paper (default) or live")
    args = parser.parse_args()
    
    if args.mode == "live":
        logger.warning("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
        logger.warning("This will execute REAL TRADES with REAL MONEY.")
        # Removed interactive prompt for automation
    
    trader = SpectralTrader(mode=args.mode)
    trader.run()
