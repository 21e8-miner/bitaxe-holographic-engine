"""
HIGH-SPEED DATA AGGREGATOR
==========================
Implements parallel fetching from all verified sources with sub-second updates.
"""

import os
import requests
import json
import time
import threading
import concurrent.futures
from datetime import datetime

# Fix for yliveticker protobuf descriptor crash
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import yliveticker

HEADERS = {"User-Agent": "Mozilla/5.0"}

class LiveDataStream:
    """Uses yliveticker for sub-second updates from Yahoo Finance WebSockets."""
    def __init__(self, symbols):
        self.symbols = symbols
        self.latest = {s: None for s in symbols}
        self.thread = None
        self.running = False

    def on_new_quote(self, ws, msg):
        try:
            # yliveticker provides a dictionary directly
            symbol = msg.get('id')
            if symbol in self.latest:
                self.latest[symbol] = {
                    'source': f'ylive_{symbol}',
                    'price': float(msg.get('price')),
                    'time': time.time()
                }
        except Exception as e:
            pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # Mapping for yliveticker (Yahoo symbols)
        # NQ futures usually NQ=F, ES futures ES=F
        yliveticker.YLiveTicker(on_ticker=self.on_new_quote, ticker_names=self.symbols)

# Global stream instance for NQ and ES
live_stream = LiveDataStream(["NQ=F", "ES=F"])

def ensure_stream_started():
    if not live_stream.running:
        live_stream.start()

def fetch_stooq_nq():
    """Stooq - Verified working for NQ futures"""
    try:
        r = requests.get('https://stooq.com/q/l/?s=nq.f&f=sd2t2ohlcv&h&e=csv', headers=HEADERS, timeout=2)
        if r.status_code == 200 and 'N/D' not in r.text:
            parts = r.text.strip().split('\n')[1].split(',')
            return {'source': 'stooq_nq', 'price': float(parts[6]), 'time': time.time()}
    except: pass
    return None

def fetch_stooq_es():
    """Stooq - ES futures for correlation"""
    try:
        r = requests.get('https://stooq.com/q/l/?s=es.f&f=sd2t2ohlcv&h&e=csv', headers=HEADERS, timeout=2)
        if r.status_code == 200 and 'N/D' not in r.text:
            parts = r.text.strip().split('\n')[1].split(',')
            return {'source': 'stooq_es', 'price': float(parts[6]), 'time': time.time()}
    except: pass
    return None

def fetch_cnbc_nqh26():
    """CNBC - Specific NQH26 contract"""
    try:
        r = requests.get('https://quote.cnbc.com/quote-html-webservice/quote.htm?partnerId=2&requestMethod=quick&exthrs=1&noform=1&fund=1&output=json&symbols=NQH26', 
                        headers=HEADERS, timeout=2)
        if r.status_code == 200:
            data = r.json()
            q = data['QuickQuoteResult']['QuickQuote']
            if isinstance(q, list): q = q[0]
            return {'source': 'cnbc_nqh26', 'price': float(q['last']), 'time': time.time(), 'last_time': q.get('last_time')}
    except: pass
    return None

def fetch_cnbc_esh26():
    """CNBC - ES contract for correlation"""
    try:
        r = requests.get('https://quote.cnbc.com/quote-html-webservice/quote.htm?partnerId=2&requestMethod=quick&exthrs=1&noform=1&fund=1&output=json&symbols=ESH26', 
                        headers=HEADERS, timeout=2)
        if r.status_code == 200:
            data = r.json()
            q = data['QuickQuoteResult']['QuickQuote']
            if isinstance(q, list): q = q[0]
            return {'source': 'cnbc_esh26', 'price': float(q['last']), 'time': time.time()}
    except: pass
    return None

def fetch_webull_spy():
    """Webull - Real-time SPY for NQ correlation"""
    try:
        r = requests.get('https://quotes-gw.webullfintech.com/api/bgw/quote/realtime?ids=913243251&includeSecu=1', 
                        headers=HEADERS, timeout=2)
        if r.status_code == 200:
            data = r.json()[0]
            return {'source': 'webull_spy', 'price': float(data['close']), 'time': time.time()}
    except: pass
    return None

def fetch_coinbase_btc():
    """Coinbase - BTC for macro correlation"""
    try:
        r = requests.get('https://api.exchange.coinbase.com/products/BTC-USD/ticker', headers=HEADERS, timeout=2)
        if r.status_code == 200:
            return {'source': 'coinbase_btc', 'price': float(r.json()['price']), 'time': time.time()}
    except: pass
    return None

def fetch_deribit_btc():
    """Deribit - BTC perpetual"""
    try:
        r = requests.get('https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL', timeout=2)
        if r.status_code == 200:
            return {'source': 'deribit_btc', 'price': float(r.json()['result']['last_price']), 'time': time.time()}
    except: pass
    return None

def fetch_kraken_btc():
    """Kraken - BTC"""
    try:
        r = requests.get('https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD', timeout=2)
        if r.status_code == 200:
            return {'source': 'kraken_btc', 'price': float(r.json()['result']['XXBTZUSD']['c'][0]), 'time': time.time()}
    except: pass
    return None

def fetch_binance_btc():
    """Binance - BTC"""
    try:
        r = requests.get('https://api.binance.us/api/v3/ticker/price?symbol=BTCUSD', timeout=2)
        if r.status_code == 200:
            return {'source': 'binance_btc', 'price': float(r.json()['price']), 'time': time.time()}
    except: pass
    return None

def fetch_cryptocompare_btc():
    """CryptoCompare - BTC"""
    try:
        r = requests.get('https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD', timeout=2)
        if r.status_code == 200:
            return {'source': 'cryptocompare_btc', 'price': float(r.json()['USD']), 'time': time.time()}
    except: pass
    return None

def fetch_nasdaq_ndx():
    """Nasdaq Official - NDX index"""
    try:
        r = requests.get('https://api.nasdaq.com/api/quote/NDX/info?assetclass=index', 
                        headers={**HEADERS, "Accept": "application/json"}, timeout=2)
        if r.status_code == 200:
            data = r.json()
            price_str = data['data']['primaryData']['lastSalePrice'].replace('$', '').replace(',', '')
            return {'source': 'nasdaq_ndx', 'price': float(price_str), 'time': time.time()}
    except: pass
    return None

def fetch_coinbase_eth():
    """Coinbase - ETH"""
    try:
        r = requests.get('https://api.coinbase.com/v2/prices/ETH-USD/spot', timeout=2)
        if r.status_code == 200:
            return {'source': 'coinbase_eth', 'price': float(r.json()['data']['amount']), 'time': time.time()}
    except: pass
    return None

def fetch_kraken_eth():
    """Kraken - ETH"""
    try:
        r = requests.get('https://api.kraken.com/0/public/Ticker?pair=ETHUSD', timeout=2)
        if r.status_code == 200:
            res = r.json()['result']
            pair = list(res.keys())[0]
            return {'source': 'kraken_eth', 'price': float(res[pair]['c'][0]), 'time': time.time()}
    except: pass
    return None

# All fetch functions
ALL_FETCHERS = [
    fetch_stooq_nq,
    fetch_stooq_es,
    fetch_cnbc_nqh26,
    fetch_cnbc_esh26,
    fetch_webull_spy,
    fetch_coinbase_btc,
    fetch_coinbase_eth,
    fetch_deribit_btc,
    fetch_kraken_btc,
    fetch_kraken_eth,
    fetch_binance_btc,
    fetch_cryptocompare_btc,
    fetch_nasdaq_ndx,
]

def run_parallel_fetch():
    """Run all fetchers in parallel and mix in live stream data."""
    ensure_stream_started()
    start = time.time()
    results = []
    
    # 1. Get snapshot from live stream (if available) with TTL Check
    for sym, data in live_stream.latest.items():
        if data:
            # QC FIX: Expire stale socket data (10s TTL)
            # If socket dies silently, this prevents using 12-hour old price
            if time.time() - data['time'] < 10.0:
                results.append(data)
    
    # 2. Run parallel HTTP fetchers
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(f): f.__name__ for f in ALL_FETCHERS}
        for future in concurrent.futures.as_completed(futures, timeout=3):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except:
                pass
    
    elapsed = (time.time() - start) * 1000
    return results, elapsed

if __name__ == "__main__":
    print("=" * 70)
    print("HIGH-SPEED PARALLEL FETCH TEST (+ LIVE STREAM)")
    print("=" * 70)
    
    # Give live stream a moment to connect
    print("Connecting to live stream...")
    time.sleep(2)
    
    for run in range(5):
        print(f"\n--- Run {run+1} ---")
        results, elapsed = run_parallel_fetch()
        
        print(f"Fetched {len(results)} sources in {elapsed:.0f}ms")
        
        # Sort by source type
        nq_sources = [r for r in results if 'nq' in r['source'].lower()]
        es_sources = [r for r in results if 'es' in r['source'].lower() or 'spy' in r['source'].lower()]
        btc_sources = [r for r in results if 'btc' in r['source'].lower()]
        idx_sources = [r for r in results if 'ndx' in r['source'].lower()]
        
        print("NQ FUTURES:")
        for r in nq_sources:
            print(f"  {r['source']:20} : {r['price']:>12,.2f}")
        
        print("\nES/SPY (Correlation):")
        for r in es_sources:
            print(f"  {r['source']:20} : {r['price']:>12,.2f}")
        
        print("\nBTC (Macro):")
        for r in btc_sources:
            print(f"  {r['source']:20} : {r['price']:>12,.2f}")
        
        print("\nETH (Gas/Settlement):")
        eth_sources = [r for r in results if 'eth' in r['source'].lower()]
        for r in eth_sources:
            print(f"  {r['source']:20} : {r['price']:>12,.2f}")
        
        print("\nNDX Index:")
        for r in idx_sources:
            print(f"  {r['source']:20} : {r['price']:>12,.2f}")
        
        if nq_sources:
            nq_prices = [r['price'] for r in nq_sources]
            print(f"\n>>> NQ Consensus: {sum(nq_prices)/len(nq_prices):,.2f}")
        
        time.sleep(1)
