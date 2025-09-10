# coinswitch_futures_api.py

import json
import time
import requests
from cryptography.hazmat.primitives.asymmetric import ed25519
from urllib.parse import urlparse, urlencode, unquote_plus
from dotenv import load_dotenv
import os
import logging
import random
from threading import Lock
from collections import defaultdict

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdaptiveRateLimiter:
    def __init__(self):
        self.endpoint_limits = {
            '/trade/api/v2/futures/klines': 0.3,
            '/trade/api/v2/futures/ticker': 0.8,
            '/trade/api/v2/futures/leverage': 0.4,
            '/trade/api/v2/futures/positions': 0.5,
            '/trade/api/v2/futures/order': 0.3,
            '/trade/api/v2/depth': 0.6,
            'default': 0.2
        }
        self.last_requests = defaultdict(float)
        self.rate_limit_detected = defaultdict(int)
        self.consecutive_successes = defaultdict(int)
        self.lock = Lock()

    def wait_for_endpoint(self, endpoint):
        with self.lock:
            rate_limit = self.endpoint_limits.get(endpoint, self.endpoint_limits['default'])
            if self.rate_limit_detected[endpoint] > 0:
                backoff_factor = min(4, self.rate_limit_detected[endpoint])
                rate_limit = rate_limit / (2 ** backoff_factor)
                logging.warning(f"Applying backoff for {endpoint}: {rate_limit:.3f} req/sec")
            now = time.time()
            last_request = self.last_requests[endpoint]
            min_interval = 1.0 / rate_limit
            if last_request > 0:
                elapsed = now - last_request
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed + random.uniform(0.5, 1.5)
                    logging.debug(f"Rate limiting {endpoint}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            self.last_requests[endpoint] = time.time()

    def record_rate_limit(self, endpoint):
        with self.lock:
            self.rate_limit_detected[endpoint] += 1
            self.consecutive_successes[endpoint] = 0
            logging.warning(f"Rate limit detected for {endpoint} (count: {self.rate_limit_detected[endpoint]})")

    def record_success(self, endpoint):
        with self.lock:
            self.consecutive_successes[endpoint] += 1
            if self.consecutive_successes[endpoint] >= 5 and self.rate_limit_detected[endpoint] > 0:
                self.rate_limit_detected[endpoint] = max(0, self.rate_limit_detected[endpoint] - 0.5)
                logging.info(f"Reducing backoff for {endpoint} due to sustained success")

class CoinSwitchFuturesAPI:
    def __init__(self, api_key=None, secret_key=None, base_url="https://coinswitch.co", exchange=None):
        self.api_key = api_key or os.getenv("COINSWITCH_API_KEY")
        self.secret_key = secret_key or os.getenv("COINSWITCH_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("COINSWITCH_API_KEY and COINSWITCH_SECRET_KEY must be set in .env or passed explicitly")
        self.base_url = base_url
        self.exchange = exchange or "EXCHANGE_2"
        self.session = requests.Session()
        self.rate_limiter = AdaptiveRateLimiter()
        self.cache = {}
        self.cache_ttl = {
            'server_time': 30,
            'ticker': 60,
            'leverage': 1800,
            'active_coins': 3600,
            'klines': 120,
            'depth': 30,
            'positions': 60,
            'default': 300
        }
        try:
            bytes.fromhex(self.secret_key)
        except ValueError as e:
            raise ValueError(f"Invalid secret_key format: {e}. Must be a hexadecimal string.")

    def _get_cache_key(self, method, path, params):
        param_str = "&".join([f"{k}={v}" for k, v in sorted((params or {}).items())])
        return f"{method}:{path}?{param_str}"

    def _get_cached(self, cache_key, cache_type='default'):
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            ttl = self.cache_ttl.get(cache_type, self.cache_ttl['default'])
            if time.time() - cached_time < ttl:
                logging.debug(f"Cache hit: {cache_key}")
                return cached_data
        return None

    def _set_cache(self, cache_key, data, cache_type='default'):
        self.cache[cache_key] = (time.time(), data)
        logging.debug(f"Cached: {cache_key}")

    def get_server_time(self):
        cache_key = "server_time"
        cached = self._get_cached(cache_key, 'server_time')
        if cached:
            return cached
        url = f"{self.base_url}/trade/api/v2/time"
        response = self.session.get(url)
        if response.status_code == 200:
            server_time = response.json().get("serverTime")
            self._set_cache(cache_key, server_time, 'server_time')
            return server_time
        else:
            raise Exception(f"Failed to get server time: {response.text}")

    def generate_signature(self, method, endpoint, params=None, payload=None):
        """
        Sign EXACTLY: METHOD + endpoint(+decoded_query_if_GET) + epoch_ms
        Do NOT include body for POST/DELETE in signature.
        """
        epoch_time = str(self.get_server_time())
        if method.upper() == "GET" and params:
            sorted_params = dict(sorted((params or {}).items()))
            query_string = urlencode(sorted_params, doseq=True)
            sign_endpoint = endpoint + ('?' + query_string if query_string else '')
            sign_endpoint = unquote_plus(sign_endpoint)
        else:
            sign_endpoint = endpoint
        signature_msg = method.upper() + sign_endpoint + epoch_time
        request_bytes = signature_msg.encode('utf-8')
        try:
            secret_key_bytes = bytes.fromhex(self.secret_key)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
            signature_bytes = private_key.sign(request_bytes)
            return signature_bytes.hex(), epoch_time, sign_endpoint
        except ValueError as e:
            raise Exception(f"Invalid secret key format: {e}")

    def _make_request(self, method, path, params=None, payload=None, cache_type='default'):
        if method.upper() == "GET":
            cache_key = self._get_cache_key(method, path, params)
            cached = self._get_cached(cache_key, cache_type)
            if cached:
                return cached

        self.rate_limiter.wait_for_endpoint(path)

        signature, epoch_time, _ = self.generate_signature(method, path, params, payload)

        # Build URL for request (encoded query for GET)
        if method.upper() == "GET" and params:
            sorted_params = dict(sorted((params or {}).items()))
            query_string = urlencode(sorted_params, doseq=True)
            url = f"{self.base_url}{path}" + ('?' + query_string if query_string else '')
        else:
            url = f"{self.base_url}{path}"

        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-SIGNATURE': signature,
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-EPOCH': epoch_time
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = self.session.post(url, headers=headers, json=payload)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, headers=headers, json=payload)
                else:
                    raise ValueError("Unsupported method")

                if response.status_code == 429:
                    self.rate_limiter.record_rate_limit(path)
                    retry_after = int(response.headers.get('Retry-After', 10))
                    wait_time = retry_after + random.uniform(3, 8)
                    logging.warning(f"Rate limit hit for {path}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 400:
                    raise Exception(f"API Error: {response.status_code} - {response.text}")

                self.rate_limiter.record_success(path)
                result = response.json()
                if method.upper() == "GET":
                    self._set_cache(cache_key, result, cache_type)
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    backoff_time = (2 ** attempt) + random.uniform(2, 5)
                    logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {backoff_time:.1f}s")
                    time.sleep(backoff_time)
                else:
                    raise

    def _clean_symbol(self, symbol):
        return symbol.replace('/', '').lower()

    # ========== Data methods ==========
    def get_futures_ticker(self, symbol):
        path = "/trade/api/v2/futures/ticker"
        clean_sym = self._clean_symbol(symbol)
        params = {"symbol": clean_sym, "exchange": self.exchange}
        return self._make_request("GET", path, params=params, cache_type='ticker')

    def get_active_coins(self):
        cached = self._get_cached('active_coins_list', 'active_coins')
        if cached:
            return cached
        common_symbols = [
            'BTCUSDT','ETHUSDT','ADAUSDT','SOLUSDT','XRPUSDT',
            'DOGEUSDT','MATICUSDT','DOTUSDT','AVAXUSDT','LINKUSDT',
            'LTCUSDT','BCHUSDT','UNIUSDT','ATOMUSDT','FILUSDT',
            'BNBUSDT','TRXUSDT','ETCUSDT','XLMUSDT','VETUSDT'
        ]
        active = []
        batch_size = 5
        for i in range(0, len(common_symbols), batch_size):
            batch = common_symbols[i:i+batch_size]
            logging.info(f"Validating symbols batch {i//batch_size + 1}/{(len(common_symbols)+batch_size-1)//batch_size}")
            for symbol in batch:
                try:
                    ticker = self.get_futures_ticker(symbol)
                    if ticker.get("data") and ticker["data"].get(self.exchange):
                        active.append({"symbol": symbol, "base_asset": symbol.replace('USDT', '')})
                        logging.info(f"✅ Verified: {symbol}")
                except Exception as e:
                    logging.debug(f"Symbol {symbol} not available: {e}")
                    continue
            if i + batch_size < len(common_symbols):
                time.sleep(2)
        self._set_cache('active_coins_list', active, 'active_coins')
        logging.info(f"Found {len(active)} active futures contracts")
        return active

    def get_candles(self, symbol, interval, start_time=None, end_time=None, max_points: int = 500):
        path = "/trade/api/v2/futures/klines"
        clean_sym = self._clean_symbol(symbol)
        if not start_time or not end_time:
            current_time = int(time.time() * 1000)
            end_time = end_time or current_time
            interval_minutes = int(interval.rstrip('m')) if isinstance(interval, str) and interval.endswith('m') else int(interval)
            start_time = start_time or (end_time - (max_points * interval_minutes * 60 * 1000))
        else:
            interval_minutes = int(interval)

        params = {
            "symbol": clean_sym,
            "exchange": self.exchange,
            "interval": str(interval_minutes),
            "start_time": str(int(start_time)),
            "end_time": str(int(end_time)),
            "limit": str(max_points)
        }
        response = self._make_request("GET", path, params=params, cache_type='klines')
        data = response.get("data", [])
        if data:
            logging.info(f"✅ Fetched {len(data)} futures klines for {clean_sym}")
            return data
        else:
            raise Exception(f"❌ No candle data available for {symbol}")

    def get_24hr_ticker_specific(self, symbol):
        return self.get_futures_ticker(symbol)

    def get_depth(self, symbol):
        path = "/trade/api/v2/depth"
        clean_sym = self._clean_symbol(symbol)
        params = {"symbol": clean_sym, "exchange": self.exchange}
        return self._make_request("GET", path, params=params, cache_type='depth')

    def get_portfolio(self):
        path = "/trade/api/v2/user/portfolio"
        return self._make_request("GET", path)

    def get_leverage(self, symbol):
        path = "/trade/api/v2/futures/leverage"
        clean_sym = self._clean_symbol(symbol)
        params = {"symbol": clean_sym, "exchange": self.exchange}
        return self._make_request("GET", path, params=params, cache_type='leverage')

    def get_positions(self, symbol=None):
        path = "/trade/api/v2/futures/positions"
        params = {"exchange": self.exchange}
        if symbol:
            params["symbol"] = self._clean_symbol(symbol)
        return self._make_request("GET", path, params=params, cache_type='positions')

    def get_transactions(self, symbol=None, type=None, transaction_id=None):
        path = "/trade/api/v2/futures/transactions"
        params = {"exchange": self.exchange}
        if symbol:
            params["symbol"] = self._clean_symbol(symbol)
        if type:
            params["type"] = type
        if transaction_id:
            params["transaction_id"] = transaction_id
        return self._make_request("GET", path, params=params)

    # ========== Trading methods ==========
    def update_leverage(self, symbol, leverage):
        path = "/trade/api/v2/futures/leverage"
        clean_sym = self._clean_symbol(symbol)
        payload = {"symbol": clean_sym, "exchange": self.exchange, "leverage": int(leverage)}
        cache_key = f"GET:{path}?symbol={clean_sym}&exchange={self.exchange}"
        if cache_key in self.cache:
            del self.cache[cache_key]
        return self._make_request("POST", path, payload=payload)

    def place_order(self, side, symbol, order_type, price=None, quantity=None, trigger_price=None, reduce_only=False):
        path = "/trade/api/v2/futures/order"
        clean_sym = self._clean_symbol(symbol)
        payload = {
            "side": side.upper(),
            "symbol": clean_sym,
            "order_type": order_type.upper(),
            "exchange": self.exchange,
            "reduce_only": bool(reduce_only)
        }
        if price is not None:
            payload["price"] = price
        if quantity is not None:
            payload["quantity"] = quantity
        if trigger_price is not None:
            payload["trigger_price"] = trigger_price
        return self._make_request("POST", path, payload=payload)

    def cancel_order(self, order_id):
        """
        Cancel futures order (DELETE per docs).
        Body: {"order_id": "...", "exchange": "EXCHANGE_2"}
        """
        path = "/trade/api/v2/futures/order"
        payload = {"order_id": order_id, "exchange": self.exchange}
        return self._make_request("DELETE", path, payload=payload)

    def add_margin(self, position_id, amount):
        path = "/trade/api/v2/futures/add_margin"
        payload = {"position_id": position_id, "amount": amount, "exchange": self.exchange}
        return self._make_request("POST", path, payload=payload)

    # ========== Utility ==========
    def clear_cache(self, cache_type=None):
        if cache_type:
            keys_to_remove = []
            for key in list(self.cache.keys()):
                if cache_type in key:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.cache[key]
            logging.info(f"Cleared {cache_type} cache entries")
        else:
            self.cache.clear()
            logging.info("Cleared all cache entries")

    def get_cache_stats(self):
        total_entries = len(self.cache)
        cache_types = {}
        for key in self.cache:
            cache_type = key.split(':')[1].split('?') if ':' in key else 'unknown'
            cache_types[cache_type] = cache_types.get(cache_type, 0) + 1
        return {
            'total_entries': total_entries,
            'by_type': cache_types,
            'rate_limit_status': dict(self.rate_limiter.rate_limit_detected),
            'success_streak': dict(self.rate_limiter.consecutive_successes)
        }

    def reset_rate_limiter(self):
        self.rate_limiter.rate_limit_detected.clear()
        self.rate_limiter.consecutive_successes.clear()
        self.rate_limiter.last_requests.clear()
        logging.info("Rate limiter state reset")

    # ---------- Order book (Depth) ----------
    def get_order_book(self, symbol: str, limit: int = None) -> dict:
        path = "/trade/api/v2/depth"
        clean_sym = self._clean_symbol(symbol)
        params = {"symbol": clean_sym, "exchange": self.exchange}
        if limit is not None:
            params["limit"] = int(limit)
        return self._make_request("GET", path, params=params, cache_type='depth')

    def get_order_book_normalized(self, symbol: str, top_n: int = 5) -> dict:
        resp = self.get_order_book(symbol)
        data = resp.get("data", {}) if isinstance(resp, dict) else {}
        sym = str(data.get("symbol", "")).replace("/", "").upper() or str(symbol).upper()
        ts = int(data.get("timestamp", int(time.time() * 1000)))
        bids = data.get("bids", []) or []
        asks = data.get("asks", []) or []

        def _to_float_pairs(levels):
            out = []
            for lvl in levels[:top_n]:
                try:
                    price = float(lvl); qty = float(lvl[4])
                    out.append([price, qty])
                except Exception:
                    continue
            return out

        bids_f = _to_float_pairs(bids)
        asks_f = _to_float_pairs(asks)
        best_bid = float(bids_f) if len(bids_f) > 0 else 0.0
        best_ask = float(asks_f) if len(asks_f) > 0 else 0.0
        bid_qty = float(sum(q for _, q in bids_f))
        ask_qty = float(sum(q for _, q in asks_f))
        den = bid_qty + ask_qty
        obi = float((bid_qty - ask_qty) / den) if den > 0 else 0.0

        return {
            "symbol": sym,
            "timestamp": ts,
            "bids": bids_f,
            "asks": asks_f,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "obi": obi,
            "bid_qty_topN": bid_qty,
            "ask_qty_topN": ask_qty,
        }
