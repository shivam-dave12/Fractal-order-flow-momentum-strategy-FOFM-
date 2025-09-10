# websocket_client.py - FIXED: Direct api_client._make_request (No 'api' Attribute) + Dynamic Subscription

import socketio
import threading
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from collections import deque

# Detailed logger for debugging - SET TO WARNING LEVEL
ws_detail_logger = logging.getLogger('websocket_detail')
ws_detail_logger.setLevel(logging.WARNING)
if not ws_detail_logger.handlers:
    handler = logging.FileHandler('websocket_detail.log')
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    ws_detail_logger.addHandler(handler)
    ws_detail_logger.propagate = False

class CoinSwitchDynamicWebSocket:
    def __init__(self, api_key: str = "", api_client=None):
        self.api_key = api_key
        self.api_client = api_client  # Direct API instance
        
        # WebSocket configuration
        self.base_url = "wss://ws.coinswitch.co"
        self.namespace = "/exchange_2"
        self.socketio_path = "/pro/realtime-rates-socket/futures/exchange_2"
        
        # Initialize socketio client with built-in reconnection
        self.sio = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=0,  # Infinite attempts
            reconnection_delay=1,
            reconnection_delay_max=5,
        )
        
        # State variables
        self.connected = False
        self.symbols_subscribed = set()
        self.live_ticker_data: Dict[str, Dict] = {}
        self.last_update: Dict[str, float] = {}
        self.order_books: Dict[str, Dict] = {}
        self.spread_hist: Dict[str, deque] = {}
        
        # Statistics
        self._obi_updates = 0
        self._ticker_updates = 0
        self._first_symbol_logged = set()
        self._subscription_confirmations = 0
        
        self.logger = logging.getLogger(__name__)
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        @self.sio.event(namespace=self.namespace)
        def connect():
            self.connected = True
            self.logger.info("✅ Connected to CoinSwitch Futures WebSocket")
            time.sleep(1)
            threading.Thread(target=self._discover_and_subscribe_all_symbols, daemon=True).start()

        @self.sio.event(namespace=self.namespace)
        def disconnect():
            self.connected = False
            self.logger.warning("🔌 Disconnected from WebSocket")

        @self.sio.event(namespace=self.namespace)
        def connect_error(data):
            ws_detail_logger.error(f"Connection error: {data}")
            self.connected = False

        @self.sio.on("FETCH_TICKER_INFO_CS_PRO", namespace=self.namespace)
        def on_ticker_data(data):
            try:
                for symbol, ticker_info in data.items():
                    if not isinstance(ticker_info, dict):
                        continue
                    
                    symbol = str(symbol).upper()
                    price = float(ticker_info.get('c', 0) or 0)
                    bid = float(ticker_info.get('b', 0) or 0)
                    ask = float(ticker_info.get('a', 0) or 0)
                    
                    processed_data = {
                        'symbol': symbol,
                        'price': price,
                        'open_price': float(ticker_info.get('o', 0) or 0),
                        'high_24h': float(ticker_info.get('h', 0) or 0),
                        'low_24h': float(ticker_info.get('l', 0) or 0),
                        'volume': float(ticker_info.get('bv', 0) or 0),
                        'quote_volume': float(ticker_info.get('qv', 0) or 0),
                        'change_24h': float(ticker_info.get('P', 0) or 0),
                        'bid': bid,
                        'ask': ask,
                        'mark_price': float(ticker_info.get('p', 0) or 0),
                        'funding_rate': float(ticker_info.get('r', 0) or 0),
                        'timestamp': datetime.now(),
                    }
                    
                    if price > 0 and ask > bid > 0:
                        spread_pct = (ask - bid) / price * 100.0
                    else:
                        spread_pct = 0.0
                    
                    processed_data['spread_pct'] = spread_pct
                    
                    dq = self.spread_hist.setdefault(symbol, deque(maxlen=240))
                    dq.append(spread_pct)
                    
                    if len(dq) > 0:
                        sorted_dq = sorted(dq)
                        processed_data['spread_pct_p50'] = float(sorted_dq[len(sorted_dq)//2])
                    else:
                        processed_data['spread_pct_p50'] = spread_pct
                    
                    self.live_ticker_data[symbol] = processed_data
                    self.last_update[symbol] = time.time()
                    self._ticker_updates += 1
                    
            except Exception as e:
                ws_detail_logger.error(f"Error processing ticker data: {e}")

        @self.sio.on("FETCH_ORDER_BOOK_CS_PRO", namespace=self.namespace)
        def on_order_book(data):
            try:
                ws_detail_logger.debug(f"Raw OB data: {data}")
                
                # CoinSwitch sends data directly
                ob_data = data if isinstance(data, dict) else {}
                
                # Extract symbol
                symbol = ob_data.get('s', '').upper()
                
                if not symbol:
                    ws_detail_logger.debug("No symbol in order book data")
                    return
                
                asks = ob_data.get('asks', [])
                bids = ob_data.get('bids', [])
                
                if not asks or not bids:
                    ws_detail_logger.debug(f"Empty order book for {symbol}")
                    return
                
                # Parse levels
                processed_asks = []
                processed_bids = []
                
                for ask in asks[:10]:
                    try:
                        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            price = float(str(ask[0]))
                            qty = float(str(ask[1]))
                            if price > 0 and qty > 0:
                                processed_asks.append((price, qty))
                    except (ValueError, IndexError):
                        continue
                
                for bid in bids[:10]:
                    try:
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(str(bid[0]))
                            qty = float(str(bid[1]))
                            if price > 0 and qty > 0:
                                processed_bids.append((price, qty))
                    except (ValueError, IndexError):
                        continue
                
                if not processed_asks or not processed_bids:
                    ws_detail_logger.debug(f"No valid order book data for {symbol}")
                    return
                
                # OBI from top 5
                top_bid_volume = sum(qty for _, qty in processed_bids[:5])
                top_ask_volume = sum(qty for _, qty in processed_asks[:5])
                total_volume = top_bid_volume + top_ask_volume
                obi = (top_bid_volume - top_ask_volume) / total_volume if total_volume > 0 else 0.0
                
                best_bid = processed_bids[0][0] if processed_bids else 0.0
                best_ask = processed_asks[0][0] if processed_asks else 0.0
                
                # Store
                self.order_books[symbol] = {
                    'asks': processed_asks,
                    'bids': processed_bids,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'obi': float(obi),
                    'timestamp': time.time(),
                }
                
                self._obi_updates += 1
                
            except Exception as e:
                ws_detail_logger.error(f"Error processing order book: {e}")

    def _discover_and_subscribe_all_symbols(self):
        """FIXED: Direct self.api_client._make_request - Subscribe to dynamic USDT symbols"""
        try:
            # FIXED: Direct call to api_client._make_request (no 'api' attribute)
            response = self.api_client._make_request(
                "GET",
                "/trade/api/v2/futures/instrument_info",
                params={"exchange": "EXCHANGE_2"},
            )
            
            # FIXED PARSING: Handle direct dict or wrapped
            if isinstance(response, dict):
                if 'code' in response and response['code'] == 200 and 'data' in response:
                    instruments = response['data']
                elif 'data' in response:
                    instruments = response['data']
                else:
                    instruments = response
            else:
                raise ValueError(f"Invalid response: {response}")
            
            # Filter active USDT perpetuals
            usdt_symbols = []
            for sym, info in instruments.items():
                if (str(sym).upper().endswith("USDT") and 
                    info.get('status', '').upper() == "TRADING" and
                    info.get('type', '').upper() == 'PERPETUAL_FUTURES'):
                    usdt_symbols.append(sym.upper())
            
            self.logger.info(f"📡 Subscribing to {len(usdt_symbols)} dynamic USDT perpetual symbols")
            
            # Batch subscribe to avoid rate limits
            batch_size = 50
            for i in range(0, len(usdt_symbols), batch_size):
                batch = usdt_symbols[i:i+batch_size]
                for symbol in batch:
                    if symbol in self.symbols_subscribed or not self.connected:
                        continue
                    
                    try:
                        # Subscribe to ticker
                        self.sio.emit(
                            "FETCH_TICKER_INFO_CS_PRO",
                            {"event": "subscribe", "pair": symbol},
                            namespace=self.namespace,
                        )
                        time.sleep(0.01)  # Short delay
                        
                        # Subscribe to order book
                        self.sio.emit(
                            "FETCH_ORDER_BOOK_CS_PRO",
                            {"event": "subscribe", "pair": symbol},
                            namespace=self.namespace,
                        )
                        
                        self.symbols_subscribed.add(symbol)
                        time.sleep(0.005)
                        
                    except Exception as e:
                        ws_detail_logger.error(f"Failed to subscribe {symbol}: {e}")
                
                time.sleep(0.5)  # Batch delay
                
            time.sleep(5)  # Final settle
            self.logger.info(f"📊 Subscriptions complete: {len(self.live_ticker_data)} tickers active")
            
        except Exception as e:
            self.logger.error(f"❌ Error discovering symbols: {e}")

    def start(self):
        return self.connect_websocket()

    def stop(self):
        self.disconnect_websocket()

    def connect_websocket(self):
        try:
            self.logger.info("🔌 Connecting to CoinSwitch WebSocket...")
            self.sio.connect(
                self.base_url,
                socketio_path=self.socketio_path,
                namespaces=[self.namespace],
                transports=['websocket'],
                wait_timeout=15,
            )
            
            # Wait for connection
            max_wait = 20
            waited = 0.0
            while not self.connected and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            
            if not self.connected:
                raise TimeoutError("Failed to connect within timeout")
            
            return True
        
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False

    def disconnect_websocket(self):
        try:
            if self.sio and self.sio.connected:
                self.sio.disconnect()
            self.connected = False
            self.logger.info("🔌 WebSocket disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")

    # Data access methods - unchanged
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        return self.order_books.get(str(symbol).upper())

    def get_ticker_data(self, symbol: str) -> Optional[Dict]:
        return self.live_ticker_data.get(str(symbol).upper())

    def get_all_tickers(self) -> Dict[str, Dict]:
        return self.live_ticker_data.copy()

    def get_all_live_data(self) -> Dict[str, Dict]:
        return self.live_ticker_data.copy()

    def get_symbol_count(self) -> int:
        return len(self.live_ticker_data)

    def is_connected(self) -> bool:
        return self.connected

    def get_price(self, symbol: str) -> Optional[float]:
        ticker = self.get_ticker_data(symbol)
        return float(ticker.get('price')) if ticker and ticker.get('price') else None

    def get_obi(self, symbol: str) -> Optional[float]:
        ob = self.get_order_book(symbol)
        return float(ob.get('obi')) if ob and ob.get('obi') is not None else None

    def get_order_book_age_sec(self, symbol: str) -> Optional[float]:
        ob = self.get_order_book(symbol)
        if not ob or 'timestamp' not in ob:
            return None
        return time.time() - float(ob['timestamp'])

    def get_order_book_sum_qty(self, symbol: str) -> Optional[float]:
        ob = self.get_order_book(symbol)
        if not ob:
            return None
        
        bids = ob.get('bids', [])[:5]
        asks = ob.get('asks', [])[:5]
        bid_qty = sum(q for _, q in bids) if bids else 0.0
        ask_qty = sum(q for _, q in asks) if asks else 0.0
        
        return float(bid_qty + ask_qty)

    def get_order_book_ready_count(self) -> int:
        count = 0
        now = time.time()
        for _, ob in self.order_books.items():
            if (now - float(ob.get('timestamp', 0))) <= 5.0 and ob.get('obi') is not None:
                count += 1
        return count

    def get_connection_stats(self) -> Dict:
        return {
            'connected': self.connected,
            'symbols_subscribed': len(self.symbols_subscribed),
            'tickers_received': len(self.live_ticker_data),
            'order_books_received': len(self.order_books),
            'obi_updates': self._obi_updates,
            'ticker_updates': self._ticker_updates,
        }

    def get_websocket_health(self) -> Dict:
        return {
            'connected': self.connected,
            'symbols_subscribed': len(self.symbols_subscribed),
            'symbols_with_data': len(self.live_ticker_data),
            'order_books_count': len(self.order_books),
            'obi_updates_total': self._obi_updates,
            'ticker_updates_total': self._ticker_updates,
        }
    
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from live ticker data"""
        return list(self.live_ticker_data.keys())