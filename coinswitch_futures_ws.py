# coinswitch_futures_ws.py - Universal WebSocket Client for CoinSwitch Futures (with OB cache/getters)

import socketio
import threading
import time
import logging

COINSWITCH_WS_URL = "wss://ws.coinswitch.co/exchange_2"  # EXCHANGE_2 namespace
NAMESPACE = "/exchange_2"  # Futures Exchange Namespace
SOCKETIO_PATH = "/pro/realtime-rates-socket/futures/exchange_2"

# Event Names per Docs
ORDER_BOOK_EVENT = "FETCH_ORDER_BOOK_CS_PRO"
TICKER_EVENT = "FETCH_TICKER_INFO_CS_PRO"
TRADES_EVENT = "FETCH_TRADES_CS_PRO"
KLINES_EVENT = "FETCH_CANDLESTICK_CS_PRO"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoinSwitchFuturesSocket")

class CoinSwitchFuturesWS:
    def __init__(self, pairs=None, subscribe_order_book=True, subscribe_ticker=True,
                 subscribe_trades=False, subscribe_klines=False, kline_interval=None):
        self.pairs = pairs or ["BTCUSDT"]
        self.subscribe_order_book = subscribe_order_book
        self.subscribe_ticker = subscribe_ticker
        self.subscribe_trades = subscribe_trades
        self.subscribe_klines = subscribe_klines
        self.kline_interval = kline_interval
        self.sio = socketio.Client(logger=False, engineio_logger=False)
        self.last_order_books = {}  # pair -> normalized dict
        self._register_handlers()

    def _register_handlers(self):
        @self.sio.event(namespace=NAMESPACE)
        def connect():
            logger.info(f"✅ Connected to CoinSwitch Futures WS (namespace: {NAMESPACE})")
            for pair in self.pairs:
                pair = pair.upper()
                if self.subscribe_order_book:
                    logger.info(f"🔔 Subscribing to ORDER_BOOK for {pair}")
                    self.sio.emit(ORDER_BOOK_EVENT, {'event': 'subscribe', 'pair': pair}, namespace=NAMESPACE)
                if self.subscribe_ticker:
                    logger.info(f"🔔 Subscribing to TICKER for {pair}")
                    self.sio.emit(TICKER_EVENT, {'event': 'subscribe', 'pair': pair}, namespace=NAMESPACE)
                if self.subscribe_trades:
                    logger.info(f"🔔 Subscribing to TRADES for {pair}")
                    self.sio.emit(TRADES_EVENT, {'event': 'subscribe', 'pair': pair}, namespace=NAMESPACE)
                if self.subscribe_klines and self.kline_interval:
                    kl_pair = f"{pair}_{self.kline_interval}"
                    logger.info(f"🔔 Subscribing to KLINES ({self.kline_interval}m) for {pair}")
                    self.sio.emit(KLINES_EVENT, {'event': 'subscribe', 'pair': kl_pair}, namespace=NAMESPACE)

        @self.sio.on(ORDER_BOOK_EVENT, namespace=NAMESPACE)
        def on_order_book(data):
            ob = data.get('data', {}) or {}
            sym = str(ob.get('symbol', '')).upper()
            if not sym:
                return
            bids = ob.get('bids', []) or []
            asks = ob.get('asks', []) or []
            ts = int(ob.get('timestamp', int(time.time() * 1000)))

            def _sum_qty(levels, topn=5):
                s = 0.0
                for lvl in levels[:topn]:
                    try:
                        s += float(lvl[1])
                    except Exception:
                        continue
                return s

            def _best(levels):
                try:
                    return float(levels) if levels and len(levels) > 0 else 0.0
                except Exception:
                    return 0.0

            bid_qty = _sum_qty(bids, topn=5)
            ask_qty = _sum_qty(asks, topn=5)
            den = bid_qty + ask_qty
            obi = float((bid_qty - ask_qty) / den) if den > 0 else 0.0
            best_bid = _best(bids)
            best_ask = _best(asks)

            # normalize top-5 as float pairs
            def _to_float_pairs(levels, topn=5):
                out = []
                for lvl in levels[:topn]:
                    try:
                        out.append([float(lvl), float(lvl[1])])
                    except Exception:
                        continue
                return out

            self.last_order_books[sym] = {
                "symbol": sym,
                "timestamp": ts,
                "bids": _to_float_pairs(bids, 5),
                "asks": _to_float_pairs(asks, 5),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "obi": obi,
                "bid_qty_top5": bid_qty,
                "ask_qty_top5": ask_qty,
            }
            logger.debug(f"[OB] {sym} obi={obi:+.2f} bid5={bid_qty:.4f} ask5={ask_qty:.4f}")

        @self.sio.on(TICKER_EVENT, namespace=NAMESPACE)
        def on_ticker(data):
            for symbol, ticker in data.items():
                logger.info(f"[TICKER] {symbol}: Last={ticker.get('c')} | Bid={ticker.get('b')} | Ask={ticker.get('a')} | 24h%={ticker.get('P')}")

        @self.sio.on(TRADES_EVENT, namespace=NAMESPACE)
        def on_trades(data):
            logger.info(f"[TRADES] {data}")

        @self.sio.on(KLINES_EVENT, namespace=NAMESPACE)
        def on_klines(data):
            logger.info(f"[KLINES] {data}")

        @self.sio.event(namespace=NAMESPACE)
        def disconnect():
            logger.warning("Disconnected from WS Exchange namespace!")

    def get_last_order_book(self, pair: str) -> dict | None:
        return self.last_order_books.get(pair.upper())

    def get_last_obi(self, pair: str) -> float | None:
        ob = self.get_last_order_book(pair)
        return ob.get("obi") if ob else None

    def start(self, run_async=False):
        def _connect():
            try:
                self.sio.connect(
                    COINSWITCH_WS_URL,
                    namespaces=[NAMESPACE],
                    transports=['websocket'],
                    socketio_path=SOCKETIO_PATH,
                    wait=True,
                    wait_timeout=10
                )
                logger.info('WS connection established.')
                self.sio.wait()
            except Exception as e:
                logger.error(f"Error connecting WS: {e}")

        if run_async:
            thread = threading.Thread(target=_connect, daemon=True)
            thread.start()
            logger.info("WS client started in background thread.")
            return thread
        else:
            _connect()

    def close(self):
        self.sio.disconnect()
        logger.info('WS disconnected.')
