# api_wrapper.py - Enhanced API Wrapper for CoinSwitch Futures API

import logging
from typing import Dict, List, Optional, Any
import time
import pandas as pd

class APIWrapper:
    """Enhanced wrapper for CoinSwitchFuturesAPI to provide additional functionality and error handling"""
    
    def __init__(self, api):
        self.api = api
        self.logger = logging.getLogger(__name__)
        self._last_request_time = 0
        self._request_count = 0
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Simple rate limiting to prevent API overload"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def get_server_time(self):
        """Get server time with error handling"""
        try:
            self._rate_limit()
            return self.api.get_server_time()
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            return None
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        try:
            self._rate_limit()
            active_coins = self.api.get_active_coins()
            return [coin['symbol'] for coin in active_coins if 'symbol' in coin]
        except Exception as e:
            self.logger.error(f"Failed to get active symbols: {e}")
            return []
    
    def get_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a symbol with enhanced error handling"""
        try:
            self._rate_limit()
            response = self.api.get_futures_ticker(symbol)
            
            if response and response.get('data'):
                data = response['data'].get('EXCHANGE_2', {})
                if data:
                    return {
                        'symbol': symbol,
                        'price': float(data.get('last_price', 0)),
                        'bid': float(data.get('bid_price', 0)),
                        'ask': float(data.get('ask_price', 0)),
                        'volume_24h': float(data.get('volume_24h', 0)),
                        'change_24h': float(data.get('price_change_percent_24h', 0)),
                        'high_24h': float(data.get('high_price_24h', 0)),
                        'low_24h': float(data.get('low_price_24h', 0))
                    }
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get order book data for a symbol"""
        try:
            self._rate_limit()
            response = self.api.get_depth(symbol)
            
            if response and response.get('data'):
                data = response['data']
                return {
                    'symbol': symbol,
                    'bids': data.get('bids', []),
                    'asks': data.get('asks', []),
                    'timestamp': data.get('timestamp', int(time.time() * 1000))
                }
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to get order book for {symbol}: {e}")
            return None
    
    def get_candles(self, symbol: str, interval: str = '1', limit: int = 100) -> List:
        """Get candlestick data with enhanced error handling"""
        try:
            self._rate_limit()
            return self.api.get_candles(symbol, interval, max_points=limit)
        except Exception as e:
            self.logger.debug(f"Failed to get candles for {symbol}: {e}")
            return []
    
    def get_positions(self) -> Optional[Dict]:
        """Get current positions with enhanced error handling"""
        try:
            self._rate_limit()
            response = self.api.get_positions()
            return response if response else {'data': []}
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {'data': []}
    
    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float, price: float = None, **kwargs) -> Optional[Dict]:
        """Place an order with enhanced error handling and validation"""
        try:
            # Validate inputs
            if not symbol or not side or not order_type:
                raise ValueError("Missing required parameters: symbol, side, or order_type")
            
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            
            if price is not None and price <= 0:
                raise ValueError(f"Invalid price: {price}")
            
            self._rate_limit()
            result = self.api.place_order(
                side=side,
                symbol=symbol,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )
            
            self.logger.info(f"Order placed: {symbol} {side} {quantity} @ {price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with enhanced error handling"""
        try:
            if not order_id:
                raise ValueError("Missing order_id")
            
            self._rate_limit()
            response = self.api.cancel_order(order_id)
            success = response is not None
            
            if success:
                self.logger.info(f"Order cancelled: {order_id}")
            else:
                self.logger.warning(f"Failed to cancel order: {order_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status with enhanced error handling"""
        try:
            if not order_id:
                return None
            
            self._rate_limit()
            return self.api.get_order_status(order_id)
            
        except Exception as e:
            self.logger.debug(f"Failed to get order status for {order_id}: {e}")
            return None
    
    def update_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol with validation"""
        try:
            if not symbol:
                raise ValueError("Missing symbol")
            
            if leverage < 1 or leverage > 125:
                raise ValueError(f"Invalid leverage: {leverage}. Must be between 1 and 125")
            
            self._rate_limit()
            response = self.api.update_leverage(symbol, leverage)
            success = response is not None
            
            if success:
                self.logger.info(f"Leverage set for {symbol}: {leverage}x")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def get_balance(self) -> Optional[Dict]:
        """Get account balance with enhanced parsing"""
        try:
            self._rate_limit()
            response = self.api._make_request("GET", "/trade/api/v2/futures/wallet_balance")
            
            if 'data' in response:
                data = response['data']
                base_asset_balances = data.get('base_asset_balances', [])
                
                for balance_info in base_asset_balances:
                    if balance_info.get('base_asset') == 'USDT':
                        balances = balance_info.get('balances', {})
                        return {
                            'available_usdt': float(balances.get('total_available_balance', 0)),
                            'total_usdt': float(balances.get('total_balance', 0)),
                            'blocked_usdt': float(balances.get('total_blocked_balance', 0)),
                            'unrealized_pnl': float(balances.get('unrealized_pnl', 0)),
                            'wallet_balance': float(balances.get('wallet_balance', 0))
                        }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return None
    
    def get_trade_info(self) -> pd.DataFrame:
        """Get trading information - returns empty DataFrame for dynamic generation"""
        try:
            # Try to get from actual API first
            if hasattr(self.api, 'get_instrument_info'):
                self._rate_limit()
                instruments = self.api.get_instrument_info()
                
                if instruments:
                    data = []
                    for symbol, info in instruments.items():
                        data.append({
                            'symbol': symbol,
                            'max_leverage': info.get('max_leverage', 10),
                            'quantity_precision': info.get('quantity_precision', 3),
                            'price_precision': info.get('price_precision', 2),
                            'taker_fee_rate': info.get('taker_fee_rate', 0.00065),
                            'min_size': info.get('min_size', 1.0)
                        })
                    return pd.DataFrame(data)
            
            # Return empty DataFrame - system will generate dynamically
            self.logger.info("🎯 Returning EMPTY trade info - system will generate dynamically")
            return pd.DataFrame(columns=[
                'symbol', 'max_leverage', 'quantity_precision', 'price_precision',
                'taker_fee_rate', 'min_size'
            ])
            
        except Exception as e:
            self.logger.warning(f"Failed to get trade info: {e}")
            return pd.DataFrame(columns=[
                'symbol', 'max_leverage', 'quantity_precision', 'price_precision',
                'taker_fee_rate', 'min_size'
            ])
    
    def get_api_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'total_requests': self._request_count,
            'last_request_time': self._last_request_time,
            'rate_limit_delay': self.rate_limit_delay
        }
    
    def __getattr__(self, name):
        """Delegate any missing methods to the underlying API"""
        return getattr(self.api, name)
