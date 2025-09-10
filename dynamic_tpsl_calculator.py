import numpy as np
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class DynamicTPSLCalculator:
    def __init__(self, min_rr: float = 3.0):
        self.min_rr = min_rr

    def calculate_dynamic_tpsl(self, symbol: str, entry_price: float, side: str,
                               ohlcv_data: Dict, order_book: Optional[Dict] = None,
                               confidence: float = 0.5) -> 'TPSLResult':
        """
        Calculate dynamic TP/SL - FIXED: Proper R:R calculation, no fallbacks
        """
        class TPSLResult:
            def __init__(self):
                self.take_profit = None
                self.stop_loss = None
                self.tp_distance_pct = 0.0
                self.sl_distance_pct = 0.0
                self.risk_reward_ratio = 0.0
                self.confidence = confidence
                self.method_used = 'dynamic'
                self.supporting_levels = []

        result = TPSLResult()

        try:
            prices = ohlcv_data.get('closes', [])
            volumes = ohlcv_data.get('volumes', [])

            # RELAXED: Accept 20+ points instead of 50+
            if not prices or len(prices) < 20 or not volumes or len(volumes) < 20 or len(prices) != len(volumes):
                logger.warning(f"⚠️ Insufficient OHLCV data for {symbol} TP/SL: {len(prices)} prices, {len(volumes)} volumes")
                return result

            prices_array = np.array(prices, dtype=float)
            volumes_array = np.array(volumes, dtype=float)

            # Scalarize all np operations
            std_price = float(np.std(prices_array)) if len(prices_array) > 1 else 0.0
            mean_price = float(np.mean(prices_array)) if len(prices_array) > 0 else 0.0
            is_real_data = std_price > mean_price * 0.0001 if mean_price > 0 else False

            if not is_real_data:
                logger.warning(f"⚠️ Synthetic/constant data for {symbol} TP/SL: std={std_price:.4f}, mean={mean_price:.4f}")
                return result

            # Enhanced volatility-based TP/SL calculation
            volatility = std_price / mean_price if mean_price > 0 else 0.02
            volatility = max(volatility, 0.01)  # Minimum 1% volatility
            
            # Dynamic multipliers based on confidence and market conditions
            confidence_multiplier = 1.0 + (confidence - 0.5) * 0.5  # 0.75 to 1.25
            
            # Calculate recent trend strength
            recent_trend = 0
            if len(prices_array) >= 10:
                recent_trend = (prices_array[-1] - prices_array[-10]) / prices_array[-10]
            
            trend_multiplier = 1.0 + abs(recent_trend) * 0.3  # Up to 30% boost for strong trends

            # Base TP/SL distances
            if side.upper() == 'LONG':
                # For LONG positions
                sl_distance_pct = volatility * 0.8 * confidence_multiplier  # Tighter SL for higher confidence
                tp_distance_pct = sl_distance_pct * self.min_rr * trend_multiplier
                
                base_sl = entry_price * (1 - sl_distance_pct)
                base_tp = entry_price * (1 + tp_distance_pct)
            else:
                # For SHORT positions
                sl_distance_pct = volatility * 0.8 * confidence_multiplier
                tp_distance_pct = sl_distance_pct * self.min_rr * trend_multiplier
                
                base_sl = entry_price * (1 + sl_distance_pct)
                base_tp = entry_price * (1 - tp_distance_pct)

            # Order book adjustment (if available)
            if order_book and order_book.get('bids') and order_book.get('asks'):
                best_bid = float(order_book.get('best_bid', entry_price))
                best_ask = float(order_book.get('best_ask', entry_price))
                obi = float(order_book.get('obi', 0))

                # Adjust based on order book imbalance
                if side.upper() == 'LONG':
                    if obi > 0.1:  # Strong buy pressure
                        base_tp *= (1 + obi * 0.05)  # Extend TP slightly
                    
                    # Snap TP to resistance levels in order book
                    asks = [float(p) for p, _ in order_book.get('asks', []) if float(p) > entry_price * 1.005]
                    if asks and len(asks) >= 2:
                        closest_resistance = min(asks)
                        if abs(closest_resistance - base_tp) / base_tp < 0.1:  # Within 10%
                            base_tp = closest_resistance * 0.998  # Just below resistance
                            
                else:  # SHORT
                    if obi < -0.1:  # Strong sell pressure
                        base_tp *= (1 - abs(obi) * 0.05)  # Extend TP slightly
                    
                    # Snap TP to support levels in order book
                    bids = [float(p) for p, _ in order_book.get('bids', []) if float(p) < entry_price * 0.995]
                    if bids and len(bids) >= 2:
                        closest_support = max(bids)
                        if abs(closest_support - base_tp) / base_tp < 0.1:  # Within 10%
                            base_tp = closest_support * 1.002  # Just above support

            # Final validation
            if side.upper() == 'LONG':
                if base_tp <= entry_price or base_sl >= entry_price:
                    logger.error(f"❌ Invalid TP/SL direction for LONG {symbol}: TP={base_tp:.6f}, SL={base_sl:.6f}, Entry={entry_price:.6f}")
                    return result
                
                gain_distance = base_tp - entry_price
                loss_distance = entry_price - base_sl
            else:
                if base_tp >= entry_price or base_sl <= entry_price:
                    logger.error(f"❌ Invalid TP/SL direction for SHORT {symbol}: TP={base_tp:.6f}, SL={base_sl:.6f}, Entry={entry_price:.6f}")
                    return result
                
                gain_distance = entry_price - base_tp
                loss_distance = base_sl - entry_price

            if loss_distance <= 0:
                logger.error(f"❌ Invalid loss distance for {symbol}: {loss_distance}")
                return result

            # Calculate final metrics
            tp_distance_pct = (gain_distance / entry_price) * 100
            sl_distance_pct = (loss_distance / entry_price) * 100
            rr_ratio = gain_distance / loss_distance

            # Ensure minimum R:R is met
            if rr_ratio < self.min_rr:
                logger.warning(f"⚠️ R:R {rr_ratio:.2f} below minimum {self.min_rr} for {symbol}, rejecting")
                return result

            # Success - populate result
            result.take_profit = round(base_tp, 6)
            result.stop_loss = round(base_sl, 6)
            result.tp_distance_pct = tp_distance_pct
            result.sl_distance_pct = sl_distance_pct
            result.risk_reward_ratio = rr_ratio
            result.supporting_levels = []

            logger.info(f"✅ Dynamic TP/SL for {symbol}: TP=${base_tp:.6f}, SL=${base_sl:.6f}, R:R={rr_ratio:.2f}:1")
            return result

        except Exception as e:
            logger.error(f"❌ TP/SL calculation failed for {symbol}: {e}")
            return result

# Export main class
__all__ = ['DynamicTPSLCalculator']
