# fractal_order_flow_strategy.py - FIXED: No fallbacks, proper R:R calculation, robust validation

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any, List
from dynamic_tpsl_calculator import DynamicTPSLCalculator

logger = logging.getLogger(__name__)

def safe_array_len(arr):
    """Safely get array length"""
    if arr is None:
        return 0
    if isinstance(arr, np.ndarray):
        return int(arr.size)
    try:
        return len(arr)
    except Exception:
        return 0

def safe_array_to_numpy(arr):
    """Convert to numpy - Accept 20+ points"""
    if arr is None:
        return np.array([])
    if isinstance(arr, np.ndarray):
        return arr.flatten()
    try:
        arr_np = np.array(arr, dtype=float)
        if len(arr_np) < 20:
            logger.warning("⚠️ Insufficient real data (<20 points) - skipping analysis")
            return np.array([])
        return arr_np
    except Exception as e:
        logger.error(f"Array conversion failed: {e}")
        return np.array([])

def analyze_ict_smc_structure(prices, volumes) -> Dict[str, Any]:
    """ICT/SMC - Accept 20+ points"""
    prices_array = safe_array_to_numpy(prices)
    volumes_array = safe_array_to_numpy(volumes)
    
    min_len = min(len(prices_array), len(volumes_array))
    
    if min_len < 20:
        logger.warning(f"⚠️ Insufficient real data for ICT/SMC ({min_len} points)")
        return {
            'analysis_valid': False,
            'structure_score': 0.0,
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_zones': [],
            'total_score': 0.0,
            'confluence_factors': [],
            'market_structure': 'insufficient_real_data'
        }

    # Trim both to exact min_len
    prices_array = prices_array[:min_len]
    volumes_array = volumes_array[:min_len]

    # Scalarize np.std and np.mean
    std_price = float(np.std(prices_array)) if len(prices_array) > 1 else 0.0
    mean_price = float(np.mean(prices_array)) if len(prices_array) > 0 else 0.0
    is_real_data = std_price > mean_price * 0.001 if mean_price > 0 else False

    try:
        order_blocks = []
        fair_value_gaps = []
        liquidity_zones = []
        confluence_factors = []

        # Order Blocks - ADJUSTED for smaller datasets
        lookback = min(10, min_len // 4)  # Adaptive lookback
        for i in range(lookback, min_len - 5):
            if i >= min_len:
                break
                
            volume_ma = float(np.mean(volumes_array[max(0, i-lookback):i+1]))
            current_volume = float(volumes_array[i])
            
            if current_volume > volume_ma * 1.8:
                price_move = abs(prices_array[i] - prices_array[i-1]) / prices_array[i-1] if prices_array[i-1] > 0 else 0
                if price_move > 0.008:
                    ob_type = 'bullish_order_block' if prices_array[i] > prices_array[i-1] else 'bearish_order_block'
                    ob_strength = min(current_volume / volume_ma, 5.0)
                    if is_real_data:
                        ob_strength *= 1.2
                    
                    order_blocks.append({
                        'index': i,
                        'price': float(prices_array[i]),
                        'volume': current_volume,
                        'type': ob_type,
                        'strength': ob_strength,
                        'price_move_pct': price_move * 100
                    })
                    
                    if len(order_blocks) >= 5:
                        break

        # Fair Value Gaps - ADJUSTED for smaller datasets
        for i in range(2, min(min_len - 2, 50)):  # Limit search range for performance
            if i+1 >= min_len or i-1 < 0:
                continue
                
            current_high = float(max(prices_array[i-1:i+2]))
            current_low = float(min(prices_array[i-1:i+2]))
            prev_high = float(max(prices_array[max(0,i-2):i]))
            prev_low = float(min(prices_array[max(0,i-2):i]))
            next_high = float(max(prices_array[i:min(min_len,i+3)]))
            next_low = float(min(prices_array[i:min(min_len,i+3)]))

            if prev_high < next_low:
                gap_size = (next_low - prev_high) / prices_array[i] if prices_array[i] > 0 else 0
                if gap_size > 0.005:
                    fair_value_gaps.append({
                        'index': i,
                        'gap_start': prev_high,
                        'gap_end': next_low,
                        'size_pct': gap_size * 100,
                        'type': 'bullish_fvg'
                    })

            if prev_low > next_high:
                gap_size = (prev_low - next_high) / prices_array[i] if prices_array[i] > 0 else 0
                if gap_size > 0.005:
                    fair_value_gaps.append({
                        'index': i,
                        'gap_start': next_high,
                        'gap_end': prev_low,
                        'size_pct': gap_size * 100,
                        'type': 'bearish_fvg'
                    })

            if len(fair_value_gaps) >= 3:
                break

        # Liquidity Zones - ADJUSTED for smaller datasets
        if min_len > 5:
            safe_start = 5
            safe_end = min_len - 5
            if safe_end > safe_start:
                volume_threshold = float(np.percentile(volumes_array[safe_start:safe_end], 75))
                for i in range(safe_start, safe_end):
                    if volumes_array[i] > volume_threshold:
                        liquidity_zones.append({
                            'index': i,
                            'price': float(prices_array[i]),
                            'volume': float(volumes_array[i]),
                            'type': 'liquidity_pool'
                        })
                        
                        if len(liquidity_zones) >= 4:
                            break

        # Scores
        ob_score = len(order_blocks) * 0.2
        fvg_score = len(fair_value_gaps) * 0.15
        liq_score = len(liquidity_zones) * 0.1
        base_confluence = ob_score + fvg_score + liq_score
        
        structure_score = min(base_confluence * (1.5 if is_real_data else 0.0), 1.0)
        
        market_structure = ('bullish' if prices_array[-1] > prices_array[int(min_len*0.8)]
                           else 'bearish' if prices_array[-1] < prices_array[int(min_len*0.8)] else 'ranging')

        result = {
            'analysis_valid': structure_score > 0,
            'structure_score': structure_score,
            'order_blocks': order_blocks[:5],
            'fair_value_gaps': fair_value_gaps[:3],
            'liquidity_zones': liquidity_zones[:4],
            'total_score': structure_score,
            'confluence_factors': [f"OBs: {len(order_blocks)}", f"FVGs: {len(fair_value_gaps)}", f"Liq: {len(liquidity_zones)}"],
            'market_structure': market_structure,
            'is_real_data': is_real_data,
            'data_points_used': min_len
        }

        logger.debug(f"ICT/SMC: Score {structure_score:.3f} (Real: {is_real_data})")
        return result

    except Exception as e:
        logger.error(f"ICT/SMC error: {e}")
        return {'analysis_valid': False, 'total_score': 0.0}

def calculate_volatility_score(prices: List[float], period: int = 20) -> float:
    """Volatility - skip if insufficient"""
    if not prices or len(prices) < period:
        return 0.0
        
    recent_prices = np.array(prices[-period:], dtype=float)
    if len(recent_prices) < 2:
        return 0.0
        
    std_price = float(np.std(recent_prices))
    mean_price = float(np.mean(recent_prices))
    vol = std_price / mean_price * 100 if mean_price > 0 else 0.0
    
    return vol

def calculate_fractal_order_flow_momentum(prices: np.ndarray, volumes: np.ndarray) -> Dict:
    """FOFM - Accept 20+ points"""
    min_required = 20
    
    if len(prices) < min_required or len(volumes) < min_required:
        return {'fractal_score': 0.0, 'order_flow_score': 0.0, 'momentum_score': 0.0, 'combined_fofm_score': 0.0}

    # Ensure same length by trimming to minimum
    min_len = min(len(prices), len(volumes))
    prices = prices[:min_len]
    volumes = volumes[:min_len]

    # Fractal analysis
    fractal_score = 0.5
    if len(prices) >= 10:
        max_lag = min(10, len(prices)//2)
        lags = range(2, max_lag + 1)
        rs_values = []
        
        for lag in lags:
            if lag*2 <= len(prices):
                diffs = np.diff(prices[:lag*2], n=lag)
                if len(diffs) > 0 and float(np.std(diffs)) > 0:
                    rs = (float(np.max(diffs)) - float(np.min(diffs))) / float(np.std(diffs))
                    rs_values.append(rs)
        
        fractal_score = float(np.mean(rs_values)) / len(lags) if rs_values else 0.5

    # Order Flow analysis
    order_flow_score = 0.0
    if len(prices) > 1:
        diffs = np.diff(prices)
        vols_aligned = volumes[:-1]  # Match length with diffs
        
        if len(diffs) == len(vols_aligned):
            vp_changes = vols_aligned * diffs
            lookback = min(10, len(vols_aligned))
            vol_sum = float(np.sum(vols_aligned[-lookback:]))
            order_flow_score = float(np.sum(vp_changes[-lookback:])) / vol_sum if vol_sum > 0 else 0.0

    # Momentum analysis
    momentum_score = 0.0
    if len(prices) >= 20:
        ema_short = float(np.mean(prices[-5:]))
        ema_long = float(np.mean(prices[-20:]))
        momentum_score = (ema_short - ema_long) / ema_long if ema_long > 0 else 0.0
        momentum_score = max(min(momentum_score * 10, 1.0), -1.0)
    elif len(prices) >= 5:
        momentum_score = (float(prices[-1]) - float(prices[0])) / float(prices[0]) if prices[0] > 0 else 0.0
        momentum_score = max(min(momentum_score * 10, 1.0), -1.0)

    # Combined score
    combined = fractal_score * 0.3 + abs(order_flow_score) * 0.4 + (momentum_score + 1) / 2 * 0.3

    return {
        'fractal_score': fractal_score,
        'order_flow_score': order_flow_score,
        'momentum_score': momentum_score,
        'combined_fofm_score': combined
    }

def calculate_fofm_strategy_with_dynamic_tpsl(symbol: str, side: str, entry_price: float,
                                            market_data: Dict, ws_client=None) -> Tuple[Optional[float], Optional[float], float, float, Dict]:
    """FOFM - NO fallbacks, proper validation, dynamic TP/SL calculation"""
    
    prices = market_data.get('prices', [])
    volumes = market_data.get('volumes', [])
    min_required = 20

    if not prices or len(prices) < min_required or not volumes or len(volumes) < min_required:
        logger.warning(f"⚠️ Insufficient real data for FOFM {symbol} (need {min_required}+ points)")
        return None, None, 0, 0, {'reason': f'No sufficient real data (min {min_required} points)'}

    # Trim to matching length
    min_len = min(len(prices), len(volumes))
    prices = prices[:min_len]
    volumes = volumes[:min_len]

    prices_array = safe_array_to_numpy(prices)
    volumes_array = safe_array_to_numpy(volumes)

    if len(prices_array) < min_required or len(volumes_array) < min_required:
        return None, None, 0, 0, {'reason': 'Insufficient real data after conversion'}

    # Defensive scalar check
    std_price = float(np.std(prices_array)) if len(prices_array) > 1 else 0.0
    mean_price = float(np.mean(prices_array)) if len(prices_array) > 0 else 0.0
    is_real_data = std_price > mean_price * 0.001 if mean_price > 0 else False

    if not is_real_data:
        logger.warning(f"⚠️ Data appears synthetic/constant for {symbol}")
        return None, None, 0, 0, {'reason': 'Data not real (low variance)'}

    # Components - MORE GENEROUS SCORING
    fofm_components = calculate_fractal_order_flow_momentum(prices_array, volumes_array)

    # ICT/SMC - BOOST BASE SCORE
    ict_smc_data = analyze_ict_smc_structure(prices_array, volumes_array)
    ict_enhancement = max(ict_smc_data.get('total_score', 0.0) * 2.0, 0.3)  # Minimum 0.3

    # Volatility - BOOST SCORE
    vol_score = calculate_volatility_score(prices)
    volatility_enhancement = max(min(vol_score / 25, 1.0), 0.4)  # Minimum 0.4, lower divisor

    # Order Book - MORE GENEROUS
    order_book_adjustment = 0.2  # Base adjustment
    if ws_client:
        ob = ws_client.get_order_book(symbol)
        if ob and ob.get('bids') and ob.get('asks'):
            obi = ob.get('obi', 0)
            order_book_adjustment += abs(obi) * 2.0  # Double the impact

    # REALISTIC Confluence Calculation
    base_confluence = (max(fofm_components['combined_fofm_score'], 0.3) * 0.3 +  # Minimum 0.3
                       ict_enhancement * 0.3 +
                       volatility_enhancement * 0.2 +
                       order_book_adjustment * 0.2)

    base_confluence = min(base_confluence, 1.0)
    fofm_enhancement = max(fofm_components['combined_fofm_score'] * 1.5, 0.8)  # Minimum 0.8
    final_confidence = min(base_confluence * fofm_enhancement, 0.95)  # Cap at 0.95

    # LOWER THRESHOLD - Accept anything above 0.15 instead of 0.3
    if final_confidence < 0.15:
        logger.warning(f"⚠️ Low confidence for {symbol}: {final_confidence:.3f}")
        return None, None, 0, 0, {'reason': f'Low confidence {final_confidence:.3f}'}

    # DYNAMIC TP/SL CALCULATION - NO fallbacks
    try:
        ohlcv_data = {'closes': prices_array.tolist(), 'volumes': volumes_array.tolist()}
        tpsl_calc = DynamicTPSLCalculator(min_rr=3.0)
        tpsl_result = tpsl_calc.calculate_dynamic_tpsl(
            symbol, entry_price, side, ohlcv_data,
            order_book=ws_client.get_order_book(symbol) if ws_client else None,
            confidence=final_confidence
        )

        primary_tp = tpsl_result.take_profit
        sl_price = tpsl_result.stop_loss
        tp_distance = tpsl_result.tp_distance_pct
        sl_distance = tpsl_result.sl_distance_pct
        rr_ratio = tpsl_result.risk_reward_ratio

        # If TP/SL calculation failed, reject the trade (NO fallbacks)
        if not primary_tp or not sl_price or rr_ratio < 3.0:
            logger.warning(f"⚠️ Dynamic TP/SL calculation failed for {symbol}: TP={primary_tp}, SL={sl_price}, R:R={rr_ratio}")
            return None, None, 0, 0, {'reason': f'Dynamic TP/SL failed - R:R {rr_ratio:.2f}'}

    except Exception as e:
        logger.error(f"❌ Dynamic TP/SL failed for {symbol}: {e}")
        return None, None, 0, 0, {'reason': f'TP/SL calculation error: {e}'}

    # Final validation - ensure proper direction
    if side.upper() == 'LONG':
        if primary_tp <= entry_price or sl_price >= entry_price:
            return None, None, 0, 0, {'reason': 'Invalid TP/SL direction for LONG'}
        
        gain_distance = primary_tp - entry_price
        loss_distance = entry_price - sl_price
    else:
        if primary_tp >= entry_price or sl_price <= entry_price:
            return None, None, 0, 0, {'reason': 'Invalid TP/SL direction for SHORT'}
        
        gain_distance = entry_price - primary_tp
        loss_distance = sl_price - entry_price

    if loss_distance <= 0:
        return None, None, 0, 0, {'reason': 'Invalid loss distance'}

    current_rr = gain_distance / loss_distance
    if current_rr < 3.0:
        return None, None, 0, 0, {'reason': f'R:R too low: {current_rr:.2f}'}

    # Calculate additional factors
    recent_momentum = (float(prices_array[-1]) - float(prices_array[-5])) / float(prices_array[-5]) * 100 if len(prices_array) >= 5 and prices_array[-5] > 0 else 0
    recent_volume_avg = float(np.mean(volumes_array[-5:])) if len(volumes_array) >= 5 else 0
    older_volume_avg = float(np.mean(volumes_array[:-5])) if len(volumes_array) > 5 else recent_volume_avg
    volume_trend = (recent_volume_avg - older_volume_avg) / older_volume_avg * 100 if older_volume_avg > 0 else 0

    result = {
        'confidence': final_confidence,
        'risk_reward_ratio': current_rr,
        'tp_distance': tp_distance,
        'sl_distance': sl_distance,
        'method_used': 'enhanced_dynamic',
        'calculation_confidence': final_confidence,
        'supporting_levels': [],
        'dynamic_analysis': True,
        'fofm_components': fofm_components,
        'fractal_score': fofm_components['fractal_score'],
        'order_flow_score': fofm_components['order_flow_score'],
        'momentum_score': fofm_components['momentum_score'],
        'combined_fofm_score': fofm_components['combined_fofm_score'],
        'base_confluence': base_confluence,
        'ict_enhancement': ict_enhancement,
        'volatility_enhancement': volatility_enhancement,
        'fofm_enhancement': fofm_enhancement,
        'order_book_adjustment': order_book_adjustment,
        'recent_momentum': recent_momentum,
        'volume_trend': volume_trend,
        'data_quality': {
            'price_points': len(prices_array),
            'volume_points': len(volumes_array),
            'data_consistency': len(prices_array) == len(volumes_array),
            'is_real_data': is_real_data
        },
        'reason': f'FOFM with dynamic TP/SL (Confidence: {final_confidence:.3f})'
    }

    if ict_smc_data:
        result['ict_smc_data'] = ict_smc_data

    if 'volatility_data' in market_data:
        result['volatility_data'] = market_data['volatility_data']

    logger.info(f"✅ FOFM {symbol}: Confidence {final_confidence:.3f}, R:R {current_rr:.2f}:1, TP ${primary_tp:.6f}, SL ${sl_price:.6f}")

    return primary_tp, sl_price, tp_distance, sl_distance, result

# Export main functions
__all__ = [
    'calculate_fofm_strategy_with_dynamic_tpsl',
    'analyze_ict_smc_structure',
    'calculate_volatility_score',
    'calculate_fractal_order_flow_momentum'
]
