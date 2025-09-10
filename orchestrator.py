# orchestrator.py - COMPLETE ENHANCED VERSION - FIXED: Proper quantity limits from API

import logging
import threading
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import queue
import concurrent.futures
import statistics
from dataclasses import dataclass

# Enhanced imports
from fractal_order_flow_strategy import (
    calculate_fofm_strategy_with_dynamic_tpsl,
    analyze_ict_smc_structure,
    calculate_volatility_score
)
from data_adapter import fetch_price_volume, get_futures_trade_info, enhanced_market_data_collection

# New integrations
from supervisor import TradeSupervisor
from pnl_tracker import EnhancedPnLTracker

# ================================================================================================
# ENHANCED DATA STRUCTURES FOR ADVANCED ANALYSIS
# ================================================================================================

@dataclass
class MarketStructure:
    """Market structure analysis results"""
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str
    trend_strength: float
    key_level_proximity: float
    structure_quality: float
    breakout_potential: float

@dataclass
class MultiTimeframeAnalysis:
    """Multi-timeframe analysis results"""
    timeframes: Dict[str, Dict]
    overall_trend: str
    trend_alignment: float
    momentum_score: float
    reversal_probability: float
    confidence_boost: float

@dataclass
class VolatilityRegime:
    """Volatility regime analysis"""
    current_volatility: float
    volatility_percentile: float
    regime_type: str # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
    optimal_position_size: float
    risk_adjustment: float

# Global state management
_position_cache = {}
_recently_traded = set()
_position_lock = threading.Lock()
_order_queue = queue.Queue()
_monitoring_threads = {}
_execution_stats = defaultdict(int)
_performance_metrics = {
    'total_executions': 0,
    'successful_executions': 0,
    'failed_executions': 0,
    'average_execution_time': 0,
    'total_pnl': 0.0,
    'win_rate': 0.0
}

# ENHANCED CONFIGURATION - Targeting 60%+ Win Rate
MAX_CONCURRENT_POSITIONS = 8 # Reduced for better focus
MAX_RETRY_ATTEMPTS = 3
DEFAULT_ORDER_TIMEOUT = 30
POSITION_MONITOR_INTERVAL = 15
EXECUTION_THREAD_POOL_SIZE = 4
MINIMUM_RISK_REWARD_RATIO = 4.0 # Increased to 4:1 for higher quality

# ENHANCED STRATEGY CONFIG - RELAXED FOR ACTUAL TRADING
ENHANCED_STRATEGY_CONFIG = {
    # Entry Signal Requirements - LOWERED
    'min_confidence_score': 0.55, # ← CHANGED from 0.80 to 0.55
    'min_risk_reward_ratio': 3.0, # ← CHANGED from 4.0 to 3.0
    'min_confluence_factors': 2, # ← CHANGED from 3 to 2
    
    # Multi-timeframe Requirements - RELAXED
    'timeframe_alignment_threshold': 0.60, # ← CHANGED from 0.75 to 0.60
    'momentum_confirmation_required': True,
    'trend_strength_minimum': 0.45, # ← CHANGED from 0.6 to 0.45
    
    # Market Structure Requirements - RELAXED
    'key_level_distance_min': 0.015, # ← CHANGED from 0.02 to 0.015
    'structure_quality_min': 0.50, # ← CHANGED from 0.7 to 0.50
    'breakout_confirmation_required': True,
    
    # Volatility Filtering - WIDENED
    'volatility_regime_filtering': True,
    'optimal_volatility_range': (0.1, 0.95), # ← CHANGED from (0.3, 0.8) to wider range
    'volatility_adjustment_enabled': True,
    
    # Risk Management
    'max_trades_per_session': 8, # ← INCREASED from 5 to 8
    'max_risk_per_trade': 0.03,
    'dynamic_position_sizing': True,
    'trailing_stops_enabled': True,
}

# POSITION SIZING CONFIG - Advanced risk management
POSITION_SIZING_CONFIG = {
    'method': 'adaptive_kelly', # Advanced Kelly Criterion
    'base_risk_pct': 0.015, # 1.5% base risk
    'max_risk_pct': 0.03, # 3% maximum risk
    'confidence_scaling': True, # Scale with confidence
    'volatility_adjustment': True, # Adjust for volatility
    'correlation_adjustment': True, # Reduce if correlated positions
    'max_position_pct': 0.2, # Max 20% per trade
}

# ================================================================================================
# COINSWITCH API NORMALIZATION
# ================================================================================================

def normalize_side_for_api(side: str) -> str:
    """Convert trading side to CoinSwitch API format"""
    side_normalized = str(side).upper().strip()
    if side_normalized in ['LONG', 'BUY']:
        return 'buy'
    elif side_normalized in ['SHORT', 'SELL']:
        return 'sell'
    else:
        raise ValueError(f"Invalid side '{side}'. Expected: LONG/SHORT or buy/sell")

def get_opposite_side_for_api(side: str) -> str:
    """Get opposite side for TP/SL orders"""
    normalized_side = normalize_side_for_api(side)
    return 'sell' if normalized_side == 'buy' else 'buy'

# ================================================================================================
# ENHANCED MARKET STRUCTURE ANALYSIS
# ================================================================================================

def debug_confidence_scores(opportunities: List[Dict]) -> None:
    """Debug function to see actual confidence scores"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 DEBUGGING {len(opportunities)} opportunity confidence scores:")
    
    for i, opp in enumerate(opportunities, 1):
        symbol = opp.get('symbol', 'UNKNOWN')
        confidence = opp.get('confidence', 0.0)
        
        # Check if it would pass various thresholds
        pass_55 = "✅" if confidence >= 0.55 else "❌"
        pass_65 = "✅" if confidence >= 0.65 else "❌"
        pass_75 = "✅" if confidence >= 0.75 else "❌"
        
        logger.info(f"   {i}. {symbol}: {confidence:.3f} | 0.55:{pass_55} 0.65:{pass_65} 0.75:{pass_75}")

def analyze_support_resistance_levels(prices: List[float], volumes: List[float] = None) -> Tuple[List[float], List[float]]:
    """
    Advanced support and resistance level detection using price action and volume
    """
    logger = logging.getLogger(__name__)
    
    if len(prices) < 50:
        return [], []

    try:
        prices_array = np.array(prices)
        
        # Find local minima and maxima
        try:
            from scipy.signal import find_peaks
            
            # Support levels (local minima)
            inverted_prices = -prices_array
            support_peaks, support_props = find_peaks(inverted_prices, distance=5, prominence=0.001)
            support_levels = [prices[i] for i in support_peaks]
            
            # Resistance levels (local maxima)
            resistance_peaks, resistance_props = find_peaks(prices_array, distance=5, prominence=0.001)
            resistance_levels = [prices[i] for i in resistance_peaks]
        except ImportError:
            # Simple fallback without scipy
            support_levels = []
            resistance_levels = []
            window = 10
            
            for i in range(window, len(prices) - window):
                # Local minimum
                if all(prices[i] <= prices[j] for j in range(i-window, i+window+1)):
                    support_levels.append(prices[i])
                # Local maximum  
                if all(prices[i] >= prices[j] for j in range(i-window, i+window+1)):
                    resistance_levels.append(prices[i])
        
        # Filter and rank by significance
        current_price = prices[-1]
        
        # Support levels below current price
        support_levels = [level for level in support_levels if level < current_price * 0.98]
        support_levels = sorted(support_levels, reverse=True)[:5] # Top 5 closest
        
        # Resistance levels above current price
        resistance_levels = [level for level in resistance_levels if level > current_price * 1.02]
        resistance_levels = sorted(resistance_levels)[:5] # Top 5 closest
        
        logger.debug(f"📊 Detected {len(support_levels)} support and {len(resistance_levels)} resistance levels")
        return support_levels, resistance_levels

    except Exception as e:
        logger.error(f"❌ Support/Resistance analysis failed: {e}")
        return [], []

def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate EMA for given period"""
    if len(prices) < period:
        return np.mean(prices)
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return float(ema)

def analyze_market_structure(symbol: str, market_data: Dict, ws_client) -> MarketStructure:
    """
    COMPREHENSIVE market structure analysis with support/resistance
    """
    logger = logging.getLogger(__name__)
    
    try:
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        
        if len(prices) < 50:
            return MarketStructure([], [], 'UNKNOWN', 0, 0, 0, 0)

        current_price = prices[-1]
        
        # Detect support and resistance levels
        support_levels, resistance_levels = analyze_support_resistance_levels(prices, volumes)
        
        # Trend analysis with multiple methods
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        ema_12 = calculate_ema(prices[-20:], 12)

        # Determine trend direction and strength
        if current_price > sma_20 > sma_50:
            trend_direction = 'BULLISH'
            trend_strength = min((current_price - sma_50) / sma_50, 0.1) * 10 # Normalize to 0-1
        elif current_price < sma_20 < sma_50:
            trend_direction = 'BEARISH'
            trend_strength = min((sma_50 - current_price) / sma_50, 0.1) * 10
        else:
            trend_direction = 'SIDEWAYS'
            trend_strength = 0.3

        # Calculate distance to key levels
        key_level_proximity = 1.0 # Default: far from levels
        
        if support_levels:
            closest_support_dist = min([abs(current_price - level) / current_price for level in support_levels])
            key_level_proximity = min(key_level_proximity, closest_support_dist)
            
        if resistance_levels:
            closest_resistance_dist = min([abs(current_price - level) / current_price for level in resistance_levels])
            key_level_proximity = min(key_level_proximity, closest_resistance_dist)

        # Structure quality assessment
        structure_factors = []
        
        # Factor 1: Clear trend
        structure_factors.append(trend_strength)
        
        # Factor 2: Well-defined levels
        level_count = len(support_levels) + len(resistance_levels)
        structure_factors.append(min(level_count / 6, 1.0)) # Normalize
        
        # Factor 3: Volume confirmation
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            recent_volume = np.mean(volumes[-5:])
            volume_factor = min(recent_volume / avg_volume, 2.0) / 2.0
            structure_factors.append(volume_factor)
        else:
            structure_factors.append(0.5)
        
        structure_quality = np.mean(structure_factors)
        
        # Breakout potential
        breakout_potential = 0.0
        if key_level_proximity < 0.02: # Very close to key level
            breakout_potential = 0.8
        elif key_level_proximity < 0.05: # Moderately close
            breakout_potential = 0.5
        else:
            breakout_potential = 0.2

        logger.info(f"📊 Market structure for {symbol}: Trend={trend_direction}, Quality={structure_quality:.2f}, Breakout={breakout_potential:.2f}")

        return MarketStructure(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            key_level_proximity=key_level_proximity,
            structure_quality=structure_quality,
            breakout_potential=breakout_potential
        )

    except Exception as e:
        logger.error(f"❌ Market structure analysis failed for {symbol}: {e}")
        return MarketStructure([], [], 'UNKNOWN', 0, 0, 0, 0)

# ================================================================================================
# MULTI-TIMEFRAME ANALYSIS
# ================================================================================================

def fetch_multi_timeframe_data(api, symbol: str) -> Dict[str, List[float]]:
    logger = logging.getLogger(__name__)
    timeframes = {'1h': 24, '4h': 48, '1d': 30}
    multi_tf_data = {}
    
    for tf, limit in timeframes.items():
        try:
            klines_data = api.get_candles(symbol.lower(), '5m', max_points=limit * 12)
            if klines_data and len(klines_data) > 0:
                try:
                    if tf == '1h':
                        # FIX: Use 'c' key for close price, not k[4]
                        prices = [float(k['c']) for i, k in enumerate(klines_data) if i % 12 == 0 and 'c' in k]
                    elif tf == '4h':
                        prices = [float(k['c']) for i, k in enumerate(klines_data) if i % 48 == 0 and 'c' in k]
                    else: # 1d
                        prices = [float(k['c']) for i, k in enumerate(klines_data) if i % 288 == 0 and 'c' in k]
                    
                    if prices:
                        multi_tf_data[tf] = prices[-limit:] if len(prices) > limit else prices
                        logger.info(f"✅ Fetched {len(multi_tf_data[tf])} {tf} candles for {symbol}")
                    else:
                        multi_tf_data[tf] = []
                        
                except Exception as parse_e:
                    logger.warning(f"⚠️ Parsing error for {tf} data for {symbol}: {parse_e}")
                    multi_tf_data[tf] = []
            else:
                multi_tf_data[tf] = []
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {tf} data for {symbol}: {e}")
            multi_tf_data[tf] = []
    
    return multi_tf_data

def analyze_multi_timeframe(symbol: str, api, market_data: Dict) -> MultiTimeframeAnalysis:
    """
    COMPREHENSIVE multi-timeframe analysis for trend confirmation
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get multi-timeframe data
        mtf_data = fetch_multi_timeframe_data(api, symbol)
        
        timeframe_analysis = {}
        trend_votes = {'BULLISH': 0, 'BEARISH': 0, 'SIDEWAYS': 0}
        momentum_scores = []
        
        for tf, prices in mtf_data.items():
            if len(prices) < 10:
                continue
                
            current_price = prices[-1]
            sma_10 = np.mean(prices[-10:])
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_10
            
            # Trend determination
            if current_price > sma_10 > sma_20:
                trend = 'BULLISH'
                momentum = min((current_price - sma_20) / sma_20 * 100, 10) / 10
            elif current_price < sma_10 < sma_20:
                trend = 'BEARISH' 
                momentum = min((sma_20 - current_price) / sma_20 * 100, 10) / 10
            else:
                trend = 'SIDEWAYS'
                momentum = 0.3
            
            # Calculate rate of change
            if len(prices) >= 5:
                roc = (prices[-1] - prices[-5]) / prices[-5] * 100
                momentum = max(momentum, abs(roc) / 10) # Boost momentum with ROC
            
            timeframe_analysis[tf] = {
                'trend': trend,
                'momentum': momentum,
                'current_price': current_price,
                'sma_10': sma_10,
                'sma_20': sma_20
            }
            
            trend_votes[trend] += 1
            momentum_scores.append(momentum)

        # Determine overall trend alignment
        total_votes = sum(trend_votes.values())
        if total_votes == 0:
            return MultiTimeframeAnalysis({}, 'UNKNOWN', 0, 0, 0.5, 0)
        
        # Find dominant trend
        dominant_trend = max(trend_votes, key=trend_votes.get)
        trend_alignment = trend_votes[dominant_trend] / total_votes
        
        # Overall momentum
        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0
        
        # Reversal probability (high when alignment is low)
        reversal_probability = 1 - trend_alignment
        
        # Confidence boost (high when alignment is high and momentum is strong)
        confidence_boost = trend_alignment * avg_momentum

        logger.info(f"📊 Multi-TF analysis for {symbol}: Trend={dominant_trend}, Alignment={trend_alignment:.2f}, Momentum={avg_momentum:.2f}")

        return MultiTimeframeAnalysis(
            timeframes=timeframe_analysis,
            overall_trend=dominant_trend,
            trend_alignment=trend_alignment,
            momentum_score=avg_momentum,
            reversal_probability=reversal_probability,
            confidence_boost=confidence_boost
        )

    except Exception as e:
        logger.error(f"❌ Multi-timeframe analysis failed for {symbol}: {e}")
        return MultiTimeframeAnalysis({}, 'UNKNOWN', 0, 0, 0.5, 0)

# ================================================================================================
# VOLATILITY REGIME ANALYSIS
# ================================================================================================

def calculate_volatility_metrics(prices: List[float], volumes: List[float] = None) -> Dict[str, float]:
    """
    Calculate comprehensive volatility metrics
    """
    if len(prices) < 20:
        return {'volatility': 0, 'percentile': 0, 'trend': 'UNKNOWN'}

    # Calculate returns
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    # Current volatility (20-period)
    current_vol = np.std(returns[-20:]) * np.sqrt(24 * 365) # Annualized
    
    # Historical volatility (full period)
    if len(returns) >= 50:
        historical_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-20)]
        vol_percentile = sum(1 for v in historical_vols if v < np.std(returns[-20:])) / len(historical_vols)
    else:
        vol_percentile = 0.5

    # Volatility trend
    if len(returns) >= 40:
        recent_vol = np.std(returns[-20:])
        prev_vol = np.std(returns[-40:-20])
        vol_trend = 'INCREASING' if recent_vol > prev_vol * 1.1 else 'DECREASING' if recent_vol < prev_vol * 0.9 else 'STABLE'
    else:
        vol_trend = 'UNKNOWN'

    return {
        'current_volatility': current_vol,
        'volatility_percentile': vol_percentile, 
        'volatility_trend': vol_trend
    }

def analyze_volatility_regime(symbol: str, market_data: Dict) -> VolatilityRegime:
    """
    ADVANCED volatility regime analysis for position sizing
    """
    logger = logging.getLogger(__name__)
    
    try:
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        
        if len(prices) < 20:
            return VolatilityRegime(0.2, 0.5, 'NORMAL', 1.0, 1.0)

        # Get volatility metrics
        vol_metrics = calculate_volatility_metrics(prices, volumes)
        current_vol = vol_metrics['current_volatility']
        vol_percentile = vol_metrics['volatility_percentile']

        # Determine regime type
        if vol_percentile < 0.2:
            regime_type = 'LOW'
            optimal_size = 1.2 # Increase size in low vol
            risk_adj = 0.8 # Reduce risk adjustment
        elif vol_percentile < 0.4:
            regime_type = 'NORMAL'
            optimal_size = 1.0
            risk_adj = 1.0
        elif vol_percentile < 0.8:
            regime_type = 'HIGH'
            optimal_size = 0.8 # Reduce size in high vol
            risk_adj = 1.3 # Increase risk adjustment
        else:
            regime_type = 'EXTREME'
            optimal_size = 0.5 # Significantly reduce size
            risk_adj = 1.8 # High risk adjustment

        logger.info(f"📊 Volatility regime for {symbol}: {regime_type} (Vol: {current_vol:.3f}, Percentile: {vol_percentile:.2f})")

        return VolatilityRegime(
            current_volatility=current_vol,
            volatility_percentile=vol_percentile,
            regime_type=regime_type,
            optimal_position_size=optimal_size,
            risk_adjustment=risk_adj
        )

    except Exception as e:
        logger.error(f"❌ Volatility regime analysis failed for {symbol}: {e}")
        return VolatilityRegime(0.2, 0.5, 'NORMAL', 1.0, 1.0)

# ================================================================================================
# ENHANCED BALANCE AND POSITION MANAGEMENT
# ================================================================================================

def get_real_time_balance(api) -> Dict[str, float]:
    """Get REAL-TIME account balance"""
    logger = logging.getLogger(__name__)
    
    for attempt in range(3):
        try:
            response = api._make_request("GET", "/trade/api/v2/futures/wallet_balance")
            if 'data' in response:
                data = response['data']
                base_asset_balances = data.get('base_asset_balances', [])
                for balance_info in base_asset_balances:
                    if balance_info.get('base_asset') == 'USDT':
                        balances = balance_info.get('balances', {})
                        balance_data = {
                            'available_usdt': float(balances.get('total_available_balance', 0)),
                            'total_usdt': float(balances.get('total_balance', 0)),
                            'blocked_usdt': float(balances.get('total_blocked_balance', 0)),
                            'unrealized_pnl': float(balances.get('unrealized_pnl', 0)),
                            'wallet_balance': float(balances.get('wallet_balance', 0))
                        }
                        logger.info(f"💰 Balance: ${balance_data['available_usdt']:.2f} USDT available")
                        return balance_data
                        
            return {'available_usdt': 0.0, 'total_usdt': 0.0, 'blocked_usdt': 0.0, 
                   'unrealized_pnl': 0.0, 'wallet_balance': 0.0}
                   
        except Exception as e:
            logger.warning(f"⚠️ Balance fetch attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(1)
                
    return {'available_usdt': 0.0, 'total_usdt': 0.0, 'blocked_usdt': 0.0,
           'unrealized_pnl': 0.0, 'wallet_balance': 0.0}

def get_symbol_trading_limits(api, symbol: str) -> Dict[str, Any]:
    """FIXED: Get ACTUAL trading limits from API first, then use proper parsing"""
    logger = logging.getLogger(__name__)
    
    # Default values
    max_leverage = 5
    min_order_qty = 1.0
    
    try:
        # STEP 1: Try to get from instrument_info first (most reliable)
        logger.debug(f"Getting instrument info for {symbol}")
        
        response = api._make_request(
            "GET", 
            "/trade/api/v2/futures/instrument_info", 
            params={"exchange": "EXCHANGE_2"}
        )
        
        # Parse instrument_info response
        if isinstance(response, dict):
            if 'code' in response and response['code'] == 200 and 'data' in response:
                instruments = response['data']
            elif 'data' in response:
                instruments = response['data']
            else:
                instruments = response
                
            # Look for the symbol in instruments
            if isinstance(instruments, dict) and symbol in instruments:
                info = instruments[symbol]
                max_leverage = max(1, int(info.get('max_leverage', 5)))
                min_order_qty = float(info.get('min_base_quantity', 1.0))
                
                logger.info(f"📊 From instrument_info - {symbol}: Max leverage={max_leverage}x, Min quantity={min_order_qty}")
                
                return {
                    'max_leverage': max_leverage,
                    'min_quantity': min_order_qty,
                    'max_quantity': float(info.get('max_base_quantity', 250000.0)),
                    'quantity_precision': int(info.get('quantity_precision', 3)),
                    'price_precision': int(info.get('price_precision', 6)),
                    'tick_size': float(info.get('tick_size', 0.000001)),
                    'taker_fee_rate': float(info.get('taker_fee_rate', 0.0006))
                }
                
    except Exception as e:
        logger.debug(f"Instrument info failed for {symbol}: {e}")
    
    # STEP 2: Try leverage endpoint as fallback
    try:
        logger.debug(f"Getting leverage info for {symbol}")
        response = api.get_leverage(symbol)
        
        if response and 'data' in response:
            data = response['data']
            if isinstance(data, dict):
                max_leverage = max(1, int(data.get('max_leverage', 5)))
                min_order_qty = float(data.get('min_quantity', 1.0))
            elif isinstance(data, list) and len(data) > 0:
                item = data[0]
                max_leverage = max(1, int(item.get('max_leverage', 5)))
                min_order_qty = float(item.get('min_quantity', 1.0))
                
        logger.info(f"📊 From leverage API - {symbol}: Max leverage={max_leverage}x, Min quantity={min_order_qty}")
        
    except Exception as e:
        logger.debug(f"Leverage API failed for {symbol}: {e}")
    
    # STEP 3: Enhanced known minimums with MORE symbols (based on error messages)
    symbol_minimums = {
        'BTCUSDT': 0.001, 'ETHUSDT': 0.01, 'BNBUSDT': 0.01,
        'ADAUSDT': 50.0, 'SOLUSDT': 0.1, 'XRPUSDT': 20.0,
        'DOGEUSDT': 100.0, 'MATICUSDT': 25.0, 'LINKUSDT': 1.0,
        'DOTUSDT': 2.0, 'AVAXUSDT': 1.0, 'UNIUSDT': 2.0,
        'LTCUSDT': 0.01, 'BCHUSDT': 0.01, 'ATOMUSDT': 2.0,
        'FILUSDT': 2.0, 'TRXUSDT': 500.0, 'XLMUSDT': 200.0,
        'VETUSDT': 1000.0, 'ETCUSDT': 5.0, 'AIOUSDT': 72.0,
        'FLOCKUSDT': 19.0, 'IPUSDT': 1.0, 'AVNTUSDT': 31.0,
        'BAKEUSDT': 74.3,  # FIXED: From the error message
        'HIFIUSDT': 1.0, '10000SATSUSDT': 25000.0,
        'OPENUSDT': 10.0, 'JELLYJELLYUSDT': 150.0, 'KAITOUSDT': 100.0,
    }
    
    if symbol in symbol_minimums:
        min_order_qty = max(min_order_qty, symbol_minimums[symbol])
        logger.info(f"📊 Using known minimum for {symbol}: {min_order_qty}")
    
    return {
        'max_leverage': max_leverage,
        'min_quantity': min_order_qty,
        'max_quantity': 250000.0,
        'quantity_precision': 3,
        'price_precision': 6,
        'tick_size': 0.000001,
        'taker_fee_rate': 0.0006
    }

def calculate_adaptive_position_size(api, symbol: str, entry_price: float,
                                   confidence: float, market_structure: MarketStructure,
                                   volatility_regime: VolatilityRegime,
                                   mtf_analysis: MultiTimeframeAnalysis) -> Tuple[float, Dict]:
    """
    FIXED: PROPER position sizing using ACTUAL API limits
    """
    logger = logging.getLogger(__name__)
    
    # STEP 1: Get balance and ACTUAL trading limits from API
    balance_data = get_real_time_balance(api)
    available_usdt = balance_data.get('available_usdt', 100.0)
    
    # STEP 2: Get ACTUAL symbol-specific limits from API
    limits = get_symbol_trading_limits(api, symbol)
    max_leverage = limits.get('max_leverage', 5)
    min_quantity = limits.get('min_quantity', 1.0)
    
    logger.info(f"📊 Trading limits for {symbol}: Max leverage={max_leverage}x, Min quantity={min_quantity}")

    # STEP 3: Base risk calculation
    base_risk = POSITION_SIZING_CONFIG['base_risk_pct']
    max_risk = POSITION_SIZING_CONFIG['max_risk_pct']

    # STEP 4: Multi-factor risk scaling
    risk_factors = []
    
    # Factor 1: Confidence scaling (30% weight)
    confidence_factor = confidence * 0.3
    risk_factors.append(confidence_factor)
    
    # Factor 2: Market structure quality (25% weight)
    structure_factor = market_structure.structure_quality * 0.25
    risk_factors.append(structure_factor)
    
    # Factor 3: Multi-timeframe alignment (25% weight)
    mtf_factor = mtf_analysis.trend_alignment * 0.25
    risk_factors.append(mtf_factor)
    
    # Factor 4: Volatility regime (20% weight)
    vol_factor = (1 - abs(volatility_regime.volatility_percentile - 0.5)) * 0.2
    risk_factors.append(vol_factor)

    # Combined risk score
    combined_score = sum(risk_factors)
    
    # Scale risk based on combined score
    if combined_score > 0.85:
        risk_pct = max_risk
    elif combined_score > 0.75:
        risk_pct = base_risk * 2.5
    elif combined_score > 0.65:
        risk_pct = base_risk * 2.0
    elif combined_score > 0.55:
        risk_pct = base_risk * 1.5
    else:
        risk_pct = base_risk

    # Volatility adjustment
    risk_pct *= volatility_regime.optimal_position_size

    # STEP 5: Calculate position size with PROPER minimum validation
    risk_amount = available_usdt * risk_pct
    
    # Choose appropriate leverage (conservative)
    target_leverage = min(3, max_leverage)  # Conservative leverage
    
    # STEP 6: Calculate base quantity
    position_value = risk_amount * target_leverage
    base_quantity = position_value / entry_price
    
    # STEP 7: CRITICAL - Ensure minimum quantity is met
    if base_quantity < min_quantity:
        logger.warning(f"⚠️ Base quantity {base_quantity:.3f} below minimum {min_quantity} for {symbol}")
        
        # OPTION 1: Use minimum quantity if we have enough balance
        min_position_value = min_quantity * entry_price
        min_required_margin = min_position_value / target_leverage
        
        if min_required_margin <= available_usdt * 0.8:  # Use max 80% of balance
            final_quantity = min_quantity
            logger.info(f"✅ Using minimum quantity {min_quantity} for {symbol}")
        else:
            # OPTION 2: Can't afford minimum, skip this trade
            logger.warning(f"❌ Cannot afford minimum quantity for {symbol}: need ${min_required_margin:.2f}, have ${available_usdt:.2f}")
            return 0.0, {
                'error': 'insufficient_balance_for_minimum',
                'min_quantity': min_quantity,
                'required_margin': min_required_margin,
                'available_balance': available_usdt
            }
    else:
        final_quantity = base_quantity
    
    # STEP 8: Final validation
    final_position_value = final_quantity * entry_price
    required_margin = final_position_value / target_leverage
    
    # Ensure we don't exceed available balance
    if required_margin > available_usdt * 0.8:  # Use max 80% of balance
        max_safe_quantity = (available_usdt * 0.8 * target_leverage) / entry_price
        if max_safe_quantity >= min_quantity:
            final_quantity = max_safe_quantity
        else:
            logger.warning(f"❌ Cannot meet minimum quantity while staying within balance limits for {symbol}")
            return 0.0, {
                'error': 'cannot_meet_minimum_within_balance_limits',
                'min_quantity': min_quantity,
                'max_safe_quantity': max_safe_quantity,
                'available_balance': available_usdt
            }

    # Recalculate final metrics
    final_position_value = final_quantity * entry_price
    required_margin = final_position_value / target_leverage
    actual_risk_pct = required_margin / available_usdt
    
    sizing_details = {
        'base_risk_pct': base_risk,
        'calculated_risk_pct': risk_pct,
        'actual_risk_pct': actual_risk_pct,
        'combined_score': combined_score,
        'confidence_factor': confidence_factor,
        'structure_factor': structure_factor,
        'mtf_factor': mtf_factor,
        'vol_factor': vol_factor,
        'target_leverage': target_leverage,
        'max_leverage': max_leverage,
        'min_quantity': min_quantity,
        'base_quantity': base_quantity,
        'final_quantity': final_quantity,
        'position_value': final_position_value,
        'required_margin': required_margin,
        'available_balance': available_usdt
    }

    logger.info(f"📊 Position sizing for {symbol}:")
    logger.info(f"   Risk: {actual_risk_pct:.1%}, Quantity: {final_quantity:.3f}, Leverage: {target_leverage}x")
    logger.info(f"   Position Value: ${final_position_value:.2f}, Required Margin: ${required_margin:.2f}")

    return final_quantity, sizing_details

# ================================================================================================
# ENHANCED EXECUTION WITH PROPER LIMITS AND LIMIT ORDERS
# ================================================================================================

def execute_enhanced_trade_decision(api, decision: Dict, ws_client, supervisor, pnl_tracker=None) -> Dict:
    """
    ENHANCED trade execution with proper limits checking and limit orders
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Extract decision parameters
        symbol = decision['symbol']
        side = decision['action']
        entry_price = decision['entry_price']
        tp_price = decision['tp_price']
        sl_price = decision['sl_price']
        strategy_result = decision['strategy_result']
        market_data = decision['market_data']
        
        logger.info(f"🚀 ENHANCED EXECUTION: {symbol} {side} @ ${entry_price:.6f}")
        
        # Step 1: Get balance
        balance_data = get_real_time_balance(api)
        available_usdt = balance_data.get('available_usdt', 0)
        
        if available_usdt < 5:
            return {'status': 'error', 'reason': f'Insufficient balance: ${available_usdt:.2f}'}
        
        logger.info(f"💰 Balance: ${available_usdt:.2f} USDT available")
        
        # Step 2: Enhanced analysis
        logger.info(f"📊 Performing enhanced analysis for {symbol}")
        
        # Market structure
        market_structure = analyze_market_structure(symbol, market_data, ws_client)
        
        # Multi-timeframe analysis  
        mtf_analysis = analyze_multi_timeframe(symbol, api, market_data)
        
        # Volatility regime
        volatility_regime = analyze_volatility_regime(symbol, market_data)
        
        # Enhanced confidence calculation
        base_confidence = strategy_result.get('confidence', 0.5)
        structure_boost = market_structure.structure_quality * 0.3
        mtf_boost = mtf_analysis.confidence_boost * 0.2
        vol_boost = min(volatility_regime.current_volatility * 0.1, 0.2)
        
        enhanced_confidence = min(base_confidence + structure_boost + mtf_boost + vol_boost, 1.0)
        
        # Composite scoring
        composite_factors = [
            market_structure.structure_quality,
            mtf_analysis.trend_alignment,
            volatility_regime.optimal_position_size,
            market_structure.breakout_potential,
            strategy_result.get('combined_fofm_score', 0.5)
        ]
        composite_score = np.mean(composite_factors)
        
        enhanced_analysis = {
            'composite_score': composite_score,
            'enhanced_confidence': enhanced_confidence,
            'recommendation': 'STRONG_BUY' if composite_score > 0.7 else 'BUY' if composite_score > 0.5 else 'HOLD'
        }
        
        logger.info(f"📊 Enhanced analysis for {symbol}:")
        logger.info(f"   Composite Score: {composite_score:.3f}")
        logger.info(f"   Enhanced Confidence: {enhanced_confidence:.3f}")
        logger.info(f"   Recommendation: {enhanced_analysis['recommendation']}")
        
        # Step 3: Calculate position size with ACTUAL API limits
        final_quantity, sizing_details = calculate_adaptive_position_size(
            api, symbol, entry_price, enhanced_confidence, 
            market_structure, volatility_regime, mtf_analysis
        )
        
        # Check for sizing errors
        if final_quantity <= 0 or 'error' in sizing_details:
            error_msg = sizing_details.get('error', 'Position sizing failed')
            logger.error(f"❌ Position sizing failed for {symbol}: {error_msg}")
            return {'status': 'error', 'reason': f'Position sizing failed: {error_msg}'}
        
        final_leverage = sizing_details['target_leverage']
        
        # Step 4: Set leverage
        logger.info(f"📊 Leverage for {symbol}: Target={final_leverage}x, Max={sizing_details['max_leverage']}x")
        logger.info(f"📊 Attempting leverage update for {symbol}: {final_leverage}x")
        
        try:
            leverage_resp = api.update_leverage(symbol, final_leverage)
            if leverage_resp:
                logger.info(f"✅ Leverage set for {symbol}: {final_leverage}x")
            else:
                logger.warning(f"⚠️ Leverage update failed for {symbol}, proceeding with default")
        except Exception as e:
            logger.warning(f"⚠️ Leverage update error for {symbol}: {e}")
        
        # Step 5: Validate TP/SL prices
        tp_side = get_opposite_side_for_api(side)
        sl_side = get_opposite_side_for_api(side)
        main_side = normalize_side_for_api(side)
        
        # Step 6: Place main LIMIT order
        logger.info(f"📊 Placing enhanced LIMIT order for {symbol}")
        
        main_payload = {
            'side': main_side,
            'symbol': symbol,
            'order_type': 'LIMIT',
            'quantity': final_quantity,
            'price': entry_price
        }
        
        logger.info(f"📊 Main order: {main_payload}")
        
        try:
            main_resp = api.place_order(**main_payload)
            if main_resp and main_resp.get('data', {}).get('order_id'):
                main_order_id = main_resp.get('data', {}).get('order_id', '')
                logger.info(f"✅ Enhanced main order placed: {main_order_id}")
            else:
                raise Exception(f"Main order failed: {main_resp}")
        except Exception as e:
            logger.error(f"❌ Main order error for {symbol}: {e}")
            return {'status': 'error', 'reason': f'Main order failed: {e}'}
        
        # Step 7: Place TP and SL orders
        tp_order_id = ""
        sl_order_id = ""
        
        # Take Profit order
        try:
            tp_payload = {
                'side': tp_side,
                'symbol': symbol,
                'order_type': 'TAKE_PROFIT_MARKET',
                'quantity': final_quantity,
                'trigger_price': tp_price,
                'reduce_only': True
            }
            
            logger.info(f"📊 Placing enhanced TP: {tp_payload}")
            tp_resp = api.place_order(**tp_payload)
            
            if tp_resp and tp_resp.get('data', {}).get('order_id'):
                tp_order_id = tp_resp.get('data', {}).get('order_id', '')
                logger.info(f"✅ Enhanced TP placed: {tp_order_id}")
            else:
                logger.warning(f"⚠️ TP order failed for {symbol}: {tp_resp}")
                
        except Exception as e:
            logger.warning(f"⚠️ TP order error for {symbol}: {e}")
        
        # Stop Loss order
        try:
            sl_payload = {
                'side': sl_side,
                'symbol': symbol,
                'order_type': 'STOP_MARKET',
                'quantity': final_quantity,
                'trigger_price': sl_price,
                'reduce_only': True
            }
            
            logger.info(f"📊 Placing enhanced SL: {sl_payload}")
            sl_resp = api.place_order(**sl_payload)
            
            if sl_resp and sl_resp.get('data', {}).get('order_id'):
                sl_order_id = sl_resp.get('data', {}).get('order_id', '')
                logger.info(f"✅ Enhanced SL placed: {sl_order_id}")
            else:
                logger.warning(f"⚠️ SL order failed for {symbol}: {sl_resp}")
                
        except Exception as e:
            logger.warning(f"⚠️ SL order error for {symbol}: {e}")
        
        # Step 8: Monitor order fill
        filled = False
        actual_entry_price = entry_price
        fill_timeout = time.time() + DEFAULT_ORDER_TIMEOUT
        
        logger.info(f"⏳ Enhanced monitoring for {symbol} ({DEFAULT_ORDER_TIMEOUT}s)")
        
        while time.time() < fill_timeout and not filled:
            try:
                order_status_resp = api.get_order_status(main_order_id)
                if not order_status_resp or not order_status_resp.get('data'):
                    time.sleep(2)
                    continue
                    
                status = order_status_resp.get('data', {}).get('status', '').upper()
                
                if status in ['FILLED', 'PARTIALLY_FILLED']:
                    filled_qty = float(order_status_resp.get('data', {}).get('filled_qty', final_quantity))
                    actual_entry_price = float(order_status_resp.get('data', {}).get('avg_price', entry_price))
                    filled = True
                    logger.info(f"✅ Enhanced fill: {symbol} {filled_qty} @ ${actual_entry_price:.6f}")
                    break
                elif status in ['CANCELLED', 'REJECTED', 'FAILED']:
                    logger.error(f"❌ Order {status} for {symbol}")
                    try:
                        if tp_order_id: api.cancel_order(tp_order_id)
                        if sl_order_id: api.cancel_order(sl_order_id)
                    except: pass
                    return {'status': 'error', 'reason': f'Order {status.lower()}'}
                elif status in ['RAISED', 'PENDING', 'NEW']:
                    logger.debug(f"⏳ Order {status} for {symbol}")
                    time.sleep(2)
                    
            except Exception as e:
                logger.debug(f"Order status error: {e}")
                time.sleep(2)
        
        if not filled:
            logger.warning(f"⚠️ Order still pending for {symbol}, continuing setup")
            actual_entry_price = entry_price
        
        # Step 9: Calculate final metrics
        final_position_value = final_quantity * actual_entry_price
        final_required_margin = final_position_value / final_leverage
        calculated_rr = strategy_result.get('risk_reward_ratio', 3.0)
        
        # Step 10: P&L tracking
        if pnl_tracker:
            try:
                pnl_tracker.log_trade_entry({
                    'symbol': symbol, 'side': side, 'entry_price': actual_entry_price,
                    'quantity': final_quantity, 'leverage': final_leverage,
                    'position_value_usdt': final_position_value,
                    'risk_reward_ratio': calculated_rr,
                    'enhanced_confidence': enhanced_confidence,
                    'composite_score': enhanced_analysis['composite_score'],
                    'market_structure_quality': market_structure.structure_quality,
                    'mtf_alignment': mtf_analysis.trend_alignment,
                    'volatility_regime': volatility_regime.regime_type,
                    'main_order_id': main_order_id,
                    'tp_order_id': tp_order_id,
                    'sl_order_id': sl_order_id,
                    'tp_price': tp_price,
                    'sl_price': sl_price
                })
                logger.info(f"📊 Enhanced P&L logged for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ P&L logging error: {e}")
        
        # Step 11: Supervisor registration
        try:
            supervisor.register_open_trade(
                trade_id=main_order_id, symbol=symbol, side=side,
                quantity=final_quantity, entry_price=actual_entry_price,
                volatility_score=volatility_regime.current_volatility,
                composite_score=enhanced_analysis['composite_score'],
                execution_tier='ENHANCED'
            )
        except Exception as e:
            logger.warning(f"⚠️ Supervisor registration error: {e}")
        
        # Step 12: Enhanced monitoring thread
        def enhanced_background_monitor():
            try:
                from enhanced_position_monitor import monitor_position_with_dynamic_tpsl
                result = monitor_position_with_dynamic_tpsl(
                    api, symbol, side, actual_entry_price,
                    {
                        'enhanced_confidence': enhanced_confidence,
                        'volatility_data': {'volatility_score': volatility_regime.current_volatility, 'is_volatile': volatility_regime.regime_type in ['HIGH', 'EXTREME']},
                        'market_structure': market_structure,
                        'mtf_analysis': mtf_analysis,
                        'volatility_regime': volatility_regime
                    },
                    ws_client, max_hold_seconds=3600,  # 1 hour max
                    pnl_tracker=pnl_tracker, supervisor=supervisor
                )
                
                if result.startswith('monitoring_active_pnl_'):
                    final_pnl = float(result.split('_')[-1])
                    logger.info(f"🏁 Enhanced monitoring complete for {symbol}: P&L ${final_pnl:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced monitoring error for {symbol}: {e}")
        
        monitor_thread = threading.Thread(target=enhanced_background_monitor, daemon=True)
        monitor_thread.start()
        _monitoring_threads[symbol] = monitor_thread
        
        # Step 13: Update performance metrics
        _performance_metrics['total_executions'] += 1
        _performance_metrics['successful_executions'] += 1
        execution_time = time.time() - start_time
        
        # Step 14: Comprehensive success result
        success_result = {
            'status': 'success',
            'symbol': symbol,
            'action': side,
            'main_order_id': main_order_id,
            'take_profit_order_id': tp_order_id,
            'stop_loss_order_id': sl_order_id,
            'entry_price': actual_entry_price,
            'take_profit_price': tp_price,
            'stop_loss_price': sl_price,
            'quantity': final_quantity,
            'leverage': final_leverage,
            'position_value': final_position_value,
            'required_margin': final_required_margin,
            'risk_reward_ratio': calculated_rr,
            'enhanced_confidence': enhanced_confidence,
            'composite_score': enhanced_analysis['composite_score'],
            'execution_time_seconds': execution_time,
            'orders_placed': {
                'main': bool(main_order_id),
                'take_profit': bool(tp_order_id),
                'stop_loss': bool(sl_order_id)
            },
            'balance_after_trade': available_usdt - final_required_margin,
            'sizing_details': sizing_details,
            'filled_immediately': filled,
            'enhancement_data': {
                'market_structure_quality': market_structure.structure_quality,
                'trend_alignment': mtf_analysis.trend_alignment,
                'volatility_regime': volatility_regime.regime_type,
                'breakout_potential': market_structure.breakout_potential,
                'momentum_score': mtf_analysis.momentum_score
            }
        }
        
        # Enhanced success logging
        logger.info(f"✅ ENHANCED TRADE EXECUTION COMPLETED!")
        logger.info(f"   🎯 {symbol} {side} @ ${actual_entry_price:.6f}")
        logger.info(f"   ⚡ Execution: {execution_time:.2f}s")
        logger.info(f"   📊 Orders: Main={'✅' if main_order_id else '❌'}, TP={'✅' if tp_order_id else '⚠️'}, SL={'✅' if sl_order_id else '⚠️'}")
        logger.info(f"   🎯 R:R: {calculated_rr:.2f}:1 (TP: ${tp_price:.6f}, SL: ${sl_price:.6f})")
        logger.info(f"   📈 Enhanced Confidence: {enhanced_confidence:.3f} (Composite: {enhanced_analysis['composite_score']:.3f})")
        logger.info(f"   📊 Quality Metrics: Structure={market_structure.structure_quality:.2f}, MTF={mtf_analysis.trend_alignment:.2f}")
        logger.info(f"   💰 Position: {final_quantity} @ {final_leverage}x = ${final_position_value:.2f}")
        logger.info(f"   🎲 Vol Regime: {volatility_regime.regime_type}, Fill: {'✅' if filled else '⏳'}")
        
        return success_result

    except Exception as e:
        _performance_metrics['failed_executions'] += 1
        execution_time = time.time() - start_time
        logger.error(f"❌ ENHANCED EXECUTION ERROR: {e}")
        return {
            'status': 'error',
            'reason': f'Enhanced execution error: {e}',
            'symbol': symbol,
            'execution_time_seconds': execution_time,
            'error_type': type(e).__name__
        }

# ================================================================================================
# ENHANCED FILTERING AND OPPORTUNITY MANAGEMENT
# ================================================================================================

def filter_opportunities_by_min_rr(opportunities: List[Dict], websocket_client, api, min_rr: float = 3.0) -> List[Dict]:
    """Enhanced filtering with comprehensive analysis and NO fallbacks"""
    logger = logging.getLogger(__name__)
    
    if not opportunities:
        logger.info("🔍 No opportunities to filter")
        return []
    
    logger.info(f"🔍 Enhanced filtering started: {len(opportunities)} opportunities")
    logger.info(f"📊 Starting enhanced filtering of {len(opportunities)} opportunities")
    
    # Debug confidence scores
    debug_confidence_scores(opportunities)
    
    # Stage 1: Confidence filtering
    stage1_qualified = []
    for opp in opportunities:
        confidence = opp.get('confidence', 0.0)
        if confidence >= ENHANCED_STRATEGY_CONFIG['min_confidence_score']:
            stage1_qualified.append(opp)
    
    logger.info(f"📊 Stage 1 (Confidence): {len(stage1_qualified)} qualified")
    
    if not stage1_qualified:
        logger.info("❌ No opportunities passed confidence filter")
        return []
    
    # Stage 2: Enhanced analysis for each opportunity
    stage2_qualified = []
    
    for i, opportunity in enumerate(stage1_qualified):
        try:
            symbol = opportunity['symbol']
            side = 'LONG' if opportunity['price_change_24h'] > 0 else 'SHORT'
            entry_price = opportunity['current_price']
            
            # Get comprehensive market data
            market_data = enhanced_market_data_collection(websocket_client, api, symbol)
            
            # Validate data quality
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            
            if not prices or len(prices) < 20 or len(prices) != len(volumes):
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(prices)} prices, {len(volumes)} volumes")
                continue
            
            # Perform enhanced analysis
            market_structure = analyze_market_structure(symbol, market_data, websocket_client)
            mtf_analysis = analyze_multi_timeframe(symbol, api, market_data)
            volatility_regime = analyze_volatility_regime(symbol, market_data)
            
            # Calculate enhanced confidence
            base_confidence = opportunity.get('confidence', 0.5)
            structure_boost = market_structure.structure_quality * 0.3
            mtf_boost = mtf_analysis.confidence_boost * 0.2
            vol_boost = min(volatility_regime.current_volatility * 0.1, 0.2)
            
            enhanced_confidence = min(base_confidence + structure_boost + mtf_boost + vol_boost, 1.0)
            
            # Composite scoring
            composite_factors = [
                market_structure.structure_quality,
                mtf_analysis.trend_alignment,
                volatility_regime.optimal_position_size,
                market_structure.breakout_potential
            ]
            
            # Add strategy-specific factors
            from fractal_order_flow_strategy import calculate_fofm_strategy_with_dynamic_tpsl
            tp_price, sl_price, tp_dist, sl_dist, strategy_result = calculate_fofm_strategy_with_dynamic_tpsl(
                symbol, side, entry_price, market_data, websocket_client
            )
            
            if not tp_price or not sl_price:
                logger.warning(f"⚠️ Strategy calculation failed for {symbol}")
                continue
            
            # Add FOFM score to composite
            composite_factors.append(strategy_result.get('combined_fofm_score', 0.5))
            composite_score = np.mean(composite_factors)
            
            # Enhanced analysis results
            enhanced_analysis = {
                'market_structure': market_structure,
                'mtf_analysis': mtf_analysis,
                'volatility_regime': volatility_regime,
                'composite_score': composite_score,
                'enhanced_confidence': enhanced_confidence,
                'recommendation': 'STRONG_BUY' if composite_score > 0.7 else 'BUY' if composite_score > 0.5 else 'HOLD'
            }
            
            # Check if passes enhanced criteria
            if (composite_score >= 0.5 and 
                enhanced_confidence >= ENHANCED_STRATEGY_CONFIG['min_confidence_score'] and
                strategy_result.get('risk_reward_ratio', 0) >= ENHANCED_STRATEGY_CONFIG['min_risk_reward_ratio']):
                
                # Add enhanced analysis to opportunity
                opportunity['enhanced_analysis'] = enhanced_analysis
                opportunity['tp_price'] = tp_price
                opportunity['sl_price'] = sl_price
                opportunity['strategy_result'] = strategy_result
                opportunity['market_data'] = market_data
                
                stage2_qualified.append(opportunity)
                
                logger.info(f"📊 Enhanced analysis for {symbol}:")
                logger.info(f"   Composite Score: {composite_score:.3f}")
                logger.info(f"   Enhanced Confidence: {enhanced_confidence:.3f}")
                logger.info(f"   Recommendation: {enhanced_analysis['recommendation']}")
            else:
                logger.debug(f"❌ {symbol} failed enhanced criteria: Score={composite_score:.3f}, Confidence={enhanced_confidence:.3f}")
                
        except Exception as e:
            logger.warning(f"⚠️ Enhanced analysis failed for {opportunity.get('symbol', 'UNKNOWN')}: {e}")
            continue
    
    logger.info(f"📊 Stage 2 (Analysis): {len(stage2_qualified)} qualified")
    
    if not stage2_qualified:
        logger.info("❌ No opportunities passed enhanced analysis")
        return []
    
    # Sort by composite score (best first)
    stage2_qualified.sort(key=lambda x: x['enhanced_analysis']['composite_score'], reverse=True)
    
    # Final result
    logger.info(f"📊 Enhanced filtering complete: {len(stage2_qualified)} top opportunities selected")
    for i, opp in enumerate(stage2_qualified, 1):
        analysis = opp['enhanced_analysis']
        logger.info(f"   {i}. {opp['symbol']}: Score={analysis['composite_score']:.3f}, Confidence={analysis['enhanced_confidence']:.3f}, Rec={analysis['recommendation']}")
    
    logger.info(f"📊 Enhanced filtering result: {len(stage2_qualified)} qualified opportunities")
    return stage2_qualified

def execute_trade_decision(api, decision: Dict, websocket_client, supervisor, pnl_tracker=None) -> Dict:
    """
    Execute trade decision using enhanced execution
    """
    return execute_enhanced_trade_decision(api, decision, websocket_client, supervisor, pnl_tracker)

# Export main functions
__all__ = [
    'execute_trade_decision',
    'filter_opportunities_by_min_rr', 
    'get_real_time_balance',
    'get_symbol_trading_limits',
    'calculate_adaptive_position_size',
    'analyze_market_structure',
    'analyze_multi_timeframe',
    'analyze_volatility_regime'
]
