# ticker_cache.py - FIXED PARSING: Handle Direct Instruments Dict Response + Dynamic USDT Perpetuals Only

import os
import pandas as pd
from datetime import datetime, timedelta
from coinswitch_futures_api import CoinSwitchFuturesAPI
import json
import logging

CACHE_DIR = 'cache'
SYMBOLS_CACHE = os.path.join(CACHE_DIR, 'futures_symbols.xlsx')
SYMBOLS_JSON = os.path.join(CACHE_DIR, 'symbols_metadata.json')
TRADE_INFO_CACHE = os.path.join(CACHE_DIR, 'trade_info.xlsx')
REFRESH_DAYS = 1  # Daily refresh for dynamic

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def _is_fresh(path: str, days: int = REFRESH_DAYS) -> bool:
    """Check if file exists and is within refresh period"""
    if not os.path.exists(path):
        return False
    
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) < timedelta(days=days)
    except Exception as e:
        logger.debug(f"Error checking file freshness: {e}")
        return False

def load_or_refresh_symbols(api: CoinSwitchFuturesAPI, force: bool = False) -> pd.DataFrame:
    """DYNAMIC: Load futures symbols from API instrument_info - Handle direct dict or wrapped response"""
    if (not force) and _is_fresh(SYMBOLS_CACHE, REFRESH_DAYS):
        try:
            logger.info(f"📁 Loading cached symbols from {SYMBOLS_CACHE}")
            df = pd.read_excel(SYMBOLS_CACHE, engine='openpyxl')
            if not df.empty and 'symbol' in df.columns:
                return df
        except Exception as e:
            logger.warning(f"Failed to load cached symbols: {e}")
    
    logger.info(f"🔄 Fetching DYNAMIC symbols from API instrument_info")
    
    try:
        # API call
        response = api._make_request(
            "GET",
            "/trade/api/v2/futures/instrument_info",
            params={"exchange": "EXCHANGE_2"}
        )
        
        # FIXED PARSING: Handle direct dict (as in provided data) or wrapped {'code': 200, 'data': dict}
        if isinstance(response, dict):
            if 'code' in response and response['code'] == 200 and 'data' in response:
                instruments = response['data']
            elif 'data' in response:
                instruments = response['data']
            else:
                # Direct instruments dict (no code/data wrapper)
                instruments = response
        else:
            raise ValueError(f"Invalid response type: {type(response)}")
        
        if not isinstance(instruments, dict) or len(instruments) == 0:
            raise ValueError("No instruments data in response")
        
        # Filter to active USDT perpetual futures ONLY
        active_symbols = []
        for sym, info in instruments.items():
            try:
                # Ensure it's USDT perpetual, trading status
                if (str(sym).upper().endswith('USDT') and 
                    info.get('status', '').upper() == 'TRADING' and 
                    info.get('type', '').upper() == 'PERPETUAL_FUTURES'):
                    
                    active_symbols.append({
                        'symbol': sym.upper(),
                        'base_asset': info.get('base_asset', '').upper(),
                        'quote_asset': info.get('quote_asset', '').upper(),
                        'max_leverage': int(info.get('max_leverage', 20)),
                        'min_size': float(info.get('min_base_quantity', 0.001)),
                        'price_precision': int(info.get('price_precision', 2)),
                        'quantity_precision': int(info.get('quantity_precision', 3)),
                        'taker_fee_rate': float(info.get('taker_fee_rate', 0.0006))
                    })
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping invalid instrument {sym}: {e}")
                continue
        
        df = pd.DataFrame(active_symbols)
        if df.empty:
            raise ValueError("No active USDT perpetual futures found in API response")
        
        # Save to cache
        with pd.ExcelWriter(SYMBOLS_CACHE, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Symbols', index=False)
        
        # Metadata
        metadata = {
            'last_refresh': datetime.now().isoformat(),
            'next_refresh': (datetime.now() + timedelta(days=REFRESH_DAYS)).isoformat(),
            'symbol_count': len(df),
            'refresh_cycle_days': REFRESH_DAYS,
            'source': 'instrument_info_api',
            'filtered_count': len(active_symbols)
        }
        
        with open(SYMBOLS_JSON, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Cached {len(df)} dynamic active USDT perpetuals from API")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch dynamic symbols: {e}")
        return pd.DataFrame()  # Empty triggers error in main

def load_or_refresh_trade_info(api: CoinSwitchFuturesAPI, force: bool = False) -> pd.DataFrame:
    """Trade info from instruments - same parsing fix"""
    if (not force) and _is_fresh(TRADE_INFO_CACHE, REFRESH_DAYS):
        try:
            logger.info(f"📁 Loading cached trade info from {TRADE_INFO_CACHE}")
            return pd.read_excel(TRADE_INFO_CACHE, engine='openpyxl')
        except Exception as e:
            logger.warning(f"Failed to load cached trade info: {e}")
    
    logger.info(f"🔄 Fetching dynamic trade info from API")
    
    try:
        # Reuse instrument_info call
        response = api._make_request(
            "GET",
            "/trade/api/v2/futures/instrument_info",
            params={"exchange": "EXCHANGE_2"}
        )
        
        # FIXED PARSING: Same as above
        if isinstance(response, dict):
            if 'code' in response and response['code'] == 200 and 'data' in response:
                instruments = response['data']
            elif 'data' in response:
                instruments = response['data']
            else:
                instruments = response
        else:
            raise ValueError(f"Invalid response type: {type(response)}")
        
        if not isinstance(instruments, dict):
            raise ValueError("No instruments data")
        
        data = []
        for sym, info in instruments.items():
            try:
                if str(sym).upper().endswith('USDT') and info.get('status', '').upper() == 'TRADING':
                    data.append({
                        'symbol': sym.upper(),
                        'max_leverage': int(info.get('max_leverage', 20)),
                        'quantity_precision': int(info.get('quantity_precision', 3)),
                        'price_precision': int(info.get('price_precision', 2)),
                        'taker_fee_rate': float(info.get('taker_fee_rate', 0.0006)),
                        'min_size': float(info.get('min_base_quantity', 0.001))
                    })
            except (ValueError, KeyError):
                continue
        
        df = pd.DataFrame(data)
        if not df.empty:
            with pd.ExcelWriter(TRADE_INFO_CACHE, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Trade_Info', index=False)
            logger.info(f"💾 Cached {len(df)} trade info from API")
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Failed to fetch trade info: {e}")
        return pd.DataFrame()

def get_cache_status() -> dict:
    """Get status - unchanged"""
    status = {}
    
    if os.path.exists(SYMBOLS_JSON):
        try:
            with open(SYMBOLS_JSON, 'r') as f:
                status['symbols'] = json.load(f)
        except Exception as e:
            status['symbols'] = {"status": f"Error: {e}"}
    else:
        status['symbols'] = {"status": "No cache - fetch from API"}
    
    if os.path.exists(TRADE_INFO_CACHE):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(TRADE_INFO_CACHE))
            status['trade_info'] = {
                'last_refresh': mtime.isoformat(),
                'next_refresh': (mtime + timedelta(days=REFRESH_DAYS)).isoformat(),
                'is_fresh': _is_fresh(TRADE_INFO_CACHE),
                'file_size': os.path.getsize(TRADE_INFO_CACHE)
            }
        except Exception as e:
            status['trade_info'] = {"status": f"Error: {e}"}
    else:
        status['trade_info'] = {"status": "No cache - fetch from API"}
    
    status['overall'] = {
        'cache_dir': CACHE_DIR,
        'symbols_fresh': os.path.exists(SYMBOLS_CACHE) and _is_fresh(SYMBOLS_CACHE),
        'trade_info_fresh': os.path.exists(TRADE_INFO_CACHE) and _is_fresh(TRADE_INFO_CACHE),
        'refresh_cycle_days': REFRESH_DAYS
    }
    
    return status

def clear_cache():
    """Clear cache - unchanged"""
    files_to_remove = [SYMBOLS_CACHE, SYMBOLS_JSON, TRADE_INFO_CACHE]
    removed_count = 0
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Removed {file_path}")
                removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"✅ Cache cleared: {removed_count} files")

def force_refresh_all(api: CoinSwitchFuturesAPI):
    """Force refresh - unchanged"""
    logger.info("🔄 Force refreshing dynamic data from API...")
    
    symbols_df = load_or_refresh_symbols(api, force=True)
    trade_info_df = load_or_refresh_trade_info(api, force=True)
    
    logger.info(f"✅ Refresh complete: {len(symbols_df)} symbols, {len(trade_info_df)} trade info")
    
    return symbols_df, trade_info_df