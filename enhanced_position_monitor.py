# enhanced_position_monitor.py - v6.3 STREAMLINED (REDUCED SPAM + BETTER INTERVALS + P&L/SUPERVISOR CALLBACKS)

import time
import logging
import threading
from datetime import datetime

def monitor_position_with_dynamic_tpsl(api, symbol: str, side: str, entry_price: float,
                                     tpsl_result: dict, websocket_client=None, max_hold_seconds: int = 1800,
                                     pnl_tracker=None, supervisor=None):
    """ENHANCED: Dynamic position monitoring with REDUCED LOGGING FREQUENCY + CALLBACKS"""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    symbol_clean = symbol.upper().replace('/', '').replace('-', '')
    
    # Get confidence and volatility data
    confidence = tpsl_result.get('confidence', 0.5)
    is_volatile = tpsl_result.get('volatility_data', {}).get('is_volatile', False)
    volatility_score = tpsl_result.get('volatility_data', {}).get('volatility_score', 0)
    
    # Determine monitoring intervals based on volatility
    if is_volatile or volatility_score > 25:
        check_interval = 15  # Faster check every 15 seconds for volatile assets
        log_interval = 60   # Log every 1 minute
        logger.info(f"🔥 Volatile asset detected - using enhanced monitoring mode (15s checks)")
    else:
        check_interval = 30  # Check every 30 seconds for normal assets
        log_interval = 90   # Log every 90 seconds
        logger.info(f"📊 Standard asset - using normal monitoring mode (30s checks)")
    
    max_hold_minutes = max_hold_seconds // 60
    logger.info(f"🔄 ENHANCED position monitoring started for {symbol}")
    logger.info(f"📊 Confidence: {confidence*100:.1f}%, Volatile: {is_volatile}, Max Hold: {max_hold_minutes}min")
    
    last_log_time = 0
    consecutive_checks_without_position = 0
    
    try:
        while (time.time() - start_time) < max_hold_seconds:
            try:
                # Check if position still exists
                position = get_open_position_for_symbol(api, symbol)
                current_time = time.time()
                
                if position:
                    consecutive_checks_without_position = 0
                    
                    # Extract position data safely
                    current_qty = float(position.get('position_size', 0))
                    current_price = float(position.get('mark_price', entry_price))
                    unrealized_pnl = float(position.get('unrealised_pnl', 0))
                    
                    # Update unrealized P&L in tracker (if provided)
                    if pnl_tracker:
                        pnl_tracker.update_unrealized_pnl(symbol, unrealized_pnl)
                    
                    # Only log periodically to reduce spam
                    if current_time - last_log_time >= log_interval:
                        elapsed_minutes = (current_time - start_time) / 60
                        price_change_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"📊 {symbol} monitoring update:")
                        logger.info(f"   ⏰ Elapsed: {elapsed_minutes:.1f}min, Qty: {current_qty}, Price: ${current_price:.6f}")
                        logger.info(f"   📈 Price change: {price_change_pct:+.2f}%, P&L: ${unrealized_pnl:.3f}")
                        last_log_time = current_time
                    
                    # Dynamic exit conditions based on volatility and confidence
                    if is_volatile and confidence < 0.4 and abs(unrealized_pnl) > 1.0:
                        logger.info(f"🚀 Quick exit triggered for volatile {symbol}: P&L ${unrealized_pnl:.3f}")
                        
                        # Trigger manual close if needed (e.g., cancel orders)
                        try:
                            api.cancel_order(decision.get('main_order_id', ''))
                            logger.info(f"🛑 Manual close initiated for {symbol}")
                        except:
                            pass
                        break
                        
                else:
                    consecutive_checks_without_position += 1
                    
                    # Only log when position is actually gone
                    if consecutive_checks_without_position == 1:
                        logger.info(f"✅ Position closed for {symbol} (natural exit - TP/SL hit)")
                    elif consecutive_checks_without_position >= 3:
                        logger.info(f"✅ Position confirmed closed for {symbol} after {consecutive_checks_without_position} checks")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.debug(f"Position check error for {symbol}: {e}")
                time.sleep(check_interval)
                continue
                
    except KeyboardInterrupt:
        logger.info(f"🛑 Position monitoring stopped by user for {symbol}")
    except Exception as e:
        logger.error(f"❌ Position monitoring error for {symbol}: {e}")
    finally:
        elapsed_total = (time.time() - start_time) / 60
        logger.info(f"🏁 Position monitoring completed for {symbol} after {elapsed_total:.1f} minutes")
    
    # Return meaningful result for P&L tracking
    try:
        final_position = get_open_position_for_symbol(api, symbol)
        if final_position:
            final_pnl = float(final_position.get('unrealised_pnl', 0))
            return f"monitoring_active_pnl_{final_pnl:.3f}"
        else:
            # On confirmed close: Trigger callbacks
            exit_price = entry_price  # Default; fetch real if possible
            try:
                # Try to get last known price from WS
                if websocket_client:
                    ticker = websocket_client.get_ticker_data(symbol)
                    if ticker:
                        exit_price = float(ticker.get('price', entry_price))
            except:
                pass
            
            exit_data = {
                'exit_price': exit_price,
                'exit_reason': 'TP/SL Hit or Natural Close',
                'total_fees': 0.01  # Estimate; refine in production
            }
            
            if pnl_tracker:
                pnl_tracker.update_trade_exit(symbol, exit_data)
            
            if supervisor:
                supervisor.on_position_closed(
                    symbol, exit_price, exit_data['exit_reason'],
                    fees_usdt=exit_data['total_fees'],
                    hold_time_seconds=int(time.time() - start_time)
                )
            
            return "position_closed_naturally"
    except:
        return "monitoring_completed"

def get_open_position_for_symbol(api, symbol: str):
    """Helper function to get position"""
    try:
        clean_sym = symbol.upper().replace('/', '').replace('-', '').replace('_', '')
        resp = api.get_positions()
        data = resp.get("data", [])
        
        for pos in data:
            pos_symbol = str(pos.get('symbol', '')).upper()
            if pos_symbol == clean_sym:
                status = pos.get('status', '').upper()
                if status in ['OPEN', 'ACTIVE', 'FILLED', 'HOLDING']:
                    qty = float(pos.get('position_size', 0))
                    if abs(qty) >= 0.0001:
                        return pos
        return None
        
    except Exception as e:
        logging.getLogger(__name__).debug(f"Position check failed: {e}")
        return None