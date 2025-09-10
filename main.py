import logging
import time
import threading
import pandas as pd
import random
from datetime import datetime
import sys
import os
import numpy as np

# Core imports
from orchestrator import (
    execute_trade_decision,
    filter_opportunities_by_min_rr,
    get_real_time_balance
)
from coinswitch_futures_api import CoinSwitchFuturesAPI
from websocket_client import CoinSwitchDynamicWebSocket
from data_adapter import enhanced_market_data_collection
from pnl_tracker import get_pnl_tracker
from supervisor import TradeSupervisor
from ticker_cache import load_or_refresh_symbols, get_cache_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extreme_volatility_trading.log')
    ]
)

logger = logging.getLogger(__name__)

class ExtremeVolatilityTradingSystem:
    def __init__(self):
        self.api = None
        self.websocket_client = None
        self.supervisor = None
        self.pnl_tracker = None
        self.running = False
        self.cycle_count = 0
        self.successful_trades = 0
        self.total_trades = 0
        self.all_symbols = []
        
    def initialize(self):
        """Initialize with FULLY DYNAMIC symbol discovery from API"""
        try:
            # Initialize API
            self.api = CoinSwitchFuturesAPI()
            logger.info("✅ API validated")
            
            # Load DYNAMIC symbols from cache/API
            symbols_df = load_or_refresh_symbols(self.api)
            if symbols_df.empty:
                raise Exception("❌ No symbols available from API - cannot proceed without real data")
            
            self.all_symbols = symbols_df['symbol'].tolist()
            logger.info(f"📁 Dynamic symbols loaded: {len(self.all_symbols)} from API")
            logger.info(f"📊 Cache status: {get_cache_status()}")
            
            # Initialize WebSocket
            self.websocket_client = CoinSwitchDynamicWebSocket(api_key="", api_client=self.api)
            self.websocket_client.start()
            
            # Wait for WS to populate
            wait_time = 0
            live_count = len(self.websocket_client.get_all_tickers())
            while live_count < len(self.all_symbols) * 0.3 and wait_time < 60:
                time.sleep(2)
                wait_time += 2
                live_count = len(self.websocket_client.get_all_tickers())
            
            if live_count < 10:
                logger.warning(f"⚠️ Low live coverage ({live_count}/{len(self.all_symbols)}); proceeding but expect skips")
            
            # Initialize Supervisor and P&L
            self.supervisor = TradeSupervisor(cooldown_minutes=15, max_risk_pct=0.25, max_daily_loss_pct=0.10)
            self.pnl_tracker = get_pnl_tracker()
            logger.info("✅ Supervisor & P&L Tracker initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def identify_volatile_symbols(self, min_change_pct: float = 10.0, min_volume_usd: float = 20000000) -> list:
        """DYNAMIC: Identify volatile symbols from API/WS real data only"""
        logger.info(f"🔍 Identifying volatile symbols (min {min_change_pct}% change, ${min_volume_usd/1e6}M vol)...")
        
        live_tickers = self.websocket_client.get_all_tickers()
        if not live_tickers:
            logger.error("❌ No live ticker data from WS - cannot identify volatiles")
            return []
        
        valid_symbols = set(self.all_symbols)
        volatile_symbols = []
        
        for symbol, ticker in live_tickers.items():
            if symbol not in valid_symbols:
                continue
            
            try:
                change_24h = abs(float(ticker.get('change_24h', 0)))
                volume_24h = float(ticker.get('quote_volume', 0) or ticker.get('volume', 0) * float(ticker.get('price', 1)))
                current_price = float(ticker.get('price', 0))
                
                if current_price <= 0:
                    continue
                
                if change_24h >= min_change_pct and volume_24h >= min_volume_usd:
                    volatile_symbols.append({
                        'symbol': symbol,
                        'price_change_24h': change_24h,
                        'volume_24h': volume_24h,
                        'current_price': current_price,
                        'tier': self._get_volatility_tier(change_24h)
                    })
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping {symbol} due to invalid ticker data: {e}")
                continue
        
        volatile_symbols.sort(key=lambda x: x['price_change_24h'], reverse=True)
        logger.info(f"📊 Identified {len(volatile_symbols)} volatile symbols from {len(live_tickers)} live tickers")
        
        return volatile_symbols[:50]
    
    def _get_volatility_tier(self, change_pct: float) -> str:
        """Assign tier based on change"""
        if change_pct >= 50:
            return "EXTREME"
        elif change_pct >= 25:
            return "HIGH"
        elif change_pct >= 18:
            return "GOOD"
        else:
            return "MODERATE"
    
    def get_extreme_volatility_opportunities(self, min_change: float = 10.0) -> list:
        """DYNAMIC: Get opportunities from volatile symbols"""
        volatiles = self.identify_volatile_symbols(min_change_pct=min_change)
        if not volatiles:
            logger.warning("⚠️ No volatile symbols identified - check API/WS connectivity")
            return []
        
        random.shuffle(volatiles)
        
        symbol_groups = {}
        for opp in volatiles:
            first_letter = opp['symbol'][0].upper()
            if first_letter not in symbol_groups:
                symbol_groups[first_letter] = []
            symbol_groups[first_letter].append(opp)
        
        diversified = []
        for group in symbol_groups.values():
            diversified.extend(group[:3])
        
        vetted = []
        balance = get_real_time_balance(self.api)
       
        for opp in diversified:
            if not self.supervisor.veto_trade(opp['symbol'], balance.get('available_usdt', 0)):
                # ADD CONFIDENCE TO THE OPPORTUNITY OBJECT
                opp['confidence'] = 0.6  # Temporary default, will be overwritten by strategy
                vetted.append(opp)

        
        logger.info(f"📊 Vetted opportunities: {len(vetted)} from {len(diversified)} (after supervisor filter)")
        return vetted[:10]
    
    def execute_trading_cycle(self):
        """Execute cycle with dynamic volatiles + strict filters"""
        self.cycle_count += 1
        print(f"\n🔄 CYCLE #{self.cycle_count} - Scanning dynamic volatiles...")

        opportunities = self.get_extreme_volatility_opportunities()
        if not opportunities:
            print("❌ No dynamic volatile opportunities found")
            return False

        qualified = filter_opportunities_by_min_rr(
            opportunities,
            self.websocket_client,
            api=self.api,
            min_rr=3.0
        )

        print(f"📊 Qualified after R:R filter: {len(qualified)}")

        # CRITICAL DEBUG - ADD THIS NOW
        print(f"🔍 DEBUG: qualified type: {type(qualified)}")
        if qualified:
            print(f"✅ PROCEEDING WITH {len(qualified)} QUALIFIED TRADES")
            for i, opp in enumerate(qualified, 1):
                symbol = opp.get('symbol', 'NO_SYMBOL')  
                has_analysis = 'enhanced_analysis' in opp
                print(f"🎯 Trade {i}: {symbol} - Enhanced Analysis: {has_analysis}")
                if has_analysis:
                    analysis = opp['enhanced_analysis']
                    score = analysis.get('composite_score', 0)
                    rec = analysis.get('recommendation', 'UNKNOWN')
                    print(f"   📊 Score: {score:.3f}, Recommendation: {rec}")
        else:
            print("❌ NO QUALIFIED TRADES")
            return False

        # Make sure execution continues
        print("🚀 STARTING TRADE EXECUTION...")


        # ← MAKE SURE THIS EXECUTION LOOP RUNS
        successful_executions = 0
        trades_executed = 0

              
        for i, opportunity in enumerate(qualified[:3], 1):
            try:
                symbol = opportunity['symbol']
                side = 'LONG' if opportunity['price_change_24h'] > 0 else 'SHORT'
                entry_price = opportunity['current_price']

                # Get real market data
                market_data = enhanced_market_data_collection(self.websocket_client, self.api, symbol)

                # FIX: avoid numpy truth-value ambiguity with explicit checks - RELAXED: Accept 20+ points
                prices_seq = market_data.get('prices')
                volumes_seq = market_data.get('volumes')
                prices_len = len(prices_seq) if prices_seq is not None else 0
                volumes_len = len(volumes_seq) if volumes_seq is not None else 0
                if prices_seq is None or prices_len < 20 or prices_len != volumes_len:
                    logger.warning(f"⚠️ Skipping {symbol}: Insufficient or mismatched real data")
                    print(f"⚠️ TRADE #{i} SKIPPED: Insufficient/mismatched real data")
                    continue

                logger.debug(f"📊 {symbol} data: {len(market_data.get('prices', []))} prices, {len(market_data.get('volumes', []))} volumes, sources: {market_data.get('data_sources', [])}")

                # Strategy with dynamic TP/SL
                from fractal_order_flow_strategy import calculate_fofm_strategy_with_dynamic_tpsl
                tp_price, sl_price, tp_dist, sl_dist, strategy_result = calculate_fofm_strategy_with_dynamic_tpsl(
                    symbol, side, entry_price, market_data, self.websocket_client
                )

                if not tp_price or strategy_result.get('risk_reward_ratio', 0) < 3.0:
                    logger.warning(f"⚠️ Skipping {symbol}: Invalid R:R {strategy_result.get('risk_reward_ratio', 0):.2f}, reason: {strategy_result.get('reason', 'Unknown')}")
                    print(f"⚠️ TRADE #{i} SKIPPED: Invalid R:R {strategy_result.get('risk_reward_ratio', 0):.2f}")
                    continue

                decision = {
                    'action': side,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'tp_distance_pct': tp_dist,
                    'sl_distance_pct': sl_dist,
                    'strategy_result': strategy_result,
                    'market_data': market_data,
                    'tier': opportunity['tier'],
                    'confidence': strategy_result.get('confidence', 0.0)
                }

                # Execute
                result = execute_trade_decision(
                    self.api, decision, self.websocket_client, self.supervisor, self.pnl_tracker
                )

                trades_executed += 1
                if result.get('status') == 'success':
                    successful_executions += 1
                    self.successful_trades += 1
                    print(f"✅ TRADE #{i} SUCCESS: {symbol} {side} (R:R {result.get('risk_reward_ratio', 0):.2f}:1)")

                    # Log to P&L
                    self.pnl_tracker.log_trade_entry({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'quantity': result.get('quantity', 0),
                        'leverage': result.get('leverage', 1),
                        'position_value_usdt': result.get('position_value', 0),
                        'risk_reward_ratio': result.get('risk_reward_ratio', 0),
                        'main_order_id': result.get('main_order_id', ''),
                        'tp_order_id': result.get('take_profit_order_id', ''),
                        'sl_order_id': result.get('stop_loss_order_id', '')
                    })
                else:
                    print(f"❌ TRADE #{i} FAILED: {result.get('reason', 'Unknown error')}")
            except Exception as e:
                logger.error(f"❌ Trade error for {opportunity['symbol']}: {e}")
                print(f"❌ TRADE #{i} FAILED: {str(e)}")

        self.total_trades += trades_executed
        if successful_executions > 0:
            print(f"\n✅ CYCLE COMPLETE - {successful_executions} TRADE(S) EXECUTED")
        else:
            print(f"\n❌ CYCLE COMPLETE - NO QUALIFIED TRADES")

        success_rate = (successful_executions / trades_executed * 100) if trades_executed > 0 else 0
        print(f"📊 Cycle: {successful_executions}/{trades_executed} ({success_rate:.1f}%)")

        # Supervisor stats
        stats = self.supervisor.get_performance_stats()
        print(f"📈 Overall Win Rate: {stats.get('win_rate', 0):.1f}%")

        return successful_executions > 0

    
    def run(self):
        """Main loop - fully dynamic"""
        if not self.initialize():
            return
        
        print("\n🎯 DYNAMIC VOLATILITY TRADING SYSTEM v8.5 - API-DRIVEN ONLY")
        print("=" * 100)
        print("🔥 Dynamic: All symbols/volatiles from API/WS - NO PRE-LISTS/FALLBACKS")
        print("📊 Volatility Filter: Real 24h Change + Volume from Live Data")
        print("🛡️ Strict: 3:1 R:R + Supervisor Veto + Real Data Only")
        print("=" * 100)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    cycle_success = self.execute_trading_cycle()
                    
                    wait_time = 5 if cycle_success else 10
                    print(f"⏳ Waiting {wait_time}s before next cycle...")
                    
                    for remaining in range(wait_time, 0, -1):
                        if not self.running:
                            break
                        time.sleep(1)
                    
                    if self.cycle_count % 20 == 0:
                        self.supervisor._cooldowns.clear()
                        logger.info("🧹 Periodic cooldown reset")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Manual stop")
                    break
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            print("\n🛑 System stopped")
        finally:
            self.running = False
            if self.websocket_client:
                self.websocket_client.stop()
            
            try:
                summary = self.pnl_tracker.get_performance_summary()
                print(f"📊 Final: {summary.get('total_trades', 0)} trades, Win {summary.get('win_rate', 0):.1f}%, P&L ${summary.get('total_pnl', 0):.3f}")
            except Exception as e:
                logger.error(f"Failed to get performance summary: {e}")
                print(f"📊 Final: {self.total_trades} trades, Win {0.0}%, P&L ${0.000}")
            print("✅ Shutdown complete")

def main():
    system = ExtremeVolatilityTradingSystem()
    system.run()

if __name__ == "__main__":
    main()