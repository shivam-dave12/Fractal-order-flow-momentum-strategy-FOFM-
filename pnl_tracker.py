# pnl_tracker.py - v2.0 COMPLETE P&L TRACKING WITH EXCEL EXPORT

import pandas as pd
import os
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
import time

class EnhancedPnLTracker:
    """ULTIMATE: Complete P&L tracking with Excel export and real-time updates"""
    
    def __init__(self, excel_file: str = "trading_pnl_log.xlsx"):
        self.excel_file = excel_file
        self.trades_df = pd.DataFrame()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Excel file with headers if it doesn't exist
        self._initialize_excel_file()
    
    def _initialize_excel_file(self):
        """Initialize Excel file with proper headers"""
        try:
            if not os.path.exists(self.excel_file):
                initial_df = pd.DataFrame(columns=[
                    'Trade_ID', 'Timestamp', 'Symbol', 'Action', 'Strategy', 'Entry_Price', 'Exit_Price',
                    'Quantity', 'Leverage', 'Position_Value_USDT', 'Realized_PnL_USDT',
                    'Unrealized_PnL_USDT', 'Fees_USDT', 'Hold_Time_Minutes', 'Exit_Reason',
                    'Confluence_Score', 'ICT_SMC_Score', 'Volatility_Score', 'Setup_Quality',
                    'Risk_Reward_Ratio', 'Execution_Tier', 'Order_IDs', 'Status'
                ])
                
                with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                    initial_df.to_excel(writer, sheet_name='Trading_Log', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Total_Trades', 'Winning_Trades', 'Losing_Trades', 'Win_Rate_%',
                                  'Total_PnL_USDT', 'Total_Fees_USDT', 'Net_PnL_USDT', 'Best_Trade_USDT',
                                  'Worst_Trade_USDT', 'Avg_Trade_USDT', 'Total_Volume_USDT'],
                        'Value': [0] * 11
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                self.logger.info(f"✅ Initialized P&L tracking Excel file: {self.excel_file}")
            else:
                # Load existing trades
                try:
                    self.trades_df = pd.read_excel(self.excel_file, sheet_name='Trading_Log')
                    self.logger.info(f"📊 Loaded existing P&L data: {len(self.trades_df)} trades")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load existing data: {e}")
                    self.trades_df = pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Excel file: {e}")
    
    def log_trade_entry(self, trade_data: Dict) -> str:
        """Log new trade entry with all details"""
        try:
            with self.lock:
                trade_id = f"TRADE_{int(time.time())}_{trade_data.get('symbol', 'UNKNOWN')}"
                
                new_trade = {
                    'Trade_ID': trade_id,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Symbol': trade_data.get('symbol', ''),
                    'Action': trade_data.get('side', ''),
                    'Strategy': trade_data.get('strategy', 'ENHANCED_FOFM_ICT_SMC'),
                    'Entry_Price': float(trade_data.get('entry_price', 0)),
                    'Exit_Price': None,  # Will be updated on exit
                    'Quantity': float(trade_data.get('quantity', 0)),
                    'Leverage': int(trade_data.get('leverage', 3)),
                    'Position_Value_USDT': float(trade_data.get('position_value_usdt', 0)),
                    'Realized_PnL_USDT': 0.0,  # Will be updated on exit
                    'Unrealized_PnL_USDT': 0.0,
                    'Fees_USDT': float(trade_data.get('fees_usdt', 0)),
                    'Hold_Time_Minutes': 0,  # Will be updated on exit
                    'Exit_Reason': '',  # Will be updated on exit
                    'Confluence_Score': float(trade_data.get('confluence_score', 0)),
                    'ICT_SMC_Score': float(trade_data.get('ict_smc_score', 0)),
                    'Volatility_Score': float(trade_data.get('volatility_score', 0)),
                    'Setup_Quality': trade_data.get('setup_quality', ''),
                    'Risk_Reward_Ratio': float(trade_data.get('risk_reward_ratio', 0)),
                    'Execution_Tier': trade_data.get('execution_tier', ''),
                    'Order_IDs': f"Main:{trade_data.get('main_order_id', '')}, TP:{trade_data.get('tp_order_id', '')}, SL:{trade_data.get('sl_order_id', '')}",
                    'Status': 'ACTIVE'
                }
                
                # Append to dataframe
                new_row_df = pd.DataFrame([new_trade])
                if self.trades_df.empty:
                    self.trades_df = new_row_df
                else:
                    self.trades_df = pd.concat([self.trades_df, new_row_df], ignore_index=True)
                
                # Export to Excel
                self._export_to_excel()
                
                self.logger.info(f"📝 Logged trade entry: {trade_id} for {trade_data.get('symbol')}")
                return trade_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log trade entry: {e}")
            return ""
    
    def update_trade_exit(self, symbol: str, exit_data: Dict):
        """Update trade with exit information"""
        try:
            with self.lock:
                # Find the most recent active trade for this symbol
                active_trades = self.trades_df[
                    (self.trades_df['Symbol'] == symbol) &
                    (self.trades_df['Status'] == 'ACTIVE')
                ]
                
                if active_trades.empty:
                    self.logger.warning(f"⚠️ No active trade found for {symbol}")
                    return
                
                # Update the most recent trade
                latest_idx = active_trades.index[-1]
                entry_time = pd.to_datetime(self.trades_df.loc[latest_idx, 'Timestamp'])
                exit_time = datetime.now()
                hold_time_minutes = (exit_time - entry_time).total_seconds() / 60
                
                # Calculate realized P&L
                entry_price = float(self.trades_df.loc[latest_idx, 'Entry_Price'])
                exit_price = float(exit_data.get('exit_price', entry_price))
                quantity = float(self.trades_df.loc[latest_idx, 'Quantity'])
                leverage = int(self.trades_df.loc[latest_idx, 'Leverage'])
                action = self.trades_df.loc[latest_idx, 'Action']
                
                # Calculate P&L based on position direction
                if action.upper() == 'LONG':
                    realized_pnl = (exit_price - entry_price) * quantity * leverage
                else:  # SHORT
                    realized_pnl = (entry_price - exit_price) * quantity * leverage
                
                # Update the trade record
                self.trades_df.loc[latest_idx, 'Exit_Price'] = exit_price
                self.trades_df.loc[latest_idx, 'Realized_PnL_USDT'] = realized_pnl
                self.trades_df.loc[latest_idx, 'Hold_Time_Minutes'] = round(hold_time_minutes, 2)
                self.trades_df.loc[latest_idx, 'Exit_Reason'] = exit_data.get('exit_reason', 'Manual')
                self.trades_df.loc[latest_idx, 'Fees_USDT'] = float(exit_data.get('total_fees', 0))
                self.trades_df.loc[latest_idx, 'Status'] = 'CLOSED'
                
                # Export updated data
                self._export_to_excel()
                
                self.logger.info(f"✅ Updated trade exit for {symbol}: P&L ${realized_pnl:.3f}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update trade exit: {e}")
    
    def update_unrealized_pnl(self, symbol: str, unrealized_pnl: float):
        """Update unrealized P&L for active position"""
        try:
            with self.lock:
                # Find active trade for symbol
                active_mask = (self.trades_df['Symbol'] == symbol) & (self.trades_df['Status'] == 'ACTIVE')
                active_trades = self.trades_df[active_mask]
                
                if not active_trades.empty:
                    latest_idx = active_trades.index[-1]
                    self.trades_df.loc[latest_idx, 'Unrealized_PnL_USDT'] = float(unrealized_pnl)
                    
                    # Export every 10 updates to avoid excessive I/O
                    if hasattr(self, '_update_counter'):
                        self._update_counter += 1
                    else:
                        self._update_counter = 1
                    
                    if self._update_counter % 10 == 0:
                        self._export_to_excel()
                        
        except Exception as e:
            self.logger.debug(f"Failed to update unrealized P&L: {e}")
    
    def _export_to_excel(self):
        """Export current data to Excel with summary statistics"""
        try:
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                # Main trading log
                self.trades_df.to_excel(writer, sheet_name='Trading_Log', index=False)
                
                # Calculate summary statistics
                closed_trades = self.trades_df[self.trades_df['Status'] == 'CLOSED']
                
                if not closed_trades.empty:
                    winning_trades = closed_trades[closed_trades['Realized_PnL_USDT'] > 0]
                    losing_trades = closed_trades[closed_trades['Realized_PnL_USDT'] < 0]
                    
                    total_trades = len(closed_trades)
                    win_count = len(winning_trades)
                    loss_count = len(losing_trades)
                    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
                    
                    total_pnl = closed_trades['Realized_PnL_USDT'].sum()
                    total_fees = closed_trades['Fees_USDT'].sum()
                    net_pnl = total_pnl - total_fees
                    
                    best_trade = closed_trades['Realized_PnL_USDT'].max() if not closed_trades.empty else 0
                    worst_trade = closed_trades['Realized_PnL_USDT'].min() if not closed_trades.empty else 0
                    avg_trade = closed_trades['Realized_PnL_USDT'].mean() if not closed_trades.empty else 0
                    total_volume = closed_trades['Position_Value_USDT'].sum()
                    
                    # R:R Analysis
                    avg_rr = closed_trades['Risk_Reward_Ratio'].mean() if not closed_trades.empty else 0
                    
                    summary_data = {
                        'Metric': [
                            'Total_Trades', 'Winning_Trades', 'Losing_Trades', 'Win_Rate_%',
                            'Total_PnL_USDT', 'Total_Fees_USDT', 'Net_PnL_USDT', 'Best_Trade_USDT',
                            'Worst_Trade_USDT', 'Avg_Trade_USDT', 'Total_Volume_USDT', 'Avg_RR_Ratio'
                        ],
                        'Value': [
                            total_trades, win_count, loss_count, round(win_rate, 2),
                            round(total_pnl, 3), round(total_fees, 3), round(net_pnl, 3),
                            round(best_trade, 3), round(worst_trade, 3), round(avg_trade, 3),
                            round(total_volume, 2), round(avg_rr, 2)
                        ]
                    }
                else:
                    summary_data = {
                        'Metric': ['Total_Trades', 'Note'],
                        'Value': [0, 'No closed trades yet']
                    }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to export to Excel: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        try:
            closed_trades = self.trades_df[self.trades_df['Status'] == 'CLOSED']
            active_trades = self.trades_df[self.trades_df['Status'] == 'ACTIVE']
            
            if closed_trades.empty:
                return {
                    'total_trades': 0,
                    'active_trades': len(active_trades),
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_rr': 0.0
                }
            
            winning_trades = closed_trades[closed_trades['Realized_PnL_USDT'] > 0]
            total_pnl = closed_trades['Realized_PnL_USDT'].sum()
            win_rate = len(winning_trades) / len(closed_trades) * 100
            avg_rr = closed_trades['Risk_Reward_Ratio'].mean()
            
            return {
                'total_trades': len(closed_trades),
                'active_trades': len(active_trades),
                'winning_trades': len(winning_trades),
                'total_pnl': round(total_pnl, 3),
                'win_rate': round(win_rate, 2),
                'avg_rr': round(avg_rr, 2),
                'best_trade': round(closed_trades['Realized_PnL_USDT'].max(), 3),
                'worst_trade': round(closed_trades['Realized_PnL_USDT'].min(), 3)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}

# Global tracker instance
pnl_tracker = None

def initialize_pnl_tracker(excel_file: str = "trading_pnl_log.xlsx"):
    """Initialize global P&L tracker"""
    global pnl_tracker
    pnl_tracker = EnhancedPnLTracker(excel_file)
    return pnl_tracker

def get_pnl_tracker():
    """Get global P&L tracker instance"""
    global pnl_tracker
    if pnl_tracker is None:
        pnl_tracker = EnhancedPnLTracker()
    return pnl_tracker
