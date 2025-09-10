# supervisor.py - Enhanced AI Supervisor: Higher frequency + Volatility awareness

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import json

JOURNAL_DIR = "journal"
JOURNAL_XLSX = os.path.join(JOURNAL_DIR, "pnl_log.xlsx")

class TradeSupervisor:
    def __init__(self,
                cooldown_minutes: int = 20,  # Reduced from 45 for higher frequency
                max_risk_pct: float = 0.30,  # Increased for more opportunities
                max_daily_loss_pct: float = 0.15):  # Slightly increased
        
        self.logger = logging.getLogger(__name__)
        self.cooldown_minutes = cooldown_minutes
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        
        # Initialize all attributes before calling methods
        self._cooldowns: Dict[str, datetime] = {}
        self._open_trades: Dict[str, Dict[str, Any]] = {}
        self._volatile_symbols: Dict[str, float] = {}  # Track volatile symbols
        
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        self._journal_df = self._load_journal()
    
    def _load_journal(self) -> pd.DataFrame:
        if os.path.exists(JOURNAL_XLSX):
            try:
                df = pd.read_excel(JOURNAL_XLSX, engine="openpyxl")
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper()
                
                if "exit_time" in df.columns and "symbol" in df.columns:
                    for sym, grp in df.groupby("symbol"):
                        last = pd.to_datetime(grp["exit_time"]).max()
                        if pd.notna(last):
                            self._cooldowns[sym] = last
                
                # Track volatile symbols from historical data
                if "volatility_score" in df.columns:
                    for _, row in df.iterrows():
                        if pd.notna(row.get("volatility_score", 0)):
                            self._volatile_symbols[row["symbol"]] = float(row["volatility_score"])
                
                return df
                
            except Exception as e:
                self.logger.warning(f"Failed reading journal: {e}")
        
        # Create empty DataFrame with proper columns
        cols = ["trade_id", "symbol", "side", "quantity", "entry_price", "exit_price", "fees_usdt",
                "realized_pnl_usdt", "rr", "reason", "entry_time", "exit_time", "volatility_score",
                "execution_tier", "hold_time_seconds"]
        return pd.DataFrame(columns=cols)
    
    def _persist_journal(self):
        try:
            self._journal_df.to_excel(JOURNAL_XLSX, index=False, engine="openpyxl")
        except Exception as e:
            self.logger.error(f"Failed writing journal: {e}")
    
    def get_cooldown_symbols(self) -> List[str]:
        now = datetime.now()
        blocked = []
        
        for sym, t in self._cooldowns.items():
            # Dynamic cooldown based on volatility
            cooldown_time = self.cooldown_minutes
            if sym in self._volatile_symbols and self._volatile_symbols[sym] > 25:
                cooldown_time = max(10, self.cooldown_minutes // 2)  # Shorter cooldown for volatile
            
            if (now - t) < timedelta(minutes=cooldown_time):
                blocked.append(sym)
        
        return blocked
    
    def is_symbol_on_cooldown(self, symbol: str) -> bool:
        sym = str(symbol).upper()
        now = datetime.now()
        last = self._cooldowns.get(sym)
        
        if not last:
            return False
        
        # Dynamic cooldown based on volatility
        cooldown_time = self.cooldown_minutes
        if sym in self._volatile_symbols and self._volatile_symbols[sym] > 25:
            cooldown_time = max(10, self.cooldown_minutes // 2)  # Shorter for volatile
        
        return (now - last) < timedelta(minutes=cooldown_time)
    
    def register_open_trade(self, trade_id: str, symbol: str, side: str, quantity: float,
                           entry_price: float, entry_time: Optional[datetime] = None,
                           volatility_score: float = 0, execution_tier: str = "STANDARD"):
        
        self._open_trades[str(symbol).upper()] = {
            "trade_id": trade_id,
            "symbol": str(symbol).upper(),
            "side": side.upper(),
            "quantity": float(quantity),
            "entry_price": float(entry_price),
            "entry_time": entry_time or datetime.now(),
            "volatility_score": float(volatility_score),
            "execution_tier": execution_tier
        }
        
        # Update volatile symbols tracking
        if volatility_score > 15:
            self._volatile_symbols[str(symbol).upper()] = volatility_score
    
    def on_position_closed(self, symbol: str, exit_price: float, reason: str,
                          fees_usdt: float = 0.0, hold_time_seconds: int = 0):
        
        sym = str(symbol).upper()
        info = self._open_trades.pop(sym, None)
        
        if not info:
            self.logger.warning(f"No open trade registered for {sym}, journaling minimal row")
            info = {"trade_id": "", "symbol": sym, "side": "NA", "quantity": 0.0,
                   "entry_price": 0.0, "entry_time": datetime.now(), "volatility_score": 0,
                   "execution_tier": "UNKNOWN"}
        
        qty = float(info.get("quantity", 0.0))
        side = info.get("side", "NA")
        entry = float(info.get("entry_price", 0.0))
        
        pnl = 0.0
        if qty > 0 and entry > 0:
            if side == "LONG":
                pnl = (exit_price - entry) * qty
            elif side == "SHORT":
                pnl = (entry - exit_price) * qty
        
        realized = pnl - abs(fees_usdt)
        
        row = {
            "trade_id": info.get("trade_id", ""),
            "symbol": sym,
            "side": side,
            "quantity": qty,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "fees_usdt": float(fees_usdt),
            "realized_pnl_usdt": float(realized),
            "rr": None,
            "reason": str(reason),
            "entry_time": info.get("entry_time"),
            "exit_time": datetime.now(),
            "volatility_score": info.get("volatility_score", 0),
            "execution_tier": info.get("execution_tier", "UNKNOWN"),
            "hold_time_seconds": hold_time_seconds
        }
        
        # Fix DataFrame concatenation
        if self._journal_df.empty:
            self._journal_df = pd.DataFrame([row])
        else:
            self._journal_df = pd.concat([self._journal_df, pd.DataFrame([row])], ignore_index=True)
        
        self._persist_journal()
        self._cooldowns[sym] = datetime.now()
    
    def veto_trade(self, symbol, daily_equity_usdt=None):
        """Enhanced veto logic with volatility consideration"""
        sym = str(symbol).upper()
        
        if self.is_symbol_on_cooldown(symbol):
            return True
        
        # More lenient for volatile assets and smaller accounts
        if daily_equity_usdt and daily_equity_usdt < 100:  # Very small accounts
            return False
        
        # Relaxed daily loss limits for accounts under $500
        if daily_equity_usdt and daily_equity_usdt < 500:
            return False
        
        if daily_equity_usdt and self._journal_df.shape[0] > 0:
            df_today = self._journal_df[
                pd.to_datetime(self._journal_df["exit_time"]).dt.date == datetime.now().date()
            ]
            
            if "realized_pnl_usdt" in df_today.columns and df_today["realized_pnl_usdt"].notna().any():
                day_pnl = df_today["realized_pnl_usdt"].fillna(0.0).sum()
                
                # More lenient for volatile symbols (they can recover quickly)
                loss_threshold = self.max_daily_loss_pct
                if sym in self._volatile_symbols and self._volatile_symbols[sym] > 20:
                    loss_threshold *= 1.5  # 50% more lenient for volatile
                
                if daily_equity_usdt > 0 and (abs(day_pnl) / daily_equity_usdt) > loss_threshold:
                    self.logger.warning(f"Daily loss limit reached for {sym}, vetoing new trades")
                    return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get enhanced performance statistics"""
        if self._journal_df.empty:
            return {"total_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0}
        
        total_trades = len(self._journal_df)
        wins = len(self._journal_df[self._journal_df["realized_pnl_usdt"] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_pnl = self._journal_df["realized_pnl_usdt"].mean()
        
        # Volatile vs non-volatile performance
        volatile_trades = self._journal_df[self._journal_df["volatility_score"] > 20]
        volatile_win_rate = 0.0
        if len(volatile_trades) > 0:
            volatile_wins = len(volatile_trades[volatile_trades["realized_pnl_usdt"] > 0])
            volatile_win_rate = volatile_wins / len(volatile_trades)
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "volatile_trades": len(volatile_trades),
            "volatile_win_rate": volatile_win_rate
        }
