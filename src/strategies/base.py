"""
Base strategy classes and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from src.data.models import MarketData, TradingSignal, Position, Order, OrderSide, OrderType


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.is_active = False
        
    @abstractmethod
    def generate_signal(self, market_data: MarketData, features: Dict = None) -> Optional[TradingSignal]:
        """Generate trading signal based on market data."""
        pass
    
    @abstractmethod
    def should_enter_position(self, signal: TradingSignal, current_positions: Dict[str, Position]) -> bool:
        """Determine if we should enter a new position."""
        pass
    
    @abstractmethod
    def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """Determine if we should exit an existing position."""
        pass
    
    def update_positions(self, market_data: MarketData):
        """Update existing positions with current market data."""
        symbol = market_data.symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = market_data.close
            position.unrealized_pnl = self.calculate_unrealized_pnl(position)
    
    def calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L for a position."""
        if position.side == OrderSide.BUY:
            return (position.current_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - position.current_price) * position.quantity
    
    def add_signal(self, signal: TradingSignal):
        """Add signal to history."""
        self.signals_history.append(signal)
        
        # Keep only last 1000 signals
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]
    
    def get_recent_signals(self, symbol: str, count: int = 10) -> List[TradingSignal]:
        """Get recent signals for a symbol."""
        symbol_signals = [s for s in self.signals_history if s.symbol == symbol]
        return symbol_signals[-count:] if symbol_signals else []
    
    def activate(self):
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Strategy {self.name} activated")
    
    def deactivate(self):
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Strategy {self.name} deactivated")


class RiskManager:
    """Risk management for trading strategies."""
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.15, 
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.06):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        
    def validate_order(self, signal: TradingSignal, portfolio_value: float, 
                      current_positions: Dict[str, Position]) -> Tuple[bool, str, float]:
        """
        Validate if an order should be placed based on risk management rules.
        
        Returns:
            tuple: (is_valid, reason, suggested_quantity)
        """
        
        # Check if we have too many positions
        if len(current_positions) >= 3:  # Max 3 concurrent positions
            return False, "Maximum concurrent positions reached", 0.0
        
        # Check daily trade limit
        if self.daily_trades >= 100:  # Max 100 trades per day
            return False, "Daily trade limit reached", 0.0
        
        # Calculate position size
        suggested_quantity = self.calculate_position_size(
            signal, portfolio_value, current_positions
        )
        
        if suggested_quantity <= 0:
            return False, "Calculated position size is zero or negative", 0.0
        
        # Check if we already have a position in this symbol
        if signal.symbol in current_positions:
            return False, f"Already have position in {signal.symbol}", 0.0
        
        # Check signal strength
        if abs(signal.signal) < 0.3:  # Minimum signal strength
            return False, "Signal strength too weak", 0.0
        
        # Check confidence
        if signal.confidence < 0.6:  # Minimum confidence
            return False, "Signal confidence too low", 0.0
        
        return True, "Order validated", suggested_quantity
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float,
                              current_positions: Dict[str, Position]) -> float:
        """Calculate appropriate position size based on risk management."""
        
        # Base position size as percentage of portfolio
        base_size = portfolio_value * self.max_position_size
        
        # Adjust based on signal strength and confidence
        signal_multiplier = abs(signal.signal) * signal.confidence
        adjusted_size = base_size * signal_multiplier
        
        # Adjust based on volatility (using predicted return as proxy)
        if abs(signal.predicted_return) > 0.05:  # High volatility
            adjusted_size *= 0.5
        
        # Calculate quantity based on current price
        quantity = adjusted_size / signal.predicted_price
        
        return quantity
    
    def check_stop_loss(self, position: Position) -> bool:
        """Check if position should be stopped out."""
        unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.quantity)
        
        if position.side == OrderSide.BUY:
            return unrealized_pnl_pct <= -self.stop_loss_pct
        else:
            return unrealized_pnl_pct <= -self.stop_loss_pct
    
    def check_take_profit(self, position: Position) -> bool:
        """Check if position should take profit."""
        unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.quantity)
        
        if position.side == OrderSide.BUY:
            return unrealized_pnl_pct >= self.take_profit_pct
        else:
            return unrealized_pnl_pct >= self.take_profit_pct
    
    def check_portfolio_drawdown(self, current_value: float) -> bool:
        """Check if portfolio drawdown exceeds limit."""
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            return drawdown >= self.max_drawdown
        
        return False
    
    def update_daily_stats(self, pnl: float):
        """Update daily trading statistics."""
        self.daily_trades += 1
        self.daily_pnl += pnl
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of each day)."""
        self.daily_trades = 0
        self.daily_pnl = 0.0


class PortfolioManager:
    """Manages portfolio and position tracking."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    def add_position(self, symbol: str, side: OrderSide, quantity: float, price: float):
        """Add a new position."""
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        
        self.positions[symbol] = position
        
        # Update cash
        if side == OrderSide.BUY:
            self.cash -= quantity * price
        else:
            self.cash += quantity * price
        
        logger.info(f"Added {side.value} position: {quantity} {symbol} at {price}")
    
    def close_position(self, symbol: str, exit_price: float):
        """Close an existing position."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        if position.side == OrderSide.BUY:
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            self.cash += position.quantity * exit_price
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
            self.cash -= position.quantity * exit_price
        
        position.realized_pnl = realized_pnl
        
        # Record trade
        trade = {
            'symbol': symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'entry_time': position.timestamp,
            'exit_time': datetime.now()
        }
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} with P&L: {realized_pnl:.2f}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Update unrealized P&L
            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions."""
        return {
            'total_positions': len(self.positions),
            'total_value': self.get_portfolio_value(),
            'cash': self.cash,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital
        }
