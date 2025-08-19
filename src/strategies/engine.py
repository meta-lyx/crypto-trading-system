"""
Strategy execution engine.
"""

import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from loguru import logger
import redis.asyncio as redis
import json

from src.strategies.base import BaseStrategy, RiskManager, PortfolioManager
from src.strategies.ml_strategy import MLTradingStrategy, EnsembleMLStrategy
from src.data.models import MarketData, TradingSignal, Position, Order, OrderSide, OrderType, OrderStatus
from src.core.config import config


class StrategyEngine:
    """Main strategy execution engine."""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.portfolio_manager = PortfolioManager(config.trading.initial_capital)
        self.risk_manager = RiskManager()
        
        self.redis_client: Optional[redis.Redis] = None
        self.is_running = False
        self.order_callbacks: List[Callable] = []
        
        # Performance tracking
        self.execution_stats = {
            'signals_processed': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'total_pnl': 0.0,
            'start_time': None
        }
        
        # Safety limits
        self.daily_loss_limit = config.trading.initial_capital * 0.05  # 5% daily loss limit
        self.emergency_stop = False
        
    async def start(self):
        """Start the strategy engine."""
        self.redis_client = redis.from_url(config.database.redis_url)
        self.is_running = True
        self.execution_stats['start_time'] = datetime.now()
        
        # Subscribe to market data
        await self._subscribe_to_market_data()
        
        # Start monitoring tasks
        asyncio.create_task(self._portfolio_monitor())
        asyncio.create_task(self._risk_monitor())
        asyncio.create_task(self._performance_monitor())
        
        logger.info("Strategy engine started")
    
    async def stop(self):
        """Stop the strategy engine."""
        self.is_running = False
        
        # Close all positions if needed
        if config.safety.enable_live_trading:
            await self._close_all_positions()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Strategy engine stopped")
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the engine."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy from the engine."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    def activate_strategy(self, strategy_name: str):
        """Activate a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].activate()
    
    def deactivate_strategy(self, strategy_name: str):
        """Deactivate a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].deactivate()
    
    def add_order_callback(self, callback: Callable):
        """Add callback for order execution."""
        self.order_callbacks.append(callback)
    
    async def _subscribe_to_market_data(self):
        """Subscribe to market data updates."""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("market_data")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    market_data = MarketData(**data)
                    await self._process_market_data(market_data)
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
    
    async def _process_market_data(self, market_data: MarketData):
        """Process incoming market data."""
        if not self.is_running or self.emergency_stop:
            return
        
        # Update portfolio with current prices
        self.portfolio_manager.update_position_price(market_data.symbol, market_data.close)
        
        # Check for exit signals on existing positions
        await self._check_position_exits(market_data)
        
        # Generate and process signals from active strategies
        for strategy in self.strategies.values():
            if strategy.is_active:
                try:
                    signal = strategy.generate_signal(market_data)
                    if signal:
                        await self._process_trading_signal(signal, strategy)
                        self.execution_stats['signals_processed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error in strategy {strategy.name}: {e}")
    
    async def _process_trading_signal(self, signal: TradingSignal, strategy: BaseStrategy):
        """Process a trading signal."""
        
        # Check if strategy wants to enter position
        if not strategy.should_enter_position(signal, self.portfolio_manager.positions):
            return
        
        # Risk management validation
        portfolio_value = self.portfolio_manager.get_portfolio_value()
        is_valid, reason, suggested_quantity = self.risk_manager.validate_order(
            signal, portfolio_value, self.portfolio_manager.positions
        )
        
        if not is_valid:
            logger.debug(f"Order rejected: {reason}")
            return
        
        # Determine order details
        side = OrderSide.BUY if signal.signal > 0 else OrderSide.SELL
        order_type = OrderType.MARKET  # For now, use market orders
        
        # Create order
        order = Order(
            id=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            symbol=signal.symbol,
            side=side,
            type=order_type,
            quantity=suggested_quantity,
            price=None,  # Market order
            timestamp=datetime.now(),
            exchange="binance"
        )
        
        # Execute order
        await self._execute_order(order, signal)
    
    async def _execute_order(self, order: Order, signal: TradingSignal = None):
        """Execute a trading order."""
        
        try:
            # For paper trading, simulate execution
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                await self._simulate_order_execution(order, signal)
            elif config.safety.enable_live_trading:
                # Call exchange API through order callbacks
                for callback in self.order_callbacks:
                    await callback(order)
            
            self.execution_stats['orders_placed'] += 1
            logger.info(f"Order executed: {order.side.value} {order.quantity} {order.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            order.status = OrderStatus.FAILED
    
    async def _simulate_order_execution(self, order: Order, signal: TradingSignal = None):
        """Simulate order execution for paper trading."""
        
        # Simulate immediate fill at current market price
        fill_price = signal.predicted_price if signal else order.price
        
        if fill_price is None:
            logger.error("Cannot simulate order execution without price")
            return
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        
        # Add position to portfolio
        self.portfolio_manager.add_position(
            order.symbol, order.side, order.quantity, fill_price
        )
        
        # Update risk manager
        self.risk_manager.update_daily_stats(0)  # No immediate P&L
        
        self.execution_stats['orders_filled'] += 1
    
    async def _check_position_exits(self, market_data: MarketData):
        """Check if any positions should be exited."""
        
        symbol = market_data.symbol
        if symbol not in self.portfolio_manager.positions:
            return
        
        position = self.portfolio_manager.positions[symbol]
        
        # Check each strategy for exit signals
        should_exit = False
        exit_reason = ""
        
        for strategy in self.strategies.values():
            if strategy.is_active and strategy.should_exit_position(position, market_data):
                should_exit = True
                exit_reason = f"Strategy {strategy.name} exit signal"
                break
        
        # Risk management exits
        if self.risk_manager.check_stop_loss(position):
            should_exit = True
            exit_reason = "Stop loss triggered"
        elif self.risk_manager.check_take_profit(position):
            should_exit = True
            exit_reason = "Take profit triggered"
        
        if should_exit:
            await self._close_position(symbol, market_data.close, exit_reason)
    
    async def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position."""
        
        if symbol not in self.portfolio_manager.positions:
            return
        
        position = self.portfolio_manager.positions[symbol]
        
        # Create exit order
        exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        
        exit_order = Order(
            id=f"exit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            symbol=symbol,
            side=exit_side,
            type=OrderType.MARKET,
            quantity=position.quantity,
            price=None,
            timestamp=datetime.now(),
            exchange="binance"
        )
        
        # Execute exit order
        if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
            # Paper trading: close position directly
            self.portfolio_manager.close_position(symbol, exit_price)
            logger.info(f"Position closed: {symbol} - {reason}")
        elif config.safety.enable_live_trading:
            # Live trading: execute through callbacks
            for callback in self.order_callbacks:
                await callback(exit_order)
    
    async def _close_all_positions(self):
        """Close all open positions."""
        for symbol in list(self.portfolio_manager.positions.keys()):
            position = self.portfolio_manager.positions[symbol]
            # Use last known price for emergency close
            await self._close_position(symbol, position.current_price, "Emergency close")
    
    async def _portfolio_monitor(self):
        """Monitor portfolio health."""
        while self.is_running:
            try:
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                
                # Check for maximum drawdown
                if self.risk_manager.check_portfolio_drawdown(portfolio_value):
                    logger.critical("Maximum drawdown exceeded - triggering emergency stop")
                    self.emergency_stop = True
                    await self._close_all_positions()
                
                # Check daily loss limit
                daily_pnl = self.portfolio_manager.get_realized_pnl() + self.portfolio_manager.get_unrealized_pnl()
                if daily_pnl < -self.daily_loss_limit:
                    logger.critical("Daily loss limit exceeded - triggering emergency stop")
                    self.emergency_stop = True
                    await self._close_all_positions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in portfolio monitor: {e}")
                await asyncio.sleep(60)
    
    async def _risk_monitor(self):
        """Monitor risk metrics."""
        while self.is_running:
            try:
                # Reset daily stats at midnight
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    self.risk_manager.reset_daily_stats()
                
                # Log risk metrics
                portfolio_summary = self.portfolio_manager.get_position_summary()
                logger.info(f"Portfolio summary: {portfolio_summary}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor(self):
        """Monitor performance metrics."""
        while self.is_running:
            try:
                # Calculate performance metrics
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                total_return = (portfolio_value - config.trading.initial_capital) / config.trading.initial_capital
                
                runtime = datetime.now() - self.execution_stats['start_time']
                
                performance_metrics = {
                    'portfolio_value': portfolio_value,
                    'total_return': total_return,
                    'runtime_hours': runtime.total_seconds() / 3600,
                    'signals_per_hour': self.execution_stats['signals_processed'] / max(runtime.total_seconds() / 3600, 1),
                    'orders_per_hour': self.execution_stats['orders_placed'] / max(runtime.total_seconds() / 3600, 1),
                    'fill_rate': self.execution_stats['orders_filled'] / max(self.execution_stats['orders_placed'], 1)
                }
                
                # Publish metrics to Redis
                if self.redis_client:
                    await self.redis_client.publish("performance_metrics", json.dumps(performance_metrics))
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    def get_engine_status(self) -> Dict:
        """Get current engine status."""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'active_strategies': [name for name, strategy in self.strategies.items() if strategy.is_active],
            'portfolio_summary': self.portfolio_manager.get_position_summary(),
            'execution_stats': self.execution_stats,
            'strategy_count': len(self.strategies)
        }
    
    def get_strategy_metrics(self) -> Dict:
        """Get metrics for all strategies."""
        metrics = {}
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'get_strategy_metrics'):
                metrics[name] = strategy.get_strategy_metrics()
            else:
                metrics[name] = {
                    'is_active': strategy.is_active,
                    'signals_count': len(strategy.signals_history)
                }
        return metrics
    
    def emergency_shutdown(self):
        """Emergency shutdown of the engine."""
        logger.critical("Emergency shutdown initiated")
        self.emergency_stop = True
        asyncio.create_task(self.stop())
