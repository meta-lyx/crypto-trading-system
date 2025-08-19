"""
Backtesting engine for strategy validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.strategies.base import BaseStrategy, PortfolioManager
from src.data.models import MarketData, TradingSignal, Position, Order, OrderSide, OrderStatus, OrderType
from src.core.config import config


@dataclass
class BacktestResult:
    """Backtesting result container."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_holding_period: float
    portfolio_history: List[Dict]
    trade_history: List[Dict]
    signals_history: List[TradingSignal]
    daily_returns: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_win': self.max_win,
            'max_loss': self.max_loss,
            'avg_holding_period': self.avg_holding_period
        }


class BacktestEngine:
    """Backtesting engine for strategy validation."""
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.trading.initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """Run backtest for a strategy."""
        
        logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Filter data by date range
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        if len(data) == 0:
            raise ValueError("No data available for backtesting")
        
        # Initialize portfolio and tracking
        portfolio = PortfolioManager(self.initial_capital)
        portfolio_history = []
        trade_history = []
        signals_history = []
        
        # Activate strategy for backtesting
        strategy.activate()
        
        # Process each data point
        for idx, row in data.iterrows():
            try:
                # Create market data object
                market_data = MarketData(
                    symbol=row.get('symbol', 'BTC/USDT'),
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    bid=row.get('bid', row['close']),
                    ask=row.get('ask', row['close']),
                    exchange=row.get('exchange', 'binance')
                )
                
                # Update position prices
                portfolio.update_position_price(market_data.symbol, market_data.close)
                
                # Check for exit signals on existing positions
                self._check_position_exits(strategy, portfolio, market_data, trade_history)
                
                # Generate signal
                signal = strategy.generate_signal(market_data)
                
                if signal:
                    signals_history.append(signal)
                    
                    # Check if strategy wants to enter position
                    if strategy.should_enter_position(signal, portfolio.positions):
                        # Execute trade
                        self._execute_trade(signal, portfolio, market_data, trade_history)
                
                # Record portfolio snapshot
                portfolio_snapshot = {
                    'timestamp': market_data.timestamp,
                    'portfolio_value': portfolio.get_portfolio_value(),
                    'cash': portfolio.cash,
                    'positions_value': portfolio.get_portfolio_value() - portfolio.cash,
                    'unrealized_pnl': portfolio.get_unrealized_pnl(),
                    'realized_pnl': portfolio.get_realized_pnl(),
                    'num_positions': len(portfolio.positions)
                }
                portfolio_history.append(portfolio_snapshot)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        # Close all remaining positions at the end
        for symbol in list(portfolio.positions.keys()):
            final_price = data[data['symbol'] == symbol]['close'].iloc[-1] if 'symbol' in data.columns else data['close'].iloc[-1]
            portfolio.close_position(symbol, final_price)
            
            # Record final trade
            trade_history.append({
                'symbol': symbol,
                'side': 'close',
                'quantity': portfolio.positions.get(symbol, {}).get('quantity', 0),
                'price': final_price,
                'timestamp': data['timestamp'].iloc[-1],
                'type': 'final_close'
            })
        
        # Calculate results
        result = self._calculate_backtest_results(
            strategy, portfolio_history, trade_history, signals_history,
            data['timestamp'].iloc[0], data['timestamp'].iloc[-1]
        )
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _filter_data_by_date(self, data: pd.DataFrame, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter data by date range."""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        if start_date:
            data = data[data['timestamp'] >= start_date]
        
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        return data.reset_index(drop=True)
    
    def _check_position_exits(self, strategy: BaseStrategy, portfolio: PortfolioManager, 
                            market_data: MarketData, trade_history: List[Dict]):
        """Check if any positions should be exited."""
        
        symbol = market_data.symbol
        if symbol not in portfolio.positions:
            return
        
        position = portfolio.positions[symbol]
        
        # Check strategy exit condition
        if strategy.should_exit_position(position, market_data):
            # Close position
            exit_price = self._apply_slippage(market_data.close, position.side, is_exit=True)
            portfolio.close_position(symbol, exit_price)
            
            # Record trade
            trade_history.append({
                'symbol': symbol,
                'side': 'sell' if position.side == OrderSide.BUY else 'buy',
                'quantity': position.quantity,
                'price': exit_price,
                'timestamp': market_data.timestamp,
                'type': 'exit',
                'pnl': position.realized_pnl
            })
    
    def _execute_trade(self, signal: TradingSignal, portfolio: PortfolioManager, 
                      market_data: MarketData, trade_history: List[Dict]):
        """Execute a trade based on signal."""
        
        # Determine trade details
        side = OrderSide.BUY if signal.signal > 0 else OrderSide.SELL
        
        # Calculate position size (simple fixed percentage for now)
        portfolio_value = portfolio.get_portfolio_value()
        position_value = portfolio_value * config.trading.max_position_size
        
        # Apply signal strength and confidence
        signal_multiplier = abs(signal.signal) * signal.confidence
        adjusted_position_value = position_value * signal_multiplier
        
        # Calculate quantity
        entry_price = self._apply_slippage(market_data.close, side)
        quantity = adjusted_position_value / entry_price
        
        # Check if we have enough cash
        required_cash = quantity * entry_price
        if side == OrderSide.BUY and required_cash > portfolio.cash:
            quantity = portfolio.cash * 0.95 / entry_price  # Use 95% of available cash
        
        if quantity <= 0:
            return
        
        # Apply commission
        commission = required_cash * self.commission_rate
        
        # Execute trade
        portfolio.add_position(signal.symbol, side, quantity, entry_price)
        
        # Deduct commission from cash
        portfolio.cash -= commission
        
        # Record trade
        trade_history.append({
            'symbol': signal.symbol,
            'side': side.value,
            'quantity': quantity,
            'price': entry_price,
            'timestamp': market_data.timestamp,
            'type': 'entry',
            'commission': commission,
            'signal_strength': signal.signal,
            'signal_confidence': signal.confidence
        })
    
    def _apply_slippage(self, price: float, side: OrderSide, is_exit: bool = False) -> float:
        """Apply slippage to price."""
        slippage = self.slippage_rate
        
        if side == OrderSide.BUY:
            # Buy orders execute at higher price
            return price * (1 + slippage)
        else:
            # Sell orders execute at lower price
            return price * (1 - slippage)
    
    def _calculate_backtest_results(
        self,
        strategy: BaseStrategy,
        portfolio_history: List[Dict],
        trade_history: List[Dict],
        signals_history: List[TradingSignal],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        
        if not portfolio_history:
            raise ValueError("No portfolio history available")
        
        # Basic metrics
        final_value = portfolio_history[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Time-based metrics
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(portfolio_history)):
            prev_value = portfolio_history[i-1]['portfolio_value']
            curr_value = portfolio_history[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        # Sharpe ratio
        if len(daily_returns) > 1:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            if std_daily_return > 0:
                sharpe_ratio = avg_daily_return / std_daily_return * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Trade statistics
        entry_trades = [t for t in trade_history if t.get('type') == 'entry']
        exit_trades = [t for t in trade_history if t.get('type') == 'exit']
        
        total_trades = len(exit_trades)
        winning_trades = len([t for t in exit_trades if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        winning_pnls = [t['pnl'] for t in exit_trades if t.get('pnl', 0) > 0]
        losing_pnls = [t['pnl'] for t in exit_trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        max_win = np.max(winning_pnls) if winning_pnls else 0
        max_loss = np.min(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average holding period
        holding_periods = []
        entry_by_symbol = {}
        
        for trade in trade_history:
            if trade.get('type') == 'entry':
                entry_by_symbol[trade['symbol']] = trade['timestamp']
            elif trade.get('type') == 'exit' and trade['symbol'] in entry_by_symbol:
                entry_time = entry_by_symbol[trade['symbol']]
                exit_time = trade['timestamp']
                holding_period = (exit_time - entry_time).total_seconds() / 3600  # hours
                holding_periods.append(holding_period)
                del entry_by_symbol[trade['symbol']]
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_holding_period=avg_holding_period,
            portfolio_history=portfolio_history,
            trade_history=trade_history,
            signals_history=signals_history,
            daily_returns=daily_returns
        )
    
    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy results."""
        
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Annualized Return': f"{result.annualized_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Total Trades': result.total_trades,
                'Avg Holding (hrs)': f"{result.avg_holding_period:.1f}",
                'Final Value': f"${result.final_value:,.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        
        report = f"""
=== BACKTEST REPORT ===
Strategy: {result.strategy_name}
Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}

PERFORMANCE METRICS:
- Initial Capital: ${result.initial_capital:,.2f}
- Final Value: ${result.final_value:,.2f}
- Total Return: {result.total_return:.2%}
- Annualized Return: {result.annualized_return:.2%}
- Sharpe Ratio: {result.sharpe_ratio:.2f}
- Maximum Drawdown: {result.max_drawdown:.2%}

TRADE STATISTICS:
- Total Trades: {result.total_trades}
- Winning Trades: {result.winning_trades}
- Losing Trades: {result.losing_trades}
- Win Rate: {result.win_rate:.2%}
- Profit Factor: {result.profit_factor:.2f}

TRADE ANALYSIS:
- Average Win: ${result.avg_win:.2f}
- Average Loss: ${result.avg_loss:.2f}
- Maximum Win: ${result.max_win:.2f}
- Maximum Loss: ${result.max_loss:.2f}
- Average Holding Period: {result.avg_holding_period:.1f} hours

RISK METRICS:
- Volatility (Daily): {np.std(result.daily_returns) * np.sqrt(252):.2%}
- Best Day: {max(result.daily_returns) if result.daily_returns else 0:.2%}
- Worst Day: {min(result.daily_returns) if result.daily_returns else 0:.2%}
"""
        
        return report


class WalkForwardAnalysis:
    """Walk-forward analysis for robust strategy validation."""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
    
    def run_walk_forward(
        self,
        strategy_class,
        data: pd.DataFrame,
        training_window: int = 252,  # days
        testing_window: int = 63,   # days
        step_size: int = 21         # days
    ) -> List[BacktestResult]:
        """Run walk-forward analysis."""
        
        logger.info("Starting walk-forward analysis")
        
        results = []
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        start_idx = 0
        
        while start_idx + training_window + testing_window < len(data):
            # Define windows
            train_end_idx = start_idx + training_window
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + testing_window
            
            # Extract data
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            # Create and train strategy (simplified for demo)
            strategy = strategy_class()
            
            # Run backtest on test period
            try:
                result = self.backtest_engine.run_backtest(
                    strategy,
                    test_data,
                    test_data['timestamp'].iloc[0],
                    test_data['timestamp'].iloc[-1]
                )
                results.append(result)
                
                logger.info(f"Walk-forward period {len(results)}: {result.total_return:.2%}")
                
            except Exception as e:
                logger.error(f"Error in walk-forward period: {e}")
            
            # Move to next period
            start_idx += step_size
        
        logger.info(f"Walk-forward analysis completed. {len(results)} periods analyzed")
        return results
    
    def analyze_stability(self, results: List[BacktestResult]) -> Dict:
        """Analyze strategy stability across walk-forward periods."""
        
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'return_stability': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'positive_periods': sum(1 for r in returns if r > 0) / len(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns),
            'total_periods': len(results)
        }


def create_backtest_charts(result: BacktestResult) -> go.Figure:
    """Create interactive charts for backtest results."""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Drawdown',
            'Daily Returns Distribution',
            'Trade P&L',
            'Cumulative Returns',
            'Position Count'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # Portfolio value
    timestamps = [p['timestamp'] for p in result.portfolio_history]
    values = [p['portfolio_value'] for p in result.portfolio_history]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=values, name='Portfolio Value', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak * 100
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=drawdown, name='Drawdown %', fill='tozeroy', 
                  line=dict(color='red')),
        row=1, col=2
    )
    
    # Daily returns histogram
    if result.daily_returns:
        fig.add_trace(
            go.Histogram(x=result.daily_returns, name='Daily Returns', nbinsx=50),
            row=2, col=1
        )
    
    # Trade P&L
    exit_trades = [t for t in result.trade_history if t.get('type') == 'exit']
    if exit_trades:
        trade_pnls = [t.get('pnl', 0) for t in exit_trades]
        trade_times = [t['timestamp'] for t in exit_trades]
        
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        
        fig.add_trace(
            go.Scatter(x=trade_times, y=trade_pnls, mode='markers', 
                      name='Trade P&L', marker=dict(color=colors)),
            row=2, col=2
        )
    
    # Cumulative returns
    cumulative_returns = [(v / result.initial_capital - 1) * 100 for v in values]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=cumulative_returns, name='Cumulative Return %', 
                  line=dict(color='green')),
        row=3, col=1
    )
    
    # Position count
    position_counts = [p['num_positions'] for p in result.portfolio_history]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=position_counts, name='Position Count', 
                  line=dict(color='orange')),
        row=3, col=2
    )
    
    fig.update_layout(height=800, title=f"Backtest Results - {result.strategy_name}")
    return fig
