"""
Prometheus metrics for the crypto trading system.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import time
from typing import Dict, List
from datetime import datetime
import asyncio
from loguru import logger

from src.core.config import config


class TradingMetrics:
    """Prometheus metrics for trading system monitoring."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Trading metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders placed',
            ['exchange', 'symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.order_execution_duration = Histogram(
            'trading_order_execution_duration_seconds',
            'Time taken to execute orders',
            ['exchange', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            'trading_unrealized_pnl_usd',
            'Unrealized P&L in USD',
            registry=self.registry
        )
        
        self.realized_pnl = Gauge(
            'trading_realized_pnl_usd',
            'Realized P&L in USD',
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'trading_positions_count',
            'Number of open positions',
            registry=self.registry
        )
        
        self.signals_generated = Counter(
            'trading_signals_generated_total',
            'Total signals generated',
            ['strategy', 'symbol', 'direction'],
            registry=self.registry
        )
        
        self.signal_confidence = Histogram(
            'trading_signal_confidence',
            'Signal confidence distribution',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        # Data metrics
        self.market_data_received = Counter(
            'data_market_data_received_total',
            'Total market data messages received',
            ['exchange', 'symbol', 'type'],
            registry=self.registry
        )
        
        self.data_processing_duration = Histogram(
            'data_processing_duration_seconds',
            'Time taken to process market data',
            ['exchange', 'symbol'],
            registry=self.registry
        )
        
        # System metrics
        self.strategy_active = Gauge(
            'system_strategy_active',
            'Whether strategy is active (1) or not (0)',
            ['strategy_name'],
            registry=self.registry
        )
        
        self.system_uptime = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.ml_model_predictions = Counter(
            'ml_model_predictions_total',
            'Total ML model predictions',
            ['model_version', 'symbol'],
            registry=self.registry
        )
        
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy',
            'ML model directional accuracy',
            ['model_version', 'symbol'],
            registry=self.registry
        )
        
        # Risk metrics
        self.max_drawdown = Gauge(
            'risk_max_drawdown_pct',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'risk_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        self.var_95 = Gauge(
            'risk_var_95_pct',
            'Value at Risk (95%)',
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'system_errors_total',
            'Total system errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Performance metrics
        self.daily_return = Gauge(
            'performance_daily_return_pct',
            'Daily return percentage',
            registry=self.registry
        )
        
        self.total_return = Gauge(
            'performance_total_return_pct',
            'Total return percentage since inception',
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'performance_win_rate_pct',
            'Win rate percentage',
            registry=self.registry
        )
        
        # Initialize start time
        self.start_time = time.time()
    
    def record_order(self, exchange: str, symbol: str, side: str, status: str):
        """Record order metrics."""
        self.orders_total.labels(exchange=exchange, symbol=symbol, side=side, status=status).inc()
    
    def record_order_execution_time(self, exchange: str, symbol: str, side: str, duration: float):
        """Record order execution time."""
        self.order_execution_duration.labels(exchange=exchange, symbol=symbol, side=side).observe(duration)
    
    def update_portfolio_metrics(self, value: float, unrealized_pnl: float, realized_pnl: float, position_count: int):
        """Update portfolio metrics."""
        self.portfolio_value.set(value)
        self.unrealized_pnl.set(unrealized_pnl)
        self.realized_pnl.set(realized_pnl)
        self.position_count.set(position_count)
    
    def record_signal(self, strategy: str, symbol: str, direction: str, confidence: float):
        """Record trading signal."""
        self.signals_generated.labels(strategy=strategy, symbol=symbol, direction=direction).inc()
        self.signal_confidence.labels(strategy=strategy, symbol=symbol).observe(confidence)
    
    def record_market_data(self, exchange: str, symbol: str, data_type: str):
        """Record market data reception."""
        self.market_data_received.labels(exchange=exchange, symbol=symbol, type=data_type).inc()
    
    def record_data_processing_time(self, exchange: str, symbol: str, duration: float):
        """Record data processing time."""
        self.data_processing_duration.labels(exchange=exchange, symbol=symbol).observe(duration)
    
    def update_strategy_status(self, strategy_name: str, is_active: bool):
        """Update strategy status."""
        self.strategy_active.labels(strategy_name=strategy_name).set(1 if is_active else 0)
    
    def update_uptime(self):
        """Update system uptime."""
        self.system_uptime.set(time.time() - self.start_time)
    
    def record_ml_prediction(self, model_version: str, symbol: str, accuracy: float = None):
        """Record ML model prediction."""
        self.ml_model_predictions.labels(model_version=model_version, symbol=symbol).inc()
        if accuracy is not None:
            self.ml_model_accuracy.labels(model_version=model_version, symbol=symbol).set(accuracy)
    
    def update_risk_metrics(self, max_drawdown: float, sharpe_ratio: float, var_95: float):
        """Update risk metrics."""
        self.max_drawdown.set(max_drawdown)
        self.sharpe_ratio.set(sharpe_ratio)
        self.var_95.set(var_95)
    
    def record_error(self, component: str, error_type: str):
        """Record system error."""
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def update_performance_metrics(self, daily_return: float, total_return: float, win_rate: float):
        """Update performance metrics."""
        self.daily_return.set(daily_return)
        self.total_return.set(total_return)
        self.win_rate.set(win_rate)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


class PerformanceMonitor:
    """Performance monitoring and calculation."""
    
    def __init__(self):
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
    def add_trade(self, trade: Dict):
        """Add a completed trade."""
        self.trade_history.append({
            **trade,
            'timestamp': datetime.now()
        })
    
    def add_portfolio_snapshot(self, snapshot: Dict):
        """Add portfolio snapshot."""
        self.portfolio_history.append({
            **snapshot,
            'timestamp': datetime.now()
        })
        
        # Calculate daily return if we have previous data
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['total_value']
            curr_value = snapshot['total_value']
            daily_return = (curr_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        import numpy as np
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        import numpy as np
        
        values = [p['total_value'] for p in self.portfolio_history]
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        
        return np.max(drawdown)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trade_history)
    
    def calculate_var_95(self) -> float:
        """Calculate Value at Risk (95th percentile)."""
        if len(self.daily_returns) < 10:
            return 0.0
        
        import numpy as np
        return np.percentile(self.daily_returns, 5) * 100  # 5th percentile (95% VaR)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.portfolio_history:
            return {}
        
        initial_value = config.trading.initial_capital
        current_value = self.portfolio_history[-1]['total_value']
        total_return = (current_value - initial_value) / initial_value
        
        daily_return = self.daily_returns[-1] if self.daily_returns else 0.0
        
        return {
            'total_return': total_return,
            'daily_return': daily_return,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'var_95': self.calculate_var_95(),
            'total_trades': len(self.trade_history),
            'portfolio_value': current_value,
            'days_active': len(self.portfolio_history)
        }


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.alert_thresholds = {
            'max_drawdown': 0.10,  # 10%
            'daily_loss': 0.05,    # 5%
            'portfolio_value_drop': 0.15,  # 15%
            'win_rate_drop': 0.30,  # Below 30%
        }
        
        self.alerts_sent: Dict[str, datetime] = {}
        self.alert_cooldown = 3600  # 1 hour cooldown between same alerts
    
    def check_alerts(self, performance_metrics: Dict, portfolio_metrics: Dict):
        """Check for alert conditions."""
        alerts = []
        
        # Check drawdown
        if performance_metrics.get('max_drawdown', 0) > self.alert_thresholds['max_drawdown']:
            alert = self._create_alert(
                'high_drawdown',
                f"Maximum drawdown exceeded: {performance_metrics['max_drawdown']:.2%}",
                'critical'
            )
            if alert:
                alerts.append(alert)
        
        # Check daily loss
        if performance_metrics.get('daily_return', 0) < -self.alert_thresholds['daily_loss']:
            alert = self._create_alert(
                'daily_loss',
                f"Daily loss exceeded: {performance_metrics['daily_return']:.2%}",
                'warning'
            )
            if alert:
                alerts.append(alert)
        
        # Check win rate
        if performance_metrics.get('win_rate', 1) < self.alert_thresholds['win_rate_drop']:
            alert = self._create_alert(
                'low_win_rate',
                f"Win rate dropped below threshold: {performance_metrics['win_rate']:.2%}",
                'warning'
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, alert_type: str, message: str, severity: str) -> Dict:
        """Create an alert if not in cooldown."""
        now = datetime.now()
        last_sent = self.alerts_sent.get(alert_type)
        
        if last_sent and (now - last_sent).total_seconds() < self.alert_cooldown:
            return None
        
        self.alerts_sent[alert_type] = now
        
        return {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': now
        }
    
    def send_alert(self, alert: Dict):
        """Send an alert (implement integration with notification services)."""
        logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
        
        # Here you could integrate with:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS services
        # - PagerDuty
        # etc.


class MonitoringService:
    """Main monitoring service."""
    
    def __init__(self):
        self.metrics = TradingMetrics()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.is_running = False
        
    async def start(self):
        """Start monitoring service."""
        self.is_running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_updater())
        asyncio.create_task(self._alert_checker())
        
        logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop monitoring service."""
        self.is_running = False
        logger.info("Monitoring service stopped")
    
    async def _metrics_updater(self):
        """Periodically update metrics."""
        while self.is_running:
            try:
                # Update uptime
                self.metrics.update_uptime()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _alert_checker(self):
        """Periodically check for alerts."""
        while self.is_running:
            try:
                # Get performance metrics
                performance_metrics = self.performance_monitor.get_performance_summary()
                
                # Check for alerts
                alerts = self.alert_manager.check_alerts(performance_metrics, {})
                
                # Send alerts
                for alert in alerts:
                    self.alert_manager.send_alert(alert)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(300)
    
    def get_metrics_endpoint(self) -> bytes:
        """Get Prometheus metrics."""
        return self.metrics.get_metrics()
