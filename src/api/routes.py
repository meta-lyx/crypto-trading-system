"""
FastAPI routes for the crypto trading system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from datetime import datetime

from src.core.config import config
from src.core.logging_setup import setup_logging
from src.strategies.engine import StrategyEngine
from src.strategies.ml_strategy import MLTradingStrategy
from src.data.streaming import DataManager
from src.trading.exchange import ExchangeManager, BinanceExchange, OrderManager
from src.monitoring.metrics import MonitoringService
from src.backtesting.engine import BacktestEngine


# Request/Response models
class SystemStatus(BaseModel):
    is_running: bool
    uptime_seconds: float
    active_strategies: List[str]
    portfolio_value: float
    total_return: float
    emergency_stop: bool


class StrategyConfig(BaseModel):
    name: str
    strategy_type: str
    parameters: Dict
    is_active: bool = False


class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: Optional[float] = None


class TradingSystemAPI:
    """Main API for the crypto trading system."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Crypto Trading System",
            description="Professional algorithmic crypto trading system with ML strategies",
            version="1.0.0"
        )
        
        # System components
        self.strategy_engine: Optional[StrategyEngine] = None
        self.data_manager: Optional[DataManager] = None
        self.exchange_manager: Optional[ExchangeManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.monitoring_service: Optional[MonitoringService] = None
        self.backtest_engine = BacktestEngine()
        
        self.is_initialized = False
        self.is_running = False
        self.start_time = datetime.now()
        
        # Setup logging
        setup_logging()
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main dashboard."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Crypto Trading System</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 40px; }
                    .card { background: #2d2d2d; padding: 20px; border-radius: 8px; margin: 20px 0; }
                    .status { display: flex; gap: 20px; }
                    .metric { flex: 1; text-align: center; }
                    .metric h3 { margin: 0; color: #4CAF50; }
                    .metric p { margin: 5px 0; font-size: 24px; font-weight: bold; }
                    button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
                    button:hover { background: #45a049; }
                    button.danger { background: #f44336; }
                    button.danger:hover { background: #da190b; }
                    .log { background: #000; padding: 15px; border-radius: 4px; height: 200px; overflow-y: scroll; font-family: monospace; }
                </style>
                <script>
                    async function fetchStatus() {
                        try {
                            const response = await fetch('/status');
                            const data = await response.json();
                            document.getElementById('status').innerText = data.is_running ? 'RUNNING' : 'STOPPED';
                            document.getElementById('uptime').innerText = Math.round(data.uptime_seconds / 3600 * 100) / 100 + ' hours';
                            document.getElementById('portfolio').innerText = '$' + data.portfolio_value.toLocaleString();
                            document.getElementById('return').innerText = (data.total_return * 100).toFixed(2) + '%';
                            document.getElementById('strategies').innerText = data.active_strategies.length;
                        } catch (error) {
                            console.error('Error fetching status:', error);
                        }
                    }
                    
                    async function startSystem() {
                        try {
                            await fetch('/start', { method: 'POST' });
                            alert('System started successfully');
                            fetchStatus();
                        } catch (error) {
                            alert('Error starting system');
                        }
                    }
                    
                    async function stopSystem() {
                        try {
                            await fetch('/stop', { method: 'POST' });
                            alert('System stopped successfully');
                            fetchStatus();
                        } catch (error) {
                            alert('Error stopping system');
                        }
                    }
                    
                    // Update status every 5 seconds
                    setInterval(fetchStatus, 5000);
                    fetchStatus();
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 Crypto Trading System</h1>
                        <p>Professional Algorithmic Trading with Deep Learning</p>
                    </div>
                    
                    <div class="card">
                        <h2>System Status</h2>
                        <div class="status">
                            <div class="metric">
                                <h3>Status</h3>
                                <p id="status">-</p>
                            </div>
                            <div class="metric">
                                <h3>Uptime</h3>
                                <p id="uptime">-</p>
                            </div>
                            <div class="metric">
                                <h3>Portfolio Value</h3>
                                <p id="portfolio">-</p>
                            </div>
                            <div class="metric">
                                <h3>Total Return</h3>
                                <p id="return">-</p>
                            </div>
                            <div class="metric">
                                <h3>Active Strategies</h3>
                                <p id="strategies">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Controls</h2>
                        <button onclick="startSystem()">🎯 Start Trading</button>
                        <button onclick="stopSystem()" class="danger">⏹️ Stop Trading</button>
                        <button onclick="window.open('/docs', '_blank')">📖 API Docs</button>
                        <button onclick="window.open('http://localhost:3000', '_blank')">📊 Grafana Dashboard</button>
                    </div>
                    
                    <div class="card">
                        <h2>Quick Links</h2>
                        <button onclick="window.open('/strategies', '_blank')">⚡ Strategies</button>
                        <button onclick="window.open('/portfolio', '_blank')">💼 Portfolio</button>
                        <button onclick="window.open('/trades', '_blank')">📈 Trades</button>
                        <button onclick="window.open('/metrics', '_blank')">📊 Metrics</button>
                    </div>
                </div>
            </body>
            </html>
            """
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_status():
            """Get system status."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            if self.strategy_engine:
                engine_status = self.strategy_engine.get_engine_status()
                return SystemStatus(
                    is_running=self.is_running,
                    uptime_seconds=uptime,
                    active_strategies=engine_status.get('active_strategies', []),
                    portfolio_value=engine_status.get('portfolio_summary', {}).get('total_value', 0),
                    total_return=engine_status.get('portfolio_summary', {}).get('total_return', 0),
                    emergency_stop=engine_status.get('emergency_stop', False)
                )
            else:
                return SystemStatus(
                    is_running=False,
                    uptime_seconds=uptime,
                    active_strategies=[],
                    portfolio_value=config.trading.initial_capital,
                    total_return=0.0,
                    emergency_stop=False
                )
        
        @self.app.post("/start")
        async def start_system(background_tasks: BackgroundTasks):
            """Start the trading system."""
            if self.is_running:
                raise HTTPException(status_code=400, detail="System is already running")
            
            try:
                background_tasks.add_task(self._start_system_components)
                return {"message": "System startup initiated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")
        
        @self.app.post("/stop")
        async def stop_system():
            """Stop the trading system."""
            if not self.is_running:
                raise HTTPException(status_code=400, detail="System is not running")
            
            try:
                await self._stop_system_components()
                return {"message": "System stopped successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")
        
        @self.app.get("/strategies")
        async def get_strategies():
            """Get all strategies."""
            if not self.strategy_engine:
                return {"strategies": []}
            
            return {"strategies": self.strategy_engine.get_strategy_metrics()}
        
        @self.app.post("/strategies/{strategy_name}/activate")
        async def activate_strategy(strategy_name: str):
            """Activate a strategy."""
            if not self.strategy_engine:
                raise HTTPException(status_code=400, detail="System not initialized")
            
            self.strategy_engine.activate_strategy(strategy_name)
            return {"message": f"Strategy {strategy_name} activated"}
        
        @self.app.post("/strategies/{strategy_name}/deactivate")
        async def deactivate_strategy(strategy_name: str):
            """Deactivate a strategy."""
            if not self.strategy_engine:
                raise HTTPException(status_code=400, detail="System not initialized")
            
            self.strategy_engine.deactivate_strategy(strategy_name)
            return {"message": f"Strategy {strategy_name} deactivated"}
        
        @self.app.get("/portfolio")
        async def get_portfolio():
            """Get portfolio information."""
            if not self.strategy_engine:
                return {"portfolio": {}}
            
            engine_status = self.strategy_engine.get_engine_status()
            return {"portfolio": engine_status.get('portfolio_summary', {})}
        
        @self.app.get("/trades")
        async def get_trades():
            """Get trade history."""
            if not self.order_manager:
                return {"trades": []}
            
            return {
                "active_orders": [order.dict() for order in self.order_manager.get_active_orders()],
                "order_history": [order.dict() for order in self.order_manager.get_order_history()],
                "statistics": self.order_manager.get_order_statistics()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get Prometheus metrics."""
            if not self.monitoring_service:
                return Response("# No metrics available\n", media_type="text/plain")
            
            metrics_data = self.monitoring_service.get_metrics_endpoint()
            return Response(metrics_data, media_type="text/plain")
        
        @self.app.post("/backtest")
        async def run_backtest(request: BacktestRequest):
            """Run a backtest."""
            try:
                # This is a simplified version - in practice, you'd load historical data
                import pandas as pd
                import numpy as np
                
                # Generate sample data for demo
                start = datetime.fromisoformat(request.start_date)
                end = datetime.fromisoformat(request.end_date)
                
                dates = pd.date_range(start, end, freq='H')
                data = pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.randn(len(dates)) * 1000,
                    'high': 50000 + np.random.randn(len(dates)) * 1000 + 500,
                    'low': 50000 + np.random.randn(len(dates)) * 1000 - 500,
                    'close': 50000 + np.random.randn(len(dates)) * 1000,
                    'volume': np.random.rand(len(dates)) * 1000000
                })
                
                # Create strategy
                strategy = MLTradingStrategy(request.strategy_name)
                
                # Run backtest
                if request.initial_capital:
                    backtest_engine = BacktestEngine(request.initial_capital)
                else:
                    backtest_engine = self.backtest_engine
                
                result = backtest_engine.run_backtest(strategy, data)
                
                return {"result": result.to_dict()}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
        
        @self.app.post("/emergency-stop")
        async def emergency_stop():
            """Emergency stop the system."""
            if self.strategy_engine:
                self.strategy_engine.emergency_shutdown()
            
            await self._stop_system_components()
            return {"message": "Emergency stop executed"}
    
    async def _start_system_components(self):
        """Start all system components."""
        try:
            # Initialize components
            self.strategy_engine = StrategyEngine()
            self.data_manager = DataManager()
            self.exchange_manager = ExchangeManager()
            self.monitoring_service = MonitoringService()
            
            # Add exchanges
            binance_exchange = BinanceExchange()
            await self.exchange_manager.add_exchange(binance_exchange)
            
            # Initialize order manager
            self.order_manager = OrderManager(self.exchange_manager)
            
            # Add order callback to strategy engine
            self.strategy_engine.add_order_callback(self.order_manager.submit_order)
            
            # Create and add ML strategy
            ml_strategy = MLTradingStrategy()
            self.strategy_engine.add_strategy(ml_strategy)
            
            # Start services
            await self.strategy_engine.start()
            await self.data_manager.start()
            await self.monitoring_service.start()
            
            self.is_running = True
            self.is_initialized = True
            
        except Exception as e:
            raise Exception(f"Failed to start system components: {str(e)}")
    
    async def _stop_system_components(self):
        """Stop all system components."""
        try:
            if self.strategy_engine:
                await self.strategy_engine.stop()
            
            if self.data_manager:
                await self.data_manager.stop()
            
            if self.exchange_manager:
                await self.exchange_manager.close_all_connections()
            
            if self.monitoring_service:
                await self.monitoring_service.stop()
            
            self.is_running = False
            
        except Exception as e:
            raise Exception(f"Failed to stop system components: {str(e)}")


# Create the FastAPI app instance
api = TradingSystemAPI()
app = api.app
