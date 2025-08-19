"""
Configuration management for the crypto trading system.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://trader:trading_password_2024@localhost:5432/crypto_trading"))
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))


class ExchangeConfig(BaseModel):
    """Exchange API configuration."""
    binance_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("BINANCE_API_KEY"))
    binance_secret_key: Optional[str] = Field(default_factory=lambda: os.getenv("BINANCE_SECRET_KEY"))
    coinbase_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("COINBASE_API_KEY"))
    coinbase_secret_key: Optional[str] = Field(default_factory=lambda: os.getenv("COINBASE_SECRET_KEY"))


class TradingConfig(BaseModel):
    """Trading configuration."""
    initial_capital: float = Field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000.0")))
    max_position_size: float = Field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.1")))
    risk_free_rate: float = Field(default_factory=lambda: float(os.getenv("RISK_FREE_RATE", "0.02")))
    max_drawdown: float = Field(default_factory=lambda: float(os.getenv("MAX_DRAWDOWN", "0.15")))
    max_daily_trades: int = Field(default_factory=lambda: int(os.getenv("MAX_DAILY_TRADES", "100")))
    
    # Trading pairs to monitor
    trading_pairs: List[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    max_concurrent_positions: int = 3


class ModelConfig(BaseModel):
    """Model configuration."""
    model_update_interval: int = Field(default_factory=lambda: int(os.getenv("MODEL_UPDATE_INTERVAL", "3600")))
    prediction_horizon: int = Field(default_factory=lambda: int(os.getenv("PREDICTION_HORIZON", "300")))
    feature_window: int = Field(default_factory=lambda: int(os.getenv("FEATURE_WINDOW", "1440")))
    
    # Model parameters
    sequence_length: int = 60  # 60 minutes of data
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prometheus_port: int = Field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


class SafetyConfig(BaseModel):
    """Safety configuration."""
    enable_live_trading: bool = Field(default_factory=lambda: os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true")
    enable_paper_trading: bool = Field(default_factory=lambda: os.getenv("ENABLE_PAPER_TRADING", "true").lower() == "true")
    
    # Emergency stops
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_position_value: float = 50000.0  # Max $50k per position


class Config(BaseModel):
    """Main configuration class."""
    database: DatabaseConfig = DatabaseConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    model: ModelConfig = ModelConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    safety: SafetyConfig = SafetyConfig()


# Global configuration instance
config = Config()
