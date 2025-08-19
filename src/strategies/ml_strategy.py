"""
Machine learning-based trading strategy.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy, RiskManager, PortfolioManager
from src.data.models import MarketData, TradingSignal, Position, OrderSide
from src.models.trainer import ModelTrainer
from src.core.config import config


class MLTradingStrategy(BaseStrategy):
    """ML-based trading strategy using transformer models."""
    
    def __init__(self, name: str = "ML_Transformer_Strategy", parameters: Dict = None):
        super().__init__(name, parameters)
        
        self.model_trainer = ModelTrainer()
        self.risk_manager = RiskManager(
            max_position_size=config.trading.max_position_size,
            max_drawdown=config.trading.max_drawdown,
            stop_loss_pct=config.trading.stop_loss_pct,
            take_profit_pct=config.trading.take_profit_pct
        )
        
        self.market_data_buffer: Dict[str, List[MarketData]] = {}
        self.min_buffer_size = config.model.sequence_length + 10
        self.prediction_cache: Dict[str, TradingSignal] = {}
        self.last_prediction_time: Dict[str, datetime] = {}
        
        # Strategy parameters
        self.signal_threshold = parameters.get('signal_threshold', 0.3)
        self.confidence_threshold = parameters.get('confidence_threshold', 0.6)
        self.prediction_refresh_interval = parameters.get('prediction_refresh_interval', 300)  # seconds
        
    def add_market_data(self, market_data: MarketData):
        """Add market data to buffer."""
        symbol = market_data.symbol
        
        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = []
        
        self.market_data_buffer[symbol].append(market_data)
        
        # Keep only recent data
        max_buffer_size = self.min_buffer_size * 2
        if len(self.market_data_buffer[symbol]) > max_buffer_size:
            self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-max_buffer_size:]
    
    def generate_signal(self, market_data: MarketData, features: Dict = None) -> Optional[TradingSignal]:
        """Generate trading signal using ML model."""
        
        if not self.is_active:
            return None
        
        symbol = market_data.symbol
        self.add_market_data(market_data)
        
        # Check if we have enough data
        if len(self.market_data_buffer.get(symbol, [])) < self.min_buffer_size:
            return None
        
        # Check if we need to refresh prediction
        now = datetime.now()
        last_prediction = self.last_prediction_time.get(symbol)
        
        if (last_prediction is None or 
            (now - last_prediction).total_seconds() > self.prediction_refresh_interval):
            
            signal = self._generate_ml_prediction(symbol, market_data)
            if signal:
                self.prediction_cache[symbol] = signal
                self.last_prediction_time[symbol] = now
                self.add_signal(signal)
            return signal
        
        # Return cached prediction if still valid
        return self.prediction_cache.get(symbol)
    
    def _generate_ml_prediction(self, symbol: str, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate prediction using ML model."""
        
        try:
            # Convert market data to DataFrame
            data_list = []
            for md in self.market_data_buffer[symbol]:
                data_list.append({
                    'timestamp': md.timestamp,
                    'open': md.open,
                    'high': md.high,
                    'low': md.low,
                    'close': md.close,
                    'volume': md.volume,
                    'symbol': md.symbol
                })
            
            df = pd.DataFrame(data_list)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Make prediction using trained model
            if self.model_trainer.model is None:
                logger.warning("ML model not loaded, cannot generate signal")
                return None
            
            predictions, confidences = self.model_trainer.predict(df)
            
            if len(predictions) == 0:
                return None
            
            # Get latest prediction
            latest_prediction = predictions[-1][0]  # Assuming single output
            latest_confidence = confidences[-1][0]
            
            # Convert prediction to signal (-1 to 1)
            signal_strength = np.tanh(latest_prediction * 5)  # Scale and bound
            
            # Calculate predicted price
            current_price = market_data.close
            predicted_return = latest_prediction
            predicted_price = current_price * (1 + predicted_return)
            
            # Create trading signal
            trading_signal = TradingSignal(
                symbol=symbol,
                timestamp=market_data.timestamp,
                signal=signal_strength,
                confidence=latest_confidence,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                features={
                    'current_price': current_price,
                    'prediction': latest_prediction,
                    'model_version': 'transformer_v1'
                },
                model_version='transformer_v1'
            )
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating ML prediction for {symbol}: {e}")
            return None
    
    def should_enter_position(self, signal: TradingSignal, current_positions: Dict[str, Position]) -> bool:
        """Determine if we should enter a position based on ML signal."""
        
        # Check signal strength and confidence
        if abs(signal.signal) < self.signal_threshold:
            return False
        
        if signal.confidence < self.confidence_threshold:
            return False
        
        # Check if we already have a position in this symbol
        if signal.symbol in current_positions:
            return False
        
        # Additional ML-specific checks
        recent_signals = self.get_recent_signals(signal.symbol, count=5)
        if len(recent_signals) >= 3:
            # Check for signal consistency
            recent_directions = [np.sign(s.signal) for s in recent_signals[-3:]]
            current_direction = np.sign(signal.signal)
            
            # Require at least 2 out of 3 recent signals in same direction
            same_direction_count = sum(1 for d in recent_directions if d == current_direction)
            if same_direction_count < 2:
                return False
        
        return True
    
    def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """Determine if we should exit a position."""
        
        # Risk management checks
        if self.risk_manager.check_stop_loss(position):
            logger.info(f"Stop loss triggered for {position.symbol}")
            return True
        
        if self.risk_manager.check_take_profit(position):
            logger.info(f"Take profit triggered for {position.symbol}")
            return True
        
        # ML-based exit signal
        signal = self.prediction_cache.get(position.symbol)
        if signal:
            # Exit if signal changes direction significantly
            if position.side == OrderSide.BUY and signal.signal < -0.2:
                logger.info(f"ML signal suggests exit for long position in {position.symbol}")
                return True
            
            if position.side == OrderSide.SELL and signal.signal > 0.2:
                logger.info(f"ML signal suggests exit for short position in {position.symbol}")
                return True
        
        # Time-based exit (hold for maximum time)
        max_hold_time = timedelta(hours=24)  # Maximum 24 hours
        if datetime.now() - position.timestamp > max_hold_time:
            logger.info(f"Maximum hold time reached for {position.symbol}")
            return True
        
        return False
    
    def get_position_side(self, signal: TradingSignal) -> OrderSide:
        """Determine position side based on signal."""
        return OrderSide.BUY if signal.signal > 0 else OrderSide.SELL
    
    def calculate_confidence_adjusted_size(self, signal: TradingSignal, base_size: float) -> float:
        """Adjust position size based on signal confidence."""
        confidence_multiplier = signal.confidence
        signal_strength_multiplier = abs(signal.signal)
        
        adjusted_size = base_size * confidence_multiplier * signal_strength_multiplier
        return adjusted_size
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy-specific metrics."""
        total_signals = len(self.signals_history)
        
        if total_signals == 0:
            return {'total_signals': 0}
        
        # Calculate signal statistics
        signal_strengths = [abs(s.signal) for s in self.signals_history]
        confidences = [s.confidence for s in self.signals_history]
        
        buy_signals = sum(1 for s in self.signals_history if s.signal > 0)
        sell_signals = sum(1 for s in self.signals_history if s.signal < 0)
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_signal_strength': np.mean(signal_strengths),
            'avg_confidence': np.mean(confidences),
            'max_signal_strength': np.max(signal_strengths),
            'min_signal_strength': np.min(signal_strengths),
            'active_symbols': len(self.market_data_buffer)
        }
    
    def load_model(self, model_path: str, feature_engineer_path: str):
        """Load trained ML model."""
        try:
            self.model_trainer.load_model(model_path, feature_engineer_path)
            logger.info(f"Loaded ML model for strategy {self.name}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            raise
    
    def retrain_model(self, training_data: pd.DataFrame):
        """Retrain the ML model with new data."""
        try:
            logger.info("Starting model retraining")
            
            # Prepare data
            train_loader, val_loader, test_loader = self.model_trainer.prepare_data(training_data)
            
            # Train model
            training_results = self.model_trainer.train(train_loader, val_loader)
            
            # Evaluate
            metrics = self.model_trainer.evaluate(test_loader)
            
            logger.info(f"Model retraining completed. Metrics: {metrics}")
            return training_results, metrics
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            raise


class EnsembleMLStrategy(BaseStrategy):
    """Ensemble of multiple ML strategies."""
    
    def __init__(self, strategies: List[MLTradingStrategy], weights: List[float] = None):
        super().__init__("Ensemble_ML_Strategy")
        
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def generate_signal(self, market_data: MarketData, features: Dict = None) -> Optional[TradingSignal]:
        """Generate ensemble signal by combining individual strategy signals."""
        
        if not self.is_active:
            return None
        
        signals = []
        for strategy in self.strategies:
            if strategy.is_active:
                signal = strategy.generate_signal(market_data, features)
                if signal:
                    signals.append(signal)
        
        if not signals:
            return None
        
        # Combine signals using weighted average
        weighted_signal = 0.0
        weighted_confidence = 0.0
        weighted_predicted_return = 0.0
        
        active_weights = []
        for i, signal in enumerate(signals):
            if i < len(self.weights):
                active_weights.append(self.weights[i])
            else:
                active_weights.append(1.0 / len(signals))
        
        # Normalize active weights
        total_active_weight = sum(active_weights)
        if total_active_weight > 0:
            active_weights = [w / total_active_weight for w in active_weights]
        
        for signal, weight in zip(signals, active_weights):
            weighted_signal += signal.signal * weight
            weighted_confidence += signal.confidence * weight
            weighted_predicted_return += signal.predicted_return * weight
        
        # Create ensemble signal
        ensemble_signal = TradingSignal(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            signal=weighted_signal,
            confidence=weighted_confidence,
            predicted_price=market_data.close * (1 + weighted_predicted_return),
            predicted_return=weighted_predicted_return,
            features={
                'ensemble_components': len(signals),
                'active_strategies': len([s for s in self.strategies if s.is_active])
            },
            model_version='ensemble_v1'
        )
        
        self.add_signal(ensemble_signal)
        return ensemble_signal
    
    def should_enter_position(self, signal: TradingSignal, current_positions: Dict[str, Position]) -> bool:
        """Use majority vote from active strategies."""
        votes = 0
        total_strategies = 0
        
        for strategy in self.strategies:
            if strategy.is_active:
                if strategy.should_enter_position(signal, current_positions):
                    votes += 1
                total_strategies += 1
        
        # Require majority consensus
        return votes > total_strategies / 2 if total_strategies > 0 else False
    
    def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """Use majority vote from active strategies for exit decisions."""
        votes = 0
        total_strategies = 0
        
        for strategy in self.strategies:
            if strategy.is_active:
                if strategy.should_exit_position(position, market_data):
                    votes += 1
                total_strategies += 1
        
        # Require majority consensus
        return votes > total_strategies / 2 if total_strategies > 0 else False
    
    def activate_strategy(self, strategy_index: int):
        """Activate a specific strategy in the ensemble."""
        if 0 <= strategy_index < len(self.strategies):
            self.strategies[strategy_index].activate()
    
    def deactivate_strategy(self, strategy_index: int):
        """Deactivate a specific strategy in the ensemble."""
        if 0 <= strategy_index < len(self.strategies):
            self.strategies[strategy_index].deactivate()
    
    def update_weights(self, new_weights: List[float]):
        """Update ensemble weights."""
        if len(new_weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        
        if abs(sum(new_weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.weights = new_weights
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_ensemble_metrics(self) -> Dict:
        """Get metrics for the ensemble and individual strategies."""
        ensemble_metrics = {
            'total_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies if s.is_active]),
            'weights': self.weights,
            'individual_metrics': {}
        }
        
        for i, strategy in enumerate(self.strategies):
            ensemble_metrics['individual_metrics'][f'strategy_{i}'] = strategy.get_strategy_metrics()
        
        return ensemble_metrics
