"""
Exchange integration and order management.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ccxt.pro as ccxt
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger

from src.data.models import Order, OrderStatus, OrderSide, OrderType, Position, MarketData
from src.core.config import config


class BaseExchange(ABC):
    """Base class for exchange integrations."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.fees = {}
        
    @abstractmethod
    async def connect(self):
        """Connect to the exchange."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order on the exchange."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        pass


class BinanceExchange(BaseExchange):
    """Binance exchange integration."""
    
    def __init__(self):
        super().__init__("binance")
        self.client: Optional[Client] = None
        self.ccxt_client: Optional[ccxt.binance] = None
        
    async def connect(self):
        """Connect to Binance."""
        try:
            if not config.exchange.binance_api_key or not config.exchange.binance_secret_key:
                raise ValueError("Binance API credentials not configured")
            
            # REST client
            self.client = Client(
                config.exchange.binance_api_key,
                config.exchange.binance_secret_key,
                testnet=not config.safety.enable_live_trading
            )
            
            # CCXT Pro client for advanced features
            self.ccxt_client = ccxt.binance({
                'apiKey': config.exchange.binance_api_key,
                'secret': config.exchange.binance_secret_key,
                'sandbox': not config.safety.enable_live_trading,
                'enableRateLimit': True,
            })
            
            # Test connection
            account_info = self.client.get_account()
            self.is_connected = True
            
            logger.info(f"Connected to Binance. Account status: {account_info.get('accountType', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Binance."""
        if self.ccxt_client:
            await self.ccxt_client.close()
        self.is_connected = False
        logger.info("Disconnected from Binance")
    
    async def place_order(self, order: Order) -> Order:
        """Place an order on Binance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            # Convert order to Binance format
            binance_side = order.side.value.upper()
            binance_type = self._convert_order_type(order.type)
            
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol.replace('/', ''),
                'side': binance_side,
                'type': binance_type,
                'quantity': order.quantity,
            }
            
            # Add price for limit orders
            if order.type == OrderType.LIMIT and order.price:
                order_params['price'] = order.price
                order_params['timeInForce'] = 'GTC'  # Good Till Cancelled
            
            # Place order
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                # Paper trading simulation
                result = await self._simulate_order_placement(order_params)
            else:
                # Live trading
                result = self.client.create_order(**order_params)
            
            # Update order with response
            order.status = OrderStatus.PENDING
            order.id = result.get('orderId', order.id)
            
            if result.get('status') == 'FILLED':
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(result.get('executedQty', 0))
                order.filled_price = float(result.get('price', 0)) if result.get('price') else None
                
                # Calculate fees
                if 'fills' in result:
                    total_fees = sum(float(fill['commission']) for fill in result['fills'])
                    order.fees = total_fees
            
            logger.info(f"Order placed: {order.id} - {order.side.value} {order.quantity} {order.symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {e}")
            order.status = OrderStatus.FAILED
            raise
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            order.status = OrderStatus.FAILED
            raise
    
    async def _simulate_order_placement(self, order_params: Dict) -> Dict:
        """Simulate order placement for paper trading."""
        # Get current market price
        symbol = order_params['symbol']
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Simulate immediate fill for market orders
        return {
            'orderId': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'status': 'FILLED',
            'executedQty': order_params['quantity'],
            'price': current_price,
            'fills': []
        }
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Binance format."""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP_LOSS: 'STOP_LOSS_LIMIT',
            OrderType.TAKE_PROFIT: 'TAKE_PROFIT_LIMIT'
        }
        return mapping.get(order_type, 'MARKET')
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                # Paper trading simulation
                logger.info(f"Simulated cancellation of order {order_id}")
                return True
            else:
                result = self.client.cancel_order(
                    symbol=symbol.replace('/', ''),
                    orderId=order_id
                )
                logger.info(f"Order cancelled: {order_id}")
                return result.get('status') == 'CANCELED'
                
        except BinanceAPIException as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                # Paper trading simulation
                return OrderStatus.FILLED
            else:
                result = self.client.get_order(
                    symbol=symbol.replace('/', ''),
                    orderId=order_id
                )
                
                status_mapping = {
                    'NEW': OrderStatus.PENDING,
                    'PARTIALLY_FILLED': OrderStatus.PENDING,
                    'FILLED': OrderStatus.FILLED,
                    'CANCELED': OrderStatus.CANCELLED,
                    'REJECTED': OrderStatus.FAILED,
                    'EXPIRED': OrderStatus.FAILED
                }
                
                return status_mapping.get(result.get('status'), OrderStatus.PENDING)
                
        except BinanceAPIException as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return OrderStatus.FAILED
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                # Paper trading simulation
                return {
                    'USDT': config.trading.initial_capital,
                    'BTC': 0.0,
                    'ETH': 0.0
                }
            else:
                account_info = self.client.get_account()
                balances = {}
                
                for balance in account_info['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:
                        balances[asset] = total
                
                return balances
                
        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        # For spot trading, we'll track positions through our portfolio manager
        # This method is more relevant for futures trading
        return []
    
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees."""
        if symbol in self.fees:
            return self.fees[symbol]
        
        try:
            if config.safety.enable_paper_trading and not config.safety.enable_live_trading:
                # Use standard Binance fees
                fees = {'maker': 0.001, 'taker': 0.001}  # 0.1%
            else:
                # Get actual fees from exchange info
                exchange_info = self.client.get_exchange_info()
                symbol_info = next(
                    (s for s in exchange_info['symbols'] if s['symbol'] == symbol.replace('/', '')),
                    None
                )
                
                if symbol_info:
                    # Use default fees if not specified
                    fees = {'maker': 0.001, 'taker': 0.001}
                else:
                    fees = {'maker': 0.001, 'taker': 0.001}
            
            self.fees[symbol] = fees
            return fees
            
        except Exception as e:
            logger.error(f"Error getting fees for {symbol}: {e}")
            return {'maker': 0.001, 'taker': 0.001}


class ExchangeManager:
    """Manages multiple exchange connections."""
    
    def __init__(self):
        self.exchanges: Dict[str, BaseExchange] = {}
        self.default_exchange = "binance"
        
    async def add_exchange(self, exchange: BaseExchange):
        """Add an exchange."""
        self.exchanges[exchange.name] = exchange
        await exchange.connect()
        logger.info(f"Added exchange: {exchange.name}")
    
    async def remove_exchange(self, exchange_name: str):
        """Remove an exchange."""
        if exchange_name in self.exchanges:
            await self.exchanges[exchange_name].disconnect()
            del self.exchanges[exchange_name]
            logger.info(f"Removed exchange: {exchange_name}")
    
    async def place_order(self, order: Order, exchange_name: str = None) -> Order:
        """Place an order on specified exchange."""
        exchange_name = exchange_name or self.default_exchange
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not available")
        
        exchange = self.exchanges[exchange_name]
        return await exchange.place_order(order)
    
    async def cancel_order(self, order_id: str, symbol: str, exchange_name: str = None) -> bool:
        """Cancel an order."""
        exchange_name = exchange_name or self.default_exchange
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not available")
        
        exchange = self.exchanges[exchange_name]
        return await exchange.cancel_order(order_id, symbol)
    
    async def get_best_price(self, symbol: str, side: OrderSide) -> Tuple[str, float]:
        """Get best price across all exchanges."""
        best_price = None
        best_exchange = None
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'get_ticker'):
                    ticker = await exchange.get_ticker(symbol)
                    price = ticker['bid'] if side == OrderSide.SELL else ticker['ask']
                    
                    if best_price is None or (
                        (side == OrderSide.SELL and price > best_price) or
                        (side == OrderSide.BUY and price < best_price)
                    ):
                        best_price = price
                        best_exchange = exchange_name
                        
            except Exception as e:
                logger.warning(f"Error getting price from {exchange_name}: {e}")
        
        return best_exchange, best_price
    
    async def get_aggregated_balance(self) -> Dict[str, float]:
        """Get aggregated balance across all exchanges."""
        aggregated_balance = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                balance = await exchange.get_balance()
                for asset, amount in balance.items():
                    if asset in aggregated_balance:
                        aggregated_balance[asset] += amount
                    else:
                        aggregated_balance[asset] = amount
                        
            except Exception as e:
                logger.error(f"Error getting balance from {exchange_name}: {e}")
        
        return aggregated_balance
    
    async def close_all_connections(self):
        """Close all exchange connections."""
        for exchange in self.exchanges.values():
            try:
                await exchange.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from {exchange.name}: {e}")


class OrderManager:
    """Manages order lifecycle and tracking."""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
    async def submit_order(self, order: Order, exchange_name: str = None) -> Order:
        """Submit an order and track it."""
        try:
            # Place order on exchange
            filled_order = await self.exchange_manager.place_order(order, exchange_name)
            
            # Track order
            self.active_orders[filled_order.id] = filled_order
            
            # Start monitoring order if not immediately filled
            if filled_order.status == OrderStatus.PENDING:
                asyncio.create_task(self._monitor_order(filled_order, exchange_name))
            else:
                # Order filled immediately, move to history
                self.order_history.append(filled_order)
                if filled_order.id in self.active_orders:
                    del self.active_orders[filled_order.id]
            
            return filled_order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.FAILED
            return order
    
    async def cancel_order(self, order_id: str, exchange_name: str = None) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return False
        
        order = self.active_orders[order_id]
        success = await self.exchange_manager.cancel_order(order_id, order.symbol, exchange_name)
        
        if success:
            order.status = OrderStatus.CANCELLED
            self.order_history.append(order)
            del self.active_orders[order_id]
            logger.info(f"Order {order_id} cancelled")
        
        return success
    
    async def _monitor_order(self, order: Order, exchange_name: str = None):
        """Monitor an order until it's filled or cancelled."""
        max_checks = 60  # Monitor for 5 minutes (60 * 5 seconds)
        check_count = 0
        
        while check_count < max_checks and order.id in self.active_orders:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                status = await self.exchange_manager.exchanges[exchange_name or self.exchange_manager.default_exchange].get_order_status(
                    order.id, order.symbol
                )
                
                order.status = status
                
                if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
                    # Order is no longer active
                    self.order_history.append(order)
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                    
                    logger.info(f"Order {order.id} finished with status: {status.value}")
                    break
                
                check_count += 1
                
            except Exception as e:
                logger.error(f"Error monitoring order {order.id}: {e}")
                check_count += 1
        
        # If order is still pending after max checks, log warning
        if order.id in self.active_orders and check_count >= max_checks:
            logger.warning(f"Order {order.id} still pending after {max_checks * 5} seconds")
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        return self.order_history[-limit:]
    
    def get_order_statistics(self) -> Dict:
        """Get order execution statistics."""
        total_orders = len(self.order_history)
        if total_orders == 0:
            return {}
        
        filled_orders = sum(1 for order in self.order_history if order.status == OrderStatus.FILLED)
        cancelled_orders = sum(1 for order in self.order_history if order.status == OrderStatus.CANCELLED)
        failed_orders = sum(1 for order in self.order_history if order.status == OrderStatus.FAILED)
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'failed_orders': failed_orders,
            'fill_rate': filled_orders / total_orders,
            'active_orders': len(self.active_orders)
        }
