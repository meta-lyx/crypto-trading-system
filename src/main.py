"""
Main entry point for the crypto trading system.
"""

import asyncio
import uvicorn
from loguru import logger

from src.core.config import config
from src.core.logging_setup import setup_logging
from src.api.routes import app


async def main():
    """Main application entry point."""
    
    # Setup logging
    setup_logging()
    logger.info("Starting Crypto Trading System")
    
    # Log configuration
    logger.info(f"Paper Trading: {config.safety.enable_paper_trading}")
    logger.info(f"Live Trading: {config.safety.enable_live_trading}")
    logger.info(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    logger.info(f"Trading Pairs: {config.trading.trading_pairs}")
    
    # Start the FastAPI server
    config_server = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False
    )
    
    server = uvicorn.Server(config_server)
    
    try:
        logger.info("Starting API server on http://0.0.0.0:8000")
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Crypto Trading System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
