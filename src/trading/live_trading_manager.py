class LiveTradingManager:
    def __init__(self):
        self.trading_enabled = False
    
    def get_live_trading_status(self):
        return {'trading_enabled': self.trading_enabled}
