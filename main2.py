import logging
import os
import signal
import sqlite3  # Import for SQLite database operations
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tigeropen.common.consts import BarPeriod, Language, Market, QuoteRight
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.domain.contract import Contract
from tigeropen.trade.domain.order import Order
from tigeropen.trade.trade_client import TradeClient

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class YahooFinanceGapTrader:
    """Gap trading system using Yahoo Finance for data and Tiger Trade for execution"""
    
    def __init__(self, tiger_id: str = None, account: str = None, is_sandbox: bool = None):
        self.tiger_id = tiger_id or os.getenv('TIGER_ID')
        self.account = account or os.getenv('TIGER_ACCOUNT')
        # Ensure boolean parsing for IS_SANDBOX
        self.is_sandbox = is_sandbox if is_sandbox is not None else os.getenv('IS_SANDBOX', 'True').lower() == 'true'
        self.client_config = self._setup_tiger_api()
        self.quote_client = QuoteClient(self.client_config)
        self.trade_client = TradeClient(self.client_config)

        # Robust parsing for numeric environment variables
        self.min_gap_percent = float(os.getenv('MIN_GAP_PERCENT', '1.5').strip().strip('"'))
        self.max_gap_percent = float(os.getenv('MAX_GAP_PERCENT', '15.0').strip().strip('"'))
        self.min_volume = int(os.getenv('MIN_VOLUME', '50000').strip().strip('"'))
        self.min_price = float(os.getenv('MIN_PRICE', '5.0').strip().strip('"'))
        self.position_size = float(os.getenv('POSITION_SIZE', '5000').strip().strip('"'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '10').strip().strip('"'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.05').strip().strip('"'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.08').strip().strip('"'))
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '600').strip().strip('"'))
        self.max_scan_attempts = int(os.getenv('MAX_SCAN_ATTEMPTS', '0').strip().strip('"'))
        self.market_hours_only = os.getenv('MARKET_HOURS_ONLY', 'False').lower().strip().strip('"') == 'true'
        
        # Initialize symbol-to-exchange map
        self.symbol_exchange_map = {}
        self.gap_prone_symbols = self._get_comprehensive_symbol_list() # This will populate the map internally

        self.running = False
        self.scan_count = 0
        self.last_successful_trade = None
        self.total_trades_executed = 0
        self.setup_logging()
        try:
            import yfinance as yf
            self.yf = yf
            self.logger.info("✅ Yahoo Finance module loaded successfully")
        except ImportError:
            self.logger.error("❌ yfinance not installed. Install with: pip install yfinance")
            raise
        
        # SQLite Database setup
        self.db_conn = None
        self._setup_database()

        self.logger.info(f"🔧 Yahoo Finance Gap Trader initialized")
        self.logger.info(f"📊 Monitoring {len(self.gap_prone_symbols)} symbols across various markets")
        self.logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")
        self.logger.info(f"🎯 Gap range: {self.min_gap_percent}% - {self.max_gap_percent}%")
        self.logger.info(f"💰 Position size: ${self.position_size}")
        self.logger.info(f"🔄 Data source: Yahoo Finance (FREE)")
        self.logger.info(f"💾 Data storage: SQLite database (trading_data.db)")

    def _setup_tiger_api(self):
        """Configures the Tiger Open API client."""
        client_config = TigerOpenClientConfig()
        # Strip quotes and newlines from the private key string
        private_key_raw = os.getenv('TIGER_RSA_PRIVATE_KEY_PYTHON', '').strip().strip('"')
        client_config.private_key = private_key_raw.replace('\\n', '\n')

        client_config.tiger_id = self.tiger_id
        client_config.account = self.account
        client_config.language = Language.en_US
        # Robust parsing for API_TIMEOUT
        api_timeout_str = os.getenv('API_TIMEOUT', '30').strip().strip('"')
        client_config.timeout = int(api_timeout_str)
        if self.is_sandbox:
            client_config.sandbox_debug = True
        return client_config

    def setup_logging(self):
        """Sets up logging for the application."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_file = os.getenv('LOG_FILE', f'yahoo_gap_trading_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_database(self):
        """Initializes the SQLite database and creates necessary tables."""
        try:
            self.db_conn = sqlite3.connect('trading_data.db')
            cursor = self.db_conn.cursor()

            # Table for Gap Opportunities
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gap_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT,
                    direction TEXT,
                    gap_percent REAL,
                    gap_amount REAL,
                    current_price REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    volume INTEGER,
                    volume_ratio REAL,
                    strength REAL,
                    price_change_from_open REAL,
                    scanned_at TEXT NOT NULL
                )
            ''')

            # Table for Executed Trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    order_id TEXT,
                    trade_time TEXT NOT NULL
                )
            ''')

            # Table for Current Positions (upsert logic will be used)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER,
                    average_cost REAL,
                    market_price REAL,
                    unrealized_pnl REAL,
                    updated_at TEXT NOT NULL
                )
            ''')

            self.db_conn.commit()
            self.logger.info("✅ SQLite database 'trading_data.db' initialized successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"❌ SQLite database error during setup: {e}")
            if self.db_conn:
                self.db_conn.close()
            self.db_conn = None # Ensure db_conn is None if setup fails

    def _close_database(self):
        """Closes the SQLite database connection."""
        if self.db_conn:
            self.db_conn.close()
            self.logger.info("✅ SQLite database connection closed.")

    def _get_comprehensive_symbol_list(self) -> List[str]:
        """
        Retrieves the list of symbols to monitor from environment variables
        for specified markets and populates the symbol_exchange_map.
        """
        all_symbols = []
        # Define default symbols and their exchanges/currencies for each market
        # These are illustrative; users should populate their .env with actual symbols.
        market_configs = {
            'HK': {'env_var': 'GAP_PRONE_SYMBOLS_HK', 'default_symbols': ['0005.HK', '0700.HK'], 'exchange': 'HKEX', 'currency': 'HKD'},
            'SG': {'env_var': 'GAP_PRONE_SYMBOLS_SG', 'default_symbols': ['D05.SG', 'U11.SG'], 'exchange': 'SGX', 'currency': 'SGD'},
            'ASX': {'env_var': 'GAP_PRONE_SYMBOLS_ASX', 'default_symbols': ['CBA.AX', 'BHP.AX'], 'exchange': 'ASX', 'currency': 'AUD'},
            'CHINA_A_SHARES_SSE': {'env_var': 'GAP_PRONE_SYMBOLS_SSE', 'default_symbols': ['600000.SS', '601398.SS'], 'exchange': 'SSE', 'currency': 'CNY'},
            'CHINA_A_SHARES_SZSE': {'env_var': 'GAP_PRONE_SYMBOLS_SZSE', 'default_symbols': ['000001.SZ', '000002.SZ'], 'exchange': 'SZSE', 'currency': 'CNY'},
            # You can add US stocks back if desired, e.g.:
            # 'US': {'env_var': 'GAP_PRONE_SYMBOLS_US', 'default_symbols': ['AAPL', 'MSFT'], 'exchange': 'SMART', 'currency': 'USD'}
        }

        for market_key, config in market_configs.items():
            symbols_str = os.getenv(config['env_var'], '').strip().strip('"')
            if symbols_str:
                market_symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
            else:
                market_symbols = config['default_symbols']
            
            if market_symbols:
                self.logger.info(f"Loading symbols for {market_key} ({config['exchange']}): {', '.join(market_symbols)}")

            for symbol in market_symbols:
                # Store symbol, exchange, and currency in the map
                self.symbol_exchange_map[symbol] = {
                    'exchange': config['exchange'],
                    'currency': config['currency']
                }
                all_symbols.append(symbol)
        
        # Remove duplicates and return a clean list of symbols
        return list(set(all_symbols))

    def get_yahoo_finance_data(self, symbols: List[str]) -> Dict:
        """Fetches historical data for given symbols from Yahoo Finance."""
        try:
            all_data = {}
            chunk_size = 10
            self.logger.info(f"📊 Yahoo Finance: Processing {len(symbols)} symbols...")
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                for symbol in chunk:
                    try:
                        ticker = self.yf.Ticker(symbol)
                        # Fetch 1 week of daily data to ensure we have at least 2 bars (prev_close and current_open)
                        hist = ticker.history(period="1wk", interval="1d")
                        if not hist.empty and len(hist) >= 2:
                            bars = []
                            for date, row in hist.iterrows():
                                # Remove timezone info for SQLite compatibility
                                timestamp = date.tz_localize(None) if date.tz is not None else date
                                bars.append({
                                    'time': timestamp,
                                    'open': float(row['Open']),
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'close': float(row['Close']),
                                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                                })
                            if len(bars) >= 2:
                                all_data[symbol] = bars
                        time.sleep(0.1) # Small delay to respect Yahoo Finance limits
                    except Exception as e:
                        self.logger.error(f"${symbol}: possibly delisted or no price data found (period=1wk) (Yahoo error = \"{e}\")")
                        continue
                if i > 0 and i % 50 == 0:
                    self.logger.info(f"📊 Processed {i}/{len(symbols)} symbols...")
                time.sleep(0.5) # Larger delay between chunks
            self.logger.info(f"✅ Yahoo Finance: Retrieved data for {len(all_data)} symbols")
            return all_data
        except Exception as e:
            self.logger.error(f"❌ Yahoo Finance error: {e}")
            return {}

    def calculate_enhanced_gaps(self, market_data: Dict) -> List[Dict]:
        """Calculates gap opportunities from market data."""
        gap_opportunities = []
        self.logger.info(f"🔍 Analyzing {len(market_data)} symbols for gaps...")
        current_scan_time = datetime.now().isoformat() # Timestamp for database entry

        for symbol, bars in market_data.items():
            try:
                if not bars or len(bars) < 2:
                    continue
                sorted_bars = sorted(bars, key=lambda x: x['time'])
                prev_bar = sorted_bars[-2] # Second to last bar is previous day's close
                current_bar = sorted_bars[-1] # Last bar is current day's open/close

                prev_close = float(prev_bar['close'])
                current_open = float(current_bar['open'])
                current_close = float(current_bar['close'])
                current_high = float(current_bar['high'])
                current_low = float(current_bar['low'])
                current_volume = int(current_bar['volume'])

                # Filter out stocks below min price or min volume
                if current_close < self.min_price or current_volume < self.min_volume:
                    continue

                gap_amount = current_open - prev_close
                gap_percent = (gap_amount / prev_close) * 100

                # Check if gap is within configured range
                if abs(gap_percent) >= self.min_gap_percent and abs(gap_percent) <= self.max_gap_percent:
                    current_range = current_high - current_low
                    price_change = current_close - current_open
                    price_change_percent = (price_change / current_open) * 100 if current_open != 0 else 0

                    recent_volumes = [bar['volume'] for bar in sorted_bars[-5:]]
                    avg_volume = sum(recent_volumes) / len(recent_volumes) if len(recent_volumes) > 0 else 0
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                    if gap_percent > 0: # Gap Up
                        signal_type = "gap_up_fade"
                        direction = "SELL"
                        strength = self._calculate_gap_strength(gap_percent, volume_ratio, current_range, "gap_up")
                        entry_price = current_close
                        stop_loss = entry_price * (1 + self.stop_loss_pct) # Stop loss above entry for short
                        take_profit = entry_price * (1 - self.take_profit_pct) # Take profit below entry for short
                    else: # Gap Down
                        signal_type = "gap_down_buy"
                        direction = "BUY"
                        strength = self._calculate_gap_strength(gap_percent, volume_ratio, current_range, "gap_down")
                        entry_price = current_close
                        stop_loss = entry_price * (1 - self.stop_loss_pct) # Stop loss below entry for long
                        take_profit = entry_price * (1 + self.take_profit_pct) # Take profit above entry for long

                    opportunity = {
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'direction': direction,
                        'gap_percent': gap_percent,
                        'gap_amount': gap_amount,
                        'current_price': current_close,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'volume': current_volume,
                        'volume_ratio': volume_ratio,
                        'strength': strength,
                        'price_change_from_open': price_change_percent,
                        'scanned_at': current_scan_time # Use the consistent scan time
                    }
                    gap_opportunities.append(opportunity)
                    direction_emoji = "🔴" if direction == "SELL" else "🟢"
                    self.logger.info(f"🎯 {symbol}: {gap_percent:+.2f}% gap {direction_emoji} "
                                   f"(Strength: {strength:.2f}, Vol: {current_volume:,})")
                    
                    # Save gap opportunity to database
                    if self.db_conn:
                        cursor = self.db_conn.cursor()
                        cursor.execute('''
                            INSERT INTO gap_opportunities (symbol, signal_type, direction, gap_percent, gap_amount,
                            current_price, entry_price, stop_loss, take_profit, volume, volume_ratio, strength,
                            price_change_from_open, scanned_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            opportunity['symbol'], opportunity['signal_type'], opportunity['direction'],
                            opportunity['gap_percent'], opportunity['gap_amount'], opportunity['current_price'],
                            opportunity['entry_price'], opportunity['stop_loss'], opportunity['take_profit'],
                            opportunity['volume'], opportunity['volume_ratio'], opportunity['strength'],
                            opportunity['price_change_from_open'], opportunity['scanned_at']
                        ))
                        self.db_conn.commit()

            except Exception as e:
                self.logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        gap_opportunities.sort(key=lambda x: x['strength'], reverse=True)
        self.logger.info(f"📈 Found {len(gap_opportunities)} gap opportunities")
        return gap_opportunities[:self.max_positions] # Return top N opportunities

    def _calculate_gap_strength(self, gap_percent: float, volume_ratio: float, price_range: float, gap_type: str) -> float:
        """Calculates a strength score for a gap opportunity."""
        strength = 0.0
        # Factor based on absolute gap percentage (capped at 4)
        gap_factor = min(abs(gap_percent) * 0.25, 4)
        strength += gap_factor
        # Factor based on volume ratio (capped at 3)
        volume_factor = min(volume_ratio * 1.0, 3)
        strength += volume_factor
        # Factor based on daily price range (capped at 2)
        range_factor = min(price_range * 0.1, 2)
        strength += range_factor
        # Small bonus for being a recent opportunity (can be dynamic)
        time_bonus = 1.0
        strength += time_bonus
        return min(strength, 10) # Cap strength at 10

    def create_stock_contract(self, symbol: str) -> Contract:
        """Creates a Tiger Trade contract object for a given symbol."""
        contract = Contract()
        contract.symbol = symbol
        contract.sec_type = 'STK'
        
        # Get exchange and currency from the pre-populated map
        market_info = self.symbol_exchange_map.get(symbol)
        if market_info:
            contract.exchange = market_info['exchange']
            contract.currency = market_info['currency']
        else:
            # Fallback for symbols not found in the map (e.g., if dynamically added or error)
            self.logger.warning(f"Symbol {symbol} not found in exchange map, defaulting to SMART/USD.")
            contract.exchange = 'SMART'
            contract.currency = 'USD'
            
        return contract

    def place_gap_trade(self, opportunity: Dict) -> Optional[Dict]:
        """Places a market order for a gap opportunity and sets a stop loss."""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            entry_price = opportunity['entry_price']
            quantity = max(1, int(self.position_size / entry_price)) # Calculate quantity based on position size
            contract = self.create_stock_contract(symbol)

            # Create and place the market order
            order = Order(self.account, contract, direction)
            order.order_type = 'MKT' # Market order
            order.quantity = quantity
            order.time_in_force = 'DAY' # Good for day
            order.outside_rth = False # Do not allow outside regular trading hours
            
            order_result = self.trade_client.place_order(order)

            if order_result:
                direction_emoji = "🔴" if direction == "SELL" else "🟢"
                self.logger.info(f"✅ {direction_emoji} {direction} {quantity} shares of {symbol}")
                self.logger.info(f"🎯 Entry: ${entry_price:.2f}")
                self.logger.info(f"🛡️ Stop: ${opportunity['stop_loss']:.2f}")
                self.logger.info(f"💰 Target: ${opportunity['take_profit']:.2f}")
                self.logger.info(f"📊 Gap: {opportunity['gap_percent']:+.2f}%")
                
                # Determine stop loss action based on initial trade direction
                stop_action = 'BUY' if direction == 'SELL' else 'SELL'
                self._place_stop_loss(contract, quantity, opportunity['stop_loss'], stop_action)
                
                self.last_successful_trade = datetime.now()
                self.total_trades_executed += 1

                # Save executed trade to database
                if self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute('''
                        INSERT INTO executed_trades (symbol, direction, quantity, entry_price, stop_loss,
                        take_profit, order_id, trade_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, direction, quantity, entry_price, opportunity['stop_loss'],
                        opportunity['take_profit'], getattr(order_result, 'order_id', 'N/A'),
                        datetime.now().isoformat()
                    ))
                    self.db_conn.commit()

                return order_result
            else:
                self.logger.warning(f"⚠️ Order placement for {symbol} failed with no result.")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error placing trade for {symbol}: {e}")
            return None

    def _place_stop_loss(self, contract: Contract, quantity: int, stop_price: float, action: str):
        """Places a stop loss order for a given contract."""
        try:
            stop_order = Order(self.account, contract, action)
            stop_order.order_type = 'STP' # Stop order
            stop_order.quantity = quantity
            stop_order.aux_price = stop_price # Stop price
            stop_order.time_in_force = 'DAY'
            stop_order.outside_rth = False
            
            stop_result = self.trade_client.place_order(stop_order)
            if stop_result:
                self.logger.info(f"🛡️ Stop loss placed at ${stop_price:.2f} for {contract.symbol}")
            else:
                self.logger.warning(f"⚠️ Stop loss placement for {contract.symbol} failed with no result.")
        except Exception as e:
            self.logger.error(f"❌ Error placing stop loss for {contract.symbol}: {e}")

    def is_market_hours(self) -> bool:
        """Checks if the current time is within NYSE market hours.
        NOTE: This function currently assumes NYSE hours. For international markets,
        you would need to implement more sophisticated logic considering different timezones
        and market open/close times for HKEX, SGX, ASX, SSE, SZSE.
        """
        if not self.market_hours_only:
            return True
        now = datetime.now()
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5: # Saturday or Sunday
            return False
        # Define market open and close times (Eastern Time assumed for NYSE)
        # Note: This simple check doesn't account for holidays or daylight saving changes
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def scan_for_opportunities(self) -> List[Dict]:
        """Performs a scan for gap opportunities."""
        self.scan_count += 1
        self.logger.info(f"🔍 Starting scan #{self.scan_count}")
        self.logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            if self.market_hours_only and not self.is_market_hours():
                self.logger.info("🕐 Outside market hours, skipping scan")
                return []
            
            market_data = self.get_yahoo_finance_data(self.gap_prone_symbols)
            if not market_data:
                self.logger.warning("⚠️ No market data retrieved from Yahoo Finance")
                return []
            
            opportunities = self.calculate_enhanced_gaps(market_data)
            
            if opportunities:
                self.logger.info(f"📊 Found {len(opportunities)} opportunities:")
                for i, opp in enumerate(opportunities, 1):
                    direction_emoji = "🔴" if opp['direction'] == 'SELL' else "🟢"
                    self.logger.info(f"  {i}. {opp['symbol']} {direction_emoji} {opp['gap_percent']:+.2f}% "
                                   f"(Strength: {opp['strength']:.2f}, Vol: {opp['volume']:,})")
            else:
                self.logger.info("❌ No gap opportunities found this scan")
            
            return opportunities
        except Exception as e:
            self.logger.error(f"❌ Error in scan cycle: {e}")
            return []

    def execute_opportunities(self, opportunities: List[Dict]) -> int:
        """Executes trades for identified opportunities."""
        executed_count = 0
        for opportunity in opportunities:
            try:
                order_result = self.place_gap_trade(opportunity)
                if order_result:
                    executed_count += 1
                    time.sleep(3) # Small delay between trades
            except Exception as e:
                self.logger.error(f"❌ Error executing trade: {e}")
                continue
        return executed_count

    def monitor_positions(self):
        """Monitors current open positions and updates the database."""
        try:
            positions = self.trade_client.get_positions()
            
            if not positions:
                self.logger.info("📊 No open positions")
                # Optionally, clear positions from DB if none are open
                # if self.db_conn:
                #     cursor = self.db_conn.cursor()
                #     cursor.execute("DELETE FROM positions")
                #     self.db_conn.commit()
                return
            
            self.logger.info("📊 Current Positions:")
            self.logger.info("-" * 50)
            total_pnl = 0
            current_time_iso = datetime.now().isoformat()

            for position in positions:
                # Use getattr for safer attribute access
                symbol = getattr(position, 'symbol', 'Unknown')
                quantity = getattr(position, 'quantity', 0)
                avg_cost = getattr(position, 'average_cost', 0)
                market_price = getattr(position, 'market_price', 0)
                unrealized_pnl = getattr(position, 'unrealized_pnl', 0)
                
                total_pnl += float(unrealized_pnl)
                pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
                
                self.logger.info(f"{pnl_emoji} {symbol}: {quantity} shares @ ${avg_cost:.2f}")
                self.logger.info(f"   Current: ${market_price:.2f} | P&L: ${unrealized_pnl:.2f}")

                # Upsert (INSERT or UPDATE) position into database
                if self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute('''
                        INSERT INTO positions (symbol, quantity, average_cost, market_price, unrealized_pnl, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            quantity = ?,
                            average_cost = ?,
                            market_price = ?,
                            unrealized_pnl = ?,
                            updated_at = ?
                    ''', (
                        symbol, quantity, avg_cost, market_price, unrealized_pnl, current_time_iso,
                        quantity, avg_cost, market_price, unrealized_pnl, current_time_iso
                    ))
                    self.db_conn.commit()

            self.logger.info("-" * 50)
            total_emoji = "📈" if total_pnl >= 0 else "📉"
            self.logger.info(f"{total_emoji} Total P&L: ${total_pnl:.2f}")
        except Exception as e:
            self.logger.error(f"❌ Error monitoring positions: {e}")

    def run_continuous_monitoring(self):
        """Runs the continuous monitoring and trading loop."""
        self.running = True
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("🚀 Starting continuous gap monitoring with Yahoo Finance...")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Symbols: {len(self.gap_prone_symbols)} from specified markets")
        self.logger.info(f"⏱️ Scan interval: {self.scan_interval} seconds")
        self.logger.info(f"🎯 Gap range: {self.min_gap_percent}% - {self.max_gap_percent}%")
        self.logger.info(f"💰 Position size: ${self.position_size}")
        self.logger.info(f"🛡️ Stop loss: {self.stop_loss_pct:.1%}")
        self.logger.info(f"🎯 Take profit: {self.take_profit_pct:.1%}")
        self.logger.info(f"📊 Max positions: {self.max_positions}")
        self.logger.info(f"🔄 Data source: Yahoo Finance (FREE - No limits!)")
        self.logger.info(f"💼 Trading via: Tiger Trade API")
        self.logger.info("=" * 80)

        try:
            while self.running:
                scan_start_time = time.time()
                opportunities = self.scan_for_opportunities()
                
                if opportunities:
                    self.logger.info(f"🎯 Executing {len(opportunities)} opportunities...")
                    executed = self.execute_opportunities(opportunities)
                    if executed > 0:
                        self.logger.info(f"✅ Successfully executed {executed} trades")
                        self.logger.info(f"📈 Total trades executed: {self.total_trades_executed}")
                        time.sleep(5) # Give some time for orders to process
                        self.monitor_positions() # Check positions after trades
                
                # Check if max scan attempts reached
                if self.max_scan_attempts > 0 and self.scan_count >= self.max_scan_attempts:
                    self.logger.info(f"⏹️ Reached maximum scan attempts ({self.max_scan_attempts})")
                    break
                
                scan_duration = time.time() - scan_start_time
                sleep_time = max(0, self.scan_interval - scan_duration)
                
                if sleep_time > 0:
                    next_scan_time = datetime.now() + timedelta(seconds=sleep_time)
                    self.logger.info(f"⏸️ Next scan at {next_scan_time.strftime('%H:%M:%S')} "
                                   f"(waiting {sleep_time:.1f}s)...")
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            self.logger.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in continuous monitoring: {e}")
        finally:
            self.running = False
            self.logger.info(f"📊 Monitoring session complete")
            self.logger.info(f"🔍 Total scans performed: {self.scan_count}")
            self.logger.info(f"📈 Total trades executed: {self.total_trades_executed}")
            if self.last_successful_trade:
                self.logger.info(f"⏰ Last trade: {self.last_successful_trade.strftime('%Y-%m-%d %H:%M:%S')}")
            self._close_database() # Ensure database connection is closed

    def _signal_handler(self, signum, frame):
        """Handles OS signals for graceful shutdown."""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False

def main():
    """Main function to initialize and run the gap trader."""
    try:
        required_vars = ['TIGER_ID', 'TIGER_ACCOUNT', 'TIGER_RSA_PRIVATE_KEY_PYTHON']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        trader = YahooFinanceGapTrader()
        trader.run_continuous_monitoring()
    except Exception as e:
        logging.error(f"❌ Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
