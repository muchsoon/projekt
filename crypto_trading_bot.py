import pandas as pd
import logging
import threading
from PyQt5.QtCore import pyqtSignal, QObject
from data_provider import DataProvider
from indicator_calculator import IndicatorCalculator
from strategy_executor import StrategyExecutor
from trade_manager import TradeManager
from backtester import Backtester
from gui import CryptoTradingBotGUI  # Import GUI
import matplotlib.pyplot as plt
import yaml

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class CryptoTradingBot(QObject):
    log_signal = pyqtSignal(str)
    data_signal = pyqtSignal(pd.DataFrame)

    def __init__(self, config_path):
        super().__init__()
        logging.debug(f"Starting initialization with config_path: {config_path}")
        self.config = self.load_config(config_path)
        logging.debug(f"Loaded config: {self.config}")
        
        # Debugowanie wartości API
        api_key = self.config.get('api_key', '')
        api_secret = self.config.get('api_secret', '')
        logging.debug(f"api_key type: {type(api_key)}, value: {api_key}")
        logging.debug(f"api_secret type: {type(api_secret)}, value: {api_secret}")
        if not isinstance(api_key, str) or not isinstance(api_secret, str):
            raise ValueError(f"API key and secret must be strings. Got: api_key={api_key}, api_secret={api_secret}")
        
        self.data_provider = DataProvider(api_key, api_secret)
        initial_capital = self.config.get('initial_capital', 10000)
        position_size = self.config.get('position_size', 0.1)  # Domyślna wartość 0.1
        logging.debug(f"initial_capital: {initial_capital}, position_size: {position_size}")
        self.trade_manager = TradeManager(initial_capital, position_size)
        self.indicator_calculator = IndicatorCalculator(self.config.get('strategy_params', {}))
        self.strategy_executor = StrategyExecutor(self.config.get('active_strategies', ['SMA']), 
                                                 self.config.get('strategy_params', {}))
        self.backtester = Backtester(self.strategy_executor, self.trade_manager, self.indicator_calculator)
        self.selected_symbol = self.config.get('symbol', 'BTC/USDT')
        self.bot_running = False
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.gui = CryptoTradingBotGUI(self)  # Przywrócenie GUI
        self.log_signal.connect(self.gui.update_log)
        self.data_signal.connect(self.gui.update_data)

    def load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(f"Config file {config_path} must contain a dictionary, got {type(config)}")
            return config
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def start(self):
        """Start the trading bot."""
        if not self.bot_running:
            self.bot_running = True
            threading.Thread(target=self.run, daemon=True).start()
            self.log_signal.emit("Bot started")
            self.update_status()

    def stop(self):
        """Stop the trading bot."""
        self.bot_running = False
        self.log_signal.emit("Bot stopped")
        self.update_status()

    def run(self):
        """Main bot trading loop."""
        self.bot_running = True
        while self.bot_running:
            try:
                df = self.data_provider.get_data(self.selected_symbol)
                df = self.indicator_calculator.calculate_all(df)
                df = self.strategy_executor.calculate_signals(df, self.trade_manager.in_position.get(self.selected_symbol, False))
                self.plot_data(df)
                self.execute_trades(df)
            except Exception as e:
                logging.error(f"Error in run loop: {str(e)}")
                self.log_signal.emit(f"Błąd w pętli bota: {str(e)}")
            threading.Event().wait(60)  # Czekaj 60 sekund między iteracjami

    def execute_trades(self, df):
        """Execute trades based on signals."""
        latest_row = df.iloc[-1]
        symbol = self.selected_symbol
        price = latest_row['Close']
        
        if latest_row.get('buy_signal', False) and not self.trade_manager.in_position.get(symbol, False):
            self.trade_manager.in_position[symbol] = True
            self.log_signal.emit(f"Buy {symbol} at {price}")
        elif latest_row.get('sell_signal', False) and self.trade_manager.in_position.get(symbol, False):
            self.trade_manager.in_position[symbol] = False
            self.log_signal.emit(f"Sell {symbol} at {price}")

    def plot_data(self, df):
        """Plot price and indicator data."""
        try:
            logging.debug(f"Plotting data with columns: {df.columns.tolist()}")

            # Ujednolicenie nazw kolumn
            column_mapping = {
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'timestamp': 'Date'
            }
            for old, new in column_mapping.items():
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})
                    logging.debug(f"Renamed column '{old}' to '{new}' in plot_data")

            # Sprawdzenie wymaganych kolumn
            required_columns = ['Close', 'Date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns for plotting: {missing_columns}")

            self.ax1.clear()
            self.ax2.clear()

            # Wykres ceny zamknięcia
            self.ax1.plot(df['Date'], df['Close'], label='Cena zamknięcia', color='blue')
            self.ax1.set_title(f"{self.selected_symbol} - Cena")
            self.ax1.set_ylabel('Cena')
            self.ax1.legend()

            # Wykres przykładowego wskaźnika (np. RSI)
            if 'RSI' in df.columns:
                self.ax2.plot(df['Date'], df['RSI'], label='RSI', color='orange')
                self.ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                self.ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                self.ax2.set_title('RSI')
                self.ax2.set_ylabel('RSI')
                self.ax2.legend()

            self.fig.tight_layout()
            self.fig.autofmt_xdate()
            self.data_signal.emit(df)

        except Exception as e:
            logging.error(f"Failed to plot data: {str(e)}")
            raise

    def fetch_initial_data(self, symbol):
        """Fetch initial data for the bot."""
        df = self.data_provider.get_data(symbol)
        df = self.indicator_calculator.calculate_all(df)
        self.plot_data(df)

    def set_symbol(self, symbol):
        """Set the trading symbol."""
        self.selected_symbol = symbol
        self.fetch_initial_data(symbol)

    def update_timeframe(self, timeframe):
        """Update the timeframe (not implemented yet)."""
        self.log_signal.emit(f"Timeframe updated to {timeframe} (not implemented)")

    def run_backtest(self, start_date, end_date):
        """Run backtest in a separate thread."""
        threading.Thread(target=self.backtest_thread, args=(start_date, end_date), daemon=True).start()

    def backtest_thread(self, start_date, end_date):
        """Backtest thread function."""
        try:
            symbol = self.selected_symbol
            df = self.data_provider.get_historical_data(symbol, start_date, end_date)
            stats = self.backtester.run(symbol, df)
            self.log_signal.emit(f"Backtest zakończony. Zysk: {stats['profit']:.2f}, Kapitał końcowy: {stats['final_capital']:.2f}")
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            self.log_signal.emit(f"Błąd podczas backtestu: {str(e)}")

    def optimize_parameters(self, start_date, end_date):
        """Optimize strategy parameters in a separate thread."""
        threading.Thread(target=self.optimize_parameters_thread, args=(start_date, end_date), daemon=True).start()

    def optimize_parameters_thread(self, start_date, end_date):
        """Optimize parameters thread function."""
        try:
            symbol = self.selected_symbol
            df = self.data_provider.get_historical_data(symbol, start_date, end_date)
            best_params, best_profit = self.backtester.optimize_parameters(symbol, df)
            self.strategy_executor.params.update(best_params)
            self.trade_manager.stop_loss = best_params['stop_loss']
            self.trade_manager.take_profit = best_params['take_profit']
            self.log_signal.emit(f"Optymalizacja zakończona. Zysk: {best_profit:.2f}, Najlepsze parametry: {best_params}")
        except Exception as e:
            logging.error(f"Optimization error: {str(e)}")
            self.log_signal.emit(f"Błąd podczas optymalizacji: {str(e)}")

    def train_ml_model(self, start_date, end_date):
        """Train ML model (not implemented yet)."""
        self.log_signal.emit("Training ML model not implemented yet")

    def update_status(self):
        """Update bot status."""
        self.log_signal.emit("Bot Status: Running" if self.bot_running else "Bot Status: Stopped")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    bot = CryptoTradingBot('config.yaml')
    bot.gui.show()
    sys.exit(app.exec_())