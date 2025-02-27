import pandas as pd
import logging
from typing import Dict, Tuple
from itertools import product

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class Backtester:
    def __init__(self, strategy_executor, trade_manager, indicator_calculator):
        self.strategy_executor = strategy_executor
        self.trade_manager = trade_manager
        self.indicator_calculator = indicator_calculator
        self.stats = {
            'profit': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_capital': self.trade_manager.capital
        }

    def run(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        try:
            logging.info(f"Uruchamianie backtestu dla {symbol} od {df.index[0]} do {df.index[-1]}")
            logging.debug(f"Input DataFrame columns: {df.columns.tolist()}")

            # Ujednolicenie nazw kolumn
            column_mapping = {
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            }
            for old, new in column_mapping.items():
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})
                    logging.debug(f"Renamed column '{old}' to '{new}'")

            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns in DataFrame: {missing_columns}")

            # Obliczanie wskaźników i sygnałów
            logging.debug("Calculating indicators and signals in backtest")
            df = self.indicator_calculator.calculate_all(df)
            df = self.strategy_executor.calculate_signals(df, self.trade_manager.in_position.get(symbol, False))

            # Inicjalizacja zmiennych backtestu
            self.stats = {
                'profit': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'final_capital': self.trade_manager.capital
            }
            in_position = False
            entry_price = 0.0
            capital_history = []

            stop_loss = getattr(self.trade_manager, 'stop_loss', 0.05)
            take_profit = getattr(self.trade_manager, 'take_profit', 0.10)

            # Pętla backtestu
            for index, row in df.iterrows():
                price = row['Close']
                capital_history.append(self.stats['final_capital'])

                if not in_position and row.get('buy_signal', False):
                    in_position = True
                    entry_price = price
                    self.trade_manager.in_position[symbol] = True
                    logging.debug(f"Buy at {price} on {index}")
                elif in_position:
                    if price <= entry_price * (1 - stop_loss):
                        in_position = False
                        exit_price = price
                        profit = exit_price - entry_price
                        self.stats['profit'] += profit
                        self.stats['final_capital'] += profit
                        self.trade_manager.capital = self.stats['final_capital']
                        self.stats['trades'] += 1
                        self.stats['losses'] += 1
                        self.trade_manager.in_position[symbol] = False
                        logging.debug(f"Stop-loss triggered at {price} on {index}, profit: {profit}")
                    elif price >= entry_price * (1 + take_profit):
                        in_position = False
                        exit_price = price
                        profit = exit_price - entry_price
                        self.stats['profit'] += profit
                        self.stats['final_capital'] += profit
                        self.trade_manager.capital = self.stats['final_capital']
                        self.stats['trades'] += 1
                        self.stats['wins'] += 1
                        self.trade_manager.in_position[symbol] = False
                        logging.debug(f"Take-profit triggered at {price} on {index}, profit: {profit}")
                    elif row.get('sell_signal', False):
                        in_position = False
                        exit_price = price
                        profit = exit_price - entry_price
                        self.stats['profit'] += profit
                        self.stats['final_capital'] += profit
                        self.trade_manager.capital = self.stats['final_capital']
                        self.stats['trades'] += 1
                        if profit > 0:
                            self.stats['wins'] += 1
                        else:
                            self.stats['losses'] += 1
                        self.trade_manager.in_position[symbol] = False
                        logging.debug(f"Sell at {price} on {index}, profit: {profit}")

            # Obliczenie statystyk
            if capital_history:
                capital_series = pd.Series(capital_history)
                returns = capital_series.pct_change().dropna()
                if not returns.empty:
                    self.stats['sharpe_ratio'] = (returns.mean() / returns.std()) * (24 ** 0.5)
                    max_drawdown = (capital_series.cummax() - capital_series).max() / capital_series.cummax().max()
                    self.stats['max_drawdown'] = max_drawdown

            logging.info(f"Backtest completed: Profit={self.stats['profit']:.2f}, Trades={self.stats['trades']}, "
                         f"Wins={self.stats['wins']}, Losses={self.stats['losses']}, "
                         f"Max Drawdown={self.stats['max_drawdown']:.2%}, Sharpe Ratio={self.stats['sharpe_ratio']:.2f}, "
                         f"Final Capital={self.stats['final_capital']:.2f}")
            return self.stats

        except Exception as e:
            logging.error(f"Błąd podczas backtestu: {str(e)}")
            raise

    def optimize_parameters(self, symbol: str, df: pd.DataFrame) -> Tuple[Dict, float]:
        try:
            logging.info(f"Optymalizacja parametrów dla {symbol} od {df.index[0]} do {df.index[-1]}")

            # Rozszerzona siatka parametrów
            param_grid = {
                'sma_short_length': [20, 30, 50],
                'sma_long_length': [100, 150, 200],
                'rsi_low': [25, 30, 35],
                'rsi_high': [65, 70, 75],
                'threshold': [0.3, 0.4, 0.5],
                'stop_loss': [0.02, 0.05, 0.10],
                'take_profit': [0.05, 0.10, 0.15]
            }

            best_profit = float('-inf')
            best_params = None

            # Iteracja po kombinacjach parametrów
            for sma_short, sma_long, rsi_low, rsi_high, thresh, stop_loss, take_profit in product(
                param_grid['sma_short_length'],
                param_grid['sma_long_length'],
                param_grid['rsi_low'],
                param_grid['rsi_high'],
                param_grid['threshold'],
                param_grid['stop_loss'],
                param_grid['take_profit']
            ):
                if sma_short >= sma_long or rsi_low >= rsi_high:
                    continue

                # Ustawienie nowych parametrów
                self.strategy_executor.params['threshold'] = thresh
                self.strategy_executor.params['rsi_low'] = rsi_low
                self.strategy_executor.params['rsi_high'] = rsi_high
                self.indicator_calculator.params['sma_short_length'] = sma_short
                self.indicator_calculator.params['sma_long_length'] = sma_long
                if hasattr(self.trade_manager, 'stop_loss'):
                    self.trade_manager.stop_loss = stop_loss
                if hasattr(self.trade_manager, 'take_profit'):
                    self.trade_manager.take_profit = take_profit

                # Reset przed każdym backtestem
                self.reset()

                # Wykonanie backtestu
                stats = self.run(symbol, df.copy())
                profit = stats['profit']

                logging.debug(f"Tested params: SMA_short={sma_short}, SMA_long={sma_long}, RSI_low={rsi_low}, "
                              f"RSI_high={rsi_high}, Threshold={thresh}, Stop Loss={stop_loss}, "
                              f"Take Profit={take_profit}, Profit={profit:.2f}")

                if profit > best_profit:
                    best_profit = profit
                    best_params = {
                        'sma_short_length': sma_short,
                        'sma_long_length': sma_long,
                        'rsi_low': rsi_low,
                        'rsi_high': rsi_high,
                        'threshold': thresh,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }

            logging.info(f"Optymalizacja zakończona. Najlepsze parametry: {best_params}, Najlepszy zysk: {best_profit:.2f}")
            return best_params, best_profit

        except Exception as e:
            logging.error(f"Błąd podczas optymalizacji parametrów: {str(e)}")
            raise

    def reset(self):
        self.stats = {
            'profit': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_capital': self.trade_manager.capital
        }
        self.trade_manager.in_position.clear()
        self.trade_manager.capital = self.stats['final_capital']