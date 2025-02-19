import sys
import random
import numpy as np
import itertools
import requests
import logging
from concurrent.futures import ProcessPoolExecutor

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QTabWidget, QHBoxLayout
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
# Import matplotlib do rysowania wykresów
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Import Plotly oraz QWebEngineView do wykresów interaktywnych
import plotly.graph_objects as go
import plotly.io as pio
from PyQt6.QtWebEngineWidgets import QWebEngineView

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ================================
# MODUŁ ANALIZY SENTYMENTU
# ================================
class SentimentAnalyzer:
    """
    Prosty moduł symulujący analizę sentymentu. Zwraca losową wartość z zakresu -1 do 1.
    W praktyce można podłączyć API analizy nastrojów lub zaawansowany model ML.
    """
    @staticmethod
    def get_sentiment():
        return random.uniform(-1, 1)


# ================================
# MODUŁY POBIERANIA DANYCH
# ================================
class MarketDataFetcher:
    """
    Pobiera rzeczywiste dane rynkowe za pomocą API Binance.
    """
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol

    def get_latest_price(self):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}"
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            return float(data["price"])
        except Exception as e:
            logging.error(f"Błąd pobierania danych: {e}")
            return None

class DataSimulator:
    """
    Symulacja danych cenowych metodą random walk.
    """
    def __init__(self):
        self.price = 100.0

    def get_next_price(self):
        change = random.uniform(-1, 1)
        self.price += change
        return self.price


# ================================
# MODUŁ WSKAŹNIKÓW TECHNICZNYCH
# ================================
class Indicators:
    """Metody statyczne do obliczania wskaźników technicznych."""
    @staticmethod
    def moving_average(prices, window=10):
        prices = np.asarray(prices)
        if prices.size < window:
            return prices.mean()
        return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]

    @staticmethod
    def rsi(prices, window=14):
        prices = np.asarray(prices)
        if prices.size < window + 1:
            return 50
        deltas = np.diff(prices[-(window+1):])
        ups = deltas[deltas > 0].sum() if np.any(deltas > 0) else 0.0
        downs = -deltas[deltas < 0].sum() if np.any(deltas < 0) else 0.0
        if downs == 0:
            return 100
        rs = ups / downs
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(prices, period):
        prices = np.asarray(prices)
        ema = []
        multiplier = 2 / (period + 1)
        for i, price in enumerate(prices):
            if i == 0:
                ema.append(price)
            else:
                ema.append((price - ema[-1]) * multiplier + ema[-1])
        return np.array(ema)

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        prices = np.asarray(prices)
        if prices.size < slow:
            return 0, 0
        ema_fast = Indicators.ema(prices, fast)
        ema_slow = Indicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line.tolist(), signal)
        return macd_line[-1], signal_line[-1]

    @staticmethod
    def bollinger_bands(prices, window=20, num_std=2):
        prices = np.asarray(prices)
        if prices.size < window:
            data = prices
        else:
            data = prices[-window:]
        avg = np.mean(data)
        std = np.std(data)
        return avg, avg + num_std * std, avg - num_std * std


# ================================
# MODUŁ PROGNOZOWANIA CEN (NOWOŚĆ)
# ================================
class PricePredictor:
    """
    Prosty moduł prognozowania ceny przy użyciu regresji liniowej na podstawie ostatnich N próbek.
    """
    @staticmethod
    def predict_next(prices, window=10):
        prices = np.asarray(prices)
        if prices.size < window:
            window = prices.size
        x = np.arange(window)
        y = prices[-window:]
        # Dopasowanie prostej regresji liniowej
        coeffs = np.polyfit(x, y, 1)  # [slope, intercept]
        slope, intercept = coeffs
        next_x = window
        prediction = slope * next_x + intercept
        return prediction


# ================================
# STRATEGIE HANDLOWE
# ================================
class TradingStrategy:
    def __init__(self, ma_window=10, rsi_window=14, rsi_overbought=70,
                 rsi_oversold=30, stop_loss=0.02, take_profit=0.04):
        self.ma_window = ma_window
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def evaluate(self, prices):
        prices_np = np.asarray(prices)
        if prices_np.size < max(self.ma_window, self.rsi_window + 1, 26):
            return "HOLD", None, None, None, None
        current_price = prices_np[-1]
        ma = Indicators.moving_average(prices_np, self.ma_window)
        rsi_val = Indicators.rsi(prices_np, self.rsi_window)
        macd_val, signal_line = Indicators.macd(prices_np)
        signal = "HOLD"
        if current_price > ma and rsi_val < self.rsi_oversold and macd_val > signal_line:
            signal = "BUY"
        elif current_price < ma and rsi_val > self.rsi_overbought and macd_val < signal_line:
            signal = "SELL"
        return signal, ma, rsi_val, macd_val, signal_line

class EnhancedTradingStrategy(TradingStrategy):
    def __init__(self, ma_window=10, rsi_window=14, rsi_overbought=70, rsi_oversold=30,
                 stop_loss=0.02, take_profit=0.04, bb_window=20, bb_std=2, trailing_stop=0.01,
                 long_ma_window=50, trend_threshold=0.05, min_volatility=0.2):
        super().__init__(ma_window, rsi_window, rsi_overbought, rsi_oversold, stop_loss, take_profit)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.trailing_stop = trailing_stop
        self.base_volatility = 1.0
        self.volatility_factor = 0.5
        self.long_ma_window = long_ma_window
        self.trend_threshold = trend_threshold
        self.min_volatility = min_volatility
        self.weights = {"ma": 0.5, "rsi": 1.0, "macd": 1.0, "bb": 0.5}
        self.score_threshold = 0.8

    def evaluate(self, prices):
        prices_np = np.asarray(prices)
        if prices_np.size < max(self.ma_window, self.rsi_window + 1, 26, self.bb_window, self.long_ma_window):
            return "HOLD", None, None, None, None, None, None, None, None, None
        current_price = prices_np[-1]
        ma = Indicators.moving_average(prices_np, self.ma_window)
        rsi_val = Indicators.rsi(prices_np, self.rsi_window)
        macd_val, macd_signal = Indicators.macd(prices_np)
        bb_mid, bb_upper, bb_lower = Indicators.bollinger_bands(prices_np, self.bb_window, self.bb_std)
        score = 0
        score += self.weights["ma"] if current_price > ma else -self.weights["ma"]
        if rsi_val < self.rsi_oversold:
            score += self.weights["rsi"]
        elif rsi_val > self.rsi_overbought:
            score -= self.weights["rsi"]
        score += self.weights["macd"] if macd_val > macd_signal else -self.weights["macd"]
        if current_price < bb_lower:
            score += self.weights["bb"]
        elif current_price > bb_upper:
            score -= self.weights["bb"]
        volatility = np.std(prices_np[-self.rsi_window:]) if prices_np.size >= self.rsi_window else 0
        if volatility < self.min_volatility:
            return "HOLD", ma, rsi_val, macd_val, macd_signal, bb_mid, bb_upper, bb_lower, None, None
        if score >= self.score_threshold:
            signal = "BUY"
        elif score <= -self.score_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        long_ma = Indicators.moving_average(prices_np, self.long_ma_window)
        past_long_ma = Indicators.moving_average(prices_np[:-1], self.long_ma_window) if prices_np.size > self.long_ma_window else long_ma
        trend = (long_ma - past_long_ma) / past_long_ma if past_long_ma != 0 else 0
        if (trend > self.trend_threshold and signal == "SELL") or (trend < -self.trend_threshold and signal == "BUY"):
            signal = "HOLD"
        effective_stop_loss = self.stop_loss * (1 + (volatility / self.base_volatility) * self.volatility_factor)
        effective_take_profit = self.take_profit * (1 + (volatility / self.base_volatility) * self.volatility_factor)
        return (signal, ma, rsi_val, macd_val, macd_signal,
                bb_mid, bb_upper, bb_lower, effective_stop_loss, effective_take_profit)

class IntelligentTradingStrategy(EnhancedTradingStrategy):
    def __init__(self, *args, multi_timeframe_window=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_timeframe_window = multi_timeframe_window

    def evaluate(self, prices):
        prices_np = np.asarray(prices)
        if prices_np.size < max(self.ma_window, self.rsi_window + 1, 26, self.bb_window, self.long_ma_window, self.multi_timeframe_window):
            return super().evaluate(prices_np)
        current_price = prices_np[-1]
        ma = Indicators.moving_average(prices_np, self.ma_window)
        rsi_val = Indicators.rsi(prices_np, self.rsi_window)
        macd_val, macd_signal = Indicators.macd(prices_np)
        bb_mid, bb_upper, bb_lower = Indicators.bollinger_bands(prices_np, self.bb_window, self.bb_std)
        score_short = (self.weights["ma"] if current_price > ma else -self.weights["ma"])
        if rsi_val < self.rsi_oversold:
            score_short += self.weights["rsi"]
        elif rsi_val > self.rsi_overbought:
            score_short -= self.weights["rsi"]
        score_short += self.weights["macd"] if macd_val > macd_signal else -self.weights["macd"]
        if current_price < bb_lower:
            score_short += self.weights["bb"]
        elif current_price > bb_upper:
            score_short -= self.weights["bb"]
        prices_long = prices_np[-self.multi_timeframe_window:]
        current_price_long = prices_long[-1]
        ma_long = Indicators.moving_average(prices_long, self.ma_window)
        rsi_long = Indicators.rsi(prices_long, self.rsi_window)
        macd_long, macd_signal_long = Indicators.macd(prices_long)
        bb_mid_long, bb_upper_long, bb_lower_long = Indicators.bollinger_bands(prices_long, self.bb_window, self.bb_std)
        score_long = (self.weights["ma"] if current_price_long > ma_long else -self.weights["ma"])
        if rsi_long < self.rsi_oversold:
            score_long += self.weights["rsi"]
        elif rsi_long > self.rsi_overbought:
            score_long -= self.weights["rsi"]
        score_long += self.weights["macd"] if macd_long > macd_signal_long else -self.weights["macd"]
        if current_price_long < bb_lower_long:
            score_long += self.weights["bb"]
        elif current_price_long > bb_upper_long:
            score_long -= self.weights["bb"]
        final_score = (score_short + score_long) / 2
        if final_score >= self.score_threshold:
            final_signal = "BUY"
        elif final_score <= -self.score_threshold:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        long_ma = Indicators.moving_average(prices_np, self.long_ma_window)
        past_long_ma = Indicators.moving_average(prices_np[:-1], self.long_ma_window) if prices_np.size > self.long_ma_window else long_ma
        trend = (long_ma - past_long_ma) / past_long_ma if past_long_ma != 0 else 0
        if (trend > self.trend_threshold and final_signal == "SELL") or (trend < -self.trend_threshold and final_signal == "BUY"):
            final_signal = "HOLD"
        volatility = np.std(prices_np[-self.rsi_window:]) if prices_np.size >= self.rsi_window else 0
        if volatility < self.min_volatility:
            return "HOLD", ma, rsi_val, macd_val, macd_signal, bb_mid, bb_upper, bb_lower, None, None
        effective_stop_loss = self.stop_loss * (1 + (volatility / self.base_volatility) * self.volatility_factor)
        effective_take_profit = self.take_profit * (1 + (volatility / self.base_volatility) * self.volatility_factor)
        return (final_signal, ma, rsi_val, macd_val, macd_signal,
                bb_mid, bb_upper, bb_lower, effective_stop_loss, effective_take_profit)

# Nowa strategia wykorzystująca prognozę ceny – zwiększa "inteligencję"
class PricePredictor:
    """
    Prosty moduł prognozowania ceny przy użyciu regresji liniowej na podstawie ostatnich N próbek.
    """
    @staticmethod
    def predict_next(prices, window=10):
        prices = np.asarray(prices)
        if prices.size < window:
            window = prices.size
        x = np.arange(window)
        y = prices[-window:]
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        next_x = window
        prediction = slope * next_x + intercept
        return prediction

class IntelligentTradingStrategyWithPrediction(AdaptiveTradingStrategy):
    """
    Strategia rozszerzona o prognozę ceny.
    Na podstawie prognozy (PricePredictor) modyfikuje ostateczny sygnał:
    Jeśli przewidywana cena jest wyraźnie wyższa od bieżącej – wymusza sygnał BUY,
    a jeśli niższa – wymusza sygnał SELL.
    """
    def evaluate(self, prices):
        base_result = super().evaluate(prices)
        if len(base_result) != 10:
            return base_result
        base_signal, ma, rsi_val, macd_val, macd_signal, bb_mid, bb_upper, bb_lower, eff_sl, eff_tp = base_result
        predicted_price = PricePredictor.predict_next(prices, window=10)
        current_price = np.asarray(prices)[-1]
        if predicted_price > current_price * 1.01:
            final_signal = "BUY"
        elif predicted_price < current_price * 0.99:
            final_signal = "SELL"
        else:
            final_signal = base_signal
        return (final_signal, ma, rsi_val, macd_val, macd_signal,
                bb_mid, bb_upper, bb_lower, eff_sl, eff_tp)

# ================================
# MODUŁ RYZYKA I OCENY WYNIKÓW
# ================================
class RiskManager:
    """
    Oblicza wielkość pozycji na podstawie salda i procentu ryzyka.
    """
    def __init__(self, balance, risk_percent=0.01):
        self.balance = balance
        self.risk_percent = risk_percent

    def calculate_position_size(self, entry_price, stop_loss_percentage):
        risk_amount = self.balance * self.risk_percent
        if stop_loss_percentage == 0:
            return 0
        return risk_amount / (entry_price * stop_loss_percentage)

def evaluate_performance(balance_history):
    balance_array = np.array(balance_history)
    returns = np.diff(balance_array) / balance_array[:-1]
    avg_return = np.mean(returns)
    std_return = np.std(returns) if np.std(returns) > 0 else 1e-9
    sharpe_ratio = avg_return / std_return * np.sqrt(252)
    peak = np.maximum.accumulate(balance_array)
    drawdown = (balance_array - peak) / peak
    max_drawdown = drawdown.min()
    return sharpe_ratio, max_drawdown

# ================================
# BACKTESTER
# ================================
class Backtester:
    def __init__(self, historical_prices, strategy, commission=0.001, initial_balance=10000):
        self.historical_prices = historical_prices
        self.strategy = strategy
        self.commission = commission
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.balance_history = [initial_balance]
        self.trades = []
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.max_price = 0

    def run(self):
        for i in range(1, len(self.historical_prices)):
            prices = self.historical_prices[:i+1]
            result = self.strategy.evaluate(prices)
            if len(result) == 5:
                signal, ma, rsi_val, macd_val, macd_signal = result
                effective_stop_loss = self.strategy.stop_loss
                effective_take_profit = self.strategy.take_profit
            else:
                (signal, ma, rsi_val, macd_val, macd_signal,
                 bb_mid, bb_upper, bb_lower, effective_stop_loss, effective_take_profit) = result
            if effective_stop_loss is None:
                effective_stop_loss = self.strategy.stop_loss
            if effective_take_profit is None:
                effective_take_profit = self.strategy.take_profit
            current_price = self.historical_prices[i]
            if self.position == "LONG":
                self.max_price = max(self.max_price, current_price)
                if current_price <= self.max_price * (1 - self.strategy.trailing_stop):
                    profit = (current_price - self.entry_price) * self.position_size - (current_price * self.commission * self.position_size)
                    self.balance += profit
                    self.trades.append(("SELL (Trailing Stop)", current_price, i, profit, self.balance))
                    self.position = None
                    self.balance_history.append(self.balance)
                    continue
                if current_price <= self.entry_price * (1 - effective_stop_loss):
                    profit = (current_price - self.entry_price) * self.position_size - (current_price * self.commission * self.position_size)
                    self.balance += profit
                    self.trades.append(("SELL (Stop Loss)", current_price, i, profit, self.balance))
                    self.position = None
                    self.balance_history.append(self.balance)
                    continue
                elif current_price >= self.entry_price * (1 + effective_take_profit):
                    profit = (current_price - self.entry_price) * self.position_size - (current_price * self.commission * self.position_size)
                    self.balance += profit
                    self.trades.append(("SELL (Take Profit)", current_price, i, profit, self.balance))
                    self.position = None
                    self.balance_history.append(self.balance)
                    continue
            if signal == "BUY" and self.position is None:
                self.position = "LONG"
                self.entry_price = current_price
                self.max_price = current_price
                risk_manager = RiskManager(self.balance)
                self.position_size = risk_manager.calculate_position_size(current_price, effective_stop_loss)
                self.trades.append(("BUY", current_price, i, self.position_size, self.balance))
            elif signal == "SELL" and self.position == "LONG":
                profit = (current_price - self.entry_price) * self.position_size - (current_price * self.commission * self.position_size)
                self.balance += profit
                self.trades.append(("SELL", current_price, i, profit, self.balance))
                self.position = None
                self.balance_history.append(self.balance)
        if self.position == "LONG":
            profit = (self.historical_prices[-1] - self.entry_price) * self.position_size - (self.historical_prices[-1] * self.commission * self.position_size)
            self.balance += profit
            self.trades.append(("SELL (End)", self.historical_prices[-1], len(self.historical_prices)-1, profit, self.balance))
            self.position = None
            self.balance_history.append(self.balance)
        return self.trades, self.balance_history

# ================================
# OPTIMALIZACJA PARAMETRÓW – RÓWNOLEGŁA
# ================================
def evaluate_params_worker(args):
    values, keys, historical_prices, strategy_class, commission = args
    params = dict(zip(keys, values))
    strategy = strategy_class(**params)
    backtester = Backtester(historical_prices, strategy, commission=commission)
    trades, _ = backtester.run()
    total_profit = sum(trade[3] for trade in trades if trade[0].startswith("SELL") and len(trade) > 3)
    return params, total_profit

def parallel_optimize_parameters(historical_prices, param_grid,
                                 strategy_class=IntelligentTradingStrategy, commission=0.001):
    best_profit = -np.inf
    best_params = None
    keys = list(param_grid.keys())
    grid = list(itertools.product(*(param_grid[key] for key in keys)))
    args_list = [(values, keys, historical_prices, strategy_class, commission) for values in grid]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_params_worker, args_list))
    for params, profit in results:
        if profit > best_profit:
            best_profit = profit
            best_params = params
    return best_params, best_profit

class OptimizeWorker(QThread):
    result_ready = pyqtSignal(object, float)

    def __init__(self, historical_prices, param_grid, parent=None):
        super().__init__(parent)
        self.historical_prices = historical_prices
        self.param_grid = param_grid

    def run(self):
        try:
            best_params, best_profit = parallel_optimize_parameters(self.historical_prices, self.param_grid)
            if best_profit is None:
                best_profit = 0.0
            self.result_ready.emit(best_params, best_profit)
        except Exception as e:
            logging.error(f"Błąd podczas optymalizacji: {e}")
            self.result_ready.emit(None, 0.0)

# ================================
# BOT HANDLOWY
# ================================
class TradingBot:
    def __init__(self, ui_log, data_source="real", symbol="BTCUSDT"):
        # Ustawienie data_source="real" powoduje korzystanie z prawdziwych danych z Binance.
        if data_source == "real":
            self.data_fetcher = MarketDataFetcher(symbol=symbol)
            ui_log.append("Używane są rzeczywiste dane rynkowe.")
        else:
            self.data_fetcher = DataSimulator()
            ui_log.append("Używana jest symulacja danych.")
        self.prices = []
        self.strategy = AdaptiveTradingStrategy()  # Możesz zamienić na IntelligentTradingStrategyWithPrediction, jeśli chcesz użyć prognozy.
        self.ui_log = ui_log
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_new_data)
        self.last_signal = None
        self.cooldown_ticks = 5
        self.cooldown_counter = 0

    def start(self, interval=1000):
        self.timer.start(interval)

    def stop(self):
        self.timer.stop()

    def on_new_data(self):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        price = (self.data_fetcher.get_latest_price() if hasattr(self.data_fetcher, "get_latest_price")
                 else self.data_fetcher.get_next_price())
        if price is None:
            self.ui_log.append("Brak danych rynkowych.")
            return
        self.prices.append(price)
        result = self.strategy.evaluate(self.prices)
        if len(result) == 5:
            signal, ma, rsi_val, macd_val, macd_signal = result
            bb_mid = bb_upper = bb_lower = None
        else:
            (signal, ma, rsi_val, macd_val, macd_signal,
             bb_mid, bb_upper, bb_lower, _, _) = result
        if signal != self.last_signal:
            log_msg = (
                f"Cena: {price:.2f}, "
                f"MA: {f'{ma:.2f}' if ma is not None else 'N/A'}, "
                f"RSI: {f'{rsi_val:.2f}' if rsi_val is not None else 'N/A'}, "
                f"MACD: {f'{macd_val:.2f}' if macd_val is not None else 'N/A'}, "
                f"Signal Line: {f'{macd_signal:.2f}' if macd_signal is not None else 'N/A'}, "
                f"BB: {f'{bb_mid:.2f}/{bb_upper:.2f}/{bb_lower:.2f}' if bb_mid is not None else 'N/A'}, "
                f"Sygnał: {signal}"
            )
            self.ui_log.append(log_msg)
            self.last_signal = signal
            self.cooldown_counter = self.cooldown_ticks

# ================================
# WIDGETY WYKRESÓW
# ================================
class ChartWidget(QWidget):
    def __init__(self, trading_bot, parent=None):
        super().__init__(parent)
        self.trading_bot = trading_bot
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(1000)

    def update_chart(self):
        prices = self.trading_bot.prices
        if not prices:
            return
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        x = list(range(len(prices)))
        ax1.plot(x, prices, label='Cena', color='black')
        ma_values = [np.mean(prices[max(0, i - self.trading_bot.strategy.ma_window + 1): i + 1])
                     for i in range(len(prices))]
        ax1.plot(x, ma_values, label=f"MA({self.trading_bot.strategy.ma_window})", color='orange')
        if hasattr(self.trading_bot.strategy, 'bb_window'):
            bb_window = self.trading_bot.strategy.bb_window
            bb_mid_values, bb_upper_values, bb_lower_values = self.compute_bb_values(prices, bb_window, self.trading_bot.strategy.bb_std)
            ax1.plot(x, bb_mid_values, label='BB Middle', linestyle='--', color='purple')
            ax1.plot(x, bb_upper_values, label='BB Upper', linestyle='--', color='red')
            ax1.plot(x, bb_lower_values, label='BB Lower', linestyle='--', color='blue')
        ax1.set_title("Cena, MA i Bollinger Bands")
        ax1.legend()
        rsi_values = self.compute_rsi_values(prices, self.trading_bot.strategy.rsi_window)
        ax2.plot(x, rsi_values, label='RSI', color='green')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought')
        ax2.axhline(30, color='blue', linestyle='--', label='Oversold')
        ax2.set_title("RSI")
        ax2.legend()
        self.canvas.draw()

    def compute_bb_values(self, prices, window, num_std):
        bb_mid_values, bb_upper_values, bb_lower_values = [], [], []
        for i in range(len(prices)):
            data = prices[max(0, i - window + 1): i + 1]
            mid = np.mean(data)
            std = np.std(data)
            bb_mid_values.append(mid)
            bb_upper_values.append(mid + num_std * std)
            bb_lower_values.append(mid - num_std * std)
        return bb_mid_values, bb_upper_values, bb_lower_values

    def compute_rsi_values(self, prices, window):
        rsi_values = []
        for i in range(len(prices)):
            if i < window + 1:
                rsi_values.append(50)
            else:
                rsi_values.append(Indicators.rsi(prices[:i+1], window))
        return rsi_values

class InteractiveChartWidget(QWidget):
    def __init__(self, trading_bot, parent=None):
        super().__init__(parent)
        self.trading_bot = trading_bot
        self.view = QWebEngineView()
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(5000)

    def update_chart(self):
        prices = self.trading_bot.prices
        if not prices:
            return
        interval = 5
        ohlc_data = []
        x_data = []
        for i in range(0, len(prices), interval):
            chunk = prices[i:i+interval]
            if not chunk:
                break
            open_val = chunk[0]
            high_val = max(chunk)
            low_val = min(chunk)
            close_val = chunk[-1]
            ohlc_data.append((open_val, high_val, low_val, close_val))
            x_data.append(i // interval)
        if not ohlc_data:
            return
        open_vals, high_vals, low_vals, close_vals = zip(*ohlc_data)
        fig = go.Figure(data=[go.Candlestick(
            x=x_data,
            open=open_vals,
            high=high_vals,
            low=low_vals,
            close=close_vals,
            name="Candlestick"
        )])
        window = 3
        ma = []
        for i in range(len(close_vals)):
            if i < window - 1:
                ma.append(None)
            else:
                ma.append(np.mean(close_vals[i-window+1:i+1]))
        fig.add_trace(go.Scatter(
            x=x_data,
            y=ma,
            mode='lines',
            name=f'MA({window})'
        ))
        fig.update_layout(
            title="Interactive Candlestick Chart with Moving Average",
            xaxis_title="Candlestick Index",
            yaxis_title="Price"
        )
        html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        self.view.setHtml(html)

# ================================
# INTERFEJS UŻYTKOWNIKA (GUI)
# ================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Crypto Trading Bot z Inteligentną Logiką")
        self.resize(900, 700)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Zakładka tradingu – inicjujemy najpierw logi
        self.tab_trading = QWidget()
        self.tabs.addTab(self.tab_trading, "Real-Time Trading")
        self.init_trading_tab()

        # Na podstawie loga tworzymy bota – używamy prawdziwych danych (data_source="real")
        self.trading_bot = TradingBot(self.txt_log, data_source="real", symbol="BTCUSDT")

        # Zakładka backtestingu
        self.tab_backtest = QWidget()
        self.tabs.addTab(self.tab_backtest, "Backtesting")
        self.init_backtest_tab()

        # Zakładka wykresów (matplotlib)
        self.tab_charts = QWidget()
        self.tabs.addTab(self.tab_charts, "Wykresy")
        self.init_charts_tab()

        # Zakładka wykresów interaktywnych (Plotly)
        self.tab_interactive = QWidget()
        self.tabs.addTab(self.tab_interactive, "Advanced Charts")
        self.init_interactive_tab()

    def init_trading_tab(self):
        layout = QVBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log)
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Trading")
        self.btn_stop = QPushButton("Stop Trading")
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        self.tab_trading.setLayout(layout)
        self.btn_start.clicked.connect(self.start_trading)
        self.btn_stop.clicked.connect(self.stop_trading)

    def init_backtest_tab(self):
        layout = QVBoxLayout()
        self.txt_backtest = QTextEdit()
        self.txt_backtest.setReadOnly(True)
        layout.addWidget(self.txt_backtest)
        btn_layout = QHBoxLayout()
        self.btn_run_backtest = QPushButton("Run Backtest")
        self.btn_optimize = QPushButton("Optimize Strategy")
        btn_layout.addWidget(self.btn_run_backtest)
        btn_layout.addWidget(self.btn_optimize)
        layout.addLayout(btn_layout)
        self.tab_backtest.setLayout(layout)
        self.btn_run_backtest.clicked.connect(self.run_backtest)
        self.btn_optimize.clicked.connect(self.optimize_strategy)

    def init_charts_tab(self):
        layout = QVBoxLayout()
        self.chart_widget = ChartWidget(self.trading_bot)
        layout.addWidget(self.chart_widget)
        self.tab_charts.setLayout(layout)

    def init_interactive_tab(self):
        layout = QVBoxLayout()
        self.interactive_chart_widget = InteractiveChartWidget(self.trading_bot)
        layout.addWidget(self.interactive_chart_widget)
        self.tab_interactive.setLayout(layout)

    def start_trading(self):
        self.txt_log.append("Uruchamianie tradingu w czasie rzeczywistym...")
        self.trading_bot.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_trading(self):
        self.trading_bot.stop()
        self.txt_log.append("Trading zatrzymany.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def run_backtest(self):
        self.txt_backtest.append("Rozpoczęcie backtestu...")
        historical_prices = self.generate_historical_data()
        # Używamy AdaptiveTradingStrategy – strategii adaptacyjnej z prognozowaniem,
        # ale możesz też wybrać IntelligentTradingStrategyWithPrediction, jeśli chcesz.
        strategy = AdaptiveTradingStrategy()
        backtester = Backtester(historical_prices, strategy)
        trades, balance_history = backtester.run()
        sharpe, max_dd = evaluate_performance(balance_history)
        self.txt_backtest.append("Wyniki backtestu:")
        for trade in trades:
            if trade[0].startswith("BUY"):
                self.txt_backtest.append(f"{trade[0]} at {trade[1]:.2f} (index {trade[2]}), Pos. Size: {trade[3]:.2f}, Balance: {trade[4]:.2f}")
            else:
                self.txt_backtest.append(f"{trade[0]} at {trade[1]:.2f} (index {trade[2]}), Profit: {trade[3]:.2f}, Balance: {trade[4]:.2f}")
        self.txt_backtest.append(f"Final Balance: {balance_history[-1]:.2f}")
        self.txt_backtest.append(f"Sharpe Ratio: {sharpe:.2f}, Max Drawdown: {max_dd*100:.2f}%")
        self.txt_backtest.append("Backtest zakończony.\n")

    def optimize_strategy(self):
        self.txt_backtest.append("Rozpoczynam optymalizację parametrów...")
        historical_prices = self.generate_historical_data()
        param_grid = {
            'ma_window': [5, 10, 15],
            'rsi_window': [10, 14, 20],
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.03, 0.04, 0.05]
        }
        self.optimize_worker = OptimizeWorker(historical_prices, param_grid)
        self.optimize_worker.result_ready.connect(self.on_optimization_finished)
        self.optimize_worker.start()

    def on_optimization_finished(self, best_params, best_profit):
        if best_params is not None:
            self.txt_backtest.append(f"Najlepsze parametry: {best_params}")
            self.txt_backtest.append(f"Łączny zysk (symulowany): {best_profit:.2f}")
        else:
            self.txt_backtest.append("Nie znaleziono optymalnych parametrów.")
        self.txt_backtest.append("Optymalizacja zakończona.\n")

    def generate_historical_data(self, length=200):
        # Pobieramy dane historyczne z prawdziwego źródła – przykładowo z Binance
        # Możesz dostosować to rozwiązanie, np. pobierając dane z innego API
        prices = []
        # Dla uproszczenia, użyjemy symulowanych danych, ale tutaj możesz zaimplementować funkcję pobierającą dane historyczne
        price = 100.0
        for _ in range(length):
            change = random.uniform(-1, 1)
            price += change
            prices.append(price)
        return prices

# ================================
# URUCHOMIENIE APLIKACJI
# ================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
