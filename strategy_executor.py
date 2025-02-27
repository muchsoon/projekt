import pandas as pd
import numpy as np
import logging
import requests
from typing import List, Dict

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class StrategyExecutor:
    def __init__(self, active_strategies: List[str], params: Dict[str, float]):
        self.active_strategies = active_strategies
        self.params = params
        self.ml_model = None

    def calculate_signals(self, df: pd.DataFrame, in_position: bool) -> pd.DataFrame:
        """
        Calculate trading signals based on active strategies.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            in_position (bool): Whether currently in a position
            
        Returns:
            pd.DataFrame: DataFrame with added buy_signal and sell_signal columns
        """
        logging.debug(f"Calculating signals for {len(df)} records, in_position={in_position}")
        df = df.copy()
        
        # Sprawdzenie dostępnych kolumn
        logging.debug(f"Input DataFrame columns: {df.columns.tolist()}")

        # Ujednolicenie nazw kolumn
        column_mapping = {
            'close': 'Close',
            'volume': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
                logging.debug(f"Renamed column '{old}' to '{new}'")

        buy_signals = pd.Series(0.0, index=df.index)
        sell_signals = pd.Series(0.0, index=df.index)
        in_position_series = pd.Series(in_position, index=df.index).astype(bool)

        for strategy in self.active_strategies:
            if strategy == 'SMA':
                sma_signals = (df['SMA_short'] > df['SMA_long']).astype(int) - (df['SMA_short'] < df['SMA_long']).astype(int)
                buy_signals += sma_signals.where(sma_signals > 0, 0)
                sell_signals += (-sma_signals).where(sma_signals < 0, 0)
            elif strategy == 'RSI':
                rsi_signals = (df['RSI'] < self.params.get('rsi_low', 30)).astype(int) - (df['RSI'] > self.params.get('rsi_high', 70)).astype(int)
                buy_signals += rsi_signals.where(rsi_signals > 0, 0)
                sell_signals += (-rsi_signals).where(rsi_signals < 0, 0)
            elif strategy == 'MACD':
                macd_signals = (df['MACD'] > df['MACD_signal']).astype(int) - (df['MACD'] < df['MACD_signal']).astype(int)
                buy_signals += macd_signals.where(macd_signals > 0, 0)
                sell_signals += (-macd_signals).where(macd_signals < 0, 0)

        # Dodatkowe filtry
        volatility_filter = (df['ATR'] > df['ATR'].shift(1)).astype(bool)
        trend_direction = (df['Close'] > df['Close'].shift(1)).astype(bool)
        volume_filter = (df['Volume'] > df['Volume'].shift(1)).astype(bool)
        momentum_filter_buy = (df['Momentum'] > 0).astype(bool)
        
        buy_signals = buy_signals * volatility_filter * trend_direction * volume_filter * momentum_filter_buy
        sell_signals = sell_signals * volatility_filter * ~trend_direction * volume_filter

        # Fear and Greed Index
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1")
            if response.status_code == 200:
                fng_data = response.json()
                fng_value = int(fng_data['data'][0]['value'])
                logging.debug(f"Fear and Greed Index: {fng_value}")
                if fng_value < 30:
                    buy_signals *= 1.5
                elif fng_value > 70:
                    sell_signals *= 1.5
        except Exception as e:
            logging.warning(f"Failed to fetch Fear and Greed Index: {str(e)}")

        # Logowanie surowych sygnałów
        logging.debug(f"Raw buy_signals: min={buy_signals.min()}, max={buy_signals.max()}")
        logging.debug(f"Raw sell_signals: min={sell_signals.min()}, max={sell_signals.max()}")

        # Filtrowanie sygnałów z progiem
        threshold = self.params.get('threshold', 0.4)
        df.loc[:, 'buy_signal'] = ((buy_signals >= threshold) & ~in_position_series).fillna(False).astype(bool)
        df.loc[:, 'sell_signal'] = ((sell_signals >= threshold) & in_position_series).fillna(False).astype(bool)

        # Logowanie filtrowanych sygnałów
        logging.debug(f"Filtered buy_signal: {df['buy_signal'].sum()} points")
        logging.debug(f"Filtered sell_signal: {df['sell_signal'].sum()} points")

        return df

    def train_ml_model(self, symbol: str, df: pd.DataFrame) -> bool:
        logging.debug(f"Training ML model not implemented yet for {symbol}")
        return False  # Do zaimplementowania później