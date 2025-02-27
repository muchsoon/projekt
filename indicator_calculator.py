import pandas as pd
import pandas_ta as ta
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class IndicatorCalculator:
    def __init__(self, params: dict = None):
        """
        Initialize the IndicatorCalculator with optional parameters.
        
        Args:
            params (dict, optional): Dictionary of strategy parameters (e.g., SMA lengths)
        """
        self.params = params if params is not None else {}

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicator columns
        """
        try:
            logging.debug(f"Calculating indicators for {len(df)} records")
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

            # Sprawdzenie wymaganych kolumn
            required_columns = ['Close', 'Open', 'High', 'Low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns in DataFrame: {missing_columns}")

            df = df.copy()
            close = df['Close']

            # Obliczanie wskaźników technicznych z parametrami z self.params, jeśli dostępne
            sma_short_length = self.params.get('sma_short_length', 50)
            sma_long_length = self.params.get('sma_long_length', 200)
            ema_short_length = self.params.get('ema_short_length', 20)
            ema_long_length = self.params.get('ema_long_length', 50)
            rsi_length = self.params.get('rsi_length', 14)

            df['SMA_short'] = ta.sma(close, length=sma_short_length)
            df['SMA_long'] = ta.sma(close, length=sma_long_length)
            df['EMA_short'] = ta.ema(close, length=ema_short_length)
            df['EMA_long'] = ta.ema(close, length=ema_long_length)
            df['RSI'] = ta.rsi(close, length=rsi_length)
            df['MACD'], df['MACD_signal'], _ = ta.macd(close, fast=12, slow=26, signal=9)
            df['ATR'] = ta.atr(df['High'], df['Low'], close, length=14)
            
            # Poprawione wywołanie SMI z logowaniem
            smi = ta.smi(high=df['High'], low=df['Low'], close=close, fast=5, slow=20, signal=3)
            logging.debug(f"SMI output type: {type(smi)}, columns: {smi.columns.tolist() if isinstance(smi, pd.DataFrame) else 'Not a DataFrame'}")
            
            # Dynamiczne przypisanie SMI i sygnału
            if isinstance(smi, pd.DataFrame):
                # Sprawdzamy dostępne kolumny i przypisujemy odpowiednie
                smi_col = next((col for col in smi.columns if 'SMI_' in col and '_s' not in col.lower()), None)
                signal_col = next((col for col in smi.columns if 'SMIs_' in col), None)
                if smi_col and signal_col:
                    df['SMI'] = smi[smi_col]
                    df['SMI_signal'] = smi[signal_col]
                else:
                    raise KeyError(f"Expected SMI columns not found in output: {smi.columns.tolist()}")
            else:
                raise TypeError(f"Expected DataFrame from ta.smi, got {type(smi)}")

            df['STOCH_k'], df['STOCH_d'] = ta.stoch(df['High'], df['Low'], close, k=14, d=3, smooth_k=3)
            ichimoku = ta.ichimoku(df['High'], df['Low'], close, tenkan=9, kijun=26, senkou=52)
            df['ICHIMOKU_tenkan'] = ichimoku[0]['ITS_9']
            df['ICHIMOKU_kijun'] = ichimoku[0]['IKS_26']
            df['ICHIMOKU_senkou_a'] = ichimoku[0]['ISA_9']
            df['ICHIMOKU_senkou_b'] = ichimoku[0]['ISB_26']
            df['ICHIMOKU_chikou'] = ichimoku[0]['ICS_26']
            df['Momentum'] = ta.mom(close, length=10)

            logging.debug(f"Calculated all indicators for {len(df)} records")
            return df

        except Exception as e:
            logging.error(f"Failed to calculate indicators: {str(e)}")
            raise