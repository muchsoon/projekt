import ccxt
import pandas as pd
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class DataProvider:
    def __init__(self, api_key: str, api_secret: str):
        logging.debug(f"Initializing DataProvider with api_key={api_key}, api_secret={api_secret}")
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        }
        logging.debug(f"Exchange config type: {type(exchange_config)}, value: {exchange_config}")
        self.exchange = ccxt.binance(exchange_config)
        logging.debug("Exchange initialized successfully")

    def get_data(self, symbol: str) -> pd.DataFrame:
        try:
            logging.debug(f"Fetching OHLCV data for {symbol} with timeframe='1h', limit=1000")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.debug(f"Successfully fetched {len(df)} records for {symbol}")
            return df
        except ccxt.NetworkError as e:
            logging.error(f"Network error fetching data for {symbol}: {str(e)}")
            raise
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error fetching data for {symbol}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
            raise

    def get_historical_data(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        try:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            logging.debug(f"Fetching historical OHLCV data for {symbol} from {start_date} to {end_date}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', since=since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            filtered_df = df[df['timestamp'] <= pd.Timestamp(end_date)]
            logging.debug(f"Successfully fetched {len(filtered_df)} historical records for {symbol}")
            return filtered_df
        except ccxt.NetworkError as e:
            logging.error(f"Network error fetching historical data for {symbol}: {str(e)}")
            raise
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error fetching historical data for {symbol}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error fetching historical data for {symbol}: {str(e)}")
            raise

    def fetch_available_symbols(self) -> list:
        return ['BTC/USDT', 'ETH/USDT']