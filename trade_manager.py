import logging
import pandas as pd
import numpy as np

class TradeManager:
    def __init__(self, initial_capital, position_size, stop_loss=0.05, take_profit=0.10, atr_multiplier_sl=1.5, atr_multiplier_tp=3.0):
        self.capital = initial_capital
        self.position_size = position_size  # Procent kapitału na pozycję
        self.base_stop_loss = stop_loss  # Bazowa wartość stop-loss
        self.base_take_profit = take_profit  # Bazowa wartość take-profit
        self.atr_multiplier_sl = atr_multiplier_sl  # Mnożnik ATR dla stop-loss
        self.atr_multiplier_tp = atr_multiplier_tp  # Mnożnik ATR dla take-profit
        self.in_position = {}
        self.positions = {}

    def calculate_position_size(self, price, atr):
        """Oblicza rozmiar pozycji na podstawie kapitału i zmienności (ATR), działa z seriami Pandas."""
        risk_amount = self.capital * self.position_size
        stop_loss_distance = atr * self.atr_multiplier_sl
        
        # Wektoryzowane sprawdzenie, gdzie stop_loss_distance == 0
        zero_stop_loss = stop_loss_distance == 0
        if zero_stop_loss.any():
            logging.warning("Wykryto ATR = 0 w niektórych rekordach, ustawienie domyślnego stop-loss")
            # Ustawienie domyślnego stop-loss tam, gdzie stop_loss_distance wynosi 0
            stop_loss_distance = pd.Series(
                np.where(zero_stop_loss, price * self.base_stop_loss, stop_loss_distance),
                index=stop_loss_distance.index
            )
        
        position_size = risk_amount / stop_loss_distance
        max_position = self.capital / price
        return pd.Series(
            np.minimum(position_size, max_position),
            index=position_size.index
        )

    def get_dynamic_levels(self, price, atr):
        """Oblicza dynamiczne stop_loss i take_profit na podstawie ATR."""
        stop_loss = max(self.base_stop_loss, self.atr_multiplier_sl * atr / price)
        take_profit = max(self.base_take_profit, self.atr_multiplier_tp * atr / price)
        return stop_loss, take_profit

    def buy(self, symbol, price, timestamp, atr):
        if symbol in self.in_position and self.in_position[symbol]:
            return f"Już w pozycji dla {symbol}"
        position_size = self.calculate_position_size(price, atr)
        cost = position_size * price
        if cost > self.capital:
            return f"Brak wystarczającego kapitału dla {symbol}: {cost} > {self.capital}"
        
        self.capital -= cost
        self.in_position[symbol] = True
        self.positions[symbol] = {'entry_price': price, 'size': position_size, 'timestamp': timestamp}
        logging.info(f"Buy {symbol}: {position_size} @ {price}, kapitał: {self.capital}")
        return f"Buy {symbol}: {position_size} @ {price}"

    def sell(self, symbol, price, timestamp):
        if symbol not in self.in_position or not self.in_position[symbol]:
            return f"Brak aktywnej pozycji dla {symbol}"
        
        position = self.positions[symbol]
        profit = position['size'] * (price - position['entry_price'])
        self.capital += position['size'] * price
        self.in_position[symbol] = False
        del self.positions[symbol]
        logging.info(f"Sell {symbol}: {profit}, kapitał: {self.capital}")
        return f"Sell {symbol}: zysk {profit}"

    def check_risk(self, symbol, current_price, df):
        if symbol not in self.in_position or not self.in_position[symbol]:
            return False
        position = self.positions[symbol]
        atr = df['ATR'].iloc[-1]
        stop_loss, take_profit = self.get_dynamic_levels(position['entry_price'], atr)
        stop_loss_price = position['entry_price'] * (1 - stop_loss)
        take_profit_price = position['entry_price'] * (1 + take_profit)
        return current_price <= stop_loss_price or current_price >= take_profit_price