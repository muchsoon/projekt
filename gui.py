import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use('Qt5Agg')  # Wymuszenie backendu Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import logging
import matplotlib.dates as mdates

plt.style.use('dark_background')

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class CryptoTradingBotGUI(QtWidgets.QWidget):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.show_plots = False
        # Stała figura i canvas
        self.candlestick_fig = plt.Figure(figsize=(12, 6), facecolor=(0.1, 0.1, 0.18, 0.9))  # #1A1A2E z opacity 0.9
        self.candlestick_canvas = FigureCanvas(self.candlestick_fig)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Crypto Trading Dashboard")
        self.setMinimumSize(1200, 800)
        layout = QtWidgets.QVBoxLayout()

        # Nowoczesny styl Glassmorphism
        dark_glass_style = """
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                           stop:0 #1A1A2E, stop:1 #0F0F1B);
                color: #E0E0E0;
                font-family: 'Inter', sans-serif;
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 8px;
                color: #00D4FF;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(0, 212, 255, 0.3);
            }
            QComboBox {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 5px;
                color: #E0E0E0;
            }
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QTextEdit {
                background: rgba(15, 15, 27, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 5px;
                color: #E0E0E0;
            }
            QGroupBox {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 10px;
                margin-top: 15px;
                color: #00D4FF;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background: rgba(0, 212, 255, 0.2);
                border-radius: 8px;
            }
            QCheckBox {
                color: #E0E0E0;
            }
            QCheckBox::indicator:checked {
                background-color: #00D4FF;
                border: 1px solid #00D4FF;
            }
            QLabel {
                color: #E0E0E0;
                font-weight: 400;
            }
            QDateTimeEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 5px;
                color: #E0E0E0;
            }
        """
        self.setStyleSheet(dark_glass_style)

        # Główny layout w stylu Bento Grid
        main_grid = QtWidgets.QGridLayout()
        main_grid.setSpacing(15)

        # Nagłówek z nazwą aplikacji
        header_label = QtWidgets.QLabel("Crypto Trading Dashboard")
        header_label.setStyleSheet("font-size: 24px; font-weight: 600; color: #00D4FF;")
        header_label.setAlignment(QtCore.Qt.AlignCenter)
        main_grid.addWidget(header_label, 0, 0, 1, 3)

        # Przyciski start/stop (małe pudełko Bento)
        control_box = QtWidgets.QGroupBox("Control Panel")
        control_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Bot")
        self.stop_button = QtWidgets.QPushButton("Stop Bot")
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_box.setLayout(control_layout)
        main_grid.addWidget(control_box, 1, 0, 1, 1)

        # Wybór symbolu i timeframe (małe pudełko Bento)
        settings_box = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QVBoxLayout()
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.addItems(self.bot.data_provider.fetch_available_symbols())
        self.timeframe_combo = QtWidgets.QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        settings_layout.addWidget(QtWidgets.QLabel("Symbol:"))
        settings_layout.addWidget(self.symbol_combo)
        settings_layout.addWidget(QtWidgets.QLabel("Timeframe:"))
        settings_layout.addWidget(self.timeframe_combo)
        settings_box.setLayout(settings_layout)
        main_grid.addWidget(settings_box, 1, 1, 1, 1)

        # Checkbox do wykresów (małe pudełko Bento)
        plot_box = QtWidgets.QGroupBox("Display")
        plot_layout = QtWidgets.QVBoxLayout()
        self.plot_checkbox = QtWidgets.QCheckBox("Show Plots")
        plot_layout.addWidget(self.plot_checkbox)
        plot_box.setLayout(plot_layout)
        main_grid.addWidget(plot_box, 1, 2, 1, 1)

        # Strategie (większe pudełko Bento)
        self.strategy_group = QtWidgets.QGroupBox("Active Strategies")
        strategy_layout = QtWidgets.QVBoxLayout()
        self.strategy_checkboxes = {
            'SMA': QtWidgets.QCheckBox("SMA"),
            'EMA': QtWidgets.QCheckBox("EMA"),
            'RSI': QtWidgets.QCheckBox("RSI"),
            'MACD': QtWidgets.QCheckBox("MACD"),
            'STOCH': QtWidgets.QCheckBox("Stochastic Oscillator"),
            'ICHIMOKU': QtWidgets.QCheckBox("Ichimoku Cloud"),
            'ML': QtWidgets.QCheckBox("Machine Learning")
        }
        for checkbox in self.strategy_checkboxes.values():
            strategy_layout.addWidget(checkbox)
        self.strategy_checkboxes['SMA'].setChecked(True)
        self.strategy_group.setLayout(strategy_layout)
        main_grid.addWidget(self.strategy_group, 2, 0, 2, 1)

        # Logi (większe pudełko Bento)
        log_box = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout()
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_box.setLayout(log_layout)
        main_grid.addWidget(log_box, 2, 1, 2, 2)

        # Wykres świecowy (największe pudełko Bento)
        main_grid.addWidget(self.candlestick_canvas, 4, 0, 2, 3)

        # Status i kapitał (małe pudełko Bento)
        status_box = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout()
        self.capital_label = QtWidgets.QLabel(f"Current Capital: {self.bot.trade_manager.capital:.2f}")
        self.status_label = QtWidgets.QLabel("Bot Status: Stopped")
        status_layout.addWidget(self.capital_label)
        status_layout.addWidget(self.status_label)
        status_box.setLayout(status_layout)
        main_grid.addWidget(status_box, 6, 0, 1, 1)

        # Przyciski akcji (małe pudełko Bento)
        action_box = QtWidgets.QGroupBox("Actions")
        action_layout = QtWidgets.QVBoxLayout()
        self.history_button = QtWidgets.QPushButton("Transaction History")
        self.backtest_button = QtWidgets.QPushButton("Run Backtest")
        self.optimize_button = QtWidgets.QPushButton("Optimize Parameters")
        self.train_ml_button = QtWidgets.QPushButton("Train XGBoost Model")
        action_layout.addWidget(self.history_button)
        action_layout.addWidget(self.backtest_button)
        action_layout.addWidget(self.optimize_button)
        action_layout.addWidget(self.train_ml_button)
        action_box.setLayout(action_layout)
        main_grid.addWidget(action_box, 6, 1, 1, 1)

        # Daty backtestu (małe pudełko Bento)
        date_box = QtWidgets.QGroupBox("Backtest Range")
        date_layout = QtWidgets.QVBoxLayout()  # Poprawiona literówka
        self.start_date_edit = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime().addMonths(-6))
        self.end_date_edit = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        date_layout.addWidget(QtWidgets.QLabel("Start Date:"))
        date_layout.addWidget(self.start_date_edit)
        date_layout.addWidget(QtWidgets.QLabel("End Date:"))
        date_layout.addWidget(self.end_date_edit)
        date_box.setLayout(date_layout)
        main_grid.addWidget(date_box, 6, 2, 1, 1)

        # Ustawienie layoutu głównego
        layout.addLayout(main_grid)
        self.setLayout(layout)

        # Połączenia sygnałów
        self.start_button.clicked.connect(self.bot.start)
        self.stop_button.clicked.connect(self.bot.stop)
        self.plot_checkbox.stateChanged.connect(self.toggle_plots)
        self.symbol_combo.currentTextChanged.connect(self.bot.set_symbol)
        self.timeframe_combo.currentTextChanged.connect(self.bot.update_timeframe)
        for checkbox in self.strategy_checkboxes.values():
            checkbox.stateChanged.connect(self.update_strategies)
        self.history_button.clicked.connect(self.view_transaction_history)
        self.backtest_button.clicked.connect(self.run_backtest)
        self.optimize_button.clicked.connect(self.optimize_parameters)
        self.train_ml_button.clicked.connect(self.train_ml_model)

        self.update_strategies()

        # Timer do aktualizacji wykresu
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_candlestick_plot)
        self.timer.start(60000)

    def toggle_plots(self):
        self.show_plots = self.plot_checkbox.isChecked()
        if self.show_plots:
            self.plot_window = PlotWindow(self.bot)
            self.plot_window.show()
        else:
            if hasattr(self, 'plot_window'):
                self.plot_window.close()

    def update_strategies(self):
        active_strategies = [name for name, checkbox in self.strategy_checkboxes.items() if checkbox.isChecked()]
        self.bot.strategy_executor.active_strategies = active_strategies
        self.bot.config['active_strategies'] = active_strategies
        self.log_text.append(f"Active strategies: {', '.join(active_strategies)}")

    def view_transaction_history(self):
        try:
            with open('transactions.csv', 'r') as file:
                history_window = QtWidgets.QWidget()
                history_window.setWindowTitle("Transaction History")
                history_window.setStyleSheet(self.styleSheet())
                history_text = QtWidgets.QTextEdit(file.read())
                history_text.setReadOnly(True)
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(history_text)
                history_window.setLayout(layout)
                history_window.show()
        except FileNotFoundError:
            self.log_text.append("No transaction history found.")

    @QtCore.pyqtSlot(str)
    def update_log(self, message):
        self.log_text.append(message)
        self.capital_label.setText(f"Current Capital: {self.bot.trade_manager.capital:.2f}")

    @QtCore.pyqtSlot(pd.DataFrame)
    def update_data(self, df):
        pass

    def update_status(self):
        self.status_label.setText("Bot Status: Running" if self.bot.bot_running else "Bot Status: Stopped")

    def run_backtest(self):
        start_date = self.start_date_edit.dateTime().toPyDateTime()
        end_date = self.end_date_edit.dateTime().toPyDateTime()
        self.bot.run_backtest(start_date, end_date)

    def optimize_parameters(self):
        start_date = self.start_date_edit.dateTime().toPyDateTime()
        end_date = self.end_date_edit.dateTime().toPyDateTime()
        self.bot.optimize_parameters(start_date, end_date)

    def train_ml_model(self):
        start_date = self.start_date_edit.dateTime().toPyDateTime()
        end_date = self.end_date_edit.dateTime().toPyDateTime()
        self.bot.train_ml_model(start_date, end_date)

    def update_candlestick_plot(self):
        symbol = self.bot.selected_symbol
        df = self.bot.data_provider.get_data(symbol).tail(300)
        if df.empty:
            logging.debug("No data to plot for symbol: {}".format(symbol))
            return

        if 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
            logging.debug("Calculating indicators and signals for symbol: {}".format(symbol))
            df = self.bot.indicator_calculator.calculate_all(df)
            df = self.bot.strategy_executor.calculate_signals(df, self.bot.trade_manager.in_position.get(symbol, False))

        df = df.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df.set_index('Date', inplace=True)
        logging.debug("Data prepared for plotting: shape={}, columns={}".format(df.shape, df.columns.tolist()))

        buys = pd.Series(index=df.index, dtype=float)
        sells = pd.Series(index=df.index, dtype=float)
        if 'buy_signal' in df.columns and 'sell_signal' in df.columns:
            buy_mask = df['buy_signal'] == True
            sell_mask = df['sell_signal'] == True
            buys[buy_mask] = df['Close'][buy_mask]
            sells[sell_mask] = df['Close'][sell_mask]
            logging.debug("Buy signals: {} points, Sell signals: {} points".format(buy_mask.sum(), sell_mask.sum()))

        # Ręczne rysowanie wykresu świecowego
        try:
            self.candlestick_fig.clear()
            ax = self.candlestick_fig.add_subplot(111)
            logging.debug("Cleared candlestick figure and added new axis, plotting new data")

            # Konwersja indeksu dat na format numeryczny dla Matplotlib
            dates = mdates.date2num(df.index.to_pydatetime())

            # Rysowanie świec
            width = 0.4  # Szerokość świecy
            for i in range(len(df)):
                open_price = df['Open'].iloc[i]
                close_price = df['Close'].iloc[i]
                high_price = df['High'].iloc[i]
                low_price = df['Low'].iloc[i]
                date = dates[i]
                color = '#00FF00' if close_price >= open_price else '#FF0000'  # Zielony dla wzrostu, czerwony dla spadku
                # Świeca (korpus)
                ax.plot([date, date], [open_price, close_price], color=color, linewidth=4, solid_capstyle='round')
                # Knoty (cienie)
                ax.plot([date, date], [low_price, high_price], color=color, linewidth=1)

            # Dodanie sygnałów kupna i sprzedaży
            if buys.notna().any():
                ax.scatter(dates[buys.notna()], buys.dropna(), marker='^', color='#00D4FF', s=100, label='Buy', zorder=5)
                logging.debug("Added buy signals to plot: {} points".format(buys.notna().sum()))
            if sells.notna().any():
                ax.scatter(dates[sells.notna()], sells.dropna(), marker='v', color='#FF3366', s=100, label='Sell', zorder=5)
                logging.debug("Added sell signals to plot: {} points".format(sells.notna().sum()))

            # Ustawienia osi
            ax.set_facecolor((0.06, 0.06, 0.11, 0.9))  # #0F0F1B z opacity 0.9
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.candlestick_fig.autofmt_xdate()  # Rotacja dat dla lepszej czytelności

            self.candlestick_canvas.draw()
            logging.debug("Candlestick plot updated and drawn in GUI")
        except Exception as e:
            logging.error(f"Failed to plot candlestick chart: {str(e)}")
            self.log_text.append(f"Błąd rysowania wykresu: {str(e)}")

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.setWindowTitle("Indicator Plots")
        self.setGeometry(100, 100, 1200, 800)
        self.canvas = FigureCanvas(self.bot.fig)
        self.setCentralWidget(self.canvas)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(60000)

    def update_plot(self):
        self.bot.plot_data()
        self.canvas.draw()