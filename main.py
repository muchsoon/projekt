import sys
from PyQt5.QtWidgets import QApplication
from crypto_trading_bot import CryptoTradingBot

if __name__ == "__main__":
    app = QApplication(sys.argv)
    bot = CryptoTradingBot('config.yaml')
    bot.gui.show()
    sys.exit(app.exec_())