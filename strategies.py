"""
Simple Trading Strategies for Backtesting

Strategies:
- BuyAndHold: Benchmark - buy and hold entire period
- SMACrossover: Buy when fast SMA crosses above slow SMA
- RSIMeanReversion: Buy oversold, sell overbought
- BollingerBands: Buy at lower band, sell at upper band
- MACDStrategy: Trade based on MACD crossovers
- MomentumStrategy: Buy strong momentum, sell weak
"""

import backtrader as bt


class BuyAndHold(bt.Strategy):
    """Buy and Hold Benchmark - invests all capital on day 1"""

    def nextstart(self):
        cash = self.broker.getcash()
        price = self.data.close[0]
        size = int(cash * 0.99 / price)
        self.buy(size=size)

    def next(self):
        pass


class SMACrossover(bt.Strategy):
    """SMA Crossover - buy when fast crosses above slow"""

    params = dict(
        fast=10,
        slow=30,
    )

    def __init__(self):
        self.fast_sma = bt.ind.SMA(period=self.p.fast)
        self.slow_sma = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(self.fast_sma, self.slow_sma)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class RSIMeanReversion(bt.Strategy):
    """RSI Mean Reversion - buy oversold (<30), sell overbought (>70)"""

    params = dict(
        period=14,
        oversold=30,
        overbought=70,
    )

    def __init__(self):
        self.rsi = bt.ind.RSI(period=self.p.period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.p.oversold:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.rsi > self.p.overbought:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class BollingerBands(bt.Strategy):
    """Bollinger Bands - buy at lower band, sell at upper band"""

    params = dict(
        period=20,
        devfactor=2.0,
    )

    def __init__(self):
        self.boll = bt.ind.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] < self.boll.lines.bot[0]:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.data.close[0] > self.boll.lines.top[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MACDStrategy(bt.Strategy):
    """MACD Strategy - buy on MACD crossover, sell on crossunder"""

    params = dict(
        fast=12,
        slow=26,
        signal=9,
    )

    def __init__(self):
        self.macd = bt.ind.MACD(
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal
        )
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MomentumStrategy(bt.Strategy):
    """Momentum - buy when price > X-day high, sell when price < X-day low"""

    params = dict(
        period=20,
    )

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.ind.Lowest(self.data.low, period=self.p.period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.highest[-1]:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.data.close[0] < self.lowest[-1]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class TripleMA(bt.Strategy):
    """Triple Moving Average - uses 3 MAs for trend confirmation"""

    params = dict(
        fast=5,
        medium=20,
        slow=50,
    )

    def __init__(self):
        self.fast = bt.ind.SMA(period=self.p.fast)
        self.medium = bt.ind.SMA(period=self.p.medium)
        self.slow = bt.ind.SMA(period=self.p.slow)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy when fast > medium > slow (uptrend)
            if self.fast[0] > self.medium[0] > self.slow[0]:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell when fast < medium (trend weakening)
            if self.fast[0] < self.medium[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# Registry of all strategies
STRATEGIES = {
    'buyhold': BuyAndHold,
    'sma': SMACrossover,
    'rsi': RSIMeanReversion,
    'bollinger': BollingerBands,
    'macd': MACDStrategy,
    'momentum': MomentumStrategy,
    'triplema': TripleMA,
}
