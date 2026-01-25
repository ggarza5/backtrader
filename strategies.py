"""
Simple Trading Strategies for Backtesting

Strategies:
- BuyAndHold: Benchmark - buy and hold entire period
- SMACrossover: Buy when fast SMA crosses above slow SMA
- RSIMeanReversion: Buy oversold, sell overbought
- BollingerBands: Buy at lower band, sell at upper band
- MACDStrategy: Trade based on MACD crossovers
- MomentumStrategy: Buy strong momentum, sell weak
- SupportResistanceBreakout: Trade breakouts from S/R levels
- IchimokuStrategy: Trade based on Ichimoku Cloud signals
"""

import backtrader as bt
import numpy as np


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


class SupportResistanceBreakout(bt.Strategy):
    """Support/Resistance Breakout - buy on resistance break, sell on support break"""

    params = dict(
        lookback=20,       # Period to find S/R levels
        stop_loss=0.05,    # 5% stop loss
    )

    def __init__(self):
        self.resistance = bt.ind.Highest(self.data.high, period=self.p.lookback)
        self.support = bt.ind.Lowest(self.data.low, period=self.p.lookback)
        self.order = None
        self.entry_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy on breakout: close > previous resistance
            if self.data.close[0] > self.resistance[-1]:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = self.data.close[0]
        else:
            # Sell on breakdown below support
            if self.data.close[0] < self.support[-1]:
                self.order = self.close()
            # Or stop loss triggered
            elif self.entry_price and self.data.close[0] < self.entry_price * (1 - self.p.stop_loss):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class IchimokuStrategy(bt.Strategy):
    """Ichimoku Cloud - buy above cloud with bullish cross, sell below cloud"""

    params = dict(
        tenkan=9,       # Conversion line (Tenkan-sen)
        kijun=26,       # Base line (Kijun-sen)
        senkou=52,      # Leading span B period
        senkou_lead=26, # Forward displacement
    )

    def __init__(self):
        self.ichimoku = bt.ind.Ichimoku(
            tenkan=self.p.tenkan,
            kijun=self.p.kijun,
            senkou=self.p.senkou,
            senkou_lead=self.p.senkou_lead,
        )
        # Tenkan/Kijun crossover
        self.cross = bt.ind.CrossOver(self.ichimoku.tenkan_sen, self.ichimoku.kijun_sen)
        self.order = None

    def next(self):
        if self.order:
            return

        # Get cloud boundaries (Senkou Span A and B)
        span_a = self.ichimoku.senkou_span_a[0]
        span_b = self.ichimoku.senkou_span_b[0]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        price = self.data.close[0]

        if not self.position:
            # Buy signal: price above cloud + bullish TK cross
            if price > cloud_top and self.cross > 0:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / price)
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell signal: price enters or goes below cloud, or bearish TK cross
            if price < cloud_bottom or self.cross < 0:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class IchimokuCloudOnly(bt.Strategy):
    """Ichimoku Cloud Only - simpler version, just trade based on cloud position"""

    params = dict(
        tenkan=9,
        kijun=26,
        senkou=52,
        senkou_lead=26,
    )

    def __init__(self):
        self.ichimoku = bt.ind.Ichimoku(
            tenkan=self.p.tenkan,
            kijun=self.p.kijun,
            senkou=self.p.senkou,
            senkou_lead=self.p.senkou_lead,
        )
        self.order = None

    def next(self):
        if self.order:
            return

        span_a = self.ichimoku.senkou_span_a[0]
        span_b = self.ichimoku.senkou_span_b[0]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        price = self.data.close[0]

        if not self.position:
            # Buy when price closes above cloud
            if price > cloud_top:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / price)
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell when price closes below cloud
            if price < cloud_bottom:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class BreakoutVolume(bt.Strategy):
    """Volume Breakout - breakout with volume confirmation"""

    params = dict(
        price_period=20,    # Period for price breakout
        volume_period=20,   # Period for volume average
        volume_mult=1.5,    # Volume must be X times average
    )

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.p.price_period)
        self.lowest = bt.ind.Lowest(self.data.low, period=self.p.price_period)
        self.vol_sma = bt.ind.SMA(self.data.volume, period=self.p.volume_period)
        self.order = None

    def next(self):
        if self.order:
            return

        # Check volume confirmation
        high_volume = self.data.volume[0] > self.vol_sma[0] * self.p.volume_mult

        if not self.position:
            # Buy on breakout above previous high with volume
            if self.data.close[0] > self.highest[-1] and high_volume:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell on breakdown below support (with or without volume)
            if self.data.close[0] < self.lowest[-1]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# =============================================================================
# STATE OF THE ART STRATEGIES
# =============================================================================

class TurtleTrading(bt.Strategy):
    """Turtle Trading System - legendary trend following with pyramiding"""

    params = dict(
        entry_period=20,     # Donchian channel for entry
        exit_period=10,      # Donchian channel for exit
        atr_period=20,       # ATR for position sizing
        risk_pct=0.01,       # Risk 1% of equity per ATR unit
        max_units=4,         # Max pyramid units
    )

    def __init__(self):
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.order = None
        self.units = 0
        self.entry_price = None

    def next(self):
        if self.order:
            return

        atr_val = self.atr[0]
        if atr_val <= 0:
            return

        if not self.position:
            # Entry: breakout above 20-day high
            if self.data.close[0] > self.entry_high[-1]:
                # Position sizing: risk 1% per unit, stop at 2 ATR
                equity = self.broker.getvalue()
                dollar_vol = atr_val  # 1 ATR move
                unit_size = int((equity * self.p.risk_pct) / dollar_vol)
                # Cap at what we can afford
                max_affordable = int(self.broker.getcash() * 0.25 / self.data.close[0])
                unit_size = min(unit_size, max_affordable)
                if unit_size > 0:
                    self.order = self.buy(size=unit_size)
                    self.units = 1
                    self.entry_price = self.data.close[0]
        else:
            # Pyramiding: add units if price moves 0.5 ATR in our favor
            if self.units < self.p.max_units:
                if self.data.close[0] > self.entry_price + (0.5 * atr_val * self.units):
                    equity = self.broker.getvalue()
                    unit_size = int((equity * self.p.risk_pct) / atr_val)
                    max_affordable = int(self.broker.getcash() * 0.25 / self.data.close[0])
                    unit_size = min(unit_size, max_affordable)
                    if unit_size > 0:
                        self.order = self.buy(size=unit_size)
                        self.units += 1

            # Exit: break below 10-day low
            if self.data.close[0] < self.exit_low[-1]:
                self.order = self.close()
                self.units = 0
                self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class KeltnerChannel(bt.Strategy):
    """Keltner Channel Breakout - volatility-based trend following"""

    params = dict(
        ema_period=20,
        atr_period=10,
        atr_mult=2.0,
    )

    def __init__(self):
        self.ema = bt.ind.EMA(period=self.p.ema_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return

        upper = self.ema[0] + self.atr[0] * self.p.atr_mult
        lower = self.ema[0] - self.atr[0] * self.p.atr_mult

        if not self.position:
            # Buy on upper band breakout
            if self.data.close[0] > upper:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell on lower band breakdown or EMA cross
            if self.data.close[0] < lower or self.data.close[0] < self.ema[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class DualMomentum(bt.Strategy):
    """Dual Momentum (Gary Antonacci) - absolute + relative momentum"""

    params = dict(
        momentum_period=252,  # 12-month lookback
        sma_period=200,       # Trend filter
    )

    def __init__(self):
        self.returns = bt.ind.PercentChange(self.data.close, period=self.p.momentum_period)
        self.sma = bt.ind.SMA(period=self.p.sma_period)
        self.order = None

    def next(self):
        if self.order:
            return

        # Absolute momentum: positive returns over lookback
        abs_momentum = self.returns[0] > 0
        # Trend filter: price above 200 SMA
        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            # Buy only if both absolute momentum and trend are positive
            if abs_momentum and trend_up:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell if either condition fails
            if not abs_momentum or not trend_up:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MeanReversionZScore(bt.Strategy):
    """Z-Score Mean Reversion - statistical mean reversion"""

    params = dict(
        period=20,
        entry_z=-2.0,   # Buy when Z < -2
        exit_z=0.0,     # Sell when Z returns to 0
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.period)
        self.std = bt.ind.StdDev(period=self.p.period)
        self.order = None

    def next(self):
        if self.order:
            return

        # Calculate Z-score
        if self.std[0] > 0:
            zscore = (self.data.close[0] - self.sma[0]) / self.std[0]
        else:
            return

        if not self.position:
            # Buy when extremely oversold (Z < -2)
            if zscore < self.p.entry_z:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell when mean reverts (Z >= 0) or stop loss (Z < -3)
            if zscore >= self.p.exit_z or zscore < -3:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class AdaptiveTrendFollowing(bt.Strategy):
    """Adaptive Trend Following with ATR-based stops and position sizing"""

    params = dict(
        fast=10,
        slow=30,
        atr_period=14,
        atr_stop_mult=3.0,    # Trailing stop = 3x ATR
        risk_pct=0.02,        # Risk 2% per trade
    )

    def __init__(self):
        self.fast_ma = bt.ind.EMA(period=self.p.fast)
        self.slow_ma = bt.ind.EMA(period=self.p.slow)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
        self.stop_price = None
        self.highest_since_entry = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                # Position sizing based on ATR
                risk_amount = self.broker.getvalue() * self.p.risk_pct
                stop_distance = self.atr[0] * self.p.atr_stop_mult
                if stop_distance > 0:
                    size = int(risk_amount / stop_distance)
                    if size > 0:
                        max_size = int(self.broker.getcash() * 0.99 / self.data.close[0])
                        size = min(size, max_size)
                        if size > 0:
                            self.order = self.buy(size=size)
                            self.stop_price = self.data.close[0] - stop_distance
                            self.highest_since_entry = self.data.close[0]
        else:
            # Update trailing stop
            if self.data.close[0] > self.highest_since_entry:
                self.highest_since_entry = self.data.close[0]
                self.stop_price = self.highest_since_entry - (self.atr[0] * self.p.atr_stop_mult)

            # Exit on trailing stop or bearish crossover
            if self.data.close[0] < self.stop_price or self.crossover < 0:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class VolatilityRegime(bt.Strategy):
    """Volatility Regime - adapts strategy based on market volatility"""

    params = dict(
        vol_period=20,
        vol_threshold=1.5,  # High vol = current vol > 1.5x average
        trend_period=50,
    )

    def __init__(self):
        # Volatility measures
        self.atr = bt.ind.ATR(period=self.p.vol_period)
        self.atr_sma = bt.ind.SMA(self.atr, period=self.p.vol_period * 5)

        # Trend measures
        self.sma = bt.ind.SMA(period=self.p.trend_period)
        self.rsi = bt.ind.RSI(period=14)

        self.order = None

    def next(self):
        if self.order:
            return

        # Determine volatility regime
        high_vol = self.atr[0] > self.atr_sma[0] * self.p.vol_threshold
        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            if high_vol:
                # High vol: mean reversion (buy oversold)
                if self.rsi[0] < 30:
                    cash = self.broker.getcash()
                    size = int(cash * 0.5 / self.data.close[0])  # Half size in high vol
                    if size > 0:
                        self.order = self.buy(size=size)
            else:
                # Low vol: trend following
                if trend_up and self.rsi[0] > 50:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            if high_vol:
                # High vol: exit on RSI overbought
                if self.rsi[0] > 70:
                    self.order = self.close()
            else:
                # Low vol: exit on trend break
                if not trend_up:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class KAMA_Strategy(bt.Strategy):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""

    params = dict(
        period=10,
        fast=2,
        slow=30,
    )

    def __init__(self):
        self.kama = bt.ind.KAMA(
            period=self.p.period,
            fast=self.p.fast,
            slow=self.p.slow
        )
        self.order = None
        self.prev_kama = None

    def next(self):
        if self.order:
            return

        if self.prev_kama is None:
            self.prev_kama = self.kama[0]
            return

        # KAMA slope
        kama_rising = self.kama[0] > self.prev_kama
        price_above = self.data.close[0] > self.kama[0]

        if not self.position:
            # Buy when KAMA rising and price above KAMA
            if kama_rising and price_above:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell when KAMA falling or price below KAMA
            if not kama_rising or not price_above:
                self.order = self.close()

        self.prev_kama = self.kama[0]

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class SuperTrend(bt.Strategy):
    """SuperTrend - popular ATR-based trend indicator"""

    params = dict(
        period=10,
        multiplier=3.0,
    )

    def __init__(self):
        self.atr = bt.ind.ATR(period=self.p.period)
        self.order = None
        self.supertrend = None
        self.direction = 0  # 1 = up, -1 = down

    def next(self):
        if self.order:
            return

        hl2 = (self.data.high[0] + self.data.low[0]) / 2
        atr_val = self.atr[0] * self.p.multiplier

        upper_band = hl2 + atr_val
        lower_band = hl2 - atr_val

        if self.supertrend is None:
            self.supertrend = upper_band
            self.direction = -1
            return

        # Update SuperTrend
        if self.direction == 1:
            # Currently in uptrend
            self.supertrend = max(lower_band, self.supertrend)
            if self.data.close[0] < self.supertrend:
                self.direction = -1
                self.supertrend = upper_band
        else:
            # Currently in downtrend
            self.supertrend = min(upper_band, self.supertrend)
            if self.data.close[0] > self.supertrend:
                self.direction = 1
                self.supertrend = lower_band

        if not self.position:
            if self.direction == 1:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.direction == -1:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class ChandelierExit(bt.Strategy):
    """Chandelier Exit - ATR-based trailing stop system"""

    params = dict(
        atr_period=22,
        atr_mult=3.0,
        lookback=22,
    )

    def __init__(self):
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.highest = bt.ind.Highest(self.data.high, period=self.p.lookback)
        self.sma = bt.ind.SMA(period=50)
        self.order = None

    def next(self):
        if self.order:
            return

        chandelier_stop = self.highest[0] - self.atr[0] * self.p.atr_mult

        if not self.position:
            # Enter on trend (price above 50 SMA)
            if self.data.close[0] > self.sma[0]:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit when price crosses below chandelier stop
            if self.data.close[0] < chandelier_stop:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class WilliamsR(bt.Strategy):
    """Williams %R with trend filter - momentum oscillator"""

    params = dict(
        period=14,
        overbought=-20,
        oversold=-80,
        trend_period=50,
    )

    def __init__(self):
        self.willr = bt.ind.WilliamsR(period=self.p.period)
        self.sma = bt.ind.SMA(period=self.p.trend_period)
        self.order = None

    def next(self):
        if self.order:
            return

        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            # Buy: oversold + uptrend
            if self.willr[0] < self.p.oversold and trend_up:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell: overbought or trend break
            if self.willr[0] > self.p.overbought or not trend_up:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class CombinedMomentumMeanReversion(bt.Strategy):
    """Combined Strategy - momentum in trends, mean reversion in ranges"""

    params = dict(
        adx_period=14,
        adx_threshold=25,    # ADX > 25 = trending
        rsi_period=14,
        ma_fast=10,
        ma_slow=30,
    )

    def __init__(self):
        self.adx = bt.ind.ADX(period=self.p.adx_period)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
        self.fast_ma = bt.ind.EMA(period=self.p.ma_fast)
        self.slow_ma = bt.ind.EMA(period=self.p.ma_slow)
        self.order = None

    def next(self):
        if self.order:
            return

        trending = self.adx[0] > self.p.adx_threshold
        ma_bullish = self.fast_ma[0] > self.slow_ma[0]

        if not self.position:
            if trending:
                # Trending: momentum strategy (buy on MA crossover)
                if ma_bullish:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
            else:
                # Ranging: mean reversion (buy oversold)
                if self.rsi[0] < 30:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            if trending:
                # Trending: exit on MA crossunder
                if not ma_bullish:
                    self.order = self.close()
            else:
                # Ranging: exit on overbought
                if self.rsi[0] > 70:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# =============================================================================
# MONTE CARLO STRATEGIES
# =============================================================================

class MonteCarloTrendSignificance(bt.Strategy):
    """Monte Carlo Trend Test - only trade when trend is statistically significant

    Uses bootstrap resampling to test if observed trend is significant vs random.
    Only enters when p-value < threshold (trend unlikely to be random noise).
    """

    params = dict(
        lookback=60,          # Period for returns
        n_simulations=500,    # Number of MC simulations
        significance=0.05,    # p-value threshold (5%)
        trend_period=20,      # SMA for trend direction
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.trend_period)
        self.order = None
        self.returns_buffer = []

    def next(self):
        if self.order:
            return

        # Collect returns
        if len(self.data) > 1:
            ret = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.returns_buffer.append(ret)
            if len(self.returns_buffer) > self.p.lookback:
                self.returns_buffer.pop(0)

        if len(self.returns_buffer) < self.p.lookback:
            return

        returns = np.array(self.returns_buffer)
        observed_trend = np.sum(returns)  # Cumulative return

        # Monte Carlo: shuffle returns and compute cumulative
        mc_trends = []
        for _ in range(self.p.n_simulations):
            shuffled = np.random.permutation(returns)
            mc_trends.append(np.sum(shuffled))

        mc_trends = np.array(mc_trends)

        # Calculate p-value (proportion of simulations with trend >= observed)
        if observed_trend > 0:
            p_value = np.mean(mc_trends >= observed_trend)
        else:
            p_value = np.mean(mc_trends <= observed_trend)

        trend_significant = p_value < self.p.significance
        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            # Only buy if uptrend is statistically significant
            if trend_significant and trend_up and observed_trend > 0:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit if trend is no longer significant or reversed
            if not trend_significant or not trend_up:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MonteCarloVaR(bt.Strategy):
    """Monte Carlo VaR Position Sizing - size positions based on simulated risk

    Uses MC simulation to estimate Value at Risk, then sizes positions
    so max loss doesn't exceed risk tolerance.
    """

    params = dict(
        lookback=60,          # Historical period for returns
        n_simulations=1000,   # Number of MC paths
        holding_period=5,     # Days to simulate forward
        var_percentile=5,     # VaR confidence (5% = 95% confidence)
        max_risk_pct=0.02,    # Max 2% portfolio at risk
        trend_period=50,      # Trend filter
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.trend_period)
        self.order = None
        self.returns_buffer = []

    def calculate_var(self, returns):
        """Monte Carlo VaR estimation"""
        if len(returns) < 20:
            return None

        mu = np.mean(returns)
        sigma = np.std(returns)

        if sigma == 0:
            return None

        # Simulate future paths
        simulated_returns = []
        for _ in range(self.p.n_simulations):
            # Random walk with drift
            path_return = 0
            for _ in range(self.p.holding_period):
                daily_return = np.random.normal(mu, sigma)
                path_return += daily_return
            simulated_returns.append(path_return)

        # VaR is the percentile loss
        var = np.percentile(simulated_returns, self.p.var_percentile)
        return var

    def next(self):
        if self.order:
            return

        # Collect returns
        if len(self.data) > 1:
            ret = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.returns_buffer.append(ret)
            if len(self.returns_buffer) > self.p.lookback:
                self.returns_buffer.pop(0)

        if len(self.returns_buffer) < 30:
            return

        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            if trend_up:
                var = self.calculate_var(np.array(self.returns_buffer))
                if var is not None and var < 0:
                    # Position size based on VaR
                    # If VaR is -5%, and we want max 2% risk, size = 2/5 = 40%
                    risk_adjusted_size = self.p.max_risk_pct / abs(var)
                    risk_adjusted_size = min(risk_adjusted_size, 0.99)  # Cap at 99%

                    cash = self.broker.getcash()
                    size = int(cash * risk_adjusted_size / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            # Exit on trend break
            if not trend_up:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MonteCarloKelly(bt.Strategy):
    """Monte Carlo Kelly Criterion - optimal position sizing with uncertainty

    Uses MC to estimate win rate and payoff ratio with confidence intervals,
    then applies Kelly with a safety factor.
    """

    params = dict(
        lookback=100,         # Trades to analyze
        n_bootstrap=500,      # Bootstrap samples
        kelly_fraction=0.5,   # Half-Kelly for safety
        trend_period=20,
    )

    def __init__(self):
        self.fast_ma = bt.ind.EMA(period=10)
        self.slow_ma = bt.ind.EMA(period=self.p.trend_period)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
        self.trade_results = []  # Store trade returns
        self.entry_price = None

    def calculate_kelly(self):
        """Bootstrap Kelly criterion estimation"""
        if len(self.trade_results) < 10:
            return 0.25  # Default conservative

        results = np.array(self.trade_results[-self.p.lookback:])

        kelly_estimates = []
        for _ in range(self.p.n_bootstrap):
            # Bootstrap sample
            sample = np.random.choice(results, size=len(results), replace=True)

            wins = sample[sample > 0]
            losses = sample[sample < 0]

            if len(wins) == 0 or len(losses) == 0:
                continue

            win_rate = len(wins) / len(sample)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))

            if avg_loss > 0:
                payoff_ratio = avg_win / avg_loss
                # Kelly formula: f = W - (1-W)/R
                kelly = win_rate - (1 - win_rate) / payoff_ratio
                kelly_estimates.append(kelly)

        if len(kelly_estimates) == 0:
            return 0.25

        # Use lower confidence bound (conservative)
        kelly = np.percentile(kelly_estimates, 25)  # 25th percentile
        kelly = max(0, min(kelly, 1)) * self.p.kelly_fraction

        return kelly

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                kelly = self.calculate_kelly()
                if kelly > 0.05:  # Minimum 5% to trade
                    cash = self.broker.getcash()
                    size = int(cash * kelly / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.entry_price = self.data.close[0]
        else:
            if self.crossover < 0:
                # Record trade result
                if self.entry_price:
                    trade_return = (self.data.close[0] - self.entry_price) / self.entry_price
                    self.trade_results.append(trade_return)
                self.order = self.close()
                self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MonteCarloRegime(bt.Strategy):
    """Monte Carlo Regime Detection - detects trending vs mean-reverting regimes

    Uses Hurst exponent estimation via MC to determine market regime,
    then applies appropriate strategy.
    """

    params = dict(
        lookback=100,
        n_lags=20,
        trend_threshold=0.55,   # H > 0.55 = trending
        revert_threshold=0.45,  # H < 0.45 = mean reverting
    )

    def __init__(self):
        # Trend following indicators
        self.fast_ma = bt.ind.EMA(period=10)
        self.slow_ma = bt.ind.EMA(period=30)

        # Mean reversion indicators
        self.rsi = bt.ind.RSI(period=14)
        self.boll = bt.ind.BollingerBands(period=20)

        self.order = None
        self.prices = []
        self.current_regime = 'unknown'

    def estimate_hurst(self, prices):
        """Estimate Hurst exponent using R/S analysis"""
        if len(prices) < 50:
            return 0.5

        prices = np.array(prices)
        n = len(prices)

        # Calculate returns
        returns = np.diff(np.log(prices))

        # R/S analysis for multiple lags
        lags = range(10, min(n // 2, self.p.n_lags + 10))
        rs_values = []
        lag_values = []

        for lag in lags:
            rs_list = []
            for start in range(0, len(returns) - lag, lag):
                chunk = returns[start:start + lag]
                if len(chunk) < lag:
                    continue

                mean_chunk = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean_chunk)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(chunk)

                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)

        if len(rs_values) < 3:
            return 0.5

        # Linear regression in log space
        log_lags = np.log(lag_values)
        log_rs = np.log(rs_values)

        # Hurst exponent is slope
        try:
            slope = np.polyfit(log_lags, log_rs, 1)[0]
            return np.clip(slope, 0, 1)
        except:
            return 0.5

    def next(self):
        if self.order:
            return

        # Collect prices
        self.prices.append(self.data.close[0])
        if len(self.prices) > self.p.lookback:
            self.prices.pop(0)

        if len(self.prices) < 50:
            return

        # Estimate Hurst exponent
        hurst = self.estimate_hurst(self.prices)

        # Determine regime
        if hurst > self.p.trend_threshold:
            self.current_regime = 'trending'
        elif hurst < self.p.revert_threshold:
            self.current_regime = 'mean_reverting'
        else:
            self.current_regime = 'random'

        if not self.position:
            if self.current_regime == 'trending':
                # Trend following: buy on MA crossover
                if self.fast_ma[0] > self.slow_ma[0]:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)

            elif self.current_regime == 'mean_reverting':
                # Mean reversion: buy at lower Bollinger
                if self.data.close[0] < self.boll.lines.bot[0]:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            if self.current_regime == 'trending':
                # Exit on MA cross down
                if self.fast_ma[0] < self.slow_ma[0]:
                    self.order = self.close()

            elif self.current_regime == 'mean_reverting':
                # Exit at upper Bollinger or RSI overbought
                if self.data.close[0] > self.boll.lines.top[0] or self.rsi[0] > 70:
                    self.order = self.close()

            else:
                # Random walk - exit to avoid noise
                if self.rsi[0] > 60 or self.fast_ma[0] < self.slow_ma[0]:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MonteCarloDrawdown(bt.Strategy):
    """Monte Carlo Drawdown Control - exits based on simulated drawdown probability

    Continuously monitors position and exits if probability of hitting
    max drawdown threshold exceeds limit.
    """

    params = dict(
        lookback=60,
        n_simulations=500,
        max_drawdown=0.10,    # 10% max drawdown tolerance
        prob_threshold=0.30,  # Exit if >30% chance of hitting max DD
        trend_period=50,
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.trend_period)
        self.atr = bt.ind.ATR(period=14)
        self.order = None
        self.returns_buffer = []
        self.entry_price = None
        self.peak_value = None

    def simulate_drawdown_prob(self, returns, current_dd):
        """Simulate probability of hitting max drawdown"""
        if len(returns) < 20:
            return 0

        mu = np.mean(returns)
        sigma = np.std(returns)

        if sigma == 0:
            return 0

        hit_count = 0
        remaining_dd = self.p.max_drawdown - current_dd

        for _ in range(self.p.n_simulations):
            cumulative = 0
            min_cumulative = 0

            # Simulate 20 days forward
            for _ in range(20):
                cumulative += np.random.normal(mu, sigma)
                min_cumulative = min(min_cumulative, cumulative)

            # Check if we'd hit max DD
            if min_cumulative < -remaining_dd:
                hit_count += 1

        return hit_count / self.p.n_simulations

    def next(self):
        if self.order:
            return

        # Collect returns
        if len(self.data) > 1:
            ret = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.returns_buffer.append(ret)
            if len(self.returns_buffer) > self.p.lookback:
                self.returns_buffer.pop(0)

        trend_up = self.data.close[0] > self.sma[0]

        if not self.position:
            if trend_up and len(self.returns_buffer) >= 30:
                cash = self.broker.getcash()
                size = int(cash * 0.99 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.peak_value = self.data.close[0]
        else:
            # Track peak and current drawdown
            if self.data.close[0] > self.peak_value:
                self.peak_value = self.data.close[0]

            current_dd = (self.peak_value - self.data.close[0]) / self.peak_value

            # Exit if already at max drawdown
            if current_dd >= self.p.max_drawdown:
                self.order = self.close()
                return

            # Monte Carlo: estimate probability of hitting max DD
            if len(self.returns_buffer) >= 30:
                dd_prob = self.simulate_drawdown_prob(
                    np.array(self.returns_buffer),
                    current_dd
                )

                if dd_prob > self.p.prob_threshold:
                    self.order = self.close()
                    return

            # Also exit on trend break
            if not trend_up:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MonteCarloMeanReversion(bt.Strategy):
    """Monte Carlo Mean Reversion - trades only when mean reversion is statistically likely

    Uses MC to test if current deviation from mean is likely to revert,
    based on historical return distribution.
    """

    params = dict(
        lookback=60,
        n_simulations=500,
        zscore_entry=-1.5,     # Enter when Z < -1.5
        reversion_prob=0.65,   # Need 65% probability of reversion
        holding_days=10,       # Simulate forward this many days
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.lookback)
        self.std = bt.ind.StdDev(period=self.p.lookback)
        self.order = None
        self.returns_buffer = []

    def simulate_reversion_prob(self, returns, current_zscore):
        """Monte Carlo probability that price reverts toward mean"""
        if len(returns) < 20:
            return 0

        mu = np.mean(returns)
        sigma = np.std(returns)

        if sigma == 0:
            return 0

        revert_count = 0

        for _ in range(self.p.n_simulations):
            # Simulate path and check if Z-score improves (moves toward 0)
            cumulative_return = 0
            for _ in range(self.p.holding_days):
                cumulative_return += np.random.normal(mu, sigma)

            # If we're below mean (negative Z), reversion means positive return
            if current_zscore < 0 and cumulative_return > 0:
                revert_count += 1
            elif current_zscore > 0 and cumulative_return < 0:
                revert_count += 1

        return revert_count / self.p.n_simulations

    def next(self):
        if self.order:
            return

        # Collect returns
        if len(self.data) > 1:
            ret = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.returns_buffer.append(ret)
            if len(self.returns_buffer) > self.p.lookback:
                self.returns_buffer.pop(0)

        if self.std[0] == 0 or len(self.returns_buffer) < 30:
            return

        zscore = (self.data.close[0] - self.sma[0]) / self.std[0]

        if not self.position:
            # Buy when oversold AND MC says reversion is likely
            if zscore < self.p.zscore_entry:
                reversion_prob = self.simulate_reversion_prob(
                    np.array(self.returns_buffer),
                    zscore
                )

                if reversion_prob >= self.p.reversion_prob:
                    cash = self.broker.getcash()
                    size = int(cash * 0.99 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            # Exit when Z returns toward mean or goes positive
            if zscore >= 0:
                self.order = self.close()
            # Stop loss if Z gets worse
            elif zscore < -3:
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
    'breakout': SupportResistanceBreakout,
    'ichimoku': IchimokuStrategy,
    'ichimoku_simple': IchimokuCloudOnly,
    'breakout_vol': BreakoutVolume,
    # State of the art
    'turtle': TurtleTrading,
    'keltner': KeltnerChannel,
    'dual_momentum': DualMomentum,
    'zscore': MeanReversionZScore,
    'adaptive': AdaptiveTrendFollowing,
    'vol_regime': VolatilityRegime,
    'kama': KAMA_Strategy,
    'supertrend': SuperTrend,
    'chandelier': ChandelierExit,
    'williams': WilliamsR,
    'combined': CombinedMomentumMeanReversion,
    # Monte Carlo
    'mc_trend': MonteCarloTrendSignificance,
    'mc_var': MonteCarloVaR,
    'mc_kelly': MonteCarloKelly,
    'mc_regime': MonteCarloRegime,
    'mc_drawdown': MonteCarloDrawdown,
    'mc_revert': MonteCarloMeanReversion,
}
