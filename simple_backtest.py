#!/usr/bin/env python
"""
Simple SMA Crossover Strategy Backtest with Buy & Hold Benchmark

Strategy: Buy when the fast SMA (10-period) crosses above the slow SMA (30-period).
          Sell when the fast SMA crosses below the slow SMA.
Benchmark: Buy and hold for the entire period.
"""

import backtrader as bt
import datetime


class BuyAndHoldStrategy(bt.Strategy):
    """Buy and Hold Benchmark Strategy - invests all available capital"""

    def __init__(self):
        self.bought = False
        self.start_price = None
        self.shares = None

    def nextstart(self):
        # Called on the first bar - buy with all available cash
        cash = self.broker.getcash()
        price = self.data.close[0]
        self.start_price = price
        self.shares = int(cash * 0.99 / price)  # 99% of cash (leave room for commission)
        self.buy(size=self.shares)
        self.bought = True

    def next(self):
        pass  # Just hold


class SMACrossoverStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy - invests all available capital"""

    params = dict(
        fast_period=10,   # Fast SMA period
        slow_period=30,   # Slow SMA period
    )

    def __init__(self):
        # Create SMA indicators
        self.fast_sma = bt.ind.SMA(self.data.close, period=self.p.fast_period)
        self.slow_sma = bt.ind.SMA(self.data.close, period=self.p.slow_period)

        # Create crossover signal
        self.crossover = bt.ind.CrossOver(self.fast_sma, self.slow_sma)

        # Track order
        self.order = None

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.2f} (shares: {order.executed.size:.0f})')
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Called when trade is closed"""
        if trade.isclosed:
            self.log(f'TRADE PROFIT: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}')

    def next(self):
        """Called on each bar"""
        # Skip if we have a pending order
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in market - look for buy signal
            if self.crossover > 0:  # Fast SMA crosses above slow SMA
                # Buy with all available cash
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int(cash * 0.99 / price)  # 99% of cash
                if size > 0:
                    self.log(f'BUY SIGNAL @ {price:.2f}')
                    self.order = self.buy(size=size)
        else:
            # In market - look for sell signal
            if self.crossover < 0:  # Fast SMA crosses below slow SMA
                self.log(f'SELL SIGNAL @ {self.data.close[0]:.2f}')
                self.order = self.close()  # Close entire position

    def stop(self):
        """Called at the end of the backtest"""
        print('\n' + '='*50)
        print('BACKTEST COMPLETE')
        print('='*50)
        print(f'Fast SMA: {self.p.fast_period} | Slow SMA: {self.p.slow_period}')
        print(f'Final Portfolio Value: ${self.broker.getvalue():,.2f}')
        print('='*50)


def run_backtest():
    starting_cash = 100000.0
    datafile = 'datas/spy-2015-2024.txt'
    fromdate = datetime.datetime(2015, 1, 1)
    todate = datetime.datetime(2024, 12, 31)

    print('='*60)
    print('SMA CROSSOVER STRATEGY vs BUY & HOLD BENCHMARK')
    print('='*60)
    print(f'Starting Portfolio Value: ${starting_cash:,.2f}')
    print(f'Data: S&P 500 (SPY) 2015-2024')
    print(f'Strategy: SMA Crossover (10/30)')
    print('='*60)

    # ===== RUN BUY & HOLD BENCHMARK =====
    print('\n[1/2] Running Buy & Hold Benchmark...')
    cerebro_bh = bt.Cerebro()

    data_bh = bt.feeds.YahooFinanceCSVData(
        dataname=datafile,
        fromdate=fromdate,
        todate=todate,
        reverse=False
    )
    cerebro_bh.adddata(data_bh)
    cerebro_bh.addstrategy(BuyAndHoldStrategy)
    cerebro_bh.broker.setcash(starting_cash)
    cerebro_bh.broker.setcommission(commission=0.001)
    cerebro_bh.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name='returns')

    results_bh = cerebro_bh.run()
    strat_bh = results_bh[0]
    final_value_bh = cerebro_bh.broker.getvalue()

    # ===== RUN SMA CROSSOVER STRATEGY =====
    print('[2/2] Running SMA Crossover Strategy...\n')
    cerebro = bt.Cerebro()

    data = bt.feeds.YahooFinanceCSVData(
        dataname=datafile,
        fromdate=fromdate,
        todate=todate,
        reverse=False
    )
    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossoverStrategy)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    # ===== COMPARISON RESULTS =====
    print('\n' + '='*60)
    print('PERFORMANCE COMPARISON')
    print('='*60)

    # Calculate returns
    strategy_return = ((final_value - starting_cash) / starting_cash) * 100
    benchmark_return = ((final_value_bh - starting_cash) / starting_cash) * 100
    alpha = strategy_return - benchmark_return

    # Get metrics
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_bh = strat_bh.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    dd_bh = strat_bh.analyzers.drawdown.get_analysis()

    # Trade stats
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.total.closed if hasattr(trades.total, 'closed') else 0
    won = trades.won.total if hasattr(trades, 'won') and hasattr(trades.won, 'total') else 0
    lost = trades.lost.total if hasattr(trades, 'lost') and hasattr(trades.lost, 'total') else 0

    # Print comparison table
    print(f'{"Metric":<25} {"Strategy":>15} {"Buy & Hold":>15}')
    print('-'*60)
    print(f'{"Final Value":<25} {"${:,.2f}".format(final_value):>15} {"${:,.2f}".format(final_value_bh):>15}')
    print(f'{"Total Return":<25} {"{:+.2f}%".format(strategy_return):>15} {"{:+.2f}%".format(benchmark_return):>15}')
    print(f'{"Sharpe Ratio":<25} {"{:.2f}".format(sharpe.get("sharperatio") or 0):>15} {"{:.2f}".format(sharpe_bh.get("sharperatio") or 0):>15}')
    print(f'{"Max Drawdown":<25} {"{:.2f}%".format(dd.max.drawdown):>15} {"{:.2f}%".format(dd_bh.max.drawdown):>15}')
    print(f'{"Total Trades":<25} {total_trades:>15} {1:>15}')
    print('-'*60)
    print(f'{"ALPHA (vs Benchmark)":<25} {"{:+.2f}%".format(alpha):>15}')
    print('='*60)

    # Strategy details
    print('\nSTRATEGY TRADE STATS:')
    print(f'  Winning Trades: {won}')
    print(f'  Losing Trades:  {lost}')
    print(f'  Win Rate:       {(won/total_trades)*100:.1f}%' if total_trades > 0 else '  Win Rate: N/A')

    # Verdict
    print('\n' + '='*60)
    if alpha > 0:
        print(f'RESULT: Strategy OUTPERFORMED Buy & Hold by {alpha:+.2f}%')
    elif alpha < 0:
        print(f'RESULT: Strategy UNDERPERFORMED Buy & Hold by {alpha:.2f}%')
    else:
        print('RESULT: Strategy performed same as Buy & Hold')
    print('='*60)

    # Plot results
    print('\nGenerating chart...')
    cerebro.plot(style='candlestick', barup='green', bardown='red')


if __name__ == '__main__':
    run_backtest()
