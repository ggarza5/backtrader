#!/usr/bin/env python
"""
Strategy Runner - Easy CLI for backtesting

Usage:
    python run.py AAPL                     # Run default (sma) on AAPL
    python run.py AAPL MSFT GOOGL          # Run on multiple stocks
    python run.py AAPL -s rsi              # Run RSI strategy
    python run.py AAPL TSLA -s macd        # Run MACD on multiple stocks
    python run.py --list                   # List available strategies
    python run.py AAPL -s all              # Run all strategies on AAPL
    python run.py AAPL --plot              # Show chart after backtest

Strategies: sma, rsi, bollinger, macd, momentum, triplema, buyhold
"""

import argparse
import datetime
import os
import sys
from glob import glob

import backtrader as bt
from strategies import STRATEGIES


def find_data_file(symbol):
    """Find data file for symbol"""
    symbol = symbol.lower()

    # Try 2015-2024 first (new data)
    path = f'datas/{symbol}-2015-2024.txt'
    if os.path.exists(path):
        return path

    # Try other patterns
    patterns = [
        f'datas/{symbol}-*.txt',
        f'datas/*{symbol}*.txt',
        f'contrib/datas/*{symbol}*.csv',
    ]

    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            return matches[0]

    return None


def run_backtest(symbol, strategy_name, cash=100000, plot=False):
    """Run a single backtest"""

    # Find data file
    datafile = find_data_file(symbol)
    if not datafile:
        print(f"  ERROR: No data file found for {symbol}")
        return None

    # Get strategy class
    strategy_cls = STRATEGIES.get(strategy_name)
    if not strategy_cls:
        print(f"  ERROR: Unknown strategy '{strategy_name}'")
        return None

    # Create cerebro
    cerebro = bt.Cerebro()

    # Add data (reverse=True because our data is newest-first)
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datafile,
        fromdate=datetime.datetime(2015, 1, 1),
        todate=datetime.datetime(2024, 12, 31),
        reverse=True
    )
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(strategy_cls)

    # Broker settings
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Get metrics
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value - cash) / cash) * 100

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get('sharperatio') or 0

    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.max.drawdown if hasattr(dd_analysis, 'max') else 0

    trades = strat.analyzers.trades.get_analysis()
    try:
        total_trades = trades.total.closed
    except (KeyError, AttributeError):
        total_trades = 0

    if plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red')

    return {
        'symbol': symbol.upper(),
        'strategy': strategy_name,
        'final_value': final_value,
        'return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': total_trades,
    }


def print_results(results, benchmark_results=None):
    """Print results table"""
    print()
    print("="*85)
    print(f"{'Symbol':<8} {'Strategy':<12} {'Final Value':>14} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("-"*85)

    for r in results:
        print(f"{r['symbol']:<8} {r['strategy']:<12} ${r['final_value']:>12,.2f} {r['return']:>+9.2f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['trades']:>7}")

    if benchmark_results:
        print("-"*85)
        for r in benchmark_results:
            print(f"{r['symbol']:<8} {'buyhold':<12} ${r['final_value']:>12,.2f} {r['return']:>+9.2f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['trades']:>7}")

    print("="*85)


def list_strategies():
    """List available strategies"""
    print("\nAvailable Strategies:")
    print("-"*50)
    for name, cls in STRATEGIES.items():
        doc = cls.__doc__.split('\n')[0] if cls.__doc__ else ''
        print(f"  {name:<12} {doc}")
    print()


def list_symbols():
    """List available data files"""
    files = glob('datas/*-2015-2024.txt')
    symbols = [os.path.basename(f).split('-')[0].upper() for f in files]
    symbols.sort()

    print(f"\nAvailable symbols ({len(symbols)}):")
    print("-"*50)

    # Print in columns
    cols = 8
    for i in range(0, len(symbols), cols):
        row = symbols[i:i+cols]
        print("  " + "  ".join(f"{s:<6}" for s in row))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('symbols', nargs='*', help='Stock symbols (e.g., AAPL MSFT)')
    parser.add_argument('-s', '--strategy', action='append', help='Strategy name(s) or "all". Can specify multiple.')
    parser.add_argument('-c', '--cash', type=float, default=100000, help='Starting cash')
    parser.add_argument('-p', '--plot', action='store_true', help='Show chart')
    parser.add_argument('--list', action='store_true', help='List strategies')
    parser.add_argument('--symbols', action='store_true', dest='list_symbols', help='List available symbols')
    parser.add_argument('--benchmark', action='store_true', help='Include buy & hold benchmark')

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    if args.list_symbols:
        list_symbols()
        return

    if not args.symbols:
        parser.print_help()
        return

    # Determine which strategies to run
    if args.strategy is None:
        strategies = ['sma']  # default
    elif 'all' in args.strategy:
        strategies = [s for s in STRATEGIES.keys() if s != 'buyhold']
    else:
        strategies = args.strategy

    # Run backtests
    print(f"\nRunning backtest(s)...")
    print(f"  Symbols: {', '.join(s.upper() for s in args.symbols)}")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Starting Cash: ${args.cash:,.2f}")

    results = []
    benchmark_results = []

    for symbol in args.symbols:
        for strat in strategies:
            result = run_backtest(symbol, strat, args.cash, args.plot and len(strategies) == 1)
            if result:
                results.append(result)

        if args.benchmark:
            bm = run_backtest(symbol, 'buyhold', args.cash, False)
            if bm:
                benchmark_results.append(bm)

    if results:
        print_results(results, benchmark_results if args.benchmark else None)

    # Summary for multi-strategy comparison
    if len(strategies) > 1 and len(args.symbols) == 1:
        print("\nBest performing strategy:")
        best = max(results, key=lambda x: x['return'])
        print(f"  {best['strategy']} with {best['return']:+.2f}% return")


if __name__ == '__main__':
    main()
