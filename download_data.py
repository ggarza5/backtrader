#!/usr/bin/env python
"""Download stock data from Stooq (reliable free source)"""

import requests
import time

def download_stock(symbol, start_date='2015-01-01', end_date='2024-12-31'):
    """Download stock data from Stooq and save to CSV"""
    print(f"Downloading {symbol}...")

    # Stooq uses different symbol format for US stocks
    stooq_symbol = f"{symbol}.US"

    # Stooq download URL
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={start_date.replace('-', '')}&d2={end_date.replace('-', '')}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content = response.text.strip()

        # Check if we got valid data
        if 'No data' in content or len(content) < 100:
            print(f"  No data found for {symbol}")
            return None

        # Stooq format: Date,Open,High,Low,Close,Volume
        # We need: Date,Open,High,Low,Close,Adj Close,Volume
        lines = content.split('\n')
        header = lines[0]

        # Check if header matches expected format
        if 'Date' not in header:
            print(f"  Unexpected data format for {symbol}")
            return None

        # Save to file with adjusted format
        output_path = f"datas/{symbol.lower()}-{start_date[:4]}-{end_date[:4]}.txt"

        with open(output_path, 'w') as f:
            # Write header
            f.write("Date,Open,High,Low,Close,Adj Close,Volume\n")

            # Write data (skip header, data is newest first so reverse it)
            data_lines = lines[1:]
            data_lines.reverse()  # Oldest first for backtrader

            for line in data_lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        date, open_p, high, low, close = parts[:5]
                        volume = parts[5] if len(parts) > 5 else '0'
                        # Add Adj Close (same as Close)
                        f.write(f"{date},{open_p},{high},{low},{close},{close},{volume}\n")

        num_rows = len(data_lines)
        print(f"  Saved to {output_path}")
        print(f"  Total rows: {num_rows}")

        return output_path

    except requests.exceptions.HTTPError as e:
        print(f"  HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


if __name__ == '__main__':
    # Comprehensive list of symbols for strategy testing
    symbols = [
        # Major indices ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
        # Sector ETFs
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE',
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV',
        # Consumer
        'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrial
        'CAT', 'BA', 'GE', 'HON', 'UPS',
        # Bonds & alternatives
        'TLT', 'GLD', 'SLV', 'USO',
        # Volatility
        'VXX',
        # International
        'EEM', 'EFA', 'FXI',
    ]

    print("="*50)
    print("DOWNLOADING STOCK DATA (from Stooq)")
    print(f"Total symbols: {len(symbols)}")
    print("="*50 + "\n")

    successful = []
    failed = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] ", end="")
        result = download_stock(symbol)
        if result:
            successful.append(symbol)
        else:
            failed.append(symbol)
        print()
        time.sleep(1)  # Be nice to the server

    print("="*50)
    print("DOWNLOAD COMPLETE")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {', '.join(failed)}")
    print("="*50)
