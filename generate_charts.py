"""Simple script to generate chart images from MT5 data."""
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_ohlc(num_bars=100, base_price=1.0900):
    """Generate synthetic OHLC data for testing when MT5 is not available."""
    dates = [datetime.now() - timedelta(hours=num_bars-i) for i in range(num_bars)]
    
    # Generate random walk
    np.random.seed(42)
    returns = np.random.randn(num_bars) * 0.0005  # Small random movements
    closes = base_price * (1 + returns).cumprod()
    
    data = []
    for i, (date, close) in enumerate(zip(dates, closes)):
        # Generate OHLC from close with some randomness
        range_pct = abs(np.random.randn()) * 0.0003
        high = close * (1 + range_pct)
        low = close * (1 - range_pct)
        open_price = low + (high - low) * np.random.rand()
        
        data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': int(1000 + np.random.rand() * 500)
        })
    
    return pd.DataFrame(data)


def fetch_eurusd_data(num_bars=100):
    """Fetch EURUSD data from MT5, fallback to synthetic data."""
    if not mt5.initialize():
        print("MT5 not available, using synthetic data")
        return generate_synthetic_ohlc(num_bars)
    
    # Get data from the last 2 weeks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    
    rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("Failed to fetch MT5 data, using synthetic data")
        return generate_synthetic_ohlc(num_bars)
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def create_candlestick_chart(df, output_path, title="EURUSD Chart"):
    """Create a simple candlestick chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw candlesticks
    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Wick
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
        
        # Body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        ax.add_patch(plt.Rectangle((idx - 0.4, bottom), 0.8, height,
                                   facecolor=color, edgecolor=color))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved chart: {output_path}")


def main():
    """Generate sample chart images."""
    output_dir = Path("data/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching EURUSD data...")
    df = fetch_eurusd_data(num_bars=200)
    
    if df is None:
        print("Failed to fetch data")
        return
    
    print(f"Fetched {len(df)} bars")
    
    # Generate 15 charts with different time windows
    num_charts = 15
    bars_per_chart = 100
    
    for i in range(num_charts):
        # Use overlapping windows from the data
        start_idx = max(0, i * 10)
        end_idx = start_idx + bars_per_chart
        
        if end_idx > len(df):
            end_idx = len(df)
            start_idx = max(0, end_idx - bars_per_chart)
        
        chart_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        output_path = output_dir / f"chart_{i+1:03d}.png"
        start_time = chart_df.iloc[0]['time'].strftime('%Y-%m-%d %H:%M')
        end_time = chart_df.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M')
        title = f"EURUSD {start_time} to {end_time}"
        
        create_candlestick_chart(chart_df, output_path, title)
    
    print(f"\nGenerated {num_charts} chart images in {output_dir}")


if __name__ == "__main__":
    main()
