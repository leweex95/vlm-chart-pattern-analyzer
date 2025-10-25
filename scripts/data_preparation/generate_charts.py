#!/usr/bin/env python3
"""Generate chart images from MetaTrader5 data."""
import argparse
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, List, Optional

import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import pandas as pd

# Constants
DEFAULT_NUM_CHARTS = 25
DEFAULT_BARS_PER_CHART = 100
DEFAULT_OUTPUT_DIR = "data/images"
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_DAYS = 30
DEFAULT_TIMEFRAME = "H1"
RANDOM_SEED = 42
CHART_DPI = 100
FIGURE_SIZE = (12, 6)

# Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


@contextmanager
def mt5_connection() -> Generator[None, None, None]:
    """Context manager for MT5 connection."""
    try:
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed. Make sure MetaTrader 5 is installed and you're logged in")
        logging.info("MT5 connected successfully")
        yield
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")


def fetch_random_eurusd_sections(
    num_charts: int = DEFAULT_NUM_CHARTS,
    bars_per_chart: int = DEFAULT_BARS_PER_CHART,
    symbol: str = DEFAULT_SYMBOL,
    days: int = DEFAULT_DAYS,
    timeframe: str = DEFAULT_TIMEFRAME
) -> Optional[List[pd.DataFrame]]:
    """Fetch random sections of EURUSD data from MT5."""
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}")

    try:
        with mt5_connection():
            # Fetch a large dataset from the past few weeks
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logging.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
            rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], start_date, end_date)

            if rates is None or len(rates) == 0:
                logging.error("Failed to fetch data from MT5")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            logging.info(f"Fetched {len(df)} bars of {symbol} data")

            # Generate random sections
            chart_sections = []
            max_start = len(df) - bars_per_chart

            if max_start <= 0:
                logging.error("Not enough data to create charts")
                return None

            # Generate random start indices
            random.seed(RANDOM_SEED)
            random_starts = random.sample(range(0, max_start), min(num_charts, max_start))

            for start_idx in sorted(random_starts):
                section = df.iloc[start_idx:start_idx + bars_per_chart].copy()
                section.reset_index(drop=True, inplace=True)
                chart_sections.append(section)

            logging.info(f"Created {len(chart_sections)} random chart sections")

            return chart_sections

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None


def create_candlestick_chart(df: pd.DataFrame, output_path: Path, title: str = f"{DEFAULT_SYMBOL} Chart") -> None:
    """Create a simple candlestick chart."""
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

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
        plt.savefig(output_path, dpi=CHART_DPI)
        plt.close()
        logging.info(f"Saved chart: {output_path}")

    except Exception as e:
        logging.error(f"Error creating chart: {e}")
        plt.close('all')
        raise


def main() -> None:
    """Generate sample chart images."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        parser = argparse.ArgumentParser(description='Generate chart images from MT5 data')
        parser.add_argument('--num-charts', type=int, default=DEFAULT_NUM_CHARTS,
                           help=f'Number of random chart sections to generate (default: {DEFAULT_NUM_CHARTS})')
        parser.add_argument('--bars-per-chart', type=int, default=DEFAULT_BARS_PER_CHART,
                           help=f'Number of bars per chart (default: {DEFAULT_BARS_PER_CHART})')
        parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                           help=f'Output directory for chart images (default: {DEFAULT_OUTPUT_DIR})')
        parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                           help=f'Trading symbol to fetch data for (default: {DEFAULT_SYMBOL})')
        parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                           help=f'Number of days of historical data to fetch (default: {DEFAULT_DAYS})')
        parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                           choices=list(TIMEFRAME_MAP.keys()),
                           help=f'Timeframe for the data (default: {DEFAULT_TIMEFRAME})')

        args = parser.parse_args()

        # Validate arguments
        if args.num_charts <= 0:
            raise ValueError("Number of charts must be positive")
        if args.bars_per_chart <= 0:
            raise ValueError("Bars per chart must be positive")
        if args.days <= 0:
            raise ValueError("Number of days must be positive")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch random sections from MT5
        chart_sections = fetch_random_eurusd_sections(
            num_charts=args.num_charts,
            bars_per_chart=args.bars_per_chart,
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe
        )

        if chart_sections is None:
            logging.error("Failed to fetch data from MT5")
            logging.error("Please ensure:")
            logging.error("  1. MetaTrader 5 is installed")
            logging.error("  2. You're logged into a broker account")
            logging.error(f"  3. {args.symbol} symbol is available")
            return

        # Generate chart images
        logging.info(f"Generating {len(chart_sections)} chart images...")
        for i, chart_df in enumerate(chart_sections):
            output_path = output_dir / f"chart_{i+1:03d}.png"
            start_time = chart_df.iloc[0]['time'].strftime('%Y-%m-%d %H:%M')
            end_time = chart_df.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M')
            title = f"{args.symbol} {start_time} to {end_time}"

            create_candlestick_chart(chart_df, output_path, title)

        logging.info(f"Successfully generated {len(chart_sections)} chart images in {output_dir}")

    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
