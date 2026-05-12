from __future__ import annotations

from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from tradingagents.dataflows.canonical_store import get_bars


INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: medium-term trend direction and dynamic support/resistance.",
    "close_200_sma": "200 SMA: long-term trend benchmark and major trend filter.",
    "close_10_ema": "10 EMA: short-term momentum and responsive trend shifts.",
    "macd": "MACD: momentum via EMA differences; useful for trend-change signals.",
    "macds": "MACD Signal: smoothed MACD line used for crossover confirmation.",
    "macdh": "MACD Histogram: visualizes MACD momentum expansion or contraction.",
    "rsi": "RSI: momentum oscillator for overbought/oversold and divergence checks.",
    "boll": "Bollinger Middle Band: 20-period moving average baseline.",
    "boll_ub": "Bollinger Upper Band: upper volatility band.",
    "boll_lb": "Bollinger Lower Band: lower volatility band.",
    "atr": "ATR: volatility measure used for stops and position sizing.",
    "vwma": "VWMA: volume-weighted moving average for trend confirmation.",
    "mfi": "MFI: price and volume momentum pressure indicator.",
}


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    data = get_bars(symbol, start_date, end_date)
    if data.empty:
        return (
            f"No canonical OHLCV data found for symbol '{symbol}' between "
            f"{start_date} and {end_date}. Run `tradingagents ingest daily` first."
        )

    output = data.rename(
        columns={
            "ts_open_utc": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    keep = ["Date", "Open", "High", "Low", "Close", "Volume", "source", "timeframe"]
    output = output[[col for col in keep if col in output.columns]]

    header = f"# Canonical stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(output)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output.to_csv(index=False)


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    indicator = indicator.lower()
    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(INDICATOR_DESCRIPTIONS)}"
        )

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=max(look_back_days, 260))
    data = get_bars(symbol, start_dt.strftime("%Y-%m-%d"), curr_date)
    if data.empty:
        return (
            f"No canonical OHLCV data found for {symbol} up to {curr_date}. "
            "Run ingestion before requesting indicators."
        )

    frame = data.rename(
        columns={
            "ts_open_utc": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
    stock_frame = wrap(frame[["Date", "Open", "High", "Low", "Close", "Volume"]])
    stock_frame["Date"] = stock_frame["Date"].dt.strftime("%Y-%m-%d")
    stock_frame[indicator]

    window_start = end_dt - relativedelta(days=look_back_days)
    rows = []
    current = end_dt
    values_by_date = {
        row["Date"]: row[indicator]
        for _, row in stock_frame.iterrows()
    }
    while current >= window_start:
        date_key = current.strftime("%Y-%m-%d")
        value = values_by_date.get(date_key, "N/A: Not a trading day or no canonical bar")
        if pd.isna(value):
            value = "N/A"
        rows.append(f"{date_key}: {value}")
        current = current - relativedelta(days=1)

    return (
        f"## {indicator} values from {window_start.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + "\n".join(rows)
        + "\n\n"
        + INDICATOR_DESCRIPTIONS[indicator]
    )

