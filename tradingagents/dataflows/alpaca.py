from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from tradingagents.dataflows.config import get_config


def _alpaca_credentials() -> tuple[str, str]:
    key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key_id or not secret_key:
        raise RuntimeError(
            "Alpaca credentials are missing. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in .env or the shell environment."
        )
    return key_id, secret_key


def _iso_window(start_date: str, end_date: str) -> tuple[str, str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(
        tzinfo=timezone.utc
    )
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def fetch_stock_bars(
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
    timeframe: str = "1Day",
    feed: str | None = None,
    adjustment: str = "raw",
) -> dict[str, pd.DataFrame]:
    key_id, secret_key = _alpaca_credentials()
    base_url = get_config().get("alpaca_data_base_url", "https://data.alpaca.markets").rstrip("/")
    start_iso, end_iso = _iso_window(start_date, end_date)
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "adjustment": adjustment,
        "limit": 10000,
    }
    if feed:
        params["feed"] = feed

    request = Request(
        f"{base_url}/v2/stocks/bars?{urlencode(params)}",
        headers={
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in payload.get("bars", {}).items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            frames[symbol.upper()] = frame
            continue
        frame = frame.rename(
            columns={
                "t": "time",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        frames[symbol.upper()] = frame[["time", "open", "high", "low", "close", "volume"]]
    return frames


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    frames = fetch_stock_bars([symbol], start_date=start_date, end_date=end_date)
    frame = frames.get(symbol.upper(), pd.DataFrame())
    if frame.empty:
        return f"No Alpaca bars found for symbol '{symbol}' between {start_date} and {end_date}"

    output = frame.rename(
        columns={
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    header = f"# Alpaca stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(output)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output.to_csv(index=False)

