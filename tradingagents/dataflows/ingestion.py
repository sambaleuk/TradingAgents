from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.canonical_store import (
    record_ingestion_run,
    upsert_bars,
    utc_now,
)


def ingest_yfinance_daily(
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
) -> dict:
    started_at = utc_now()
    rows_in = 0
    rows_written = 0
    errors: list[str] = []

    for symbol in symbols:
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                errors.append(f"{symbol}: no rows returned")
                continue
            data = data.reset_index()
            rows_in += len(data)
            rows_written += upsert_bars(symbol, data, source="yfinance", timeframe="D1")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    status = "completed" if not errors else "partial"
    if errors and rows_written == 0:
        status = "failed"

    finished_at = utc_now()
    record_ingestion_run(
        source="yfinance",
        symbols=symbols,
        timeframe="D1",
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        rows_in=rows_in,
        rows_written=rows_written,
        errors="\n".join(errors),
    )
    return {
        "source": "yfinance",
        "status": status,
        "rows_in": rows_in,
        "rows_written": rows_written,
        "errors": errors,
    }


MT5_TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def ingest_metatrader_bars(
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
    timeframe: str = "D1",
) -> dict:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 Python package is not installed. Install it in the runtime "
            "environment and ensure the MetaTrader 5 terminal is available."
        ) from exc

    timeframe_key = timeframe.upper()
    if timeframe_key not in MT5_TIMEFRAMES:
        raise ValueError(f"Unsupported MetaTrader timeframe: {timeframe}")

    mt5_timeframe = getattr(mt5, MT5_TIMEFRAMES[timeframe_key])
    started_at = utc_now()
    rows_in = 0
    rows_written = 0
    errors: list[str] = []

    if not mt5.initialize():
        code, message = mt5.last_error()
        raise RuntimeError(f"MetaTrader 5 initialize failed: {code} {message}")

    try:
        utc_from = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        utc_to = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        ).replace(tzinfo=timezone.utc)

        for symbol in symbols:
            try:
                if not mt5.symbol_select(symbol, True):
                    errors.append(f"{symbol}: symbol_select failed")
                    continue
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, utc_from, utc_to)
                if rates is None or len(rates) == 0:
                    errors.append(f"{symbol}: no rows returned")
                    continue
                data = pd.DataFrame(rates)
                data["time"] = pd.to_datetime(data["time"], unit="s", utc=True)
                data = data.rename(columns={"tick_volume": "volume"})
                rows_in += len(data)
                rows_written += upsert_bars(
                    symbol,
                    data[["time", "open", "high", "low", "close", "volume", "spread"]],
                    source="metatrader",
                    timeframe=timeframe_key,
                )
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
    finally:
        mt5.shutdown()

    status = "completed" if not errors else "partial"
    if errors and rows_written == 0:
        status = "failed"

    record_ingestion_run(
        source="metatrader",
        symbols=symbols,
        timeframe=timeframe_key,
        started_at=started_at,
        finished_at=utc_now(),
        status=status,
        rows_in=rows_in,
        rows_written=rows_written,
        errors="\n".join(errors),
    )
    return {
        "source": "metatrader",
        "status": status,
        "rows_in": rows_in,
        "rows_written": rows_written,
        "errors": errors,
    }

