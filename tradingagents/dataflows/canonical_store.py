from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from tradingagents.dataflows.config import get_config


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS instruments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    broker_symbol TEXT,
    tv_symbol TEXT,
    asset_class TEXT,
    exchange TEXT,
    timezone TEXT,
    currency TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bars (
    instrument_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts_open_utc TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    spread REAL,
    ingested_at TEXT NOT NULL,
    quality_flags TEXT DEFAULT '',
    PRIMARY KEY (instrument_id, source, timeframe, ts_open_utc),
    FOREIGN KEY (instrument_id) REFERENCES instruments(id)
);

CREATE INDEX IF NOT EXISTS idx_bars_lookup
ON bars (instrument_id, timeframe, ts_open_utc);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    symbols TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    rows_in INTEGER DEFAULT 0,
    rows_written INTEGER DEFAULT 0,
    errors TEXT DEFAULT ''
);
"""


def canonical_store_path() -> Path:
    config = get_config()
    configured = config.get("canonical_store_path")
    if configured:
        return Path(configured).expanduser()
    return Path(config["data_cache_dir"]).expanduser() / "canonical_market_data.sqlite"


@contextmanager
def connect(path: str | Path | None = None):
    db_path = Path(path).expanduser() if path else canonical_store_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_instrument(conn: sqlite3.Connection, symbol: str, **metadata) -> int:
    normalized = symbol.strip().upper()
    now = utc_now()
    conn.execute(
        """
        INSERT INTO instruments (
            symbol, broker_symbol, tv_symbol, asset_class, exchange, timezone,
            currency, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            broker_symbol = COALESCE(excluded.broker_symbol, instruments.broker_symbol),
            tv_symbol = COALESCE(excluded.tv_symbol, instruments.tv_symbol),
            asset_class = COALESCE(excluded.asset_class, instruments.asset_class),
            exchange = COALESCE(excluded.exchange, instruments.exchange),
            timezone = COALESCE(excluded.timezone, instruments.timezone),
            currency = COALESCE(excluded.currency, instruments.currency)
        """,
        (
            normalized,
            metadata.get("broker_symbol"),
            metadata.get("tv_symbol"),
            metadata.get("asset_class"),
            metadata.get("exchange"),
            metadata.get("timezone"),
            metadata.get("currency"),
            now,
        ),
    )
    row = conn.execute(
        "SELECT id FROM instruments WHERE symbol = ?", (normalized,)
    ).fetchone()
    return int(row["id"])


def normalize_bars_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    frame = data.copy()
    rename = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=rename)
    if "date" in frame.columns and "ts_open_utc" not in frame.columns:
        frame = frame.rename(columns={"date": "ts_open_utc"})
    if "time" in frame.columns and "ts_open_utc" not in frame.columns:
        frame = frame.rename(columns={"time": "ts_open_utc"})

    required = ["ts_open_utc", "open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"bars data missing required columns: {missing}")

    frame["ts_open_utc"] = pd.to_datetime(frame["ts_open_utc"], utc=True)
    for col in ["open", "high", "low", "close", "volume", "spread"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["ts_open_utc", "open", "high", "low", "close"])
    frame = frame.sort_values("ts_open_utc")
    return frame


def upsert_bars(
    symbol: str,
    data: pd.DataFrame,
    *,
    source: str,
    timeframe: str = "D1",
    path: str | Path | None = None,
) -> int:
    frame = normalize_bars_frame(data)
    if frame.empty:
        return 0

    with connect(path) as conn:
        instrument_id = ensure_instrument(conn, symbol)
        ingested_at = utc_now()
        rows = []
        for row in frame.to_dict("records"):
            rows.append(
                (
                    instrument_id,
                    source,
                    timeframe,
                    row["ts_open_utc"].isoformat(),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    None if pd.isna(row.get("volume")) else float(row.get("volume", 0)),
                    None if pd.isna(row.get("spread")) else float(row.get("spread", 0)),
                    ingested_at,
                    "",
                )
            )
        conn.executemany(
            """
            INSERT INTO bars (
                instrument_id, source, timeframe, ts_open_utc, open, high, low,
                close, volume, spread, ingested_at, quality_flags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, source, timeframe, ts_open_utc) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                spread = excluded.spread,
                ingested_at = excluded.ingested_at,
                quality_flags = excluded.quality_flags
            """,
            rows,
        )
    return len(rows)


def get_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    timeframe: str = "D1",
    source_priority: Iterable[str] | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    sources = list(source_priority or get_config().get("canonical_source_priority", []))
    source_filter = ""
    params: list = [symbol.strip().upper(), timeframe, start_date, end_date]
    if sources:
        placeholders = ",".join("?" for _ in sources)
        source_filter = f"AND b.source IN ({placeholders})"
        params.extend(sources)

    with connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                b.source,
                b.timeframe,
                b.ts_open_utc,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.spread,
                b.quality_flags
            FROM bars b
            JOIN instruments i ON i.id = b.instrument_id
            WHERE i.symbol = ?
              AND b.timeframe = ?
              AND date(b.ts_open_utc) >= date(?)
              AND date(b.ts_open_utc) <= date(?)
              {source_filter}
            ORDER BY b.ts_open_utc ASC
            """,
            params,
        ).fetchall()

    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty or not sources:
        return frame

    priority = {source: index for index, source in enumerate(sources)}
    frame["_priority"] = frame["source"].map(priority).fillna(len(priority))
    frame = (
        frame.sort_values(["ts_open_utc", "_priority"])
        .drop_duplicates(subset=["ts_open_utc"], keep="first")
        .drop(columns=["_priority"])
        .sort_values("ts_open_utc")
    )
    return frame


def record_ingestion_run(
    *,
    source: str,
    symbols: list[str],
    timeframe: str,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    rows_in: int = 0,
    rows_written: int = 0,
    errors: str = "",
    path: str | Path | None = None,
) -> None:
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                source, symbols, timeframe, started_at, finished_at, status,
                rows_in, rows_written, errors
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                ",".join(symbols),
                timeframe,
                started_at,
                finished_at,
                status,
                rows_in,
                rows_written,
                errors,
            ),
        )

