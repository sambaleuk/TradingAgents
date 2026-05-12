import pandas as pd
import pytest

from tradingagents.dataflows.canonical_store import get_bars, upsert_bars
from tradingagents.dataflows.canonical_vendor import get_indicator, get_stock_data
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor


@pytest.fixture()
def canonical_store(tmp_path):
    path = tmp_path / "canonical.sqlite"
    set_config(
        {
            "canonical_store_path": str(path),
            "canonical_source_priority": ["metatrader", "yfinance"],
        }
    )
    return path


def test_upsert_and_read_canonical_bars(canonical_store):
    data = pd.DataFrame(
        [
            {"Date": "2026-01-02", "Open": 100, "High": 105, "Low": 99, "Close": 104, "Volume": 1000},
            {"Date": "2026-01-03", "Open": 104, "High": 106, "Low": 102, "Close": 103, "Volume": 1100},
        ]
    )

    written = upsert_bars("SPY", data, source="yfinance", timeframe="D1", path=canonical_store)
    assert written == 2

    bars = get_bars("SPY", "2026-01-01", "2026-01-04", path=canonical_store)
    assert len(bars) == 2
    assert bars.iloc[0]["close"] == 104


def test_canonical_vendor_formats_stock_data(canonical_store):
    data = pd.DataFrame(
        [{"Date": "2026-01-02", "Open": 100, "High": 105, "Low": 99, "Close": 104, "Volume": 1000}]
    )
    upsert_bars("SPY", data, source="yfinance", timeframe="D1", path=canonical_store)

    result = get_stock_data("SPY", "2026-01-01", "2026-01-04")

    assert "Canonical stock data for SPY" in result
    assert "104" in result


def test_route_to_vendor_supports_canonical(canonical_store):
    data = pd.DataFrame(
        [{"Date": "2026-01-02", "Open": 100, "High": 105, "Low": 99, "Close": 104, "Volume": 1000}]
    )
    upsert_bars("SPY", data, source="yfinance", timeframe="D1", path=canonical_store)
    set_config({"data_vendors": {"core_stock_apis": "canonical"}})

    result = route_to_vendor("get_stock_data", "SPY", "2026-01-01", "2026-01-04")

    assert "Canonical stock data for SPY" in result


def test_canonical_indicator_reports_missing_data(canonical_store):
    result = get_indicator("SPY", "rsi", "2026-01-10", 5)

    assert "No canonical OHLCV data found" in result
