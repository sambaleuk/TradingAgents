import json
from unittest.mock import patch

import pytest

from tradingagents.dataflows.alpaca import fetch_stock_bars
from tradingagents.dataflows.canonical_store import get_bars
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.ingestion import ingest_alpaca_daily


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture()
def alpaca_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    set_config(
        {
            "canonical_store_path": str(tmp_path / "canonical.sqlite"),
            "alpaca_data_base_url": "https://example.test",
        }
    )


def test_fetch_stock_bars_parses_alpaca_response(alpaca_env):
    payload = {
        "bars": {
            "SPY": [
                {"t": "2026-04-20T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
            ]
        }
    }

    with patch("tradingagents.dataflows.alpaca.urlopen", return_value=_FakeResponse(payload)):
        frames = fetch_stock_bars(["SPY"], start_date="2026-04-20", end_date="2026-04-20")

    assert list(frames["SPY"].columns) == ["time", "open", "high", "low", "close", "volume"]
    assert frames["SPY"].iloc[0]["close"] == 1.5


def test_ingest_alpaca_daily_writes_canonical_store(alpaca_env):
    payload = {
        "bars": {
            "SPY": [
                {"t": "2026-04-20T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
            ]
        }
    }

    with patch("tradingagents.dataflows.alpaca.urlopen", return_value=_FakeResponse(payload)):
        result = ingest_alpaca_daily(["SPY"], start_date="2026-04-20", end_date="2026-04-20")

    bars = get_bars("SPY", "2026-04-20", "2026-04-20")
    assert result["status"] == "completed"
    assert result["rows_written"] == 1
    assert bars.iloc[0]["source"] == "alpaca"


def test_alpaca_credentials_required(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Alpaca credentials are missing"):
        fetch_stock_bars(["SPY"], start_date="2026-04-20", end_date="2026-04-20")

