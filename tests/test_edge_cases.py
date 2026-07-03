from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from stock_analyser.agents.base import _parse_signal
from stock_analyser.state import QueryPlan


# ── _parse_signal edge cases ──────────────────────────────────────────────────

BULLET = {"text": "Revenue grew", "source": "Yahoo Finance — Income Statement", "url": "https://finance.yahoo.com"}

def _signal_json(signal="bullish", confidence=0.8, bullets=None) -> str:
    import json
    return json.dumps({
        "signal": signal,
        "confidence": confidence,
        "bullets": bullets or [BULLET],
    })


def test_parse_signal_strips_markdown_json_fence():
    raw = f"```json\n{_signal_json()}\n```"
    sig = _parse_signal(raw)
    assert sig.signal == "bullish"


def test_parse_signal_strips_plain_code_fence():
    raw = f"```\n{_signal_json()}\n```"
    sig = _parse_signal(raw)
    assert sig.signal == "bullish"


def test_parse_signal_zero_bullets_raises():
    import json
    raw = json.dumps({"signal": "bullish", "confidence": 0.8, "bullets": []})
    with pytest.raises(ValueError, match="0 bullets"):
        _parse_signal(raw)


def test_parse_signal_six_bullets_truncated_to_five():
    sig = _parse_signal(_signal_json(bullets=[BULLET] * 6))
    assert len(sig.bullets) == 5


def test_parse_signal_string_bullets_normalised_to_dicts():
    import json
    raw = json.dumps({"signal": "bullish", "confidence": 0.8, "bullets": ["plain text bullet"]})
    sig = _parse_signal(raw)
    assert isinstance(sig.bullets[0], dict)
    assert sig.bullets[0]["text"] == "plain text bullet"
    assert sig.bullets[0]["source"] == ""
    assert sig.bullets[0]["url"] == ""


def test_parse_signal_malformed_json_raises():
    with pytest.raises(Exception):
        _parse_signal("not valid json at all {{{")


def test_parse_signal_missing_bullets_key_raises():
    import json
    with pytest.raises((KeyError, Exception)):
        _parse_signal(json.dumps({"signal": "bullish", "confidence": 0.8}))


# ── ticker_input_node edge cases ──────────────────────────────────────────────

def test_ticker_input_node_raises_value_error_on_invalid(monkeypatch):
    from stock_analyser.graph import ticker_input_node
    monkeypatch.setattr(
        "stock_analyser.graph._resolve_ticker",
        lambda raw: (_ for _ in ()).throw(ValueError("Could not find a valid ticker"))
    )
    with pytest.raises(ValueError, match="Could not find a valid ticker"):
        ticker_input_node({"ticker": "ZZZZNOTREAL"})


def test_ticker_input_node_wraps_unexpected_exception(monkeypatch):
    from stock_analyser.graph import ticker_input_node
    def boom(raw):
        raise RuntimeError("network error")
    monkeypatch.setattr("stock_analyser.graph._resolve_ticker", boom)
    with pytest.raises(ValueError, match="Could not validate"):
        ticker_input_node({"ticker": "AAPL"})


# ── _resolve_ticker edge cases ────────────────────────────────────────────────

def test_resolve_ticker_search_exception_falls_to_value_error(monkeypatch):
    from stock_analyser.graph import _resolve_ticker
    fail_ticker = MagicMock(info={}, history=MagicMock(return_value=MagicMock(empty=True)))
    monkeypatch.setattr("stock_analyser.graph.yf.Ticker", lambda t: fail_ticker)
    monkeypatch.setattr(
        "stock_analyser.graph.yf.Search",
        lambda q: (_ for _ in ()).throw(RuntimeError("search API down"))
    )
    with pytest.raises(ValueError, match="Could not find a valid ticker"):
        _resolve_ticker("SOMECOMPANY")


def test_resolve_ticker_non_equity_quotes_skipped_then_fallback(monkeypatch):
    from stock_analyser.graph import _resolve_ticker
    # Direct ticker fails
    fail_ticker = MagicMock(info={}, history=MagicMock(return_value=MagicMock(empty=True)))
    # Search returns a non-EQUITY quote first, then an EQUITY with valid price
    valid_ticker = MagicMock(info={"regularMarketPrice": 110.0})

    def mock_ticker_factory(symbol):
        return valid_ticker if symbol == "MU" else fail_ticker

    mock_search = MagicMock()
    mock_search.quotes = [
        {"symbol": "MU-ETF", "quoteType": "ETF", "exchange": "NMS"},   # skipped: not EQUITY
        {"symbol": "MU", "quoteType": "EQUITY", "exchange": "NMS"},    # accepted
    ]
    monkeypatch.setattr("stock_analyser.graph.yf.Ticker", mock_ticker_factory)
    monkeypatch.setattr("stock_analyser.graph.yf.Search", lambda q: mock_search)
    assert _resolve_ticker("micron") == "MU"


# ── Cache bypass reset on agent exception ─────────────────────────────────────

def test_bypass_reset_even_when_fundamentals_qa_raises():
    """The finally block must reset bypass even if the agent body raises."""
    from stock_analyser.agents.fundamentals import fundamentals_qa_agent
    from stock_analyser.tools.cache import _bypass

    plan = QueryPlan(
        relevant_agents=["fundamentals"],
        reasoning="",
        needs_fresh_data=True,
        answer_from_state=False,
    )
    with patch("stock_analyser.agents.fundamentals._run_tool_loop", side_effect=RuntimeError("boom")):
        fundamentals_qa_agent({"ticker": "AAPL", "query_plan": plan})

    assert _bypass.get() is False


def test_bypass_reset_even_when_sentiment_qa_raises():
    from stock_analyser.agents.sentiment import sentiment_qa_agent
    from stock_analyser.tools.cache import _bypass

    plan = QueryPlan(
        relevant_agents=["sentiment"],
        reasoning="",
        needs_fresh_data=True,
        answer_from_state=False,
    )
    with patch("stock_analyser.agents.sentiment._run_tool_loop", side_effect=RuntimeError("boom")):
        sentiment_qa_agent({"ticker": "AAPL", "query_plan": plan})

    assert _bypass.get() is False
