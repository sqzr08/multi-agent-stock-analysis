from __future__ import annotations

import pytest

from stock_analyser.tools.cache import clear_cache
from stock_analyser.tools.technical_tools import compute_indicators
from stock_analyser.tools.yfinance_tools import get_fundamentals, get_financials, get_price_history
from stock_analyser.tools.fred_tools import get_macro_indicators, get_sector_performance
from stock_analyser.tools.alphavantage_tools import get_news_sentiment
from stock_analyser.tools.risk_tools import compute_risk_metrics, get_insider_short_data


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


def test_get_fundamentals_returns_required_fields():
    result = get_fundamentals.invoke({"ticker": "AAPL"})
    assert "error" not in result
    for key in ["trailingPE", "forwardPE", "sector", "industry", "marketCap",
                "profitMargins", "returnOnEquity", "debtToEquity"]:
        assert key in result


def test_get_financials_returns_all_statement_keys():
    result = get_financials.invoke({"ticker": "AAPL"})
    assert "error" not in result
    assert set(result.keys()) == {
        "income_statement", "balance_sheet", "cash_flow",
        "quarterly_income_statement", "quarterly_balance_sheet", "quarterly_cash_flow",
    }


def test_get_price_history_returns_ohlcv_and_52w_levels():
    result = get_price_history.invoke({"ticker": "AAPL"})
    assert "error" not in result
    assert set(result["ohlcv"].keys()) == {"Open", "High", "Low", "Close", "Volume"}
    assert "52w_high" in result and "52w_low" in result
    assert result["52w_high"] >= result["52w_low"]


def test_compute_indicators_returns_all_keys():
    price_data = get_price_history.invoke({"ticker": "AAPL"})
    assert "error" not in price_data
    result = compute_indicators(price_data)
    assert "error" not in result
    assert set(result.keys()) == {
        "current_price", "rsi_14", "macd_signal", "bb_position",
        "ma50", "ma200", "price_vs_ma50", "price_vs_ma200",
        "volume_trend", "atr_14", "support", "resistance",
        "recent_swing_low", "recent_swing_high",
    }


def test_get_macro_indicators_returns_required_fields():
    result = get_macro_indicators.invoke({})
    assert "error" not in result
    for key in ["fed_funds_rate", "cpi", "cpi_yoy_pct", "treasury_10y",
                "usd_index", "unemployment_rate"]:
        assert key in result


def test_get_sector_performance_returns_required_fields():
    result = get_sector_performance.invoke({"sector_etf": "XLK"})
    assert "error" not in result
    assert result["sector_etf"] == "XLK"
    for key in ["etf_6m_return_pct", "spy_6m_return_pct", "relative_performance_pct"]:
        assert key in result


def test_get_news_sentiment_returns_required_fields():
    result = get_news_sentiment.invoke({"ticker": "AAPL"})
    assert "error" not in result
    for key in ["overall_sentiment_label", "overall_sentiment_score", "article_count",
                "bullish_pct", "bearish_pct", "neutral_pct", "sample_headlines", "topics"]:
        assert key in result
    assert result["overall_sentiment_label"] in {
        "Bullish", "Somewhat-Bullish", "Neutral", "Somewhat-Bearish", "Bearish"
    }


def test_compute_risk_metrics_returns_required_fields():
    result = compute_risk_metrics.invoke({"ticker": "AAPL"})
    assert "error" not in result
    assert set(result.keys()) == {
        "beta", "annualised_volatility", "var_95_daily", "max_drawdown", "sharpe_ratio"
    }
    assert result["max_drawdown"] <= 0
    assert result["annualised_volatility"] > 0


def test_get_insider_short_data_returns_required_fields():
    result = get_insider_short_data.invoke({"ticker": "AAPL"})
    assert "error" not in result
    for key in ["short_percent_of_float", "insider_buys_90d",
                "insider_sells_90d", "next_earnings_date"]:
        assert key in result
    assert isinstance(result["insider_buys_90d"], int)
    assert isinstance(result["insider_sells_90d"], int)
