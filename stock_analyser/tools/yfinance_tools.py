from __future__ import annotations

import yfinance as yf
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.tools.cache import cached


# ── private cached implementations ───────────────────────────────────────────

@cached(ttl_seconds=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_fundamentals_cached(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        keys = [
            "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
            "enterpriseToEbitda", "profitMargins", "returnOnEquity", "returnOnAssets",
            "debtToEquity", "currentRatio", "quickRatio", "revenueGrowth",
            "earningsGrowth", "dividendYield", "marketCap", "sector", "industry",
            "fullTimeEmployees",
        ]
        return {k: info.get(k) for k in keys}
    except Exception as e:
        return {"error": str(e)}


@cached(ttl_seconds=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_financials_cached(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        result = {}
        for name, df in [
            ("income_statement",           t.financials),
            ("balance_sheet",              t.balance_sheet),
            ("cash_flow",                  t.cashflow),
            ("quarterly_income_statement", t.quarterly_financials),
            ("quarterly_balance_sheet",    t.quarterly_balance_sheet),
            ("quarterly_cash_flow",        t.quarterly_cashflow),
        ]:
            if df is not None and not df.empty:
                d = df.to_dict()
                result[name] = {
                    str(col): {str(idx): val for idx, val in rows.items()}
                    for col, rows in d.items()
                }
            else:
                result[name] = {}
        return result
    except Exception as e:
        return {"error": str(e)}


@cached(ttl_seconds=300)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_price_history_cached(ticker: str, period: str = "1y") -> dict:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return {"error": "No price data returned"}
        hist.index = hist.index.astype(str)
        ohlcv = hist[["Open", "High", "Low", "Close", "Volume"]].to_dict()
        ohlcv = {col: {str(k): v for k, v in rows.items()} for col, rows in ohlcv.items()}
        return {
            "ohlcv": ohlcv,
            "52w_high": float(hist["High"].max()),
            "52w_low": float(hist["Low"].min()),
        }
    except Exception as e:
        return {"error": str(e)}


@cached(ttl_seconds=300)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_market_indices_cached() -> dict:
    try:
        indices = [
            ("S&P 500", "^GSPC"),
            ("Dow Jones", "^DJI"),
            ("Nasdaq Composite", "^IXIC"),
        ]
        snapshots = []
        for name, symbol in indices:
            hist = yf.Ticker(symbol).history(period="2d")
            if hist.empty or len(hist) < 2:
                snapshots.append({
                    "name": name, "symbol": symbol,
                    "last": None, "change_pct": None, "direction": "neutral",
                })
                continue
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change_pct = round((last - prev) / prev * 100, 2) if prev != 0 else 0.0
            snapshots.append({
                "name": name,
                "symbol": symbol,
                "last": round(last, 2),
                "change_pct": change_pct,
                "direction": "rise" if change_pct >= 0 else "fall",
            })
        return {"indices": snapshots}
    except Exception as e:
        return {"error": str(e)}


# ── public @tool wrappers (unchanged signatures/docstrings) ───────────────────

@tool
def get_fundamentals(ticker: str) -> dict:
    """
    Fetches key fundamental ratios and company metadata for a stock from Yahoo Finance.
    Use this first to assess valuation, profitability, and financial health.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
    Returns dict with P/E, P/B, P/S, EV/EBITDA, margins, ROE, ROA,
    debt/equity, current ratio, revenue/earnings growth, dividend yield,
    market cap, sector, industry, and employee count.
    """
    return _get_fundamentals_cached(ticker)


@tool
def get_financials(ticker: str) -> dict:
    """
    Fetches the income statement, balance sheet, and cash flow statement for a stock.
    Use this to analyse revenue trends, profit growth, asset quality, and cash generation.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
    Returns dict with keys: income_statement, balance_sheet, cash_flow (annual figures)
    and quarterly_income_statement, quarterly_balance_sheet, quarterly_cash_flow
    (quarterly figures) — all as nested dicts keyed by period-end date string (e.g. '2026-03-31').
    Use quarterly keys to answer questions about specific fiscal quarters.
    """
    return _get_financials_cached(ticker)


@tool
def get_price_history(ticker: str, period: str = "1y") -> dict:
    """
    Fetches historical OHLCV price data for a stock over a given period.
    Use this to get raw price data before computing technical indicators.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
        period: Lookback period string e.g. '1y', '6mo', '2y' (default '1y')
    Returns dict with ohlcv (Open/High/Low/Close/Volume as date-keyed dicts),
    52w_high, and 52w_low.
    """
    return _get_price_history_cached(ticker, period)


@tool
def get_market_indices() -> dict:
    """
    Fetches the latest daily price data for the three main US market indices:
    S&P 500 (^GSPC), Dow Jones Industrial Average (^DJI), and Nasdaq Composite (^IXIC).
    Use this to understand the current broad market direction and sentiment.
    Args: None — ticker-independent, returns data for all three indices at once.
    Returns dict with 'indices' list, each entry containing name, symbol,
    last price, daily change_pct, and direction ('rise' or 'fall').
    """
    return _get_market_indices_cached()
