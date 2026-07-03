from __future__ import annotations

import datetime
import ssl
import certifi
import yfinance as yf
from fredapi import Fred
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import FRED_API_KEY
from stock_analyser.tools.cache import cached

# Fix macOS SSL certificate verification for the FRED API
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

_fred = Fred(api_key=FRED_API_KEY)


# ── private cached implementations ───────────────────────────────────────────

@cached(ttl_seconds=86400)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_macro_indicators_cached() -> dict:
    try:
        series_map = {
            "FEDFUNDS": "fed_funds_rate",
            "CPIAUCSL": "cpi",
            "DGS10": "treasury_10y",
            "DTWEXBGS": "usd_index",
            "UNRATE": "unemployment_rate",
        }
        result = {}
        for sid, key in series_map.items():
            s = _fred.get_series(sid)
            result[key] = float(s.dropna().iloc[-1]) if s is not None and not s.empty else None

        cpi = _fred.get_series("CPIAUCSL").dropna()
        if len(cpi) >= 13:
            result["cpi_yoy_pct"] = round((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100, 2)
        else:
            result["cpi_yoy_pct"] = None

        return result
    except Exception as e:
        return {"error": str(e)}


@cached(ttl_seconds=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_sector_performance_cached(sector_etf: str = "XLK") -> dict:
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=182)

        etf_hist = yf.Ticker(sector_etf).history(start=str(start), end=str(end))["Close"]
        spy_hist = yf.Ticker("SPY").history(start=str(start), end=str(end))["Close"]

        if etf_hist.empty or spy_hist.empty:
            return {"error": "No price data for ETF comparison"}

        etf_return = float((etf_hist.iloc[-1] / etf_hist.iloc[0] - 1) * 100)
        spy_return = float((spy_hist.iloc[-1] / spy_hist.iloc[0] - 1) * 100)

        return {
            "sector_etf": sector_etf,
            "etf_6m_return_pct": round(etf_return, 2),
            "spy_6m_return_pct": round(spy_return, 2),
            "relative_performance_pct": round(etf_return - spy_return, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ── public @tool wrappers ─────────────────────────────────────────────────────

@tool
def get_macro_indicators() -> dict:
    """
    Fetches key US macroeconomic indicators from the FRED database.
    Use this to understand the current interest rate, inflation, and employment environment.
    Args: None
    Returns dict with fed_funds_rate, cpi (index level), cpi_yoy_pct,
    treasury_10y yield, usd_index, and unemployment_rate — all as floats.
    """
    return _get_macro_indicators_cached()


@tool
def get_sector_performance(sector_etf: str = "XLK") -> dict:
    """
    Fetches the 6-month return of a sector ETF compared to the S&P 500.
    Use this to assess whether the stock's sector is outperforming or underperforming the market.
    Args:
        sector_etf: Sector ETF ticker symbol e.g. 'XLK' (Tech), 'XLV' (Health),
                    'XLF' (Finance), 'XLE' (Energy), 'XLY' (Consumer Discretionary)
    Returns dict with sector_etf, etf_6m_return_pct, spy_6m_return_pct,
    and relative_performance_pct (ETF minus SPY).
    """
    return _get_sector_performance_cached(sector_etf)
