from __future__ import annotations

import datetime
import numpy as np
import yfinance as yf
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.tools.cache import cached


# ── private cached implementations ───────────────────────────────────────────

@cached(ttl_seconds=300)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _compute_risk_metrics_cached(ticker: str) -> dict:
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=730)

        hist = yf.Ticker(ticker).history(start=str(start), end=str(end))["Close"]
        spy = yf.Ticker("SPY").history(start=str(start), end=str(end))["Close"]

        if hist.empty:
            return {"error": "No price data"}

        returns = hist.pct_change().dropna()
        spy_returns = spy.pct_change().dropna()
        returns, spy_returns = returns.align(spy_returns, join="inner")

        ann_vol = float(returns.std() * np.sqrt(252))

        cov = np.cov(returns, spy_returns)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else None

        var_95 = float(np.percentile(returns, 5))

        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        rf_daily = 0.045 / 252
        excess = returns - rf_daily
        sharpe = float((excess.mean() / returns.std()) * np.sqrt(252)) if returns.std() != 0 else None

        return {
            "beta": round(beta, 3) if beta is not None else None,
            "annualised_volatility": round(ann_vol, 4),
            "var_95_daily": round(var_95, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


@cached(ttl_seconds=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_insider_short_data_cached(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        short_pct = info.get("shortPercentOfFloat")

        cutoff = datetime.date.today() - datetime.timedelta(days=90)
        buy_count = 0
        sell_count = 0
        try:
            transactions = t.insider_transactions
            if transactions is not None and not transactions.empty:
                transactions.index = transactions.index.tz_localize(None) if transactions.index.tz else transactions.index
                recent = transactions[transactions.index.date >= cutoff] if hasattr(transactions.index, 'date') else transactions
                for _, row in recent.iterrows():
                    shares = row.get("Shares", 0) or 0
                    text = str(row.get("Transaction", "")).lower()
                    if "sale" in text or "sell" in text:
                        sell_count += 1
                    elif "purchase" in text or "buy" in text or "acquisition" in text:
                        buy_count += 1
        except Exception:
            pass

        next_earnings = None
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date:
                        next_earnings = str(earnings_date[0]) if isinstance(earnings_date, list) else str(earnings_date)
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    next_earnings = str(cal.loc["Earnings Date"].iloc[0])
        except Exception:
            pass

        return {
            "short_percent_of_float": short_pct,
            "insider_buys_90d": buy_count,
            "insider_sells_90d": sell_count,
            "next_earnings_date": next_earnings,
        }
    except Exception as e:
        return {"error": str(e)}


# ── public @tool wrappers ─────────────────────────────────────────────────────

@tool
def compute_risk_metrics(ticker: str) -> dict:
    """
    Computes quantitative risk metrics for a stock using 2 years of daily price data.
    Use this to assess volatility, market sensitivity, downside risk, and risk-adjusted returns.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
    Returns dict with beta (vs SPY), annualised_volatility, var_95_daily (Value at Risk),
    max_drawdown, and sharpe_ratio.
    """
    return _compute_risk_metrics_cached(ticker)


@tool
def get_insider_short_data(ticker: str) -> dict:
    """
    Fetches short interest, insider transaction activity, and the next earnings date for a stock.
    Use this alongside compute_risk_metrics to get a complete risk and positioning picture.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
    Returns dict with short_percent_of_float, insider_buys_90d, insider_sells_90d,
    and next_earnings_date as a string.
    """
    return _get_insider_short_data_cached(ticker)
