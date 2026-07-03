from __future__ import annotations

# Populated at runtime by alphavantage_tools.get_news_sentiment
article_url_cache: dict[str, str] = {}


def get_source_url(source: str, ticker: str = "") -> str:
    ticker = ticker.upper()
    urls = {
        "Yahoo Finance — Company Profile":
            f"https://finance.yahoo.com/quote/{ticker}/profile",
        "Yahoo Finance — Income Statement":
            f"https://finance.yahoo.com/quote/{ticker}/financials",
        "Yahoo Finance — Balance Sheet":
            f"https://finance.yahoo.com/quote/{ticker}/balance-sheet",
        "Yahoo Finance — Cash Flow Statement":
            f"https://finance.yahoo.com/quote/{ticker}/cash-flow",
        "Yahoo Finance — Key Ratios":
            f"https://finance.yahoo.com/quote/{ticker}/key-statistics",
        "Yahoo Finance — Price History":
            f"https://finance.yahoo.com/quote/{ticker}/history",
        "Yahoo Finance — Sector ETF Performance":
            f"https://finance.yahoo.com/quote/{ticker}",
        "Yahoo Finance — Short Interest":
            f"https://finance.yahoo.com/quote/{ticker}/key-statistics",
        "Yahoo Finance — Insider Transactions":
            f"https://finance.yahoo.com/quote/{ticker}/insider-transactions",
        "Yahoo Finance — Earnings Calendar":
            f"https://finance.yahoo.com/quote/{ticker}/analysis",
        "Yahoo Finance — News":
            f"https://finance.yahoo.com/quote/{ticker}/news",
        "Alpha Vantage — News Sentiment":
            "https://www.alphavantage.co/query?function=NEWS_SENTIMENT",
        "Alpha Vantage — Analyst Ratings":
            "https://www.alphavantage.co/query?function=OVERVIEW",
        "FRED — Federal Funds Rate":
            "https://fred.stlouisfed.org/series/FEDFUNDS",
        "FRED — Inflation (CPI)":
            "https://fred.stlouisfed.org/series/CPIAUCSL",
        "FRED — 10-Year Treasury Yield":
            "https://fred.stlouisfed.org/series/DGS10",
        "FRED — US Dollar Index":
            "https://fred.stlouisfed.org/series/DTWEXBGS",
        "FRED — Unemployment Rate":
            "https://fred.stlouisfed.org/series/UNRATE",
        "Computed — Moving Averages":
            "https://www.investopedia.com/terms/m/movingaverage.asp",
        "Computed — RSI (Relative Strength Index)":
            "https://www.investopedia.com/terms/r/rsi.asp",
        "Computed — MACD":
            "https://www.investopedia.com/terms/m/macd.asp",
        "Computed — Bollinger Bands":
            "https://www.investopedia.com/terms/b/bollingerbands.asp",
        "Computed — Volume Analysis":
            "https://www.schwab.com/learn/story/trading-volume-as-market-indicator",
        "Computed — Beta":
            "https://www.investopedia.com/terms/b/beta.asp",
        "Computed — Volatility":
            "https://www.investopedia.com/terms/v/volatility.asp",
        "Computed — Value at Risk":
            "https://www.investopedia.com/terms/v/var.asp",
        "Computed — Max Drawdown":
            "https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp",
        "Computed — Sharpe Ratio":
            "https://www.investopedia.com/terms/s/sharperatio.asp",
    }
    return urls.get(source, "")


def build_url_block(sources: list[str], ticker: str = "") -> str:
    """Returns a formatted reference block of source → URL pairs for agent prompts."""
    lines = []
    for source in sources:
        url = get_source_url(source, ticker)
        lines.append(f'  "{source}": "{url}"')
    return "\n".join(lines)
