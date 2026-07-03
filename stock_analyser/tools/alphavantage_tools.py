from __future__ import annotations

import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from stock_analyser.tools.source_urls import article_url_cache  # noqa: E402
from stock_analyser.tools.cache import cached  # noqa: E402

_BASE_URL = "https://www.alphavantage.co/query"

TRUSTED_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "seekingalpha.com", "barrons.com", "forbes.com",
    "businessinsider.com", "apnews.com", "finance.yahoo.com", "benzinga.com",
}

_DEFAULT = {
    "overall_sentiment_label": "Neutral",
    "overall_sentiment_score": 0.0,
    "article_count": 0,
    "bullish_pct": 0.0,
    "bearish_pct": 0.0,
    "neutral_pct": 0.0,
    "sample_headlines": [],
    "article_urls": {},
    "topics": [],
}


# ── private cached implementation ─────────────────────────────────────────────

@cached(ttl_seconds=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_news_sentiment_cached(ticker: str) -> dict:
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 50,
            "apikey": api_key,
        }
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        feed = data.get("feed", [])
        if not feed:
            return _DEFAULT.copy()

        trusted_feed = [a for a in feed if a.get("source_domain", "") in TRUSTED_DOMAINS]
        active_feed = trusted_feed if trusted_feed else feed

        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        scores = []
        headlines = []
        topic_counter: dict[str, int] = {}

        for article in active_feed:
            ticker_sentiment_label = None
            ticker_sentiment_score = None
            relevance_score = None
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    try:
                        relevance_score = float(ts.get("relevance_score", 0.0))
                    except (TypeError, ValueError):
                        relevance_score = 0.0
                    ticker_sentiment_label = ts.get("ticker_sentiment_label", "")
                    ticker_sentiment_score = ts.get("ticker_sentiment_score")
                    break

            if relevance_score is not None and relevance_score < 0.5:
                continue

            label = (ticker_sentiment_label or article.get("overall_sentiment_label", "Neutral")).lower()
            score = ticker_sentiment_score or article.get("overall_sentiment_score", 0.0)

            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

            if "bullish" in label:
                counts["bullish"] += 1
            elif "bearish" in label:
                counts["bearish"] += 1
            else:
                counts["neutral"] += 1

            title = article.get("title", "")
            url = article.get("url", "")
            if title and url:
                article_url_cache[title] = url
            if len(headlines) < 3 and title:
                headlines.append(title)

            for topic in article.get("topics", []):
                t = topic.get("topic", "")
                if t:
                    topic_counter[t] = topic_counter.get(t, 0) + 1

        total = counts["bullish"] + counts["bearish"] + counts["neutral"]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        if avg_score >= 0.35:
            overall_label = "Bullish"
        elif avg_score >= 0.15:
            overall_label = "Somewhat-Bullish"
        elif avg_score <= -0.35:
            overall_label = "Bearish"
        elif avg_score <= -0.15:
            overall_label = "Somewhat-Bearish"
        else:
            overall_label = "Neutral"

        top_topics = sorted(topic_counter, key=topic_counter.get, reverse=True)[:5]

        if total == 0:
            return _DEFAULT.copy()

        return {
            "overall_sentiment_label": overall_label,
            "overall_sentiment_score": avg_score,
            "article_count": total,
            "bullish_pct": round(counts["bullish"] / total * 100, 1),
            "bearish_pct": round(counts["bearish"] / total * 100, 1),
            "neutral_pct": round(counts["neutral"] / total * 100, 1),
            "sample_headlines": headlines,
            "article_urls": {h: article_url_cache.get(h, "") for h in headlines},
            "topics": top_topics,
        }
    except Exception as e:
        return {"error": str(e)}


# ── public @tool wrapper ──────────────────────────────────────────────────────

@tool
def get_news_sentiment(ticker: str) -> dict:
    """
    Fetches latest news articles and pre-computed sentiment scores
    for a stock ticker using Alpha Vantage News & Sentiment API.
    Use this to assess current market sentiment and media tone
    toward the company. No NLP library needed — scores are
    returned directly by the API.
    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
    Returns dict with:
        - overall_sentiment_label: 'Bearish' | 'Somewhat-Bearish' |
          'Neutral' | 'Somewhat-Bullish' | 'Bullish'
        - overall_sentiment_score: float -1.0 to 1.0
        - article_count: int
        - bullish_pct: float (% of articles with bullish label)
        - bearish_pct: float (% of articles with bearish label)
        - neutral_pct: float (% of articles with neutral label)
        - sample_headlines: list of 3 most recent article titles
        - topics: list of top topics mentioned across articles
    """
    return _get_news_sentiment_cached(ticker)
