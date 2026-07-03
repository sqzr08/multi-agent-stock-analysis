from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import llm
from stock_analyser.state import AgentSignal, AgentAnswer, GraphState
from stock_analyser.agents.base import _parse_signal
from stock_analyser.tools.yfinance_tools import get_price_history
from stock_analyser.tools.technical_tools import compute_indicators
from stock_analyser.tools.source_urls import build_url_block

_APPROVED_SOURCES = [
    "Yahoo Finance — Price History",
    "Computed — RSI (Relative Strength Index)",
    "Computed — MACD",
    "Computed — Bollinger Bands",
    "Computed — Moving Averages",
    "Computed — Volume Analysis",
]

_PROMPT = """You are a technical analysis expert. Given the following technical indicators for a stock, provide a JSON signal.

Indicators:
{data}

Respond ONLY with valid JSON in this exact format:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "bullets": [
    {{"text": "<insight>", "source": "<source name>", "url": "<url>"}},
    ... (1 to 5 bullets total)
  ]
}}

BULLET COUNT RULES:
- Return 1–5 bullets based on data richness — never pad to hit a fixed number
- 1–2 bullets: sparse data or most indicators are unavailable
- 3 bullets: normal case with several clear findings
- 4–5 bullets: rich data where every extra point adds genuinely new information
- Skip any finding where the underlying data was None or unavailable
- Most important finding always first
- Never repeat the same finding in different words across two bullets

WRITING RULES FOR BULLETS:
- Write as if explaining to a friend who has never invested before
- Never use raw financial jargon without explaining it
- If you must use a technical term, define it in the same bullet
- Each bullet text must follow this structure: "[What the data shows] — [what this means for the stock]"
- Good example: "Revenue grew 12% this year (the company made 12% more money than last year) — this suggests the business is expanding and demand for its products is strong"
- Bad example: "Revenue CAGR of 12% YoY indicates positive topline momentum"
- Keep each bullet to 1-2 sentences maximum
- Focus on what it means for the investor, not the metric itself
- Avoid: CAGR, basis points, bps, TTM, YoY, QoQ, EV/EBITDA, alpha, beta, drawdown, VaR

URL RULES:
- Every bullet must include a "source" and "url" field
- Do not invent URLs — only use URLs from the approved list below
- Match the source name exactly as shown in the approved list

Approved sources and URLs for {ticker}:
{url_block}"""

_STRICT_PROMPT = """Respond ONLY with a JSON object. No markdown, no code fences.
Return between 1 and 5 bullets. Return only bullets where you have actual data to support the claim. Do not pad with filler points.
Each bullet must be an object with "text", "source", and "url" keys.
Format: {{"signal": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, "bullets": [{{"text":"...","source":"...","url":"..."}}]}}

Data: {data}"""

_QA_PROMPT = """You are a technical analysis expert. Answer a user's follow-up question about \
a stock's price action and technical indicators.

Ticker: {ticker}
User question: {question}

Technical indicators already computed:
{indicators}

Use this data to answer the question directly. Respond with ONLY a JSON object:
{{
  "answer": "<direct answer in plain English, 2-4 sentences>",
  "sources": [{{"source": "<name>", "url": "<url>"}}],
  "confidence": <float 0.0-1.0>
}}

Writing rules: plain English, explain any technical terms, be specific with numbers, no markdown."""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    return _extract_text(response.content)


def technical_agent(state: GraphState) -> dict:
    ticker = state["ticker"]
    try:
        price_data = get_price_history.invoke({"ticker": ticker})
        indicators = compute_indicators(price_data)
        data = json.dumps(indicators, default=str)
        url_block = build_url_block(_APPROVED_SOURCES, ticker)

        text = _call_llm(_PROMPT.format(data=data, ticker=ticker, url_block=url_block))
        try:
            signal = _parse_signal(text)
        except Exception:
            text = _call_llm(_STRICT_PROMPT.format(data=data))
            signal = _parse_signal(text)

        return {
            "technical_signal": signal,
            "technical_indicators": indicators,
        }
    except Exception as e:
        return {
            "technical_signal": AgentSignal(
                signal="neutral", confidence=0.0,
                bullets=[{"text": "Data unavailable", "source": "", "url": ""}],
                error=str(e),
            ),
            "technical_indicators": {},
        }


def technical_qa_agent(state: GraphState) -> dict:
    plan = state.get("query_plan")
    ticker = (plan.other_ticker if plan and plan.asked_about_other_stock else state["ticker"])
    question = (plan.resolved_question if plan and plan.resolved_question else state.get("question", ""))
    needs_fresh = (plan and plan.needs_fresh_data) or (plan and plan.asked_about_other_stock)
    from stock_analyser.tools.cache import _bypass
    token = _bypass.set(True) if needs_fresh else None
    try:
        # Use cached indicators only for the session ticker; always fetch fresh for other stocks
        indicators = (not needs_fresh) and (state.get("technical_indicators") or {})
        if not indicators:
            price_data = get_price_history.invoke({"ticker": ticker})
            indicators = compute_indicators(price_data)

        indicators_str = json.dumps(indicators, default=str)
        prompt = _QA_PROMPT.format(ticker=ticker, question=question, indicators=indicators_str)
        text = _call_llm(prompt).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        answer_obj = AgentAnswer(
            answer=parsed.get("answer", ""),
            sources=parsed.get("sources", []),
            confidence=float(parsed.get("confidence", 0.5)),
        )
        return {"agent_answers": {"technical": answer_obj}}
    except Exception as e:
        return {"agent_answers": {"technical": AgentAnswer(
            answer=f"Technical data unavailable: {e}",
            sources=[],
            confidence=0.0,
        )}}
    finally:
        if token is not None:
            _bypass.reset(token)
