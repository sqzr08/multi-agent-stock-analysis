from __future__ import annotations

import json
from langchain_core.messages import HumanMessage

from stock_analyser.config import llm
from stock_analyser.state import AgentSignal, AgentAnswer, GraphState
from stock_analyser.agents.base import _parse_signal, _run_tool_loop
from stock_analyser.tools.alphavantage_tools import get_news_sentiment

_TOOLS = [get_news_sentiment]

_PROMPT = """You are a market sentiment analyst. Your job is to assess how the media and market \
currently feel about stock {ticker} based on recent news.

You have one tool available:
- get_news_sentiment: fetches recent headlines and pre-computed sentiment scores from Alpha Vantage. \
Pass the ticker symbol. The response includes "article_urls" — a dict mapping each headline title \
to its direct article URL. Use those direct URLs when citing news headlines.

CRITICAL: You MUST call the tool before writing any analysis. \
Do NOT use your training knowledge to judge whether the ticker is valid or currently trading — \
the tool is authoritative. If it returns data, the stock is actively traded and \
you must analyse it from those results. Never skip the tool call based on what you think you know.

Call the tool to fetch sentiment data. Once you have the results, return your analysis as a JSON \
object with exactly these keys:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0-1.0>,
  "bullets": [
    {{"text": "<insight>", "source": "<source name>", "url": "<url>"}},
    ... (1 to 5 bullets total)
  ]
}}

Return ONLY the JSON object — no markdown, no explanation, no code fences.

BULLET COUNT RULES:
- Return exactly one bullet for each headline listed in "sample_headlines" — no more, no less
- Each bullet must correspond to exactly one article from "sample_headlines" and its direct URL from "article_urls"
- If a headline has no matching URL in "article_urls", skip it — do not create a bullet for it
- Do NOT create bullets from overall_sentiment_label, overall_sentiment_score, bullish_pct, bearish_pct, neutral_pct, or any other aggregated field
- Do NOT create a bullet that summarises overall market mood, investor sentiment, or media tone
- Most relevant finding first

WRITING RULES FOR BULLETS:
- Write as if explaining to a friend who has never invested before
- Never use raw financial jargon without explaining it
- If you must use a technical term, define it in the same bullet
- Each bullet text must follow this structure: "[What the article reports] — [what this means for the stock]"
- Good example: "A Reuters report says Apple suppliers are ramping up production ahead of the holiday season — this suggests strong expected demand for new iPhone models"
- Bad example: "Overall sentiment is neutral with 34% bearish articles indicating negative media tone"
- Keep each bullet to 1-2 sentences maximum
- Focus on what the article says and what it means for the investor
- Avoid: CAGR, basis points, bps, TTM, YoY, QoQ, EV/EBITDA, alpha, beta, drawdown, VaR

URL RULES:
- Every bullet must include a "source" and "url" field
- The only valid URLs are direct article URLs from the "article_urls" field in the tool response
- Do NOT use feed or index page URLs (e.g. finance.yahoo.com/quote/{ticker}/news, alphavantage.co/query)
- Do NOT include a bullet if its URL is not a direct article URL from "article_urls"
- Set "source" to the publication name derived from the article URL domain (e.g. "Reuters", "Bloomberg", "CNBC")
- Do not invent URLs"""

_QA_PROMPT = """You are a market sentiment analyst. Answer a user's follow-up question about \
a stock's news sentiment.

Ticker: {ticker}
User question: {question}

You have one tool: get_news_sentiment — call it if you need fresh data.

Respond with ONLY a JSON object:
{{
  "answer": "<direct answer in plain English, 2-4 sentences>",
  "sources": [{{"source": "<name>", "url": "<url>"}}],
  "confidence": <float 0.0-1.0>
}}

Writing rules: plain English, be specific about headlines or scores when available, no markdown."""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)



def sentiment_agent(state: GraphState) -> dict:
    ticker = state["ticker"]
    try:
        llm_with_tools = llm.bind_tools(_TOOLS)
        tool_map = {t.name: t for t in _TOOLS}
        messages = [HumanMessage(content=_PROMPT.format(ticker=ticker))]

        response = _run_tool_loop(messages, llm_with_tools, tool_map)
        text = _extract_text(response.content)
        try:
            signal = _parse_signal(text)
        except Exception:
            messages.append(HumanMessage(
                content='Return ONLY a JSON object with keys: signal, confidence, bullets. '
                        'Each bullet must correspond to exactly one article from "sample_headlines" '
                        'using its direct URL from "article_urls". No aggregated-sentiment bullets. '
                        'bullets is a list of objects each with "text", "source", "url" keys. No other text.'
            ))
            response = llm_with_tools.invoke(messages)
            signal = _parse_signal(_extract_text(response.content))

        _FEED_PATTERNS = ("/quote/", "alphavantage.co/query", "finance.yahoo.com/news")
        valid_bullets = [
            b for b in signal.bullets
            if b.get("url") and not any(p in b["url"] for p in _FEED_PATTERNS)
        ]
        if valid_bullets:
            signal = AgentSignal(
                signal=signal.signal,
                confidence=signal.confidence,
                bullets=valid_bullets,
                error=signal.error,
            )

        return {"sentiment_signal": signal}
    except Exception as e:
        return {"sentiment_signal": AgentSignal(
            signal="neutral", confidence=0.0,
            bullets=[{"text": "Data unavailable", "source": "", "url": ""}],
            error=str(e),
        )}


def sentiment_qa_agent(state: GraphState) -> dict:
    plan = state.get("query_plan")
    ticker = (plan.other_ticker if plan and plan.asked_about_other_stock else state["ticker"])
    question = (plan.resolved_question if plan and plan.resolved_question else state.get("question", ""))
    from stock_analyser.tools.cache import _bypass
    needs_bypass = (plan and plan.needs_fresh_data) or (plan and plan.asked_about_other_stock)
    token = _bypass.set(True) if needs_bypass else None
    try:
        llm_with_tools = llm.bind_tools(_TOOLS)
        tool_map = {t.name: t for t in _TOOLS}
        messages = [HumanMessage(content=_QA_PROMPT.format(ticker=ticker, question=question))]

        response = _run_tool_loop(messages, llm_with_tools, tool_map, session_ticker=ticker)
        text = _extract_text(response.content).strip()
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
        return {"agent_answers": {"sentiment": answer_obj}}
    except Exception as e:
        return {"agent_answers": {"sentiment": AgentAnswer(
            answer=f"Sentiment data unavailable: {e}",
            sources=[],
            confidence=0.0,
        )}}
    finally:
        if token is not None:
            _bypass.reset(token)
