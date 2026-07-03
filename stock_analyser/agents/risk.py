from __future__ import annotations

import json
from langchain_core.messages import HumanMessage

from stock_analyser.config import llm
from stock_analyser.state import AgentSignal, AgentAnswer, GraphState
from stock_analyser.agents.base import _parse_signal, _run_tool_loop
from stock_analyser.tools.risk_tools import compute_risk_metrics, get_insider_short_data
from stock_analyser.tools.source_urls import build_url_block

_TOOLS = [compute_risk_metrics, get_insider_short_data]

_APPROVED_SOURCES = [
    "Computed — Beta",
    "Computed — Volatility",
    "Computed — Value at Risk",
    "Computed — Max Drawdown",
    "Computed — Sharpe Ratio",
    "Yahoo Finance — Short Interest",
    "Yahoo Finance — Insider Transactions",
    "Yahoo Finance — Earnings Calendar",
]

_PROMPT = """You are a risk analyst. Your job is to assess the risk profile of stock {ticker} \
and return a signal where bullish = low risk and bearish = high risk.

You have two tools available:
- compute_risk_metrics: computes how sensitive the stock is to the market, how volatile it is, \
potential daily losses, its worst historical drop, and its risk-adjusted return — call this first
- get_insider_short_data: fetches how many investors are betting against the stock, whether \
company insiders are buying or selling their own shares, and the next earnings date — call this second

CRITICAL: You MUST call BOTH tools before writing any analysis. \
Do NOT use your training knowledge to judge whether the ticker is valid or currently trading — \
the tools are authoritative. If the tools return numeric data, the stock is actively traded and \
you must analyse it from those numbers. Never skip tool calls based on what you think you know.

Call both tools. Once you have the data, return your analysis as a JSON object with exactly \
these keys:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0-1.0>,
  "bullets": [
    {{"text": "<insight>", "source": "<source name>", "url": "<url>"}},
    ... (1 to 5 bullets total)
  ]
}}

Remember: bullish = low risk / safe profile, bearish = high risk / dangerous profile.
Return ONLY the JSON object — no markdown, no explanation, no code fences.

BULLET COUNT RULES:
- Return 1–5 bullets based on data richness — never pad to hit a fixed number
- 1–2 bullets: sparse data or most risk metrics returned None
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
- Avoid: CAGR, basis points, bps, TTM, YoY, QoQ, EV/EBITDA, alpha — use plain English instead:
  - beta → "moves X% when the market moves 1%"
  - drawdown → "fell X% from its highest price"
  - VaR → "on a bad day the stock could drop around X%"

URL RULES:
- Every bullet must include a "source" and "url" field
- Do not invent URLs — only use URLs from the approved list below
- For computed metrics (beta, volatility, drawdown, Sharpe, VaR) use the matching Investopedia URL
- For short interest and insider data use the Yahoo Finance URLs
- Match the source name exactly as shown in the approved list

Approved sources and URLs for {ticker}:
{url_block}"""

_QA_PROMPT = """You are a risk analyst. Answer a user's follow-up question about \
a stock's risk profile.

Ticker: {ticker}
User question: {question}

Tools available:
- compute_risk_metrics: volatility, market sensitivity, value-at-risk, max drawdown, Sharpe ratio
- get_insider_short_data: short interest, insider buys/sells, next earnings date

Call whichever tool(s) are needed. Then respond with ONLY a JSON object:
{{
  "answer": "<direct answer in plain English, 2-4 sentences>",
  "sources": [{{"source": "<name>", "url": "<url>"}}],
  "confidence": <float 0.0-1.0>
}}

Writing rules: plain English, explain risk terms in everyday language, be specific with numbers, no markdown."""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)



def risk_agent(state: GraphState) -> dict:
    ticker = state["ticker"]
    try:
        url_block = build_url_block(_APPROVED_SOURCES, ticker)
        llm_with_tools = llm.bind_tools(_TOOLS)
        tool_map = {t.name: t for t in _TOOLS}
        messages = [HumanMessage(content=_PROMPT.format(ticker=ticker, url_block=url_block))]

        response = _run_tool_loop(messages, llm_with_tools, tool_map)
        text = _extract_text(response.content)
        try:
            signal = _parse_signal(text)
        except Exception:
            messages.append(HumanMessage(
                content='Return ONLY a JSON object with keys: signal, confidence, bullets. '
                        'bullets must be a list of 1–5 objects each with "text", "source", "url" keys. '
                        'Return only bullets where you have actual data to support the claim. '
                        'Do not pad with filler points. No other text.'
            ))
            response = llm_with_tools.invoke(messages)
            signal = _parse_signal(_extract_text(response.content))

        return {"risk_signal": signal}
    except Exception as e:
        return {"risk_signal": AgentSignal(
            signal="neutral", confidence=0.0,
            bullets=[{"text": "Data unavailable", "source": "", "url": ""}],
            error=str(e),
        )}


def risk_qa_agent(state: GraphState) -> dict:
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
        return {"agent_answers": {"risk": answer_obj}}
    except Exception as e:
        return {"agent_answers": {"risk": AgentAnswer(
            answer=f"Risk data unavailable: {e}",
            sources=[],
            confidence=0.0,
        )}}
    finally:
        if token is not None:
            _bypass.reset(token)
