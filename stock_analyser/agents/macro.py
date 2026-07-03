from __future__ import annotations

import json
from langchain_core.messages import HumanMessage

from stock_analyser.config import llm
from stock_analyser.state import AgentSignal, AgentAnswer, GraphState
from stock_analyser.agents.base import _parse_signal, _run_tool_loop
from stock_analyser.tools.fred_tools import get_macro_indicators, get_sector_performance
from stock_analyser.tools.source_urls import build_url_block, get_source_url

_TOOLS = [get_macro_indicators, get_sector_performance]

_SECTOR_ETF_MAP = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Industrials": "XLI", "Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU", "Communication Services": "XLC",
}

_FRED_SOURCES = [
    "FRED — Federal Funds Rate",
    "FRED — Inflation (CPI)",
    "FRED — 10-Year Treasury Yield",
    "FRED — US Dollar Index",
    "FRED — Unemployment Rate",
]

_PROMPT = """You are a macroeconomic analyst. Your job is to assess how the current macro \
environment affects stock {ticker} in the {sector} sector (ETF: {sector_etf}).

You have two tools available:
- get_macro_indicators: fetches US interest rates, inflation (CPI YoY), 10-year Treasury yield, \
USD index, and unemployment rate from FRED — call this first
- get_sector_performance: fetches the 6-month return of a sector ETF vs the S&P 500 — \
call this with sector_etf="{sector_etf}" to assess relative sector momentum

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

Return ONLY the JSON object — no markdown, no explanation, no code fences.

BULLET COUNT RULES:
- Return 1–5 bullets based on data richness — never pad to hit a fixed number
- 1–2 bullets: sparse data or most macro indicators returned None
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
- For FRED data bullets, use the specific FRED series URL matching the indicator discussed
- For sector ETF performance bullets, use the Yahoo Finance Sector ETF Performance URL
- Match the source name exactly as shown in the approved list

Approved sources and URLs:
{url_block}"""

_QA_PROMPT = """You are a macroeconomic analyst. Answer a user's follow-up question about \
how the macro environment affects a stock.

Ticker: {ticker}
User question: {question}

Tools available:
- get_macro_indicators: US interest rates, inflation, treasury yield, USD, unemployment
- get_sector_performance: sector ETF return vs S&P 500

Call whichever tool(s) are needed. Then respond with ONLY a JSON object:
{{
  "answer": "<direct answer in plain English, 2-4 sentences>",
  "sources": [{{"source": "<name>", "url": "<url>"}}],
  "confidence": <float 0.0-1.0>
}}

Writing rules: plain English, explain any economic terms, be specific with numbers, no markdown."""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)



def _get_sector_etf(ticker: str) -> tuple[str, str]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Technology")
        return sector, _SECTOR_ETF_MAP.get(sector, "SPY")
    except Exception:
        return "Technology", "XLK"


def macro_agent(state: GraphState) -> dict:
    ticker = state["ticker"]
    try:
        sector, sector_etf = _get_sector_etf(ticker)

        fred_lines = [f'  "{s}": "{get_source_url(s)}"' for s in _FRED_SOURCES]
        etf_url = get_source_url("Yahoo Finance — Sector ETF Performance", sector_etf)
        fred_lines.append(f'  "Yahoo Finance — Sector ETF Performance": "{etf_url}"')
        url_block = "\n".join(fred_lines)

        llm_with_tools = llm.bind_tools(_TOOLS)
        tool_map = {t.name: t for t in _TOOLS}
        messages = [HumanMessage(content=_PROMPT.format(
            ticker=ticker, sector=sector, sector_etf=sector_etf, url_block=url_block
        ))]

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

        return {"macro_signal": signal}
    except Exception as e:
        return {"macro_signal": AgentSignal(
            signal="neutral", confidence=0.0,
            bullets=[{"text": "Data unavailable", "source": "", "url": ""}],
            error=str(e),
        )}


def macro_qa_agent(state: GraphState) -> dict:
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
        return {"agent_answers": {"macro": answer_obj}}
    except Exception as e:
        return {"agent_answers": {"macro": AgentAnswer(
            answer=f"Macro data unavailable: {e}",
            sources=[],
            confidence=0.0,
        )}}
    finally:
        if token is not None:
            _bypass.reset(token)
