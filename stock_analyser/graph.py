from __future__ import annotations

import asyncio
import json

import yfinance as yf
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Send
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import llm
from stock_analyser.state import GraphState, IndexSnapshot, MarketReport, QueryPlan
from stock_analyser.agents.fundamentals import fundamentals_agent, fundamentals_qa_agent
from stock_analyser.agents.technical import technical_agent, technical_qa_agent
from stock_analyser.agents.sentiment import sentiment_agent, sentiment_qa_agent
from stock_analyser.agents.macro import macro_agent, macro_qa_agent
from stock_analyser.agents.risk import risk_agent, risk_qa_agent
from stock_analyser.agents.synthesis import synthesis_agent
from stock_analyser.agents.planner import planner_agent
from stock_analyser.agents.responder import responder_agent
from stock_analyser.tools.yfinance_tools import _get_market_indices_cached
from stock_analyser.memory.analysis_store import save_analysis, get_recent_analyses

# ── Graph constants ───────────────────────────────────────────────────────────

_ANALYSIS_AGENTS = ("fundamentals", "technical", "sentiment", "macro", "risk", "market")
_QA_AGENTS = {
    "fundamentals": fundamentals_qa_agent,
    "technical": technical_qa_agent,
    "sentiment": sentiment_qa_agent,
    "macro": macro_qa_agent,
    "risk": risk_qa_agent,
}

_USER_ID = "default"

_MARKET_PROMPT = """You are a brief market commentator. The three main US indices moved as follows today:

{index_lines}

Write two short fields:
- trend_forecast: exactly 1 sentence describing today's market direction and what it may mean short-term.
- summary: exactly 2 sentences covering overall market mood and the single most important driver to watch.

Rules: plain English, no jargon, no specific price predictions.

Respond ONLY with valid JSON:
{{"trend_forecast": "...", "summary": "..."}}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _llm_market_text(index_lines: str) -> tuple[str, str]:
    response = llm.invoke(_MARKET_PROMPT.format(index_lines=index_lines))
    raw = response.content if isinstance(response.content, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in response.content
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())
    return parsed.get("trend_forecast", ""), parsed.get("summary", "")


def market_node(state: GraphState) -> dict:
    """Deterministic market snapshot — no tool loop, single LLM prose call."""
    try:
        raw = _get_market_indices_cached()
        raw_indices = raw.get("indices", []) if isinstance(raw, dict) else []

        snapshots = [
            IndexSnapshot(
                name=idx["name"],
                symbol=idx["symbol"],
                last=idx.get("last"),
                change_pct=idx.get("change_pct"),
                direction=idx.get("direction", "neutral"),
            )
            for idx in raw_indices
        ]

        index_lines = "\n".join(
            f"  {s.name} ({s.symbol}): "
            f"{'rose' if s.direction == 'rise' else 'fell' if s.direction == 'fall' else 'was flat'} "
            f"{abs(s.change_pct or 0):.2f}% to ${s.last:,.2f}"
            if s.last is not None
            else f"  {s.name} ({s.symbol}): data unavailable"
            for s in snapshots
        )

        try:
            trend_forecast, summary = _llm_market_text(index_lines)
        except Exception:
            rises = sum(1 for s in snapshots if s.direction == "rise")
            direction = "higher" if rises >= 2 else "lower"
            trend_forecast = f"Major US indices moved {direction} today."
            summary = "Market data was retrieved but commentary is temporarily unavailable."

        return {"market_report": MarketReport(
            indices=snapshots,
            trend_forecast=trend_forecast,
            summary=summary,
        )}

    except Exception as e:
        return {"market_report": MarketReport(
            indices=[],
            trend_forecast="Market data unavailable.",
            summary=f"Could not retrieve index data: {e}",
        )}


_US_EXCHANGES = {"NMS", "NYQ", "PCX", "NGM", "NCM", "ASE", "BTS"}


def _has_price_data(info: dict) -> bool:
    return bool(info) and any(
        info.get(k) is not None
        for k in ("regularMarketPrice", "currentPrice", "navPrice")
    )


def _resolve_ticker(raw: str) -> str:
    """Return a validated ticker symbol. Tries raw input first, then yf.Search for company names."""
    # 1. Try raw input directly (handles valid ticker symbols)
    try:
        info = yf.Ticker(raw).info
        if _has_price_data(info):
            return raw
        if not yf.Ticker(raw).history(period="5d").empty:
            return raw
    except Exception:
        pass

    # 2. Resolve via search (handles company names like "micron" → "MU")
    try:
        quotes = yf.Search(raw).quotes
        # Prefer US-listed equities first
        for q in quotes:
            if q.get("quoteType") == "EQUITY" and q.get("exchange") in _US_EXCHANGES:
                symbol = q["symbol"]
                if _has_price_data(yf.Ticker(symbol).info):
                    return symbol
        # Fallback: first result regardless of exchange or type
        for q in quotes:
            symbol = q.get("symbol", "")
            if symbol and _has_price_data(yf.Ticker(symbol).info):
                return symbol
    except Exception:
        pass

    raise ValueError(
        f"Could not find a valid ticker for '{raw}'. "
        "Please enter a ticker symbol directly (e.g. MU for Micron Technology)."
    )


def ticker_input_node(state: GraphState) -> dict:
    raw = state["ticker"].strip()
    try:
        ticker = _resolve_ticker(raw.upper())
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not validate '{raw}': {e}")
    return {"ticker": ticker}


async def load_memory_node(state: GraphState) -> dict:
    """Loads the 3 most recent analyses for this ticker from SQLite before the agent fan-out."""
    ticker = state["ticker"]
    rows = await asyncio.to_thread(get_recent_analyses, ticker, _USER_ID, 3)
    return {"prior_analyses": rows}


async def persist_memory_node(state: GraphState) -> dict:
    """Saves the completed analysis to SQLite after the report is rendered."""
    await asyncio.to_thread(save_analysis, _USER_ID, state)
    return {}


_SIGNAL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
_REC_EMOJI    = {"BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}
_DIR_EMOJI    = {"rise": "▲", "fall": "▼", "neutral": "─"}
_DISCLAIMER   = (
    "⚠️  This report is for informational purposes only and does not constitute financial advice. "
    "Always do your own research before making investment decisions."
)


def format_report(state: GraphState) -> str:
    ticker = state.get("ticker", "N/A")
    lines = [
        "=" * 60,
        f"  STOCK ANALYSIS REPORT — {ticker}",
        "=" * 60,
    ]

    mkt = state.get("market_report")
    if mkt:
        lines.append("\n[BROAD MARKET]")
        for idx in mkt.indices:
            dir_emoji = _DIR_EMOJI.get(idx.direction, "─")
            last_str = f"${idx.last:,.2f}" if idx.last is not None else "N/A"
            chg_str = f"{idx.change_pct:+.2f}%" if idx.change_pct is not None else ""
            lines.append(f"  {dir_emoji} {idx.name} ({idx.symbol}): {last_str}  {chg_str}")
        if mkt.trend_forecast:
            lines.append(f"  Forecast: {mkt.trend_forecast}")

    for field, label in [
        ("fundamentals_signal", "Fundamentals"),
        ("technical_signal",    "Technical"),
        ("sentiment_signal",    "Sentiment"),
        ("macro_signal",        "Macro"),
        ("risk_signal",         "Risk"),
    ]:
        sig = state.get(field)
        if sig is None:
            lines.append(f"\n[{label}]  ⚠️  No data")
            continue
        emoji = _SIGNAL_EMOJI.get(sig.signal, "🟡")
        lines.append(f"\n[{label}]  {emoji} {sig.signal.upper()}  (confidence: {sig.confidence:.0%})")
        if sig.error:
            lines.append(f"  ⚠️  {sig.error}")
        for bullet in sig.bullets:
            if isinstance(bullet, dict):
                lines.append(f"  • {bullet.get('text', '')}")
                if bullet.get("source"):
                    lines.append(f"    ↳ Source: {bullet['source']}")
                if bullet.get("url"):
                    lines.append(f"    ↳ Link: {bullet['url']}")
            else:
                lines.append(f"  • {bullet}")

    lines.append("\n" + "=" * 60)

    rec = state.get("recommendation", "N/A")
    score = state.get("score")
    score_str = f"  |  score: {score:+.4f}" if score is not None else ""
    lines.append(f"  RECOMMENDATION:  {_REC_EMOJI.get(rec, '')} {rec}{score_str}")

    lines.append("=" * 60)

    summary = state.get("summary")
    if summary:
        lines.append(f"\nSummary:\n{summary}")

    lines.append("\n" + "=" * 60)
    lines.append(_DISCLAIMER)
    lines.append("=" * 60)
    return "\n".join(lines)


def report_node(state: GraphState) -> dict:
    return {"report": format_report(state)}


def scope_guard_node(state: GraphState) -> dict:
    """Resets query_plan so stale state from a previous turn cannot bleed in.
    Routing decisions (out_of_scope vs asked_about_other_stock) are delegated to the planner.
    """
    return {"query_plan": None}


def _route_entry(state: GraphState) -> str:
    if state.get("question") and state.get("recommendation"):
        return "scope_guard"
    return "ticker_input"


def _route_after_planner(state: GraphState) -> list:
    plan = state.get("query_plan")
    if not plan or plan.out_of_scope or plan.answer_from_state:
        return [Send("responder", state)]
    if plan.asked_about_other_stock:
        # Fan out to all 5 agents — no state to answer from for the other ticker
        return [Send(f"{agent}_qa", state) for agent in _QA_AGENTS]
    valid = [a for a in plan.relevant_agents if a in _QA_AGENTS]
    if not valid:
        return [Send("responder", state)]
    return [Send(f"{agent}_qa", state) for agent in valid]


_ALLOWED_MODULES = [
    ("stock_analyser.state", "AgentSignal"),
    ("stock_analyser.state", "MarketReport"),
    ("stock_analyser.state", "IndexSnapshot"),
    ("stock_analyser.state", "QueryPlan"),
    ("stock_analyser.state", "AgentAnswer"),
]


def build_graph() -> StateGraph:
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MODULES)
    checkpointer = MemorySaver(serde=serde)
    builder = StateGraph(GraphState)

    # ── Analysis nodes ────────────────────────────────────────────────────────
    builder.add_node("ticker_input", ticker_input_node)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("fundamentals", fundamentals_agent)
    builder.add_node("technical", technical_agent)
    builder.add_node("sentiment", sentiment_agent)
    builder.add_node("macro", macro_agent)
    builder.add_node("risk", risk_agent)
    builder.add_node("market", market_node)
    builder.add_node("synthesis", synthesis_agent)
    builder.add_node("report", report_node)
    builder.add_node("persist_memory", persist_memory_node)

    # ── Q&A nodes ─────────────────────────────────────────────────────────────
    builder.add_node("scope_guard", scope_guard_node)
    builder.add_node("planner", planner_agent)
    builder.add_node("fundamentals_qa", fundamentals_qa_agent)
    builder.add_node("technical_qa", technical_qa_agent)
    builder.add_node("sentiment_qa", sentiment_qa_agent)
    builder.add_node("macro_qa", macro_qa_agent)
    builder.add_node("risk_qa", risk_qa_agent)
    builder.add_node("responder", responder_agent)

    # ── Entry routing ─────────────────────────────────────────────────────────
    builder.add_conditional_edges(
        START,
        _route_entry,
        {"ticker_input": "ticker_input", "scope_guard": "scope_guard"},
    )
    builder.add_edge("scope_guard", "planner")

    # ── Analysis path: ticker_input → load_memory → fan-out → synthesis → report → persist → END
    builder.add_edge("ticker_input", "load_memory")
    for agent in _ANALYSIS_AGENTS:
        builder.add_edge("load_memory", agent)
    for agent in _ANALYSIS_AGENTS:
        builder.add_edge(agent, "synthesis")
    builder.add_edge("synthesis", "report")
    builder.add_edge("report", "persist_memory")
    builder.add_edge("persist_memory", END)

    # ── Q&A path ──────────────────────────────────────────────────────────────
    builder.add_conditional_edges("planner", _route_after_planner)
    for agent in _QA_AGENTS:
        builder.add_edge(f"{agent}_qa", "responder")
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
