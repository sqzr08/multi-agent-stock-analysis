from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import llm
from stock_analyser.state import GraphState, QueryPlan

_VALID_AGENTS = {"fundamentals", "technical", "sentiment", "macro", "risk"}

_PROMPT = """You are a routing agent for a stock analysis system. A user has already received a full \
analysis report for {session_ticker} and is now asking a follow-up question. Your job is to:
1. Resolve any cross-turn references ("that", "the P/E you mentioned", "compare it to last time").
2. Decide which specialist agents (if any) need to run to answer.

SCOPE RULES:
- If the question is about {session_ticker} only → route to the smallest sufficient set of agents (≤3).
- If the question is about a DIFFERENT stock (by ticker, company name, or description such as "Nvidia", \
"the EV maker", "its competitor") → set asked_about_other_stock=true, extract other_ticker, \
set relevant_agents to all five agents, set needs_fresh_data=true, set answer_from_state=false.
- If the question has NOTHING to do with stocks or investing (e.g. weather, sports, cooking) \
→ set out_of_scope=true and relevant_agents=[].
- Comparative questions ("Is {session_ticker} better than MSFT?") count as asked_about_other_stock — \
extract the other ticker and fan out to all agents so both sides can be covered.

User question: {question}

Recent conversation history (most recent last):
{history}

Existing analysis summary (for {session_ticker}):
{state_summary}

Available agents and what they cover:
- fundamentals: valuation ratios, revenue/profit growth, balance sheet health, dividends
- technical: price trends, moving averages, RSI, MACD, support/resistance levels, volume
- sentiment: recent news headlines, media tone, bullish/bearish article breakdown
- macro: interest rates, inflation, employment, sector ETF performance vs market
- risk: volatility, market sensitivity, short interest, insider transactions, earnings date

Note: broad market questions (S&P 500, Dow Jones, Nasdaq, "how is the market doing?") are answered \
from the stored market_report in state — use answer_from_state=true, route to no agents.

Respond ONLY with valid JSON:
{{
  "resolved_question": "<self-contained, de-referenced version of the question — no pronouns, no 'that', no 'it'>",
  "relevant_agents": ["agent_name", ...],
  "reasoning": "<one sentence why>",
  "needs_fresh_data": true | false,
  "answer_from_state": true | false,
  "out_of_scope": false,
  "asked_about_other_stock": true | false,
  "other_ticker": "<UPPERCASE ticker symbol or null>"
}}

Rules:
- resolved_question must be fully self-contained so a fresh agent with no history can understand it.
- For {session_ticker} questions: answer_from_state=true if existing analysis is sufficient; \
  otherwise route to ≤3 agents.
- For asked_about_other_stock=true: relevant_agents must be all five \
  ["fundamentals","technical","sentiment","macro","risk"], needs_fresh_data=true, answer_from_state=false.
- For out_of_scope=true: relevant_agents=[], answer_from_state=false.
- asked_about_other_stock and out_of_scope are mutually exclusive — only one can be true.
- other_ticker must be a valid uppercase ticker symbol (e.g. "MSFT"), or null if not applicable.
- Agent names must be from: fundamentals, technical, sentiment, macro, risk."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in response.content)
    return str(response.content)


def _format_history(messages: list) -> str:
    if not messages:
        return "(no prior conversation)"
    # Take the last 6 messages (3 exchanges) to keep the prompt focused
    recent = messages[-6:]
    lines = []
    for msg in recent:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = str(msg.content)[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_state_summary(state: GraphState) -> str:
    lines = [f"Ticker: {state.get('ticker', 'N/A')}"]
    lines.append(
        f"Recommendation: {state.get('recommendation', 'N/A')} "
        f"(score: {state.get('score', 'N/A')})"
    )

    for field, label in [
        ("fundamentals_signal", "Fundamentals"),
        ("technical_signal", "Technical"),
        ("sentiment_signal", "Sentiment"),
        ("macro_signal", "Macro"),
        ("risk_signal", "Risk"),
    ]:
        sig = state.get(field)
        if sig:
            bullets_preview = " | ".join(
                b.get("text", "")[:80] for b in sig.bullets[:2] if isinstance(b, dict)
            )
            lines.append(f"{label}: {sig.signal} ({sig.confidence:.0%}) — {bullets_preview}")

    mkt = state.get("market_report")
    if mkt:
        idx_strs = [
            f"{i.name} {'+' if i.direction == 'rise' else '-'}{abs(i.change_pct or 0):.2f}%"
            for i in mkt.indices
        ]
        lines.append(f"Market today: {', '.join(idx_strs)}")
        lines.append(f"Market summary: {mkt.summary}")

    prior = state.get("prior_analyses") or []
    if prior:
        p = prior[0]
        lines.append(
            f"Previous analysis ({p.get('date')}): {p.get('recommendation')} "
            f"(score: {p.get('score')})"
        )

    return "\n".join(lines)


def planner_agent(state: GraphState) -> dict:
    question = state.get("question", "")
    session_ticker = state.get("ticker", "")
    history = _format_history(state.get("messages") or [])
    state_summary = _build_state_summary(state)

    try:
        text = _call_llm(_PROMPT.format(
            question=question,
            session_ticker=session_ticker,
            history=history,
            state_summary=state_summary,
        ))
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())

        out_of_scope = bool(parsed.get("out_of_scope", False))
        asked_about_other = bool(parsed.get("asked_about_other_stock", False))
        other_ticker = parsed.get("other_ticker") or None
        if asked_about_other:
            agents = list(_VALID_AGENTS)
        elif out_of_scope:
            agents = []
        else:
            agents = [a for a in parsed.get("relevant_agents", []) if a in _VALID_AGENTS]
        plan = QueryPlan(
            relevant_agents=agents,
            reasoning=parsed.get("reasoning", ""),
            needs_fresh_data=bool(parsed.get("needs_fresh_data", False)),
            answer_from_state=bool(parsed.get("answer_from_state", not agents)),
            resolved_question=parsed.get("resolved_question") or question,
            out_of_scope=out_of_scope,
            asked_about_other_stock=asked_about_other,
            other_ticker=other_ticker,
        )
    except Exception:
        try:
            strict = (
                f"Respond ONLY with JSON. Question: {question}\n"
                'Format: {"resolved_question":"<question>","relevant_agents":[],'
                '"reasoning":"fallback","needs_fresh_data":false,"answer_from_state":true,'
                '"out_of_scope":false,"asked_about_other_stock":false,"other_ticker":null}'
            )
            text = _call_llm(strict)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            plan = QueryPlan(
                relevant_agents=[a for a in parsed.get("relevant_agents", []) if a in _VALID_AGENTS],
                reasoning=parsed.get("reasoning", "fallback"),
                needs_fresh_data=False,
                answer_from_state=True,
                resolved_question=parsed.get("resolved_question") or question,
                out_of_scope=bool(parsed.get("out_of_scope", False)),
                asked_about_other_stock=False,
                other_ticker=None,
            )
        except Exception as e:
            plan = QueryPlan(
                relevant_agents=[],
                reasoning=f"Planner error: {e}",
                needs_fresh_data=False,
                answer_from_state=True,
                resolved_question=question,
                out_of_scope=False,
                asked_about_other_stock=False,
                other_ticker=None,
            )

    return {"query_plan": plan}
