from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from stock_analyser.state import AgentSignal, AgentAnswer, QueryPlan
from stock_analyser.agents.planner import planner_agent
from stock_analyser.agents.fundamentals import fundamentals_agent, fundamentals_qa_agent
from stock_analyser.agents.technical import technical_agent, technical_qa_agent
from stock_analyser.agents.sentiment import sentiment_agent, sentiment_qa_agent
from stock_analyser.agents.macro import macro_agent, macro_qa_agent
from stock_analyser.agents.risk import risk_agent, risk_qa_agent
from stock_analyser.agents.responder import responder_agent
from stock_analyser.tools.cache import _bypass


BULLET = {"text": "Revenue grew 10%", "source": "Yahoo Finance — Income Statement", "url": "https://finance.yahoo.com"}

def _signal(direction: str, confidence: float = 0.8) -> AgentSignal:
    return AgentSignal(signal=direction, confidence=confidence, bullets=[BULLET])

def _all_signals(direction: str) -> dict:
    return {
        "fundamentals_signal": _signal(direction),
        "technical_signal":    _signal(direction),
        "sentiment_signal":    _signal(direction),
        "macro_signal":        _signal(direction),
        "risk_signal":         _signal(direction),
    }

def _base_state(**overrides) -> dict:
    state = {
        "ticker": "AAPL",
        "market_report": None,
        "technical_indicators": {},
        "recommendation": None,
        "score": None,
        "summary": None,
        "report": None,
        "prior_analyses": [],
        "messages": [],
        "question": None,
        "query_plan": None,
        "agent_answers": {},
        "answer": None,
        **_all_signals("neutral"),
    }
    state.update(overrides)
    return state


# ── planner_agent ─────────────────────────────────────────────────────────────

@patch("stock_analyser.agents.planner._call_llm")
def test_planner_sets_asked_about_other_stock(mock_llm):
    mock_llm.return_value = json.dumps({
        "resolved_question": "What is MSFT's P/E ratio?",
        "relevant_agents": ["fundamentals", "technical", "sentiment", "macro", "risk"],
        "reasoning": "Question is about MSFT",
        "needs_fresh_data": True,
        "answer_from_state": False,
        "out_of_scope": False,
        "asked_about_other_stock": True,
        "other_ticker": "MSFT",
    })
    state = _base_state(question="How is Microsoft doing?")
    result = planner_agent(state)
    plan = result["query_plan"]
    assert plan.asked_about_other_stock is True
    assert plan.other_ticker == "MSFT"
    assert plan.needs_fresh_data is True
    assert len(plan.relevant_agents) == 5
    assert not plan.out_of_scope


@patch("stock_analyser.agents.planner._call_llm")
def test_planner_sets_out_of_scope_for_non_stock_question(mock_llm):
    mock_llm.return_value = json.dumps({
        "resolved_question": "What is the weather today?",
        "relevant_agents": [],
        "reasoning": "Not a stock question",
        "needs_fresh_data": False,
        "answer_from_state": False,
        "out_of_scope": True,
        "asked_about_other_stock": False,
        "other_ticker": None,
    })
    state = _base_state(question="What is the weather today?")
    result = planner_agent(state)
    plan = result["query_plan"]
    assert plan.out_of_scope is True
    assert plan.asked_about_other_stock is False
    assert plan.relevant_agents == []


@patch("stock_analyser.agents.planner._call_llm")
def test_planner_fallback_plan_on_llm_error(mock_llm):
    mock_llm.side_effect = Exception("LLM down")
    state = _base_state(question="What is the P/E?")
    result = planner_agent(state)
    plan = result["query_plan"]
    # Should return a safe fallback plan, not raise
    assert plan is not None
    assert isinstance(plan.relevant_agents, list)


@patch("stock_analyser.agents.fundamentals._run_tool_loop")
def test_fundamentals_qa_returns_agent_answer_shape(mock_loop):
    mock_loop.return_value = MagicMock(
        content='{"answer":"P/E is 28.","sources":[{"source":"Yahoo Finance","url":"https://finance.yahoo.com"}],"confidence":0.85}'
    )
    state = _base_state(
        question="What is the P/E?",
        query_plan=QueryPlan(
            relevant_agents=["fundamentals"],
            reasoning="",
            needs_fresh_data=False,
            answer_from_state=False,
            resolved_question="What is AAPL's P/E?",
        ),
    )
    result = fundamentals_qa_agent(state)
    assert "agent_answers" in result
    assert "fundamentals" in result["agent_answers"]
    answer = result["agent_answers"]["fundamentals"]
    assert isinstance(answer, AgentAnswer)
    assert answer.answer
    assert 0.0 <= answer.confidence <= 1.0


@patch("stock_analyser.agents.fundamentals._run_tool_loop")
def test_fundamentals_qa_returns_fallback_on_error(mock_loop):
    mock_loop.side_effect = Exception("tool failed")
    state = _base_state(
        question="What is the P/E?",
        query_plan=QueryPlan(
            relevant_agents=["fundamentals"],
            reasoning="",
            needs_fresh_data=False,
            answer_from_state=False,
        ),
    )
    result = fundamentals_qa_agent(state)
    assert "agent_answers" in result
    answer = result["agent_answers"]["fundamentals"]
    assert answer.confidence == 0.0


# ── responder_agent ───────────────────────────────────────────────────────────

@patch("stock_analyser.agents.responder._call_llm")
def test_responder_returns_answer(mock_llm):
    mock_llm.return_value = "Apple's P/E is 28, which is above the market average."
    state = _base_state(
        question="What is the P/E?",
        query_plan=QueryPlan(
            relevant_agents=[],
            reasoning="",
            needs_fresh_data=False,
            answer_from_state=True,
            resolved_question="What is AAPL's P/E ratio?",
        ),
    )
    result = responder_agent(state)
    assert result["answer"]
    assert len(result["messages"]) == 2   # HumanMessage + AIMessage


def test_responder_out_of_scope_no_llm_call():
    state = _base_state(
        question="What is the weather?",
        query_plan=QueryPlan(
            relevant_agents=[],
            reasoning="not about stocks",
            needs_fresh_data=False,
            answer_from_state=False,
            out_of_scope=True,
        ),
    )
    with patch("stock_analyser.agents.responder._call_llm") as mock_llm:
        result = responder_agent(state)
        mock_llm.assert_not_called()
    assert result["answer"]
    # Refusal should mention the user can still ask about other stocks
    assert "stock" in result["answer"].lower()


# ── Analysis agents: signal key + neutral fallback ────────────────────────────

VALID_SIGNAL = (
    '{"signal":"bullish","confidence":0.8,"bullets":[{"text":"Revenue grew",'
    '"source":"Yahoo Finance — Income Statement","url":"https://finance.yahoo.com"}]}'
)


@patch("stock_analyser.agents.fundamentals._run_tool_loop")
def test_fundamentals_agent_returns_signal(mock_loop):
    mock_loop.return_value = MagicMock(content=VALID_SIGNAL)
    result = fundamentals_agent({"ticker": "AAPL"})
    assert "fundamentals_signal" in result
    assert result["fundamentals_signal"].signal == "bullish"


@patch("stock_analyser.agents.fundamentals._run_tool_loop")
def test_fundamentals_agent_returns_neutral_on_error(mock_loop):
    mock_loop.side_effect = Exception("tool failed")
    result = fundamentals_agent({"ticker": "AAPL"})
    sig = result["fundamentals_signal"]
    assert sig.signal == "neutral"
    assert sig.confidence == 0.0
    assert sig.error


@patch("stock_analyser.agents.technical._call_llm")
@patch("stock_analyser.agents.technical.compute_indicators")
@patch("stock_analyser.agents.technical.get_price_history")
def test_technical_agent_returns_signal_and_indicators(mock_price, mock_compute, mock_llm):
    mock_price.invoke = MagicMock(return_value={"close": [150.0]})
    mock_compute.return_value = {"rsi_14": 55.0, "current_price": 150.0}
    mock_llm.return_value = VALID_SIGNAL
    result = technical_agent({"ticker": "AAPL"})
    assert "technical_signal" in result
    assert "technical_indicators" in result
    assert result["technical_signal"].signal == "bullish"
    assert result["technical_indicators"] == {"rsi_14": 55.0, "current_price": 150.0}


@patch("stock_analyser.agents.technical.get_price_history")
def test_technical_agent_returns_neutral_on_error(mock_price):
    mock_price.invoke = MagicMock(side_effect=Exception("API down"))
    result = technical_agent({"ticker": "AAPL"})
    assert result["technical_signal"].signal == "neutral"
    assert result["technical_signal"].confidence == 0.0
    assert result["technical_indicators"] == {}


@patch("stock_analyser.agents.sentiment._run_tool_loop")
def test_sentiment_agent_returns_signal(mock_loop):
    mock_loop.return_value = MagicMock(content=VALID_SIGNAL)
    result = sentiment_agent({"ticker": "AAPL"})
    assert "sentiment_signal" in result
    assert result["sentiment_signal"].signal == "bullish"


@patch("stock_analyser.agents.sentiment._run_tool_loop")
def test_sentiment_agent_returns_neutral_on_error(mock_loop):
    mock_loop.side_effect = Exception("API down")
    result = sentiment_agent({"ticker": "AAPL"})
    assert result["sentiment_signal"].signal == "neutral"
    assert result["sentiment_signal"].confidence == 0.0


@patch("stock_analyser.agents.macro._run_tool_loop")
def test_macro_agent_returns_signal(mock_loop):
    mock_loop.return_value = MagicMock(content=VALID_SIGNAL)
    with patch("stock_analyser.agents.macro._get_sector_etf", return_value=("Technology", "XLK")):
        result = macro_agent({"ticker": "AAPL"})
    assert "macro_signal" in result
    assert result["macro_signal"].signal == "bullish"


@patch("stock_analyser.agents.macro._run_tool_loop")
def test_macro_agent_returns_neutral_on_error(mock_loop):
    mock_loop.side_effect = Exception("FRED unavailable")
    with patch("stock_analyser.agents.macro._get_sector_etf", return_value=("Technology", "XLK")):
        result = macro_agent({"ticker": "AAPL"})
    assert result["macro_signal"].signal == "neutral"
    assert result["macro_signal"].confidence == 0.0


@patch("stock_analyser.agents.risk._run_tool_loop")
def test_risk_agent_returns_signal(mock_loop):
    mock_loop.return_value = MagicMock(content=VALID_SIGNAL)
    result = risk_agent({"ticker": "AAPL"})
    assert "risk_signal" in result
    assert result["risk_signal"].signal == "bullish"


@patch("stock_analyser.agents.risk._run_tool_loop")
def test_risk_agent_returns_neutral_on_error(mock_loop):
    mock_loop.side_effect = Exception("risk data unavailable")
    result = risk_agent({"ticker": "AAPL"})
    assert result["risk_signal"].signal == "neutral"
    assert result["risk_signal"].confidence == 0.0


# ── Remaining QA agents: bypass + fallback ───────────────────────────────────

def _other_stock_plan(other_ticker: str = "MSFT") -> QueryPlan:
    return QueryPlan(
        relevant_agents=["fundamentals", "technical", "sentiment", "macro", "risk"],
        reasoning="",
        needs_fresh_data=True,
        answer_from_state=False,
        asked_about_other_stock=True,
        other_ticker=other_ticker,
    )

def _session_plan_qa(**kwargs) -> QueryPlan:
    defaults = dict(
        relevant_agents=["technical"],
        reasoning="",
        needs_fresh_data=False,
        answer_from_state=False,
        resolved_question="What is the RSI?",
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


# --- technical_qa_agent ---

@patch("stock_analyser.agents.technical._call_llm")
@patch("stock_analyser.agents.technical.compute_indicators")
@patch("stock_analyser.agents.technical.get_price_history")
def test_technical_qa_sets_bypass_for_other_stock(mock_price, mock_compute, mock_llm):
    bypass_seen = []
    def capture(*args, **kwargs):
        bypass_seen.append(_bypass.get())
        return '{"answer":"ok","sources":[],"confidence":0.5}'
    mock_price.invoke = MagicMock(return_value={"close": [300.0]})
    mock_compute.return_value = {"rsi_14": 60.0}
    mock_llm.side_effect = capture
    state = _base_state(question="How is MSFT?", query_plan=_other_stock_plan("MSFT"))
    technical_qa_agent(state)
    assert bypass_seen[0] is True


@patch("stock_analyser.agents.technical.get_price_history")
def test_technical_qa_returns_fallback_on_error(mock_price):
    mock_price.invoke = MagicMock(side_effect=Exception("network error"))
    state = _base_state(
        technical_indicators={},
        question="What is the RSI?",
        query_plan=_session_plan_qa(needs_fresh_data=True),
    )
    result = technical_qa_agent(state)
    assert "technical" in result["agent_answers"]
    assert result["agent_answers"]["technical"].confidence == 0.0


# --- sentiment_qa_agent ---

@patch("stock_analyser.agents.sentiment._run_tool_loop")
def test_sentiment_qa_returns_fallback_on_error(mock_loop):
    mock_loop.side_effect = Exception("API rate limit")
    state = _base_state(question="How is news?", query_plan=_session_plan_qa(relevant_agents=["sentiment"]))
    result = sentiment_qa_agent(state)
    assert "sentiment" in result["agent_answers"]
    assert result["agent_answers"]["sentiment"].confidence == 0.0


# --- macro_qa_agent ---

@patch("stock_analyser.agents.macro._run_tool_loop")
def test_macro_qa_returns_fallback_on_error(mock_loop):
    mock_loop.side_effect = Exception("FRED down")
    state = _base_state(question="What are rates?", query_plan=_session_plan_qa(relevant_agents=["macro"]))
    result = macro_qa_agent(state)
    assert "macro" in result["agent_answers"]
    assert result["agent_answers"]["macro"].confidence == 0.0


# --- risk_qa_agent ---

@patch("stock_analyser.agents.risk._run_tool_loop")
def test_risk_qa_returns_fallback_on_error(mock_loop):
    mock_loop.side_effect = Exception("risk data unavailable")
    state = _base_state(question="What is the beta?", query_plan=_session_plan_qa(relevant_agents=["risk"]))
    result = risk_qa_agent(state)
    assert "risk" in result["agent_answers"]
    assert result["agent_answers"]["risk"].confidence == 0.0
