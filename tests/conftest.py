from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from stock_analyser.state import (
    AgentSignal, AgentAnswer, MarketReport, IndexSnapshot, QueryPlan,
)


# ── Reusable model instances ──────────────────────────────────────────────────

BULLET = {"text": "Revenue grew 10%", "source": "Yahoo Finance — Income Statement", "url": "https://finance.yahoo.com"}

@pytest.fixture
def bullish_signal():
    return AgentSignal(signal="bullish", confidence=0.8, bullets=[BULLET])

@pytest.fixture
def bearish_signal():
    return AgentSignal(signal="bearish", confidence=0.8, bullets=[BULLET])

@pytest.fixture
def neutral_signal():
    return AgentSignal(signal="neutral", confidence=0.5, bullets=[BULLET])

@pytest.fixture
def sample_market_report():
    return MarketReport(
        indices=[IndexSnapshot(name="S&P 500", symbol="^GSPC", last=5200.0, change_pct=0.5, direction="rise")],
        trend_forecast="Markets moved higher today.",
        summary="Broad risk-on sentiment. Watch Fed commentary.",
    )

@pytest.fixture
def base_state(bullish_signal, sample_market_report):
    """Minimal GraphState dict representing a completed analysis for AAPL."""
    return {
        "ticker": "AAPL",
        "fundamentals_signal": bullish_signal,
        "technical_signal": bullish_signal,
        "sentiment_signal": neutral_signal,
        "macro_signal": neutral_signal,
        "risk_signal": bullish_signal,
        "market_report": sample_market_report,
        "technical_indicators": {"current_price": 185.0, "rsi_14": 55.0},
        "recommendation": "BUY",
        "score": 0.42,
        "summary": "Apple looks solid.",
        "report": "=== REPORT ===",
        "prior_analyses": [],
        "messages": [],
        "question": None,
        "query_plan": None,
        "agent_answers": {},
        "answer": None,
    }

@pytest.fixture
def session_plan():
    """QueryPlan for a normal session-ticker Q&A."""
    return QueryPlan(
        relevant_agents=["fundamentals"],
        reasoning="P/E is a fundamentals metric",
        needs_fresh_data=False,
        answer_from_state=False,
        resolved_question="What is AAPL's trailing P/E ratio?",
    )

@pytest.fixture
def other_stock_plan():
    """QueryPlan for a question about a different stock."""
    return QueryPlan(
        relevant_agents=["fundamentals", "technical", "sentiment", "macro", "risk"],
        reasoning="Question is about MSFT, not the session ticker",
        needs_fresh_data=True,
        answer_from_state=False,
        resolved_question="What is MSFT's trailing P/E ratio?",
        asked_about_other_stock=True,
        other_ticker="MSFT",
    )

@pytest.fixture
def out_of_scope_plan():
    return QueryPlan(
        relevant_agents=[],
        reasoning="Not a stock question",
        needs_fresh_data=False,
        answer_from_state=False,
        out_of_scope=True,
        resolved_question="What is the weather today?",
    )


# ── LLM mock helpers ──────────────────────────────────────────────────────────

class MockLLMResponse:
    """Mimics a LangChain LLM response with no tool calls."""
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []


def make_mock_llm(json_content: str) -> MagicMock:
    """Returns a mock LLM whose bind_tools().invoke() returns json_content."""
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.return_value = MockLLMResponse(json_content)
    mock_llm.invoke.return_value = MockLLMResponse(json_content)
    return mock_llm


VALID_SIGNAL_JSON = (
    '{"signal":"bullish","confidence":0.75,'
    '"bullets":[{"text":"Revenue grew","source":"Yahoo Finance — Income Statement","url":"https://finance.yahoo.com"}]}'
)

VALID_ANSWER_JSON = (
    '{"answer":"The P/E ratio is 28.","sources":[{"source":"Yahoo Finance","url":"https://finance.yahoo.com"}],"confidence":0.8}'
)
