from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from api import app
from stock_analyser.state import AgentSignal, QueryPlan

BULLET = {"text": "Revenue grew", "source": "Yahoo Finance — Income Statement", "url": "https://finance.yahoo.com"}

_MOCK_ANALYSE_RESULT = {
    "ticker": "AAPL",
    "fundamentals_signal": AgentSignal(signal="bullish", confidence=0.8, bullets=[BULLET]),
    "technical_signal":    AgentSignal(signal="bullish", confidence=0.7, bullets=[BULLET]),
    "sentiment_signal":    AgentSignal(signal="neutral", confidence=0.5, bullets=[BULLET]),
    "macro_signal":        AgentSignal(signal="neutral", confidence=0.5, bullets=[BULLET]),
    "risk_signal":         AgentSignal(signal="bullish", confidence=0.75, bullets=[BULLET]),
    "market_report": None,
    "recommendation": "BUY",
    "score": 0.42,
    "summary": "Apple looks solid.",
    "report": "=== REPORT ===",
}

_MOCK_QA_RESULT = {
    "answer": "Apple's P/E ratio is approximately 28.",
    "query_plan": QueryPlan(
        relevant_agents=["fundamentals"],
        reasoning="P/E is a fundamentals metric",
        needs_fresh_data=False,
        answer_from_state=False,
        resolved_question="What is AAPL's P/E ratio?",
    ),
}

_MOCK_OTHER_STOCK_QA_RESULT = {
    "answer": "Microsoft's P/E ratio is approximately 34.",
    "query_plan": QueryPlan(
        relevant_agents=["fundamentals", "technical", "sentiment", "macro", "risk"],
        reasoning="Question about MSFT",
        needs_fresh_data=True,
        answer_from_state=False,
        asked_about_other_stock=True,
        other_ticker="MSFT",
    ),
}

_MOCK_OUT_OF_SCOPE_QA_RESULT = {
    "answer": "That question is outside what I can help with here.",
    "query_plan": QueryPlan(
        relevant_agents=[],
        reasoning="Not a stock question",
        needs_fresh_data=False,
        answer_from_state=False,
        out_of_scope=True,
    ),
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── /analyse endpoint ─────────────────────────────────────────────────────────

def test_analyse_valid_ticker_returns_200(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_ANALYSE_RESULT
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-1"})
    assert r.status_code == 200


def test_analyse_response_has_required_fields(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_ANALYSE_RESULT
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-2"})
    data = r.json()
    for field in ("ticker", "recommendation", "score", "summary", "report"):
        assert field in data, f"Missing field: {field}"


def test_analyse_recommendation_is_valid(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_ANALYSE_RESULT
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-3"})
    assert r.json()["recommendation"] in ("BUY", "HOLD", "SELL")


def test_analyse_invalid_ticker_returns_422(client):
    with patch("stock_analyser.graph._resolve_ticker",
               side_effect=ValueError("Could not find a valid ticker for 'ZZZZNOTREAL'")):
        r = client.post("/analyse", json={"ticker": "ZZZZNOTREAL", "thread_id": "test-analyse-4"})
    assert r.status_code == 422


def test_analyse_internal_error_returns_500(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.side_effect = RuntimeError("unexpected graph error")
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-5"})
    assert r.status_code == 500


def test_analyse_signals_serialised_as_dicts(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_ANALYSE_RESULT
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-6"})
    data = r.json()
    sig = data.get("fundamentals_signal")
    assert isinstance(sig, dict)
    assert "signal" in sig
    assert "confidence" in sig
    assert "bullets" in sig


def test_analyse_excludes_internal_fields(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_ANALYSE_RESULT
        r = client.post("/analyse", json={"ticker": "AAPL", "thread_id": "test-analyse-7"})
    data = r.json()
    for field in ("messages", "agent_answers", "prior_analyses", "technical_indicators"):
        assert field not in data, f"Internal field leaked: {field}"


# ── /qa endpoint ──────────────────────────────────────────────────────────────

def test_qa_returns_200(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_QA_RESULT
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "What is the P/E?", "thread_id": "test-qa-1"
        })
    assert r.status_code == 200


def test_qa_response_has_answer_and_routed_to(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_QA_RESULT
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "What is the P/E?", "thread_id": "test-qa-2"
        })
    data = r.json()
    assert "answer" in data
    assert "routed_to" in data
    assert isinstance(data["routed_to"], list)


def test_qa_routed_to_reflects_plan_agents(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_QA_RESULT
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "What is the P/E?", "thread_id": "test-qa-3"
        })
    assert r.json()["routed_to"] == ["fundamentals"]


def test_qa_other_stock_routes_to_all_five(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_OTHER_STOCK_QA_RESULT
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "How is Microsoft doing?", "thread_id": "test-qa-4"
        })
    data = r.json()
    assert len(data["routed_to"]) == 5


def test_qa_out_of_scope_returns_answer_not_error(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = _MOCK_OUT_OF_SCOPE_QA_RESULT
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "What is the weather?", "thread_id": "test-qa-5"
        })
    assert r.status_code == 200          # never 500 — graceful refusal
    assert r.json()["answer"]
    assert r.json()["routed_to"] == []


def test_qa_graph_error_returns_500(client):
    with patch.object(app.state.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.side_effect = RuntimeError("graph crashed")
        r = client.post("/qa", json={
            "ticker": "AAPL", "question": "What is the P/E?", "thread_id": "test-qa-6"
        })
    assert r.status_code == 500
