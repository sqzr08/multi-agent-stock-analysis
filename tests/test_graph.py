from __future__ import annotations

import pytest

from stock_analyser.graph import (
    _route_after_planner,
    _QA_AGENTS,
)
from stock_analyser.state import QueryPlan


def _make_plan(**kwargs) -> QueryPlan:
    defaults = dict(
        relevant_agents=[],
        reasoning="",
        needs_fresh_data=False,
        answer_from_state=False,
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def _make_state(plan=None, ticker="AAPL") -> dict:
    return {"query_plan": plan, "ticker": ticker}


# ── _route_after_planner ──────────────────────────────────────────────────────

def test_route_out_of_scope_goes_to_responder():
    plan = _make_plan(out_of_scope=True)
    sends = _route_after_planner(_make_state(plan))
    assert len(sends) == 1
    assert sends[0].node == "responder"

def test_route_answer_from_state_goes_to_responder():
    plan = _make_plan(answer_from_state=True)
    sends = _route_after_planner(_make_state(plan))
    assert len(sends) == 1
    assert sends[0].node == "responder"

def test_route_no_plan_goes_to_responder():
    sends = _route_after_planner(_make_state(plan=None))
    assert len(sends) == 1
    assert sends[0].node == "responder"

def test_route_asked_about_other_stock_fans_out_all_five():
    plan = _make_plan(
        relevant_agents=list(_QA_AGENTS),
        answer_from_state=False,
        needs_fresh_data=True,
        asked_about_other_stock=True,
        other_ticker="MSFT",
    )
    sends = _route_after_planner(_make_state(plan))
    nodes = {s.node for s in sends}
    assert nodes == {f"{a}_qa" for a in _QA_AGENTS}
    assert len(sends) == 5

def test_route_subset_agents_for_session_ticker():
    plan = _make_plan(
        relevant_agents=["fundamentals", "risk"],
        answer_from_state=False,
    )
    sends = _route_after_planner(_make_state(plan))
    nodes = {s.node for s in sends}
    assert nodes == {"fundamentals_qa", "risk_qa"}

def test_route_no_valid_agents_falls_back_to_responder():
    plan = _make_plan(
        relevant_agents=["nonexistent_agent"],
        answer_from_state=False,
    )
    sends = _route_after_planner(_make_state(plan))
    assert len(sends) == 1
    assert sends[0].node == "responder"

def test_route_asked_about_other_stock_ignores_relevant_agents_list():
    # Even if relevant_agents is empty, all 5 should fan out
    plan = _make_plan(
        relevant_agents=[],
        answer_from_state=False,
        asked_about_other_stock=True,
        other_ticker="TSLA",
        needs_fresh_data=True,
    )
    sends = _route_after_planner(_make_state(plan))
    assert len(sends) == 5


# ── _resolve_ticker (live API) ────────────────────────────────────────────────

def test_resolve_ticker_direct_symbol():
    from stock_analyser.graph import _resolve_ticker
    assert _resolve_ticker("AAPL") == "AAPL"

def test_resolve_ticker_falls_back_to_search():
    from stock_analyser.graph import _resolve_ticker
    assert _resolve_ticker("micron") == "MU"

def test_resolve_ticker_raises_on_no_match():
    from stock_analyser.graph import _resolve_ticker
    with pytest.raises(ValueError, match="Could not find a valid ticker"):
        _resolve_ticker("ZZZZNOTREAL")
