from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages


class AgentSignal(BaseModel):
    signal: str  # "bullish" | "bearish" | "neutral"
    confidence: float = Field(ge=0.0, le=1.0)
    bullets: list[dict] = Field(min_length=1, max_length=5)
    error: Optional[str] = None

    @field_validator("bullets")
    @classmethod
    def validate_bullets(cls, v):
        if not 1 <= len(v) <= 5:
            raise ValueError(f"bullets must have 1–5 items, got {len(v)}")
        for bullet in v:
            if "text" not in bullet or "source" not in bullet:
                raise ValueError("each bullet must have 'text' and 'source' keys")
        return v


class IndexSnapshot(BaseModel):
    name: str
    symbol: str
    last: Optional[float]
    change_pct: Optional[float]
    direction: str  # "rise" | "fall" | "neutral"


class MarketReport(BaseModel):
    indices: list[IndexSnapshot]
    trend_forecast: str
    summary: str


class QueryPlan(BaseModel):
    relevant_agents: list[str]
    reasoning: str
    needs_fresh_data: bool
    answer_from_state: bool
    resolved_question: Optional[str] = None   # de-referenced, self-contained question for QA agents
    out_of_scope: bool = False                # truly unrelated question (not about any stock); refuse
    asked_about_other_stock: bool = False     # question is about a different ticker than session_ticker
    other_ticker: Optional[str] = None        # the other ticker symbol extracted from the question


class AgentAnswer(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


def _merge_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    return {**(a or {}), **(b or {})}


class GraphState(TypedDict):
    ticker: str
    fundamentals_signal: Optional[AgentSignal]
    technical_signal: Optional[AgentSignal]
    sentiment_signal: Optional[AgentSignal]
    macro_signal: Optional[AgentSignal]
    risk_signal: Optional[AgentSignal]
    market_report: Optional[MarketReport]
    technical_indicators: Optional[dict]
    recommendation: Optional[str]
    score: Optional[float]
    summary: Optional[str]
    report: Optional[str]
    prior_analyses: list                          # rows from SQLite, most-recent first
    messages: Annotated[list, add_messages]       # conversation history (HumanMessage / AIMessage)
    # Q&A fields
    question: Optional[str]
    query_plan: Optional[QueryPlan]
    agent_answers: Annotated[Optional[dict], _merge_dicts]
    answer: Optional[str]
