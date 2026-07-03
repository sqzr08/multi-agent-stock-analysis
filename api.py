from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_analyser.graph import build_graph

_ANALYSIS_FIELDS = {
    "ticker",
    "fundamentals_signal",
    "technical_signal",
    "sentiment_signal",
    "macro_signal",
    "risk_signal",
    "market_report",
    "recommendation",
    "score",
    "summary",
    "report",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Stock Analyser API", lifespan=lifespan)


def _serialise(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _serialise_state(state: dict, fields: set) -> dict:
    subset = {k: v for k, v in state.items() if k in fields}
    return json.loads(json.dumps(subset, default=_serialise))


class AnalyseRequest(BaseModel):
    ticker: str
    thread_id: str


class QARequest(BaseModel):
    ticker: str
    question: str
    thread_id: str


class QAResponse(BaseModel):
    answer: str
    routed_to: list[str]


@app.post("/analyse")
async def analyse(req: AnalyseRequest) -> dict[str, Any]:
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.thread_id, "user_id": "default"}}
    try:
        result = await graph.ainvoke({"ticker": req.ticker.upper()}, config=config)
        return _serialise_state(result, _ANALYSIS_FIELDS)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest) -> QAResponse:
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.thread_id, "user_id": "default"}}
    try:
        result = await graph.ainvoke(
            {"ticker": req.ticker, "question": req.question},
            config=config,
        )
        plan = result.get("query_plan")
        routed_to = list(plan.relevant_agents) if plan and plan.relevant_agents else []
        answer = result.get("answer", "I was unable to generate an answer.")
        return QAResponse(answer=answer, routed_to=routed_to)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
