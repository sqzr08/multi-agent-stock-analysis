from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from stock_analyser.state import AgentSignal


def _parse_signal(text: str) -> AgentSignal:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    bullets = []
    for b in parsed["bullets"]:
        if isinstance(b, str):
            bullets.append({"text": b, "source": "", "url": ""})
        else:
            bullets.append({
                "text": b.get("text", ""),
                "source": b.get("source", ""),
                "url": b.get("url", ""),
            })
    bullets = bullets[:5]
    if not bullets:
        raise ValueError("LLM returned 0 bullets")
    return AgentSignal(
        signal=parsed["signal"],
        confidence=float(parsed["confidence"]),
        bullets=bullets,
    )


def _run_tool_loop(
    messages: list,
    llm_with_tools,
    tool_map: dict,
    session_ticker: str | None = None,
) -> object:
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            args = dict(tool_call["args"])
            if session_ticker and "ticker" in args:
                args["ticker"] = session_ticker
            result = tool_map[tool_call["name"]].invoke(args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return response
