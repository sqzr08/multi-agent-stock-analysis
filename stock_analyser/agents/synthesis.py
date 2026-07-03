from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import llm
from stock_analyser.state import AgentSignal, GraphState

_WEIGHTS = {
    "fundamentals": 0.30,
    "risk": 0.20,
    "technical": 0.20,
    "macro": 0.15,
    "sentiment": 0.15,
}

_SIGNAL_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

_PROMPT = """You are a senior investment analyst. Given the following weighted analysis from 5 specialist agents, produce a final investment recommendation.

Agent signals and weighted score: {data}

Based on this analysis, provide a JSON response:
{{
  "recommendation": "BUY" | "HOLD" | "SELL",
  "summary": "<2-3 sentence overall assessment>"
}}

WRITING RULES FOR SUMMARY:
- Write the summary as 2-3 sentences a beginner can read and immediately understand
- Start with the overall picture: is this stock looking good, bad, or mixed right now?
- Mention the 1-2 most important reasons behind the recommendation
- End with the single most important thing to watch next
- Good example: "Overall, Apple looks solid right now — the business is growing steadily and the stock price has good momentum. The main concern is that interest rates staying high could slow down growth. The upcoming earnings report on May 29 will be the key moment to watch."
- Bad example: "Mixed signals across technical and fundamental dimensions with macro headwinds presenting headwinds to near-term price appreciation."

Respond ONLY with valid JSON. No markdown, no explanation."""

_STRICT_PROMPT = """Respond ONLY with a JSON object. No markdown, no code fences.
Format: {{"recommendation": "BUY"|"HOLD"|"SELL", "summary": "..."}}

Data: {data}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content)
    return str(response.content)



def synthesis_agent(state: GraphState) -> dict:
    signals: dict[str, AgentSignal | None] = {
        "fundamentals": state.get("fundamentals_signal"),
        "technical": state.get("technical_signal"),
        "sentiment": state.get("sentiment_signal"),
        "macro": state.get("macro_signal"),
        "risk": state.get("risk_signal"),
    }

    weighted_score = 0.0
    signal_summary = {}
    for name, weight in _WEIGHTS.items():
        sig = signals.get(name)
        if sig and not sig.error:
            score = _SIGNAL_SCORE.get(sig.signal, 0.0) * sig.confidence
            weighted_score += score * weight
            signal_summary[name] = {
                "signal": sig.signal,
                "confidence": sig.confidence,
                "weighted_contribution": round(score * weight, 4),
                "bullets": sig.bullets,
            }
        else:
            signal_summary[name] = {
                "signal": "neutral",
                "confidence": 0.0,
                "error": str(sig.error) if sig else "missing",
            }

    weighted_score = round(weighted_score, 4)

    if weighted_score > 0.3:
        rule_rec = "BUY"
    elif weighted_score < -0.3:
        rule_rec = "SELL"
    else:
        rule_rec = "HOLD"

    data_dict: dict = {
        "weighted_score": weighted_score,
        "rule_based_recommendation": rule_rec,
        "agent_signals": signal_summary,
    }

    data = json.dumps(data_dict, default=str)

    try:
        text = _call_llm(_PROMPT.format(data=data))
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
    except Exception:
        try:
            text = _call_llm(_STRICT_PROMPT.format(data=data))
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
        except Exception:
            parsed = {
                "recommendation": rule_rec,
                "summary": "Synthesis unavailable; recommendation based on weighted signal score.",
            }

    rec = parsed.get("recommendation", rule_rec)

    return {
        "score": weighted_score,
        "recommendation": rec,
        "summary": parsed.get("summary", ""),
    }
