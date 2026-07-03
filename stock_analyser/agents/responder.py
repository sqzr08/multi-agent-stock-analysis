from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyser.config import llm
from stock_analyser.state import GraphState

_PROMPT = """You are a conversational stock analysis assistant. Answer the user's follow-up question \
using the information provided below. Be helpful, clear, and beginner-friendly.

Conversation history (most recent last):
{history}

Current question: {question}

Analysis context:
{context}

Instructions:
- Answer directly and concisely (2-5 sentences)
- Use plain English — explain any financial terms you mention
- If this question references something from the conversation history ("that level", "the P/E you mentioned"), \
  resolve it from the history and incorporate it naturally
- If citing specific data points, mention where they come from
- If the data is insufficient to fully answer, say so honestly
- Do NOT make specific price predictions or give personalised financial advice
- End with the most important thing the user should keep in mind

Write your answer as plain text (no JSON, no markdown headers)."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in response.content)
    return str(response.content)


def _format_history(messages: list) -> str:
    if not messages:
        return "(no prior conversation)"
    recent = messages[-6:]
    lines = []
    for msg in recent:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = str(msg.content)[:500]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_context(state: GraphState) -> str:
    plan = state.get("query_plan")
    session_ticker = state.get("ticker", "N/A")

    lines = [
        f"Session ticker (original analysis): {session_ticker}",
        f"Recommendation for {session_ticker}: {state.get('recommendation', 'N/A')} "
        f"(score: {state.get('score', 'N/A')})",
    ]

    if plan and plan.asked_about_other_stock and plan.other_ticker:
        lines.append(f"\n[Agent findings below are for {plan.other_ticker}, NOT {session_ticker}]")

    for field, label in [
        ("fundamentals_signal", "Fundamentals"),
        ("technical_signal", "Technical"),
        ("sentiment_signal", "Sentiment"),
        ("macro_signal", "Macro"),
        ("risk_signal", "Risk"),
    ]:
        sig = state.get(field)
        if sig:
            bullets_text = " | ".join(b.get("text", "") for b in sig.bullets if isinstance(b, dict))
            lines.append(f"{label} ({sig.signal}, {sig.confidence:.0%}): {bullets_text}")

    agent_answers = state.get("agent_answers") or {}
    if agent_answers:
        answer_ticker = (plan.other_ticker if plan and plan.asked_about_other_stock else session_ticker)
        lines.append(f"\nAgent findings for {answer_ticker}:")
        for agent_name, answer_obj in agent_answers.items():
            if hasattr(answer_obj, "answer"):
                lines.append(f"  {agent_name.title()}: {answer_obj.answer}")
            elif isinstance(answer_obj, dict):
                lines.append(f"  {agent_name.title()}: {answer_obj.get('answer', '')}")

    mkt = state.get("market_report")
    if mkt:
        idx_strs = [
            f"{i.name}: {'+' if i.direction == 'rise' else '-'}{abs(i.change_pct or 0):.2f}%"
            for i in mkt.indices
        ]
        lines.append(f"Market today: {', '.join(idx_strs)}")

    prior = state.get("prior_analyses") or []
    if prior:
        p = prior[0]
        lines.append(
            f"Previous analysis ({p.get('date')}): {p.get('recommendation')} "
            f"(score: {p.get('score')})"
        )

    summary = state.get("summary")
    if summary:
        lines.append(f"Overall summary: {summary}")

    return "\n".join(lines)


def responder_agent(state: GraphState) -> dict:
    question = state.get("question", "")
    if not question:
        return {"answer": "No question was provided."}

    plan = state.get("query_plan")

    # Out-of-scope refusal — planner flagged question as unrelated to stocks/investing
    if plan and plan.out_of_scope:
        session_ticker = state.get("ticker", "this stock")
        answer = (
            f"That question is outside what I can help with here. I can answer questions about "
            f"{session_ticker} or any other stock — just ask about a specific company or ticker."
        )
        return {
            "answer": answer,
            "messages": [HumanMessage(content=question), AIMessage(content=answer)],
        }

    # Use the planner's resolved question for the agent prompt if available
    display_question = (plan.resolved_question if plan and plan.resolved_question else question)

    history = _format_history(state.get("messages") or [])
    context = _build_context(state)

    try:
        answer = _call_llm(_PROMPT.format(
            history=history,
            question=display_question,
            context=context,
        ))
        answer = answer.strip()
    except Exception as e:
        answer = f"I was unable to generate a response: {e}"

    # Append this exchange to the conversation history via add_messages reducer
    return {
        "answer": answer,
        "messages": [HumanMessage(content=question), AIMessage(content=answer)],
    }
