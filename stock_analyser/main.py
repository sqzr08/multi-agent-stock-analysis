from __future__ import annotations

import asyncio
import uuid

from stock_analyser.graph import build_graph


async def main() -> None:
    ticker = input("Enter ticker symbol: ").strip().upper()
    if not ticker:
        print("No ticker entered. Exiting.")
        return

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nAnalysing {ticker} — running agents in parallel...\n")
    try:
        result = await graph.ainvoke({"ticker": ticker}, config=config)
        print(result["report"])
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Q&A loop
    print("\n" + "─" * 60)
    print("Ask a follow-up question (or press Enter / type 'exit' to quit).")
    print("─" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        try:
            qa_result = await graph.ainvoke(
                {"ticker": ticker, "question": question},
                config=config,
            )
            plan = qa_result.get("query_plan")
            if plan and plan.relevant_agents:
                print(f"\n[Routed to: {', '.join(plan.relevant_agents)}]")
            answer = qa_result.get("answer", "No answer generated.")
            print(f"\n{answer}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
