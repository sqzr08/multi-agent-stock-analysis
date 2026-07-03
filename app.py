from __future__ import annotations

import uuid

import httpx
import streamlit as st

from stock_analyser.state import AgentSignal, MarketReport

API_BASE = "http://localhost:8000"
_SIGNAL_KEYS = [
    "fundamentals_signal", "technical_signal", "sentiment_signal",
    "macro_signal", "risk_signal",
]


def _parse_result(data: dict) -> dict:
    for key in _SIGNAL_KEYS:
        if data.get(key):
            data[key] = AgentSignal.model_validate(data[key])
    if data.get("market_report"):
        data["market_report"] = MarketReport.model_validate(data["market_report"])
    return data

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Stock Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Multi-Agent Stock Analyser")
    st.markdown("---")
    st.markdown(
        "A multi-agent system that analyses a stock across **5 dimensions** in parallel:\n\n"
        "- 📊 **Fundamentals** — valuation & financials\n"
        "- 📉 **Technical** — price trends & indicators\n"
        "- 📰 **Sentiment** — news & market mood\n"
        "- 🌍 **Macro** — interest rates & economy\n"
        "- ⚠️ **Risk** — volatility & insider activity\n"
    )
    st.markdown("---")
    st.caption("Powered by Gemini · LangGraph")

# ── helpers ───────────────────────────────────────────────────────────────────
_SIGNAL_COLOR = {"bullish": "#00C49F", "bearish": "#FF4C4C", "neutral": "#FFD700"}
_SIGNAL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
_REC_COLOR = {"BUY": "#00C49F", "HOLD": "#FFD700", "SELL": "#FF4C4C"}
_REC_EMOJI = {"BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}
_DIR_EMOJI = {"rise": "▲", "fall": "▼", "neutral": "─"}


def _recommendation_banner(rec: str, score: float | None) -> None:
    color = _REC_COLOR.get(rec, "#FFD700")
    emoji = _REC_EMOJI.get(rec, "")
    score_str = f"score {score:+.4f}" if score is not None else ""
    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin: 16px 0;
        ">
            <div style="font-size:0.9rem; color:#aaa; margin-bottom:6px;">RECOMMENDATION</div>
            <div style="font-size:2.5rem; font-weight:800; color:{color};">
                {emoji} {rec}
            </div>
            <div style="font-size:0.85rem; color:#aaa;">{score_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _market_report_block(mkt: MarketReport) -> None:
    st.markdown("#### 🏛️ Broad Market")
    cols = st.columns(3)
    for col, idx in zip(cols, mkt.indices):
        with col:
            with st.container(border=True):
                dir_emoji = _DIR_EMOJI.get(idx.direction, "─")
                chg_str = f"{idx.change_pct:+.2f}%" if idx.change_pct is not None else ""
                last_str = f"${idx.last:,.2f}" if idx.last is not None else "N/A"
                color = "#00C49F" if idx.direction == "rise" else "#FF4C4C" if idx.direction == "fall" else "#FFD700"
                st.markdown(
                    f"**{idx.name}**<br>"
                    f"<span style='color:{color}; font-size:1.1rem;'>{dir_emoji} {last_str}</span><br>"
                    f"<span style='color:{color}; font-size:0.85rem;'>{chg_str}</span>",
                    unsafe_allow_html=True,
                )
    if mkt.trend_forecast:
        st.caption(mkt.trend_forecast)


# ── main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:4px;'>📈 Multi-Agent Stock Analyser</h1>"
    "<p style='text-align:center; color:#888; margin-bottom:24px;'>"
    "Enter a ticker symbol to run a full multi-agent analysis</p>",
    unsafe_allow_html=True,
)

col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker_input = st.text_input(
        label="Ticker",
        placeholder="e.g. AAPL, TSLA, NVDA",
        label_visibility="collapsed",
    )
with col_btn:
    analyse = st.button("Analyse", use_container_width=True, type="primary")

# ── run analysis ──────────────────────────────────────────────────────────────
if analyse and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    with st.spinner(f"Running analysis for **{ticker}** — this takes ~30 seconds…"):
        try:
            thread_id = str(uuid.uuid4())
            resp = httpx.post(
                f"{API_BASE}/analyse",
                json={"ticker": ticker, "thread_id": thread_id},
                timeout=120,
            )
            resp.raise_for_status()
            result = _parse_result(resp.json())
            st.session_state["last_result"] = result
            st.session_state["last_ticker"] = ticker
            st.session_state["thread_id"] = thread_id
            st.session_state["chat_history"] = []
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# ── render analysis results ───────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    ticker = st.session_state.get("last_ticker", "")

    st.markdown(f"### Results for `{ticker}`")

    # Broad market snapshot
    mkt = result.get("market_report")
    if mkt:
        _market_report_block(mkt)
        st.markdown("---")

    # Agent summary row
    snap_cols = st.columns(5)
    snap_agents = [
        ("Fundamentals", result.get("fundamentals_signal")),
        ("Technical",    result.get("technical_signal")),
        ("Sentiment",    result.get("sentiment_signal")),
        ("Macro",        result.get("macro_signal")),
        ("Risk",         result.get("risk_signal")),
    ]
    for col, (name, sig) in zip(snap_cols, snap_agents):
        with col:
            with st.container(border=True):
                if sig is None:
                    st.markdown(f"**{name}**\n\n⚠️ N/A")
                else:
                    emoji = "🟢" if sig.signal == "bullish" else "🔴" if sig.signal == "bearish" else "🟡"
                    st.markdown(f"**{name}**")
                    st.markdown(f"{emoji} {sig.signal.capitalize()}")
                    st.caption(f"{sig.confidence:.0%} confidence")

    agents_data = [
        ("Fundamentals", result.get("fundamentals_signal")),
        ("Technical",    result.get("technical_signal")),
        ("Sentiment",    result.get("sentiment_signal")),
        ("Macro",        result.get("macro_signal")),
        ("Risk",         result.get("risk_signal")),
    ]

    for agent_name, signal in agents_data:
        if signal is None:
            continue

        st.markdown("---")

        col_label, col_signal, col_conf = st.columns([2, 2, 6])

        with col_label:
            st.markdown(f"### {agent_name}")

        with col_signal:
            emoji = (
                "🟢" if signal.signal == "bullish"
                else "🔴" if signal.signal == "bearish"
                else "🟡"
            )
            st.markdown(f"### {emoji} {signal.signal.capitalize()}")

        with col_conf:
            st.progress(
                signal.confidence,
                text=f"Confidence: {signal.confidence:.0%}"
            )

        if signal.error:
            st.warning(f"⚠️ {signal.error}")
            continue

        with st.container(border=True):
            for i, bullet in enumerate(signal.bullets, 1):
                st.write(f"**{i}.** {bullet['text']}")
                st.caption(f"📌 Source: {bullet['source']}")
                if bullet.get("url"):
                    st.markdown(f"[🔗 View Source]({bullet['url']})")
                st.markdown("")

    st.markdown("---")

    # Recommendation banner
    rec = result.get("recommendation", "HOLD")
    score = result.get("score")
    _recommendation_banner(rec, score)

    # Summary
    summary = result.get("summary")
    if summary:
        st.markdown("#### Summary")
        st.info(summary)

    # Disclaimer
    st.caption(
        "⚠️ This analysis is for informational purposes only and does not constitute financial advice. "
        "Always do your own research before making investment decisions."
    )

    # Debug expander
    with st.expander("Debug — raw state JSON", expanded=False):
        import json

        def _serialise(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        st.json(json.loads(json.dumps(result, default=_serialise)))

    # ── Chat interface ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Ask a follow-up question")
    st.caption("Questions are answered using cached analysis data — no re-fetching unless needed.")

    chat_history = st.session_state.get("chat_history", [])
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("routed_to"):
                st.caption(f"Routed to: {', '.join(msg['routed_to'])}")
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about this stock…"):
        chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    thread_id = st.session_state["thread_id"]
                    resp = httpx.post(
                        f"{API_BASE}/qa",
                        json={"ticker": ticker, "question": prompt, "thread_id": thread_id},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    qa_result = resp.json()
                    routed_to = qa_result.get("routed_to", [])
                    answer = qa_result.get("answer", "I was unable to generate an answer.")

                    if routed_to:
                        st.caption(f"Routed to: {', '.join(routed_to)}")
                    st.markdown(answer)

                    chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "routed_to": routed_to,
                    })
                    st.session_state["chat_history"] = chat_history
                except Exception as e:
                    err = f"Error generating answer: {e}"
                    st.error(err)
                    chat_history.append({"role": "assistant", "content": err})
                    st.session_state["chat_history"] = chat_history
