# Setup Guide — Multi-Agent Stock Analyser

## Prerequisites

- **Python 3.14+** (the project was built and tested on 3.14; a module-level SSL patch in `fred_tools.py` is required for the FRED API on macOS Python 3.14)
- No other system-level dependencies
- Alternatively, **Docker** — see [Option C — Docker](#option-c--docker) below, which needs no local Python install at all

---

## 1. Get the Code

```bash
git clone https://github.com/sqzr08/multi-agent-stock-analysis.git
cd multi-agent-stock-analysis
```

---

## 2. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Environment Variables

The app requires three API keys. Create a `.env` file at the **project root** (same level as `api.py`):

```bash
# .env
GOOGLE_API_KEY=
FRED_API_KEY=
ALPHA_VANTAGE_API_KEY=
```

| Variable | What it's for | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini LLM (`gemini-3.1-flash-lite-preview`) — powers all five analysis agents | [Google AI Studio](https://aistudio.google.com) → *Get API key* |
| `FRED_API_KEY` | Federal Reserve Economic Data (interest rates, CPI, unemployment) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) → *Request API key* (free, instant) |
| `ALPHA_VANTAGE_API_KEY` | News & Sentiment API — used by the sentiment agent | [alphavantage.co](https://www.alphavantage.co/support/#api-key) → *Get Free API Key* (free tier: 25 requests/day) |

All three keys are **required** — the app raises a `KeyError` on startup if any are missing.

---

## 5. Running the Project

### Option A — Web UI (Streamlit + FastAPI)

Open **two terminals**:

**Terminal 1 — start the backend:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — start the frontend:**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Enter a ticker symbol (e.g. `AAPL`) and click **Analyse**.

API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### Option B — CLI

```bash
python3 -m stock_analyser.main
```

You'll be prompted for a ticker. After the report prints, a Q&A loop starts — type questions or press Enter / `exit` to quit.

---

### Option C — Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or another Docker Compose–compatible engine). No local Python install or virtual environment needed — steps 2–3 above are skipped entirely.

Make sure your `.env` file (step 4) exists at the project root, then run:

```bash
docker compose up --build
```

This builds one shared image and starts two containers:

| Service | Port | Notes |
|---|---|---|
| `api` | `8000` | FastAPI backend, owns the LangGraph + MemorySaver |
| `web` | `8501` | Streamlit frontend, talks to `api` over the compose network |

Open [http://localhost:8501](http://localhost:8501) in your browser. Long-term memory (`analyses.db`) persists across restarts in a named volume (`memory-db`), so analysis history survives `docker compose down`.

Stop the stack with:
```bash
docker compose down
```

Add `-v` to also delete the persisted SQLite volume:
```bash
docker compose down -v
```

---

## 6. Verify It Works

**CLI:**
```
Enter ticker symbol: AAPL

Analysing AAPL — running agents in parallel...

============================================================
  STOCK ANALYSIS REPORT — AAPL
============================================================
...
  RECOMMENDATION:  ✅ BUY  |  score: +0.3200
============================================================
```

You should see a full report with five agent sections (Fundamentals, Technical, Sentiment, Macro, Risk), a recommendation, and a summary. The analysis takes roughly 20–40 seconds.

**Web UI:** entering `AAPL` and clicking Analyse should render five agent cards, a recommendation banner, and a chat input at the bottom.

---

## 7. Run the Tests

```bash
python3 -m pytest tests/ -v
```

67 tests total. Most use mocks and run instantly. `test_tools.py` makes live API calls (yfinance, FRED, Alpha Vantage) and requires valid keys and a network connection.

---
