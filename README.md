# Setup Guide — Multi-Agent Stock Analyser

## Prerequisites

- **Python 3.14+** (the project was built and tested on 3.14; a module-level SSL patch in `fred_tools.py` is required for the FRED API on macOS Python 3.14)
- No other system-level dependencies

---

## 1. Get the Code

```bash
git clone <repo-url>
cd "Stock Analysis"
```

Or if you have the folder already, just `cd` into it.

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

## Troubleshooting

**Alpha Vantage returns `{"Information": "..."}` instead of news data**
The free tier allows 25 requests per day. The sentiment agent will fall back to a neutral signal — other agents are unaffected. Upgrade to a paid plan or wait until the next day.

**`KeyError: 'GOOGLE_API_KEY'` (or FRED / Alpha Vantage)**
The `.env` file is missing or in the wrong location. It must be at the project root (the same folder as `api.py`, not inside `stock_analyser/`).

**FRED API SSL error on macOS**
The code already applies a fix at module level in `fred_tools.py` using `certifi`. If you still see SSL errors, ensure `certifi` is installed (`pip show certifi`) and that you are using the virtual environment's Python, not a system Python.

**`ValueError: Could not find a valid ticker for '...'`**
The ticker wasn't recognised by yfinance. Try the exact ticker symbol (e.g. `MU` instead of `Micron`), or check that the symbol is listed on a US exchange.
