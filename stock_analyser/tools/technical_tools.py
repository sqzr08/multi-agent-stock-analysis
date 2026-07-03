from __future__ import annotations

import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0):
    mid = close.rolling(length).mean()
    dev = close.rolling(length).std()
    return mid - std * dev, mid + std * dev


def compute_indicators(price_data: dict) -> dict:
    try:
        ohlcv = price_data.get("ohlcv", {})
        df = pd.DataFrame({
            "Open": ohlcv.get("Open", {}),
            "High": ohlcv.get("High", {}),
            "Low": ohlcv.get("Low", {}),
            "Close": ohlcv.get("Close", {}),
            "Volume": ohlcv.get("Volume", {}),
        })
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        df = df.astype(float)

        close = df["Close"]
        current_price = float(close.iloc[-1])

        rsi_series = _rsi(close, 14)
        rsi_14 = float(rsi_series.iloc[-1])

        macd_line, signal_line = _macd(close)
        macd_signal = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"

        bb_lower, bb_upper = _bbands(close)
        bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
        bb_position = (current_price - float(bb_lower.iloc[-1])) / bb_range if bb_range != 0 else 0.5

        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        price_vs_ma50 = (current_price - ma50) / ma50
        price_vs_ma200 = (current_price - ma200) / ma200

        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
        curr_vol = float(df["Volume"].iloc[-1])
        volume_trend = "above_average" if curr_vol > avg_vol else "below_average"

        # ATR-14
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        # Structural support/resistance over 50-day window
        support = float(df["Low"].rolling(50).min().iloc[-1])
        resistance = float(df["High"].rolling(50).max().iloc[-1])

        # Recent swing levels over 20-day window
        recent_swing_low = float(df["Low"].rolling(20).min().iloc[-1])
        recent_swing_high = float(df["High"].rolling(20).max().iloc[-1])

        return {
            "current_price": current_price,
            "rsi_14": rsi_14,
            "macd_signal": macd_signal,
            "bb_position": round(bb_position, 4),
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "price_vs_ma50": round(price_vs_ma50, 4),
            "price_vs_ma200": round(price_vs_ma200, 4),
            "volume_trend": volume_trend,
            "atr_14": round(atr_14, 4),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "recent_swing_low": round(recent_swing_low, 2),
            "recent_swing_high": round(recent_swing_high, 2),
        }
    except Exception as e:
        return {"error": str(e)}
