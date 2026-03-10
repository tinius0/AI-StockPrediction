# stock_model.py
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's smoothed RSI using EMA-style averaging."""
    deltas = np.diff(prices)
    rsi = np.zeros_like(prices)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range measures volatility."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = np.convolve(tr, np.ones(period) / period, mode='full')[:len(tr)]
    return atr


def rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean no look ahead. First `window-1` values use expanding mean."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start: i + 1].mean()
    return out


FEATURE_NAMES = ["Close", "High", "Low", "Volume", "SMA20", "RSI", "Returns", "BB_Width", "ATR"]


def get_feature_names() -> list[str]:
    return FEATURE_NAMES


def prepare_data(ticker: str = "AAPL", warmup: int = 20):
    """
    Download OHLCV data and engineer features.

    Returns
    -------
    X : (N, F) feature matrix  — row t = info available *before* day t+1
    y : (N,)  target vector    — close price on day t+1
    close : full close array (for downstream charting)
    """
    df = yf.download(ticker, period="5y", interval="1d", progress=False)

    close = df["Close"].values.flatten().astype(float)
    high  = df["High"].values.flatten().astype(float)
    low   = df["Low"].values.flatten().astype(float)
    vol   = df["Volume"].values.flatten().astype(float)

    sma20 = rolling_sma(close, 20)

    std20 = np.array([close[max(0, i-19): i+1].std() for i in range(len(close))])
    bb_width = np.where(sma20 != 0, (2 * 2 * std20) / sma20, 0.0)  # normalised band width

    rsi = compute_rsi(close)
    atr = compute_atr(high, low, close)

    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    features = np.column_stack([close, high, low, vol, sma20, rsi, returns, bb_width, atr])

    # Align: X[t] → features known at close of day t; y[t] → close of day t+1
    X = features[:-1]
    y = close[1:]

    # Drop warmup rows where indicators are unreliable
    X = X[warmup:]
    y = y[warmup:]

    return X, y, close