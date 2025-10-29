# utils/indicators.py
import pandas as pd
import ta
import numpy as np

def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.dropna()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['vol_avg20'] = df['Volume'].rolling(20).mean().replace(0, np.nan)
    df['vol_spike'] = df['Volume'] / df['vol_avg20']
    df = df.dropna()
    return df

def detect_candle_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, c, h, l = last['Open'], last['Close'], last['High'], last['Low']
    po, pc = prev['Open'], prev['Close']
    body = abs(c - o)
    prev_body = abs(pc - po)
    # Bullish Engulfing
    if (c > o) and (pc < po) and (o < pc) and (c > po):
        return "Bullish Engulfing"
    # Bearish Engulfing
    if (c < o) and (pc > po) and (o > pc) and (c < po):
        return "Bearish Engulfing"
    # Hammer
    lower_shadow = min(o, c) - l
    if lower_shadow > (body * 2) and body < ((h - l) * 0.4):
        return "Hammer"
    # Doji
    if body <= 0.1 * (h - l):
        return "Doji"
    return "No pattern"
