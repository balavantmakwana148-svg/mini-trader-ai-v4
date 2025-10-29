# app.py
from fastapi import FastAPI
import joblib, yfinance as yf, numpy as np, requests, os
from utils.indicators import add_indicators, detect_candle_pattern

app = FastAPI(title="Mini Trader AI v2.1")

# Load model (make sure model/trend_model.pkl exists in repo)
MODEL_PATH = "model/trend_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model/trend_model.pkl not found in repo. Upload your trained model.")
model = joblib.load(MODEL_PATH)

# Telegram creds from environment (set in Render)
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    if not TOKEN or not CHAT_ID:
        print("Telegram TOKEN/CHAT_ID not found in environment. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        print("Telegram status:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram error:", e)

# Intraday universe (change as you like)
SYMBOLS = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","LT.NS","SBIN.NS","AXISBANK.NS"]

def get_features(symbol, period="7d", interval="5m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    df = add_indicators(df)
    if df.shape[0] < 20:
        return None
    last = df.iloc[-1]
    features = {
        "symbol": symbol,
        "rsi": float(last['rsi']),
        "ema20": float(last['ema20']),
        "ema50": float(last['ema50']),
        "macd": float(last['macd']),
        "atr": float(last['atr']),
        "vol_spike": float(last['vol_spike']),
        "close": float(last['Close']),
        "pattern": detect_candle_pattern(df)
    }
    return features

def evaluate_candidate(f, model, vol_spike_min=1.5):
    # Adjust X features to match model training order !!
    # IF your model trained on [rsi, ema20, ema50, macd], keep this order
    X = np.array([[f['rsi'], f['ema20'], f['ema50'], f['macd']]])
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        # fallback if model returns direct predict
        prob = float(model.predict(X)[0])
    ema_conf = 1 if f['ema20'] > f['ema50'] else 0
    vol_conf = 1 if f['vol_spike'] >= vol_spike_min else 0
    # pattern bonus/penalty
    pattern_bonus = 0.05 if "Bullish" in f['pattern'] else (-0.05 if "Bearish" in f['pattern'] else 0)
    score = prob * 0.8 + ema_conf * 0.1 + vol_conf * 0.1 + pattern_bonus
    return prob, score

def calc_sl_tp(entry, atr, risk_percent=1.0, account_balance=100000):
    sl = entry - 1.5 * atr
    tp1 = entry + 1.0 * atr
    tp2 = entry + 2.0 * atr
    risk_money = account_balance * (risk_percent/100.0)
    qty = max(1, int(risk_money / max(0.0001, (entry - sl))))
    return round(sl,2), round(tp1,2), round(tp2,2), qty

@app.get("/")
def home():
    return {"status":"ok","msg":"Mini Trader AI v2.1 running"}

@app.get("/pretrade")
def pretrade():
    results = []
    for s in SYMBOLS:
        try:
            f = get_features(s, period="7d", interval="5m")
            if f is None:
                continue
            prob, score = evaluate_candidate(f, model)
            if score >= 0.65 and prob >= 0.60:
                sl, tp1, tp2, qty = calc_sl_tp(f['close'], f['atr'])
                f.update({"prob": round(prob,3), "score": round(score,3), "SL": sl, "TP1": tp1, "TP2": tp2, "qty": qty})
                results.append(f)
        except Exception as e:
            print("Error processing", s, e)
            continue
    results = sorted(results, key=lambda x: -x['score'])
    if results:
        top = results[0]
        msg = (f"🔔 PRE-TRADE ALERT\nSymbol: {top['symbol']}\nConfidence: {top['prob']}\nPattern: {top['pattern']}\nEntry: {top['close']}\nSL: {top['SL']}\nTP1: {top['TP1']}\nTP2: {top['TP2']}\nQty: {top['qty']}")
        send_telegram(msg)
    return {"candidates": results, "count": len(results)}
