import os
import requests
import json
import re
import yfinance as yf
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================= AI API KEY =================
API_KEY = os.environ.get("GEMINI_API_KEY")  # এখানে শুধু key বসাবে Render এ
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("signals.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            market TEXT,
            tf TEXT,
            signal TEXT,
            accuracy TEXT,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= MARKET DATA =================
def get_data(symbol, tf):
    try:
        if "BTC" in symbol:
            ticker = "BTC-USD"
        elif "XAU" in symbol or "GOLD" in symbol:
            ticker = "GC=F"
        else:
            ticker = symbol + "=X"

        df = yf.download(ticker, period="2d", interval=f"{tf}m", progress=False)

        if df is None or df.empty:
            return None

        df = df.tail(60)

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))

        # EMA
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        return df

    except:
        return None


# ================= CANDLE ANALYSIS =================
def candle_analysis(df):

    last = df.iloc[-1]

    o = last["Open"]
    c = last["Close"]
    h = last["High"]
    l = last["Low"]

    body = abs(c - o)
    full = h - l

    color = "GREEN" if c > o else "RED"
    if c == o:
        color = "DOJI"

    body_ratio = body / full if full != 0 else 0

    if body_ratio > 0.7:
        movement = "IMPULSE"
    elif body_ratio > 0.4:
        movement = "MODERATE"
    else:
        movement = "WEAK"

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if lower_wick > body * 1.5:
        pressure = "BUY_REJECTION"
    elif upper_wick > body * 1.5:
        pressure = "SELL_REJECTION"
    else:
        pressure = "NONE"

    score = 0
    if color == "GREEN":
        score += 1
    else:
        score -= 1

    if movement == "IMPULSE":
        score += 2

    if pressure == "BUY_REJECTION":
        score += 2
    elif pressure == "SELL_REJECTION":
        score -= 2

    return {
        "color": color,
        "movement": movement,
        "pressure": pressure,
        "score": score
    }


# ================= SMC ANALYSIS =================
def analyze(df):

    last = df.iloc[-1]

    candle = candle_analysis(df)

    # Trend
    trend = "UP" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "DOWN"

    # Structure (simple HH/HL)
    structure = "UPTREND" if df["Close"].iloc[-1] > df["Close"].rolling(10).mean().iloc[-1] else "DOWNTREND"

    # Support / Resistance
    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    # Liquidity zone
    liquidity = (support + resistance) / 2

    # RSI
    rsi = float(last["RSI"])

    # Market type
    volatility = df["Close"].pct_change().std()
    market_type = "RANGE" if volatility < 0.002 else "TREND"

    # Fake breakout
    fake_breakout = False
    if last["Close"] > resistance * 1.002 or last["Close"] < support * 0.998:
        fake_breakout = True

    return {
        "trend": trend,
        "structure": structure,
        "support": support,
        "resistance": resistance,
        "liquidity": liquidity,
        "rsi": rsi,
        "market_type": market_type,
        "fake_breakout": fake_breakout,
        "candle": candle
    }


# ================= SIGNAL ENGINE =================
def signal_engine(a):

    score = 0

    if a["trend"] == "UP":
        score += 2
    else:
        score -= 2

    if "UP" in a["structure"]:
        score += 2
    else:
        score -= 2

    if a["rsi"] < 30:
        score += 2
    elif a["rsi"] > 70:
        score -= 2

    score += a["candle"]["score"]

    if a["market_type"] == "RANGE":
        score -= 1

    if a["fake_breakout"]:
        score -= 3

    if score >= 5:
        return "CALL", score
    elif score <= -5:
        return "PUT", score
    else:
        return "WAIT", score


# ================= AI CONFIRMATION =================
def ai_confirm(prompt):

    try:
        r = requests.post(GEMINI_URL, json={
            "contents": [{"parts": [{"text": prompt}]}]
        })

        res = r.json()
        return res["candidates"][0]["content"]["parts"][0]["text"]

    except:
        return "AI ERROR"


# ================= API =================
@app.route("/get_analysis")
def get_analysis():

    market = request.args.get("market", "EURUSD")
    tf = request.args.get("timeframe", "1")

    df = get_data(market, tf)

    if df is None:
        return jsonify({"signal": "WAIT", "reason": "No Data"})

    a = analyze(df)

    signal, score = signal_engine(a)

    prompt = f"""
Market: {market}
Signal: {signal}
Score: {score}

Trend: {a['trend']}
Structure: {a['structure']}
RSI: {a['rsi']}
Market Type: {a['market_type']}
Candle: {a['candle']}
Fake Breakout: {a['fake_breakout']}

Explain if this trade is valid or not in short.
"""

    ai = ai_confirm(prompt)

    # save log
    conn = sqlite3.connect("signals.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO signals (time, market, tf, signal, accuracy, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        market,
        tf,
        signal,
        f"{min(100, abs(score)*10)}%",
        ai
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "market": market,
        "signal": signal,
        "score": score,
        "accuracy": f"{min(100, abs(score)*10)}%",
        "reason": ai
    })


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
