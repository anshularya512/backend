import os
import math
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv

# Load .env locally (Railway will use environment variables)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Stock Research & AI Backend")

# Allow your Vercel/Next/v0 origin and localhost dev
origins = [
    os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    os.getenv("FRONTEND_ORIGIN_2", "https://v0.app"),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- helpers ------------------------------------------------
def to_native(o):
    # convert numpy types to python native for json
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if pd.isna(o):
        return None
    return o

def safe_jsonify(d: dict):
    return json.loads(json.dumps(d, default=to_native))

def try_ticker(symbol: str):
    # returns yfinance Ticker object and fetched info or None
    try:
        t = yf.Ticker(symbol)
        info = t.info
        # sometimes info exists but missing price — validate
        if not info or ("regularMarketPrice" not in info and "previousClose" not in info):
            return None
        return t
    except Exception:
        return None

def compute_technicals(history: pd.DataFrame):
    # expects history with 'Close' and 'Volume' columns and index is datetime
    res = {}
    if history is None or history.empty:
        return res
    closes = history["Close"].dropna()
    # SMA
    res["sma20"] = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None
    res["sma50"] = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
    res["sma200"] = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
    # momentum % over 20 days
    if len(closes) >= 21:
        res["momentum"] = float((closes.iloc[-1] / closes.iloc[-21] - 1) * 100)
    else:
        res["momentum"] = None
    # support/resistance simple: recent min/max
    res["support"] = float(closes.tail(30).min()) if len(closes) >= 1 else None
    res["resistance"] = float(closes.tail(30).max()) if len(closes) >= 1 else None
    # RSI simple
    delta = closes.diff().dropna()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / down).replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    res["rsi"] = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
    # macd simple using EMA
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    res["macd"] = float(macd.iloc[-1]) if not macd.empty else None
    res["macdSignal"] = float(macd_signal.iloc[-1]) if not macd_signal.empty else None
    res["macdHistogram"] = float(macd.iloc[-1] - macd_signal.iloc[-1]) if not macd.empty and not macd_signal.empty else None
    # volume trend
    res["volumeTrend"] = "high" if history["Volume"].iloc[-1] > history["Volume"].rolling(20).mean().iloc[-1] else "normal"
    return res

# --- stock endpoint -----------------------------------------
@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    s = symbol.strip().upper()
    # Try variants: as-is, .NS, .BO
    candidates = [s]
    if "." not in s:
        candidates = [s + ".NS", s + ".BO", s]  # prefer NSE first
    ticker = None
    used_symbol = None
    for cand in candidates:
        t = try_ticker(cand)
        if t:
            ticker = t
            used_symbol = cand
            break
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    info = ticker.info
    # Get price and quote fields safely
    price = info.get("regularMarketPrice") or info.get("previousClose") or None
    change = info.get("regularMarketChange") or (price - info.get("previousClose")) if price and info.get("previousClose") else None
    changePercent = info.get("regularMarketChangePercent") or ( (change / info.get("previousClose"))*100 if change and info.get("previousClose") else None)

    # fetch history to compute technicals
    try:
        hist = ticker.history(period="1y", interval="1d")
    except Exception:
        hist = pd.DataFrame()

    tech = compute_technicals(hist)

    # fundamentals basic fields (some may be missing)
    fundamentals = {
        "revenue": info.get("totalRevenue") or None,
        "revenueGrowth": info.get("revenueGrowth") or None,
        "eps": info.get("trailingEps") or None,
        "epsGrowth": info.get("earningsGrowth") or None,
        "pe": info.get("trailingPE") or None,
        "forwardPE": info.get("forwardPE") or None,
        "pb": info.get("priceToBook") or None,
        "ps": info.get("priceToSalesTrailing12Months") or None,
        "peg": info.get("pegRatio") or None,
        "debtToEquity": info.get("debtToEquity") or None,
        "roe": info.get("returnOnEquity") or None,
        "roa": info.get("returnOnAssets") or None,
        "profitMargin": info.get("profitMargins") or None,
        "grossMargin": info.get("grossMargins") or None,
        "operatingMargin": info.get("operatingMargins") or None,
        "currentRatio": info.get("currentRatio") or None,
        "quickRatio": info.get("quickRatio") or None,
        "dividendYield": info.get("dividendRate") or None,
        "beta": info.get("beta") or None,
        "fiftyDayMA": info.get("fiftyDayAverage") or None,
        "twoHundredDayMA": info.get("twoHundredDayAverage") or None,
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or ""
    }

    response = {
        "symbol": used_symbol,
        "name": info.get("longName") or info.get("shortName") or used_symbol,
        "price": price,
        "change": change,
        "changePercent": changePercent,
        "marketCap": info.get("marketCap") or None,
        "marketCapFormatted": None,
        "marketCapCategory": None,
        "pe": info.get("trailingPE") or None,
        "eps": info.get("trailingEps") or None,
        "volume": info.get("regularMarketVolume") or None,
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "sector": fundamentals["sector"],
        "industry": fundamentals["industry"],
        "fundamentals": fundamentals,
        "technicals": {
            "trend": "bullish" if (change is not None and change > 0) else ("bearish" if change is not None else None),
            "rsi": tech.get("rsi"),
            "macd": tech.get("macd"),
            "macdSignal": tech.get("macdSignal"),
            "macdHistogram": tech.get("macdHistogram"),
            "sma20": tech.get("sma20"),
            "sma50": tech.get("sma50"),
            "sma200": tech.get("sma200"),
            "support": tech.get("support"),
            "resistance": tech.get("resistance"),
            "volumeTrend": tech.get("volumeTrend"),
            "momentum": tech.get("momentum"),
        }
    }

    # format market cap string and category (simple)
    cap = response["marketCap"]
    if cap:
        if cap >= 1e12:
            response["marketCapFormatted"] = f"{cap/1e12:.2f}T"
            response["marketCapCategory"] = "Large Cap"
        elif cap >= 2e10:
            response["marketCapFormatted"] = f"{cap/1e9:.2f}B"
            response["marketCapCategory"] = "Mid Cap"
        else:
            response["marketCapFormatted"] = f"{cap/1e6:.2f}M"
            response["marketCapCategory"] = "Small Cap"
    else:
        response["marketCapFormatted"] = "N/A"
        response["marketCapCategory"] = "N/A"

    return safe_jsonify(response)


# --- AI analyze endpoint ------------------------------------
class AnalyzeRequest(BaseModel):
    symbol: str

@app.post("/ai/analyze")
async def analyze(req: AnalyzeRequest):
    s = req.symbol.strip().upper()
    # fetch stock data using our endpoint logic (call function directly)
    try:
        stock_resp = await get_stock(s)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=f"Stock lookup failed: {e.detail}")

    # Build prompt for OpenAI
    prompt = f"""
You are a helpful stock research assistant. Provide:
1) short executive summary (2-3 sentences)
2) 1-line recommendation: strong_buy / buy / hold / sell / strong_sell
3) top 3 risks as bullets
4) top 3 opportunities as bullets
5) a simple numeric investment grade 0-100

Respond in JSON with keys: summary, recommendation, risks (array), opportunities (array), investmentGrade, fundamentalAnalysis, technicalAnalysis, priceTarget (low, mid, high), keyMetrics object.

Stock raw data:
{json.dumps(stock_resp, indent=2)}
"""

    if OPENAI_API_KEY:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # change if unavailable — adapt
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
            )
            reply = completion["choices"][0]["message"]["content"]
            # Try to parse JSON from reply; if not parseable, return a simple structure
            try:
                parsed = json.loads(reply)
                return parsed
            except Exception:
                # if model returned plain text, include it in summary
                return {
                    "summary": reply,
                    "recommendation": "hold",
                    "risks": [],
                    "opportunities": [],
                    "investmentGrade": 50,
                    "fundamentalAnalysis": "",
                    "technicalAnalysis": "",
                    "priceTarget": {"low": None, "mid": None, "high": None},
                    "keyMetrics": {}
                }
        except Exception as e:
            # if OpenAI call failed, fallthrough to simple analysis
            print("OpenAI error:", e)

    # Fallback deterministic analysis (no OpenAI)
    fund = stock_resp.get("fundamentals", {}) or {}
    tech = stock_resp.get("technicals", {}) or {}
    risks = []
    opp = []
    grade = 50

    # sample rules
    dte = fund.get("debtToEquity") or 0
    if dte and dte > 1:
        risks.append("High debt-to-equity")
        grade -= 10
    if (fund.get("profitMargin") or 0) < 0:
        risks.append("Negative profit margins")
        grade -= 15
    if (tech.get("rsi") or 0) > 70:
        risks.append("Overbought (high RSI)")
        grade -= 5

    if (fund.get("revenueGrowth") or 0) and (fund.get("revenueGrowth") > 0.1):
        opp.append("Strong revenue growth")
        grade += 10
    if (tech.get("momentum") or 0) and tech.get("momentum") > 5:
        opp.append("Positive recent momentum")
        grade += 5
    # clamp grade
    grade = max(0, min(100, grade))

    summary = f"{stock_resp.get('name')} current price {stock_resp.get('price')}. Automated rule-of-thumb grade {grade}."
    recommendation = "hold"
    if grade >= 80:
        recommendation = "strong_buy"
    elif grade >= 60:
        recommendation = "buy"
    elif grade >= 40:
        recommendation = "hold"
    elif grade >= 20:
        recommendation = "sell"
    else:
        recommendation = "strong_sell"

    result = {
        "summary": summary,
        "recommendation": recommendation,
        "risks": risks,
        "opportunities": opp,
        "investmentGrade": grade,
        "fundamentalAnalysis": f"DebtToEquity: {fund.get('debtToEquity')}, PE: {fund.get('pe')}",
        "technicalAnalysis": f"RSI: {tech.get('rsi')}, Momentum: {tech.get('momentum')}",
        "priceTarget": {"low": None, "mid": None, "high": None},
        "keyMetrics": {
            "valuation": "fair",
            "quality": "medium",
            "momentum": "positive" if (tech.get("momentum") or 0) > 0 else "neutral",
            "safety": "low" if (fund.get("debtToEquity") or 0) > 1 else "medium"
        }
    }
    return result
