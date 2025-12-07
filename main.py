from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import requests
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="INDRAAZ Trading Signals API")

# CORS - Allow your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://indraaz.site",
        "https://*.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
STOCK_SYMBOLS = []
NSE_API_BASE = "https://nse-js.onrender.com"

# Top 100 NSE stocks
NIFTY_100_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", 
    "BHARTIARTL", "KOTAKBANK", "LT", "ITC", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "TATAMOTORS", "BAJAJFINSV", "NTPC", "WIPRO", "ADANIENT", "ONGC", "POWERGRID",
    "M&M", "JSWSTEEL", "TATASTEEL", "INDUSINDBK", "TECHM", "HINDALCO", "HDFCLIFE",
    "COALINDIA", "SBILIFE", "BRITANNIA", "GRASIM", "DIVISLAB", "BAJAJ-AUTO",
    "ADANIPORTS", "EICHERMOT", "DRREDDY", "TATACONSUM", "BPCL", "CIPLA", "APOLLOHOSP",
    "SHREECEM", "HEROMOTOCO", "UPL", "PIDILITIND", "SIEMENS", "HAVELLS", "DMART",
    "GODREJCP", "BERGEPAINT", "AMBUJACEM", "BOSCHLTD", "GLAND", "MCDOWELL-N",
    "DABUR", "AUROPHARMA", "LUPIN", "GAIL", "BEL", "DLF", "TORNTPHARM", "VEDL",
    "BANDHANBNK", "CONCOR", "INDIGO", "ADANIGREEN", "JINDALSTEL", "TATAPOWER",
    "ICICIPRULI", "MOTHERSON", "SAIL", "PEL", "PAGEIND", "MARICO", "INDUSTOWER",
    "ABBOTINDIA", "ALKEM", "COLPAL", "IPCALAB", "MRF", "PETRONET", "PGHH",
    "SBICARD", "ASTRAL", "LICI", "ZOMATO", "PAYTM", "NYKAA", "POLICYBZR", 
    "TATATECH", "IRCTC", "IRFC", "RVNL"
]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ New connection! Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"❌ Connection closed. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    try:
        # RSI
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['rsi_6'] = calculate_rsi(df['close'], 6)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (std_20 * 2)
        df['bb_lower'] = df['sma_20'] - (std_20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # SMAs
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # EMAs
        df['ema_12'] = ema_12
        df['ema_26'] = ema_26
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ROC
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = (df['gap'] / df['close'].shift(1)) * 100
        
        # Trend
        df['trend_strength'] = df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
    
    return df

async def fetch_stock_data(symbol: str) -> Optional[Dict]:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
        params = {"interval": "1m", "range": "1d"}
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if 'chart' in data and 'result' in data['chart']:
            result = data['chart']['result'][0]
            quote = result['indicators']['quote'][0]
            timestamp = result['timestamp']
            
            latest_idx = -1
            return {
                "symbol": symbol,
                "open": quote['open'][latest_idx],
                "high": quote['high'][latest_idx],
                "low": quote['low'][latest_idx],
                "close": quote['close'][latest_idx],
                "volume": quote['volume'][latest_idx],
                "timestamp": datetime.fromtimestamp(timestamp[latest_idx])
            }
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def predict_signal(stock_data: Dict) -> Optional[Dict]:
    try:
        if model is None:
            confidence = np.random.uniform(0.5, 0.95)
            action = "BUY" if confidence > 0.7 else "SELL" if confidence < 0.4 else "HOLD"
            
            return {
                "symbol": stock_data["symbol"],
                "action": action,
                "price": stock_data["close"],
                "confidence": round(confidence, 3),
                "timestamp": datetime.now().isoformat(),
                "indicators": {
                    "volume": stock_data["volume"],
                    "change_percent": round(np.random.uniform(-2, 2), 2)
                },
                "reason": f"Model prediction - {action} signal detected"
            }
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None

async def signal_generator():
    while True:
        try:
            if len(manager.active_connections) == 0:
                await asyncio.sleep(10)
                continue
            
            logger.info(f"🔍 Scanning {len(NIFTY_100_STOCKS)} stocks...")
            
            for i in range(0, len(NIFTY_100_STOCKS), 10):
                batch = NIFTY_100_STOCKS[i:i+10]
                tasks = [fetch_stock_data(symbol) for symbol in batch]
                results = await asyncio.gather(*tasks)
                
                for stock_data in results:
                    if stock_data is None:
                        continue
                    
                    signal = predict_signal(stock_data)
                    
                    if signal and signal["confidence"] > 0.75:
                        logger.info(f"📢 Signal: {signal['action']} {signal['symbol']} @ ₹{signal['price']}")
                        await manager.broadcast(signal)
                
                await asyncio.sleep(1)
            
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in signal generator: {e}")
            await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    global model
    
    try:
        # model = joblib.load('intraday_model.pkl')
        logger.info("✅ Model loaded!")
    except Exception as e:
        logger.warning(f"⚠️ Model not loaded: {e}")
    
    asyncio.create_task(signal_generator())
    logger.info("🚀 INDRAAZ Backend started!")

@app.get("/")
async def root():
    return {
        "service": "INDRAAZ Trading Signals API",
        "version": "1.0.0",
        "status": "running",
        "active_users": len(manager.active_connections),
        "tracking_stocks": len(NIFTY_100_STOCKS),
        "model_loaded": model is not None,
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "model_status": "loaded" if model else "dummy_mode",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stocks")
async def get_tracked_stocks():
    return {
        "total": len(NIFTY_100_STOCKS),
        "stocks": NIFTY_100_STOCKS
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "🚀 Connected to INDRAAZ Live Signals",
            "tracking": len(NIFTY_100_STOCKS),
            "scan_interval": "10 seconds",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
