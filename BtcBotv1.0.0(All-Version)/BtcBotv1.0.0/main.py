from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ccxt
import pandas as pd
import ta
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()

# CORS middleware biar aman walau frontend dan backend satu origin, bisa kamu sesuaikan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend di folder frontend (index.html)


class ChatRequest(BaseModel):
    message: str
    timeframe: str = "1h"

MODEL_PATH = "ensemble_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load model dan scaler jika ada
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    rf = RandomForestClassifier(n_estimators=50)
    xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss')
    sgd = SGDClassifier(loss='log_loss')
    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('sgd', sgd)], voting='hard')
    scaler = StandardScaler()

def fetch_ohlcv(symbol, timeframe="1h", limit=2000):
    try:
        exchange = ccxt.indodax({'timeout': 10000, 'enableRateLimit': True})
        markets = exchange.load_markets()
        symbol = symbol.upper()

        matched_symbol = None
        for m in markets.values():
            if m.get('symbol') == symbol:
                matched_symbol = m['symbol']
                break

        if not matched_symbol:
            return pd.DataFrame(), f"⚠️ Symbol '{symbol}' tidak ditemukan di Indodax."

        ohlcv = exchange.fetch_ohlcv(matched_symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"⚠️ Error: {str(e)}"

def preprocess(df):
    df = df.copy()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd_indicator = ta.trend.MACD(df["close"])
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_signal"] = df.apply(lambda row: 1 if row["close"] < row["bb_low"] else (-1 if row["close"] > row["bb_high"] else 0), axis=1)
    df.dropna(inplace=True)

    df["future_return"] = (df["close"].shift(-3) - df["close"]) / df["close"]
    df["target"] = df["future_return"].apply(lambda x: 1 if x > 0.015 else (0 if abs(x) <= 0.005 else -1))
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal"]
    X = df[features]
    y = df["target"]
    global model, scaler
    if len(set(y)) < 2:
        return "⚠️ Data target hanya ada satu kelas, training dibatalkan."
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return "✅ Model berhasil dilatih."

def analyze(symbol, timeframe):
    df, error = fetch_ohlcv(symbol, timeframe)
    if error:
        return error
    df = preprocess(df)
    if df.empty:
        return "⚠️ Data OHLCV atau indikator tidak cukup setelah pembersihan."

    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal"]
    X = df[features]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    signal = preds[-1]
    if signal == 1:
        return "📈 Sinyal: BUY (potensi naik lebih dari 1.5%)"
    elif signal == -1:
        return "📉 Sinyal: SELL (potensi turun lebih dari 1.5%)"
    else:
        return "🤝 Sinyal: HOLD (tidak ada sinyal kuat)"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    message = request.message.upper()
    timeframe = request.timeframe
    df, error = fetch_ohlcv(message, timeframe)
    if error:
        return {"response": error}
    df = preprocess(df)
    training_msg = train_model(df)
    prediction_msg = analyze(message, timeframe)
    return {"response": f"{training_msg}\n{prediction_msg}"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
