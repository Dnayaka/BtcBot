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
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    message: str
    timeframe: str = "1h"

MODEL_PATH = "ensemble_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load or initialize model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model dan scaler berhasil dimuat.")
else:
    rf = RandomForestClassifier(n_estimators=50)
    xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss')
    sgd = SGDClassifier(loss='log_loss')
    # Voting soft agar bisa pakai predict_proba
    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('sgd', sgd)], voting='soft')
    scaler = StandardScaler()
    logging.info("Model dan scaler baru dibuat.")

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
        logging.warning("Data target hanya ada satu kelas, training dibatalkan.")
        return "⚠️ Data target hanya ada satu kelas, training dibatalkan."
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info("Model berhasil dilatih dan disimpan.")
    return "✅ Model berhasil dilatih."

def analyze(symbol, timeframe):
    df, error = fetch_ohlcv(symbol, timeframe)
    if error:
        return {"error": error}

    df = preprocess(df)
    if df.empty:
        return {"error": "⚠️ Data OHLCV atau indikator tidak cukup setelah pembersihan."}

    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal"]
    X = df[features]
    X_scaled = scaler.transform(X)

    try:
        preds_proba = model.predict_proba(X_scaled)
        last_prob = preds_proba[-1]
        last_pred = model.predict(X_scaled)[-1]
    except AttributeError:
        last_pred = model.predict(X_scaled)[-1]
        last_prob = None

    last_row = df.iloc[-1]

    # Filter sinyal berdasarkan EMA tren dan probabilitas confidence
    if last_row["ema50"] < last_row["ema200"]:
        filtered_signal = 0  # HOLD, tren turun
    elif last_prob is not None and max(last_prob) < 0.7:
        filtered_signal = 0  # HOLD, confidence rendah
    else:
        filtered_signal = last_pred

    signals_map = {
        1: ("BUY", "potensi naik lebih dari 1.5%"),
        0: ("HOLD", "tidak ada sinyal kuat"),
        -1: ("SELL", "potensi turun lebih dari 1.5%"),
    }

    signal_name, signal_desc = signals_map.get(filtered_signal, ("HOLD", "tidak ada sinyal kuat"))

    return {
        "prediction_signal": signal_name,
        "prediction_desc": signal_desc,
        "raw_prediction": int(filtered_signal),
        "probability": last_prob.tolist() if last_prob is not None else None,
        "ema50": float(last_row["ema50"]),
        "ema200": float(last_row["ema200"]),
        "macd": float(last_row["macd"]),
        "macd_signal": float(last_row["macd_signal"]),
    }

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    message = request.message.upper()
    timeframe = request.timeframe

    logging.info(f"Request received: symbol={message}, timeframe={timeframe}")

    df, error = fetch_ohlcv(message, timeframe)
    if error:
        logging.error(error)
        return {"response": error}

    df = preprocess(df)

    training_msg = train_model(df)
    prediction_result = analyze(message, timeframe)

    if "error" in prediction_result:
        response_text = prediction_result["error"]
    else:
        response_text = (
            f"{training_msg}\n"
            f"Sinyal: {prediction_result['prediction_signal']} ({prediction_result['prediction_desc']})"
        )

    logging.info(f"Response: {response_text}")

    return {
        "response": response_text,
        "details": prediction_result
    }

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
