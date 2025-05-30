import ccxt
import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import ta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Inisialisasi model dan scaler
def load_or_init_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Model & scaler berhasil dimuat.")
    else:
        rf = RandomForestClassifier(n_estimators=50)
        xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss')
        sgd = SGDClassifier(loss='log_loss')
        model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('sgd', sgd)], voting='soft')
        scaler = StandardScaler()
        logger.info("⚠️ Model & scaler baru dibuat.")
    return model, scaler

model, scaler = load_or_init_model()

# Fetch data dari Indodax
def fetch_ohlcv(symbol, timeframe="1h", limit=2000):
    try:
        exchange = ccxt.indodax({'timeout': 10000, 'enableRateLimit': True})
        exchange.load_markets()
        symbol = symbol.upper() + "/IDR"

        if symbol not in exchange.symbols:
            return pd.DataFrame(), f"⚠️ Symbol '{symbol}' tidak tersedia di Indodax."

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df, None
    except Exception as e:
        logger.error(f"Gagal fetch data: {e}")
        return pd.DataFrame(), f"❌ Error fetch data: {str(e)}"

# Preprocessing dan feature engineering
def preprocess(df):
    df = df.copy()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_signal"] = df.apply(lambda row: 1 if row["close"] < row["bb_low"] else (-1 if row["close"] > row["bb_high"] else 0), axis=1)

    df["future_return"] = (df["close"].shift(-3) - df["close"]) / df["close"]
    df["target"] = df["future_return"].apply(lambda x: 1 if x > 0.015 else (0 if abs(x) <= 0.005 else -1))
    df.dropna(inplace=True)

    return df

# Training model
def train_model(df):
    global model, scaler
    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal"]
    X = df[features]
    y = df["target"]

    if len(set(y)) < 2:
        logger.warning("Hanya ada satu kelas target, model tidak dilatih.")
        return "⚠️ Data hanya punya satu jenis target, training dibatalkan."

    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("✅ Model berhasil dilatih dan disimpan.")
    return "✅ Model berhasil dilatih."

# Analisa dan prediksi
def analyze(symbol, timeframe):
    df, error = fetch_ohlcv(symbol, timeframe)
    if error:
        return {"error": error}
    
    df = preprocess(df)
    if df.empty:
        return {"error": "⚠️ Data kosong atau tidak valid setelah preprocessing."}

    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal"]
    X = df[features]
    X_scaled = scaler.transform(X)

    try:
        probs = model.predict_proba(X_scaled)[-1]
        pred = model.predict(X_scaled)[-1]
    except Exception as e:
        logger.warning(f"Gagal prediksi: {e}")
        return {"error": f"❌ Gagal memproses prediksi: {str(e)}"}

    last = df.iloc[-1]
    filtered = (
        0 if last["ema50"] < last["ema200"] else
        0 if max(probs) < 0.7 else
        pred
    )

    mapping = {
        1: ("BUY", "potensi naik > 1.5%"),
        0: ("HOLD", "tidak ada sinyal kuat"),
        -1: ("SELL", "potensi turun > 1.5%"),
    }
    signal, desc = mapping.get(filtered, ("HOLD", "tidak dikenali"))

    return {
        "prediction_signal": signal,
        "prediction_desc": desc,
        "raw_prediction": int(filtered),
        "probability": probs.tolist() if probs is not None else None,
        "ema50": float(last["ema50"]),
        "ema200": float(last["ema200"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
    }

# Fungsi utama yang dipanggil FastAPI
def train_and_analyze(symbol, timeframe):
    df, error = fetch_ohlcv(symbol, timeframe)
    if error:
        return {"response": error, "details": {"error": error}}

    df = preprocess(df)
    train_msg = train_model(df)
    analysis = analyze(symbol, timeframe)

    if "error" in analysis:
        return {"response": analysis["error"], "details": analysis}

    return {
        "response": f"{train_msg}\nSinyal: {analysis['prediction_signal']} ({analysis['prediction_desc']})",
        "details": analysis,
    }
