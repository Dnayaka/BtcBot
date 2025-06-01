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
from ta.momentum import RSIIndicator, StochasticOscillator
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
CACHE = {}
CACHE_TTL = 300  # Cache 5 menit

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

def fetch_ohlcv(symbol, timeframe="1h", limit=2000):
    now = time.time()
    cache_key = f"{symbol}_{timeframe}"

    if cache_key in CACHE:
        data, timestamp = CACHE[cache_key]
        if now - timestamp < CACHE_TTL:
            logger.info(f"📦 Menggunakan data cache untuk {symbol}")
            return data.copy(), None, timestamp

    try:
        exchange = ccxt.indodax({'timeout': 10000, 'enableRateLimit': True})
        exchange.load_markets()
        pair = symbol.upper() + "/IDR"

        if pair not in exchange.symbols:
            return pd.DataFrame(), f"⚠️ Symbol '{pair}' tidak tersedia di Indodax.", None

        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        CACHE[cache_key] = (df.copy(), now)
        return df, None, now

    except Exception as e:
        logger.error(f"Gagal fetch data: {e}")
        return pd.DataFrame(), f"❌ Error fetch data: {str(e)}", None

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

    # EMA Cross
    df["ema_cross"] = 0
    df.loc[
        (df["ema50"].shift(1) < df["ema200"].shift(1)) & (df["ema50"] > df["ema200"]),
        "ema_cross"
    ] = 1  # Golden Cross
    df.loc[
        (df["ema50"].shift(1) > df["ema200"].shift(1)) & (df["ema50"] < df["ema200"]),
        "ema_cross"
    ] = -1  # Death Cross

    # RSI dan Stochastic
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"])
    df["stoch"] = stoch.stoch()

    # Label target untuk training: 3 kelas (buy=1, hold=0, sell=-1)
    df["future_return"] = (df["close"].shift(-3) - df["close"]) / df["close"]
    df["target"] = df["future_return"].apply(lambda x: 1 if x > 0.015 else (0 if abs(x) <= 0.005 else -1))
    df["ema_trend"] = (df["ema50"] > df["ema200"]).astype(int)


    df.dropna(inplace=True)
    return df

def train_model(symbol, timeframe):
    global model, scaler
    df, error, _ = fetch_ohlcv(symbol, timeframe)
    if error:
        return {"error": error}

    df = preprocess(df)
    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal", "rsi", "stoch", "ema_cross", "ema_trend"]
    X = df[features]
    y = df["target"]

    if len(set(y)) < 2:
        return {"error": "⚠️ Data hanya punya satu jenis target, training dibatalkan."}

    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("✅ Model berhasil dilatih dan disimpan.")
    return {"response": "✅ Model berhasil dilatih."}

def format_rupiah(value):
    try:
        if value == 0:
            return "Rp0"
        elif value < 1:
            return f"Rp{value:,.8f}".replace(",", "#").replace(".", ",").replace("#", ".").rstrip("0").rstrip(",")
        else:
            return f"Rp{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
    except:
        return "RpN/A"

def analyze_only(symbol, timeframe):
    df, error, fetched_at = fetch_ohlcv(symbol, timeframe)
    if error:
        return {"error": error}

    df = preprocess(df)
    if df.empty:
        return {"error": "⚠️ Data kosong atau tidak valid setelah preprocessing."}

    features = ["ema50", "ema200", "macd", "macd_signal", "bb_high", "bb_low", "bb_signal", "rsi", "stoch", "ema_cross", "ema_trend"]
    X = df[features]

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return {"error": f"Gagal scaling data: {str(e)}"}

    try:
        probs = model.predict_proba(X_scaled)[-1]
        pred = model.predict(X_scaled)[-1]
    except Exception as e:
        return {"error": f"Gagal prediksi: {str(e)}"}

    last = df.iloc[-1]

    filtered = (
        0 if last["ema_trend"] == 0 else
        0 if max(probs) < 0.7 else
        pred
    )

    # Logika filter sinyal
    if last["ema50"] < last["ema200"]:
        filtered = 0
    elif max(probs) < 0.7:
        filtered = 0
    else:
        filtered = pred

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
        "rsi": float(last["rsi"]),
        "rsi_desc": "Overbought" if last["rsi"] > 70 else ("Oversold" if last["rsi"] < 30 else "Neutral"),
        "stoch": float(last["stoch"]),
        "stoch_desc": "Overbought" if last["stoch"] > 80 else ("Oversold" if last["stoch"] < 20 else "Neutral"),
        "current_price": float(last["close"]),
        "formatted_price": format_rupiah(last["close"]),
        "last_updated": int(fetched_at) if fetched_at else None,
        "formatted_ema50": format_rupiah(last["ema50"]),
        "formatted_ema200": format_rupiah(last["ema200"]),
        "ema_cross": int(last["ema_cross"]),
        "ema_cross_desc": (
            "📈 Golden Cross (potensi bullish)" if last["ema_cross"] == 1 else
            "📉 Death Cross (potensi bearish)" if last["ema_cross"] == -1 else
            "➡️ Tidak ada persilangan"
        ),
        "ema_trend": "📈 Bullish (EMA50 > EMA200)" if last["ema_trend"] else "📉 Bearish (EMA50 < EMA200)"
    }
