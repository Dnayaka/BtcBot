from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)
import requests
from datetime import datetime
import os

# Variabel global pilihan default
current_coin = "BTC"
current_tf = "1h"

# Password untuk train (harus diset sebagai env variable juga)
TRAIN_PASSWORD = os.getenv("TRAIN_PASSWORD") or "Dnayaka#011211"  # ganti sesuai kebutuhan

def build_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("BTC", callback_data='coin_BTC'),
            InlineKeyboardButton("ETH", callback_data='coin_ETH'),
            InlineKeyboardButton("DOGE", callback_data='coin_DOGE'),
        ],
        [
            InlineKeyboardButton("1h", callback_data='tf_1h'),
            InlineKeyboardButton("4h", callback_data='tf_4h'),
            InlineKeyboardButton("1d", callback_data='tf_1d'),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

def format_analysis(data):
    d = data.get("details", {})
    ts = d.get("last_updated")
    if ts:
        try:
            dt_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt_str = str(ts)
    else:
        dt_str = "N/A"

    text = (
        f"✅ *Analisis selesai untuk {data.get('response', '').split()[-1]}*\n\n"
        f"*Sinyal:* {d.get('prediction_signal', 'N/A')} ({d.get('prediction_desc', '')})\n"
        f"*Raw Prediction:* {d.get('raw_prediction', 'N/A')}\n"
        f"*Pola:* "{d.get('pattern_desc', 'N/A')}
        f"*Probability:* {', '.join(f'{p:.3f}' for p in d.get('probability', []))}\n"
        f"*Harga sekarang:* {d.get('formatted_price', 'N/A')}\n"
        f"*EMA50:* {d.get('formatted_ema50', 'N/A')}\n"
        f"*EMA200:* {d.get('formatted_ema200', 'N/A')}\n"
        f"*EMA Cross:* {d.get('ema_cross', 'N/A')} ({d.get('ema_cross_desc', '')})\n"
        f"*EMA Trend:* {d.get('ema_trend', 'N/A')}\n"
        f"*MACD:* {d.get('macd', 0):,.2f}\n"
        f"*MACD Signal:* {d.get('macd_signal', 0):,.2f}\n"
        f"*RSI:* {d.get('rsi', 0):.2f} ({d.get('rsi_desc', '')})\n"
        f"*Stochastic:* {d.get('stoch', 0):.2f} ({d.get('stoch_desc', '')})\n"
        f"\n_Last updated: {dt_str}"
    )
    return text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Pilih coin dan timeframe untuk analisis:",
        reply_markup=build_keyboard()
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_coin, current_tf
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("coin_"):
        current_coin = data.split("_")[1]
    elif data.startswith("tf_"):
        current_tf = data.split("_")[1]

    # Update pesan tombol dengan pilihan baru
    await query.edit_message_text(
        text=(
            f"✅ Pilihan saat ini:\n"
            f"Coin: {current_coin}\n"
            f"Timeframe: {current_tf}\n\n"
            "Ketik /analyze untuk mulai analisis.\n"
            "Ketik /train <password> untuk melatih model (admin only)."
        ),
        reply_markup=build_keyboard()
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payload = {
        "message": current_coin,
        "timeframe": current_tf
    }
    try:
        res = requests.post("http://localhost:8000/chat", json=payload)
        res.raise_for_status()
        data = res.json()
        text = format_analysis(data)
    except Exception as e:
        text = f"❌ Gagal ambil data: {e}"

    # Kirim pesan baru dengan hasil analisis + tombol tetap ada
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=build_keyboard())

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("❌ Mohon sertakan password admin. Contoh: /train Dnayaka#011211")
        return

    password = args[0]

    if password != TRAIN_PASSWORD:
        await update.message.reply_text("🔒 Password salah. Akses ditolak.")
        return

    payload = {
        "symbol": current_coin.lower(),
        "timeframe": current_tf,
        "password": password
    }
    try:
        res = requests.post("http://localhost:8000/train", json=payload)
        res.raise_for_status()
        data = res.json()
        await update.message.reply_text(f"✅ {data.get('response', 'Model berhasil dilatih.')}")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal melatih model: {e}")

if __name__ == "__main__":
    TOKEN = os.getenv("7577395786:AAH7eBIMY8_e3mCk8hc3T3Quqj478yNlZ8I") or "7577395786:AAH7eBIMY8_e3mCk8hc3T3Quqj478yNlZ8I"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("train", train))

    print("Bot started...")
    app.run_polling()
