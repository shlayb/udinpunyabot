import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.dates as mdates
import io
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()
TOKEN = "7888566326:AAF2rpd_i5bv655JcjRCp18TjEECDwivM48"
CHAT_ID = 1343733029

# Global variables for GRU
model_gru = None
price_scaler = None
X_scaled_global = None
window_size = 30

last_sent_prices = {}
user_targets = {}
user_portfolios = {}

# === FUNGSI TRAIN GRU SEKALI ===
def train_gru_model(symbol="ADRO"):
    global model_gru, price_scaler, X_scaled_global

    ticker = f"{symbol}.JK"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=1825)

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if data.empty:
        raise Exception("❗ Data saham tidak tersedia.")

    data['Returns'] = data['Close'].pct_change()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data = data.dropna()

    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Returns', 'MA_20']
    X_data = data[features].values
    y_data = data['Close'].values

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    price_scaler = MinMaxScaler()
    y_scaled = price_scaler.fit_transform(y_data.reshape(-1, 1)).reshape(-1)

    X_scaled_global = X_scaled

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i+window_size])
        y_seq.append(y_scaled[i+window_size])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    split = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential([
        GRU(50, input_shape=(window_size, X_scaled.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=400, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

    model_gru = model

def get_price_info(symbol):
    ticker = yf.Ticker(f"{symbol}.JK")
    data = ticker.history(period="1d", interval="1m")
    if data.empty:
        return None, None, None

    latest = data.iloc[-1]["Close"]
    open_price = data.iloc[0]["Open"]
    percent_change = ((latest - open_price) / open_price) * 100
    return latest, open_price, percent_change

def generate_stock_chart(symbol):
    ticker = yf.Ticker(f"{symbol}.JK")
    data = ticker.history(period="1d", interval="5m")

    if data.empty:
        return None

    times = data.index
    prices = data["Close"]
    color = 'green' if prices[-1] >= prices[0] else 'red'

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, prices, color=color, linewidth=2)
    ax.set_title(f"{symbol}.JK - Grafik Hari Ini", fontsize=14, color='white')
    ax.set_xlabel("Waktu", color='white')
    ax.set_ylabel("Harga (Rp)", color='white')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

async def prediksi_besok(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global model_gru, price_scaler, X_scaled_global, window_size

    if len(context.args) != 1:
        await update.message.reply_text("Gunakan: /tmrw <saham>")
        return

    symbol = context.args[0].upper()

    if not symbol.isalnum():
        await update.message.reply_text("Nama saham tidak valid. Gunakan huruf dan angka saja.")
        return

    try:
        train_gru_model(symbol)
    except Exception as e:
        await update.message.reply_text(str(e))
        return

    if model_gru is None or price_scaler is None or X_scaled_global is None:
        await update.message.reply_text("Model belum siap. Coba lagi nanti.")
        return

    last_seq = X_scaled_global[-window_size:]
    pred_scaled = model_gru.predict(last_seq.reshape(1, window_size, X_scaled_global.shape[1]), verbose=0)
    pred_price = price_scaler.inverse_transform(pred_scaled)[0][0]

    await update.message.reply_text(f"\U0001F4C8 Prediksi harga {symbol}.JK untuk besok: Rp {pred_price:,.2f}")

async def send_price_update(context: ContextTypes.DEFAULT_TYPE):
    symbols = ["ADRO", "ENRG", "MBMA"]

    for symbol in symbols:
        price, open_price, percent = get_price_info(symbol)
        if price is None:
            continue

        last_price = last_sent_prices.get(symbol)
        if price != last_price:
            arrow = "🟢🟢🟢" if percent >= 0 else "🔴🔴🔴"
            sign = "+" if percent >= 0 else ""
            message = (
                f"{arrow}\n"
                f"Harga {symbol}.JK: Rp{price:.2f}\n"
                f"Open hari ini: Rp{open_price:.2f}\n"
                f"Perubahan: {sign}{percent:.2f}%"
            )
            await context.bot.send_message(chat_id=CHAT_ID, text=message)
            last_sent_prices[symbol] = price

        for chat_id, targets in user_targets.items():
            target_price = targets.get(symbol)
            if target_price and price >= target_price:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚨 ALERT: Harga {symbol}.JK telah mencapai Rp{price:.2f} (Target: Rp{target_price:.2f})"
                )
                del user_targets[chat_id][symbol]


async def set_target(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 2:
            raise ValueError("Format salah. Gunakan: /settarget <saham> <harga>")

        symbol = context.args[0].upper()
        if symbol not in ["ADRO", "ENRG"]:
            await update.message.reply_text("❗ Saham tidak dikenal. Gunakan: ADRO atau ENRG.")
            return

        target = float(context.args[1])
        chat_id = update.effective_chat.id

        if chat_id not in user_targets:
            user_targets[chat_id] = {}

        user_targets[chat_id][symbol] = target
        await update.message.reply_text(f"🎯 Target harga untuk {symbol}.JK diset ke Rp{target:.2f}")
    except ValueError:
        await update.message.reply_text("❗ Format salah. Gunakan: /settarget <saham> <harga>")


async def set_porto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 2:
            raise ValueError("Gunakan: /setporto <saham> <jumlah lot>")

        symbol = context.args[0].upper()
        if symbol not in ["ADRO", "ENRG", "MBMA"]:
            await update.message.reply_text("❗ Saham tidak dikenal. Gunakan: ADRO atau ENRG.")
            return

        jumlah_lot = int(context.args[1])
        chat_id = update.effective_chat.id

        if chat_id not in user_portfolios:
            user_portfolios[chat_id] = {}

        user_portfolios[chat_id][symbol] = jumlah_lot
        await update.message.reply_text(f"📊 Portofolio disimpan: {jumlah_lot} lot {symbol}.JK")
    except ValueError:
        await update.message.reply_text("❗ Format salah. Gunakan: /setporto <saham> <jumlah lot>")


async def show_porto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in user_portfolios or not user_portfolios[chat_id]:
        await update.message.reply_text("📭 Portofolio kamu kosong.")
        return

    msg = "📈 *Portofolio Kamu Saat Ini:*\n"
    total_value = 0.0

    for symbol, lot in user_portfolios[chat_id].items():
        price, _, _ = get_price_info(symbol)
        if price is None:
            continue
        lembar = lot * 100
        nilai = price * lembar
        total_value += nilai
        msg += f"\n{symbol}.JK — {lot} lot ({lembar} lembar)\nHarga: Rp{price:.2f}\nNilai: Rp{nilai:,.2f}\n"

    msg += f"\n💰 *Total Nilai:* Rp{total_value:,.2f}"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cek(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = ["ADRO", "ENRG", "MBMA"]
    message = ""

    for symbol in symbols:
        price, open_price, percent = get_price_info(symbol)
        if price is None:
            continue

        arrow = "🟢🟢🟢" if percent >= 0 else "🔴🔴🔴"
        sign = "+" if percent >= 0 else ""
        message += (
            f"{arrow}\n"
            f"Harga {symbol}.JK: Rp{price:.2f}\n"
            f"Open hari ini: Rp{open_price:.2f}\n"
            f"Perubahan: {sign}{percent:.2f}%\n\n"
        )

    if not message:
        message = "❗ Gagal mengambil data harga saat ini."
    await update.message.reply_text(message)


async def graf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Gunakan format: /graf <saham>")
        return

    symbol = context.args[0].upper()
    if symbol not in ["ADRO", "ENRG"]:
        await update.message.reply_text("❗ Saham tidak dikenali. Hanya mendukung ADRO & ENRG.")
        return

    chart = generate_stock_chart(symbol)
    if chart is None:
        await update.message.reply_text("❗ Gagal mengambil data grafik.")
        return

    await update.message.reply_photo(photo=chart, caption=f"Grafik {symbol}.JK hari ini 📈")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot pemantau saham aktif! 🔍\n"
        "Gunakan:\n"
        " - /settarget <saham> <harga>\n"
        " - /setporto <saham> <lot>\n"
        " - /porto\n"
        " - /cek\n"
        " - /graf <saham>\n"
        " - /tmrw <saham>"
    )


async def main():
    print("Melatih model GRU untuk prediksi harga...")
    train_gru_model("ADRO")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tmrw", prediksi_besok))
    app.add_handler(CommandHandler("settarget", set_target))
    app.add_handler(CommandHandler("setporto", set_porto))
    app.add_handler(CommandHandler("porto", show_porto))
    app.add_handler(CommandHandler("cek", cek))
    app.add_handler(CommandHandler("graf", graf))

    job_queue = app.job_queue
    job_queue.run_repeating(send_price_update, interval=30, first=5)

    print("Bot berjalan...")
    await app.run_polling()


if __name__ == '__main__':
    import asyncio
    import nest_asyncio
    
    # Apply nest_asyncio to allow nested use of asyncio.run and loop.run_until_complete
    nest_asyncio.apply()
    
    # Now we can safely run our async code
    asyncio.run(main())
