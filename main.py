import time
import yfinance as yf
import nest_asyncio
import asyncio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
#import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)
TOKEN = "7888566326:AAF2rpd_i5bv655JcjRCp18TjEECDwivM48"
CHAT_ID = 1343733029
#TOKEN = os.getenv("TOKEN")
#CHAT_ID = int(os.getenv("CHAT_ID"))

last_sent_prices = {}
user_targets = {}
user_portfolios = {}


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


async def send_price_update(context: ContextTypes.DEFAULT_TYPE):
    symbols = ["ADRO", "ENRG"]

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
        if symbol not in ["ADRO", "ENRG"]:
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
    symbols = ["ADRO", "ENRG"]
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
        " - /graf <saham>"
    )


async def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settarget", set_target))
    app.add_handler(CommandHandler("setporto", set_porto))
    app.add_handler(CommandHandler("porto", show_porto))
    app.add_handler(CommandHandler("cek", cek))
    app.add_handler(CommandHandler("graf", graf))

    job_queue = app.job_queue
    job_queue.run_repeating(send_price_update, interval=30, first=5)

    print("Bot berjalan...")
    await app.run_polling()


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
