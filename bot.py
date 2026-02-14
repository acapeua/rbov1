import ccxt
import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from ta.momentum import RSIIndicator

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

RSI_LOW = float(os.getenv("RSI_LOW", 30))
RSI_HIGH = float(os.getenv("RSI_HIGH", 70))

TIMEFRAMES = os.getenv("TIMEFRAMES").split(",")
SYMBOL = os.getenv("SYMBOL")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL"))

exchange = ccxt.bybit({
    "enableRateLimit": True,
})
last_state = {}  # guarda estado por timeframe

def send(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def get_rsi(tf):
    candles = exchange.fetch_ohlcv(SYMBOL, tf, limit=100)
    df = pd.DataFrame(candles, columns=["t","o","h","l","c","v"])
    rsi = RSIIndicator(df["c"], window=14).rsi()
    return rsi.iloc[-2], rsi.iloc[-1]  # anterior, atual

def run():
    send("🤖 Bot RSI iniciado com sucesso.")
    while True:
        for tf in TIMEFRAMES:
            try:
                prev_rsi, curr_rsi = get_rsi(tf)
                key = f"{SYMBOL}_{tf}"

                # Inicializa estado
                if key not in last_state:
                    last_state[key] = curr_rsi
                    continue

                # Cruzamento para baixo (sobrevendido)
                if prev_rsi > RSI_LOW and curr_rsi <= RSI_LOW:
                    send(f"🔻 RSI CRUZOU ABAIXO\n{SYMBOL}\nTF: {tf}\nRSI: {curr_rsi:.2f}")

                # Cruzamento para cima (sobrecomprado)
                if prev_rsi < RSI_HIGH and curr_rsi >= RSI_HIGH:
                    send(f"🔺 RSI CRUZOU ACIMA\n{SYMBOL}\nTF: {tf}\nRSI: {curr_rsi:.2f}")

                last_state[key] = curr_rsi

            except Exception as e:
                print(f"Erro {tf}:", e)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()
