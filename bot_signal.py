import ccxt
import pandas as pd
import numpy as np
import time
import requests
import os
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

load_dotenv()

# ==================== CONFIGURAÇÕES ====================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAMES = os.getenv("TIMEFRAMES").split(",")   # Ex: "15m,1h,4h"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 60))

# Parâmetros dos indicadores
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 70))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))

ADX_THRESHOLD = float(os.getenv("ADX_THRESHOLD", 25))
BB_STD = float(os.getenv("BB_STD", 2.0))
VOLUME_MULT = float(os.getenv("VOLUME_MULT", 1.2))

EMA_FAST = int(os.getenv("EMA_FAST", 50))
EMA_SLOW = int(os.getenv("EMA_SLOW", 200))

RISK_PERCENT = float(os.getenv("RISK_PERCENT", 2.0))
TP_RATIO = float(os.getenv("TP_RATIO", 2.0))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", 4))

# ==================== INICIALIZAÇÃO ====================
exchange = ccxt.bybit({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

current_position = {
    "symbol": None,
    "side": None,
    "entry_price": None,
    "entry_time": None
}

# ==================== FUNÇÕES AUXILIARES ====================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

def fetch_ohlcv(tf, limit=200):
    return exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)

def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # RSI
    rsi = RSIIndicator(close, window=RSI_PERIOD).rsi()

    # EMAs
    ema_fast = EMAIndicator(close, window=EMA_FAST).ema_indicator()
    ema_slow = EMAIndicator(close, window=EMA_SLOW).ema_indicator()

    # MACD
    macd = MACD(close)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()

    # ADX
    adx = ADXIndicator(high, low, close, window=14).adx()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=BB_STD)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_width = (bb_upper - bb_lower) / close

    # ATR
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()

    # OBV
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Volume médio
    avg_volume = volume.rolling(window=20).mean()

    return {
        'rsi': rsi,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'adx': adx,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_width': bb_width,
        'atr': atr,
        'obv': obv,
        'volume': volume,
        'avg_volume': avg_volume,
        'close': close,
        'high': high,
        'low': low
    }

def get_indicators_for_tf(tf):
    candles = fetch_ohlcv(tf, limit=200)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    ind = calculate_indicators(df)

    # Últimos dois valores para detectar cruzamentos
    result = {
        'rsi': ind['rsi'].iloc[-2:].values,
        'ema_fast': ind['ema_fast'].iloc[-1],
        'ema_slow': ind['ema_slow'].iloc[-1],
        'macd_line': ind['macd_line'].iloc[-2:].values,
        'macd_signal': ind['macd_signal'].iloc[-2:].values,
        'adx': ind['adx'].iloc[-1],
        'bb_upper': ind['bb_upper'].iloc[-2:].values,  # dois últimos
        'bb_lower': ind['bb_lower'].iloc[-2:].values,  # dois últimos
        'atr': ind['atr'].iloc[-1],
        'obv': ind['obv'].iloc[-2:].values,
        'volume': ind['volume'].iloc[-1],
        'avg_volume': ind['avg_volume'].iloc[-1],
        'close': ind['close'].iloc[-1],
        'close_prev': ind['close'].iloc[-2] if len(ind['close']) > 1 else None,
    }
    return result

def rsi_cross_score(prev_rsi, curr_rsi):
    if prev_rsi <= RSI_OVERSOLD and curr_rsi > RSI_OVERSOLD:
        return 1
    elif prev_rsi >= RSI_OVERBOUGHT and curr_rsi < RSI_OVERBOUGHT:
        return -1
    return 0

def macd_cross_score(prev_macd, curr_macd, prev_signal, curr_signal):
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        return 1
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        return -1
    return 0

def bb_reentry_score(prev_close, curr_close, prev_bb_lower, curr_bb_lower, prev_bb_upper, curr_bb_upper):
    """
    Long: preço estava abaixo da banda inferior e agora voltou para dentro (ou acima)
    Short: preço estava acima da banda superior e agora voltou para dentro (ou abaixo)
    """
    long_score = 0
    short_score = 0
    # Long: candle anterior abaixo da banda inferior, atual acima dela (pode estar dentro)
    if prev_close < prev_bb_lower and curr_close > curr_bb_lower:
        long_score = 1
    # Short: candle anterior acima da banda superior, atual abaixo dela
    if prev_close > prev_bb_upper and curr_close < curr_bb_upper:
        short_score = 1
    return long_score, short_score

def obv_trend_score(prev_obv, curr_obv):
    if curr_obv > prev_obv:
        return 1
    elif curr_obv < prev_obv:
        return -1
    return 0

def volume_score(volume, avg_volume):
    return 1 if volume > avg_volume * VOLUME_MULT else 0

def adx_score(adx):
    return 1 if adx > ADX_THRESHOLD else 0

def trend_alignment_score(ema_fast, ema_slow):
    return 1 if ema_fast > ema_slow else -1

def generate_signal(tf_entry, tf_confirm, tf_trend):
    entry = get_indicators_for_tf(tf_entry)
    confirm = get_indicators_for_tf(tf_confirm)
    trend = get_indicators_for_tf(tf_trend)

    if any(np.isnan(x) for x in [entry['adx'], confirm['adx'], trend['adx']]):
        return None, None, None

    bull_score = 0
    bear_score = 0

    # Tendência (peso 2)
    trend_dir = trend_alignment_score(trend['ema_fast'], trend['ema_slow'])
    if trend_dir == 1:
        bull_score += 2
    else:
        bear_score += 2

    # ADX
    if adx_score(trend['adx']):
        bull_score += 1
        bear_score += 1
    if adx_score(confirm['adx']):
        bull_score += 1
        bear_score += 1

    # RSI entrada
    rsi_entry = rsi_cross_score(entry['rsi'][0], entry['rsi'][1])
    if rsi_entry == 1:
        bull_score += 2
    elif rsi_entry == -1:
        bear_score += 2

    # MACD entrada
    macd_entry = macd_cross_score(entry['macd_line'][0], entry['macd_line'][1],
                                  entry['macd_signal'][0], entry['macd_signal'][1])
    if macd_entry == 1:
        bull_score += 1
    elif macd_entry == -1:
        bear_score += 1

    # Volume
    if volume_score(entry['volume'], entry['avg_volume']):
        bull_score += 1
        bear_score += 1

    # OBV confirmação
    obv_dir = obv_trend_score(confirm['obv'][0], confirm['obv'][1])
    if obv_dir == 1:
        bull_score += 1
    elif obv_dir == -1:
        bear_score += 1

    # Bollinger Bands (entrada)
    bb_long, bb_short = bb_reentry_score(
        entry['close_prev'], entry['close'],
        entry['bb_lower'][0], entry['bb_lower'][1],
        entry['bb_upper'][0], entry['bb_upper'][1]
    )
    if bb_long:
        bull_score += 1
    if bb_short:
        bear_score += 1

    # Decisão final
    if bull_score >= SCORE_THRESHOLD and bull_score > bear_score:
        reasons = []
        if rsi_entry == 1: reasons.append("RSI saiu de oversold")
        if macd_entry == 1: reasons.append("MACD bullish cross")
        if obv_dir == 1: reasons.append("OBV em alta")
        if volume_score(entry['volume'], entry['avg_volume']): reasons.append("Volume alto")
        if adx_score(trend['adx']): reasons.append("ADX forte")
        if trend_dir == 1: reasons.append("Tendência de alta")
        if bb_long: reasons.append("Retorno das bandas de Bollinger")
        reason = ", ".join(reasons)
        return "long", reason, entry['close']

    elif bear_score >= SCORE_THRESHOLD and bear_score > bull_score:
        reasons = []
        if rsi_entry == -1: reasons.append("RSI saiu de overbought")
        if macd_entry == -1: reasons.append("MACD bearish cross")
        if obv_dir == -1: reasons.append("OBV em queda")
        if volume_score(entry['volume'], entry['avg_volume']): reasons.append("Volume alto")
        if adx_score(trend['adx']): reasons.append("ADX forte")
        if trend_dir == -1: reasons.append("Tendência de baixa")
        if bb_short: reasons.append("Retorno das bandas de Bollinger")
        reason = ", ".join(reasons)
        return "short", reason, entry['close']

    return None, None, None

def calculate_sl_tp(entry_price, side, atr_value):
    sl_atr = atr_value * 1.5
    sl_percent = entry_price * (RISK_PERCENT / 100.0)
    sl_amount = max(sl_atr, sl_percent)
    if side == "long":
        sl = entry_price - sl_amount
        tp = entry_price + sl_amount * TP_RATIO
    else:
        sl = entry_price + sl_amount
        tp = entry_price - sl_amount * TP_RATIO
    return sl, tp

def calculate_leverage_suggestion(entry_price, sl_price, risk_percent, atr=None):
    if entry_price <= 0 or sl_price <= 0:
        return 5
    sl_distance_percent = abs(entry_price - sl_price) / entry_price
    if sl_distance_percent <= 0.001:
        return 10
    leverage = (risk_percent / 100) / sl_distance_percent
    leverage = int(leverage)

    if leverage <= 1:
        leverage = 1
    elif leverage <= 2:
        leverage = 2
    elif leverage <= 3:
        leverage = 3
    elif leverage <= 5:
        leverage = 5
    elif leverage <= 10:
        leverage = 10
    elif leverage <= 20:
        leverage = 20
    else:
        leverage = 25

    if atr:
        atr_percent = atr / entry_price
        if atr_percent > 0.03:
            leverage = max(1, int(leverage * 0.7))
        elif atr_percent < 0.01 and leverage < 10:
            leverage = min(10, int(leverage * 1.2))
    return leverage

def format_signal_message(side, reason, entry_price, sl, tp, leverage):
    emoji = "🟢 LONG" if side == "long" else "🔴 SHORT"
    if side == "long":
        risk_reward = abs((tp - entry_price) / (entry_price - sl))
    else:
        risk_reward = abs((entry_price - tp) / (sl - entry_price))

    msg = f"{emoji} SINAL DE {side.upper()}\n"
    msg += f"┌─────────────────────────┐\n"
    msg += f"│ Par: {SYMBOL}\n"
    msg += f"│ Preço entrada: ${entry_price:,.2f}\n"
    msg += f"│ Stop Loss: ${sl:,.2f} ({abs((sl - entry_price)/entry_price*100):.2f}%)\n"
    msg += f"│ Take Profit: ${tp:,.2f} ({abs((tp - entry_price)/entry_price*100):.2f}%)\n"
    msg += f"│ Risco/Recompensa: 1:{risk_reward:.2f}\n"
    msg += f"│ Alavancagem sugerida: {leverage}x\n"
    msg += f"│\n"
    msg += f"│ Motivo: {reason}\n"
    msg += f"└─────────────────────────┘\n"
    msg += "#BTC #Futuros"
    return msg

# ==================== LOOP PRINCIPAL ====================
def run():
    send_telegram("🤖 Bot Gerador de Sinais BTC Futuros (versão com BB e alavancagem) iniciado.")
    send_telegram("📊 Configuração: 3 timeframes + ADX + OBV + BB + Sistema de Pontuação")

    if len(TIMEFRAMES) < 3:
        send_telegram("❌ ERRO: Defina três timeframes no .env (ex: 15m,1h,4h)")
        return

    tf_entry = TIMEFRAMES[0].strip()
    tf_confirm = TIMEFRAMES[1].strip()
    tf_trend = TIMEFRAMES[2].strip()

    while True:
        try:
            side, reason, entry_price = generate_signal(tf_entry, tf_confirm, tf_trend)

            if side and current_position['symbol'] is None:
                entry_indic = get_indicators_for_tf(tf_entry)
                atr_value = entry_indic['atr']
                sl, tp = calculate_sl_tp(entry_price, side, atr_value)

                leverage = calculate_leverage_suggestion(
                    entry_price,
                    sl,
                    RISK_PERCENT,
                    atr_value
                )

                msg = format_signal_message(side, reason, entry_price, sl, tp, leverage)
                send_telegram(msg)

                current_position['symbol'] = SYMBOL
                current_position['side'] = side
                current_position['entry_price'] = entry_price
                current_position['entry_time'] = time.time()

                time.sleep(60 * 5)  # 5 min de cooldown

            if current_position['symbol'] is not None:
                if time.time() - current_position['entry_time'] > 7200:
                    current_position['symbol'] = None
                    current_position['side'] = None
                    send_telegram("⏰ Posição antiga removida da memória (simulação).")

        except Exception as e:
            print(f"Erro no loop: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()