#!/usr/bin/env python3
"""
OmniTrade AI – Fully Async Telegram Crypto Trading Bot (Replit Ready)
Uses httpx for all network requests → no event-loop blocking.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------- LOGGING ----------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("OmniTradeAI")

# ---------- ENVIRONMENT VARIABLES (Replit Secrets) ----------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
BINANCE_API_KEY = os.environ["BINANCE_API_KEY"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# ---------- DEEPSEEK CONFIG ----------
DEEPSEEK_CHAT_MODEL = "deepseek-chat"           # V3
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"   # R1
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
BINANCE_REST_URL = "https://api.binance.com/api/v3"

# ---------- SYSTEM PROMPTS ----------
SYSTEM_PROMPT_GENERAL = (
    "You are 'OmniTrade AI', a world-class crypto trading assistant on Telegram. "
    "You speak English and Roman Urdu fluently, adapting your reply language to the user. "
    "You handle casual conversations, basic crypto questions, and beginner education. "
    "Keep answers friendly, concise, and helpful. "
    "Always remind users that trading involves risk and they should never invest more than they can afford to lose. "
    "Never give financial advice that guarantees profits."
)

SYSTEM_PROMPT_SIGNAL = (
    "You are 'OmniTrade AI', a professional institutional crypto trader and market analyst. "
    "You are an expert in technical analysis (RSI, MACD, Bollinger Bands, EMA, Fibonacci, Volume Profile, Price Action, Smart Money concepts) and risk management. "
    "You respond in a clear, structured, data‑driven manner. You always provide a market bias (LONG/SHORT), entry strategy, stop‑loss, and take‑profit targets when asked for a signal. "
    "You adapt your language to the user (English or Roman Urdu). "
    "Your analysis is based on the provided real‑time market data and technical indicators. "
    "You NEVER guarantee profits and always include a risk warning.\n\n"
    "When generating a trade signal, strictly follow this format:\n"
    "📊 [SYMBOL] Analysis ([TIMEFRAME])\n\n"
    "Market Bias: [LONG/SHORT]\n\n"
    "Entry Strategy: [Direct Market / Limit Order / Breakout Confirmation]\n"
    "Entry Zone: [price range]\n"
    "Stop Loss: [price]\n"
    "Take Profit Targets:\n"
    "  TP1: [price]\n"
    "  TP2: [price]\n"
    "  TP3: [price]\n\n"
    "Reasoning: [brief explanation based on indicators, structure, volume]\n\n"
    "⚠️ Risk Warning: This is not financial advice. Use proper risk management."
)

# ---------- TECHNICAL INDICATORS (Pure Python) ----------
def compute_rsi(close_prices: List[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(close_prices) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = close_prices[i] - close_prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macd(close_prices: List[float],
                 fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
    """MACD line, signal line, and histogram (EMA based)."""
    if len(close_prices) < slow + signal:
        return None
    def ema(data, period):
        k = 2 / (period + 1)
        result = [data[0]]
        for price in data[1:]:
            result.append(price * k + result[-1] * (1 - k))
        return result
    ema_fast = ema(close_prices, fast)
    ema_slow = ema(close_prices, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]
    signal_line = ema(macd_line, signal)
    histogram = macd_line[-1] - signal_line[-1]
    return {
        "macd": round(macd_line[-1], 6),
        "signal": round(signal_line[-1], 6),
        "histogram": round(histogram, 6)
    }

# ---------- BINANCE ASYNC FETCHING ----------
async def fetch_binance_ticker(symbol: str) -> Optional[dict]:
    """Fetch 24hr ticker using httpx."""
    url = f"{BINANCE_REST_URL}/ticker/24hr"
    params = {"symbol": symbol.upper()}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Binance ticker error for {symbol}: {e}")
        return None

async def fetch_binance_klines(symbol: str, interval: str = "15m", limit: int = 100) -> Optional[List[list]]:
    """Fetch candlestick data."""
    url = f"{BINANCE_REST_URL}/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Binance klines error for {symbol}: {e}")
        return None

async def get_market_data(symbol: str) -> Dict[str, Any]:
    """
    Assemble real‑time market snapshot:
    - Current price, 24h change, high, low, volume
    - RSI (15m)
    - MACD (15m)
    """
    ticker = await fetch_binance_ticker(symbol)
    klines = await fetch_binance_klines(symbol, "15m", 100)
    data = {
        "symbol": symbol.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        "price": None,
        "change_24h": None,
        "high_24h": None,
        "low_24h": None,
        "volume_24h": None,
        "rsi_15m": None,
        "macd_15m": None,
    }
    if ticker:
        data["price"] = float(ticker["lastPrice"])
        data["change_24h"] = float(ticker["priceChangePercent"])
        data["high_24h"] = float(ticker["highPrice"])
        data["low_24h"] = float(ticker["lowPrice"])
        data["volume_24h"] = float(ticker["volume"])
    if klines:
        closes = [float(candle[4]) for candle in klines]
        data["rsi_15m"] = compute_rsi(closes, 14)
        macd = compute_macd(closes)
        if macd:
            data["macd_15m"] = macd
    return data

# ---------- DEEPSEEK ASYNC CALL ----------
async def call_deepseek(model: str, system_prompt: str, user_message: str) -> str:
    """Post to DeepSeek chat completions using httpx."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3 if model == DEEPSEEK_REASONER_MODEL else 0.7,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return "❌ AI service temporarily unavailable. Please try again later."

# ---------- INTENT DETECTION ----------
SIGNAL_KEYWORDS = [
    "signal", "analysis", "analyze", "trade", "setup", "entry", "long",
    "short", "scalp", "target", "tp", "sl", "chart", "rsi", "macd",
    "bollinger", "support", "resistance", "breakout", "trend", "volume",
    "indicator", "technical", "price action", "strategy"
]

def is_signal_request(text: str) -> bool:
    """Detect if the user is asking for a trading signal or deep analysis."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SIGNAL_KEYWORDS)

# Coin → Binance symbol mapping
COIN_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT",
    "SOL": "SOLUSDT", "XRP": "XRPUSDT", "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT", "MATIC": "MATICUSDT", "DOT": "DOTUSDT",
    "LTC": "LTCUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "ATOM": "ATOMUSDT", "UNI": "UNIUSDT", "ETC": "ETCUSDT",
    "FIL": "FILUSDT", "APT": "APTUSDT", "ARB": "ARBUSDT",
    "OP": "OPUSDT", "INJ": "INJUSDT", "SUI": "SUIUSDT",
    "PEPE": "PEPEUSDT", "WIF": "WIFUSDT", "SHIB": "SHIBUSDT"
}

def extract_symbol(text: str, default: str = "BTCUSDT") -> str:
    """Extract a trading pair from the user's message."""
    # Explicit coin name
    for coin, symbol in COIN_MAP.items():
        if re.search(rf"\b{coin}\b", text, re.IGNORECASE):
            return symbol
    # Symbol like ETHUSDT
    match = re.search(r"\b([A-Z]{2,10}USDT)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return default

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message."""
    user = update.effective_user
    text = (
        f"👋 Assalam-o-Alaikum {user.first_name}!\n\n"
        "I'm **OmniTrade AI** – your advanced crypto trading assistant.\n"
        "I can:\n"
        "• Chat casually about crypto\n"
        "• Provide real‑time market data 📊\n"
        "• Generate professional trade signals (Long/Short) with TP/SL\n"
        "• Analyze technical indicators & price action\n\n"
        "Just send me a message! Try:\n"
        "`BTC ka analysis karo`\n"
        "`ETH signal do`\n"
        "`Market kaisa hai?`\n\n"
        "⚠️ *All analysis is educational, not financial advice.*"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information."""
    text = (
        "🆘 **OmniTrade AI Help**\n\n"
        "• **/start** – Welcome message\n"
        "• **/help** – This help menu\n"
        "• **/signal [symbol]** – Force a trade signal (e.g., `/signal ETH`)\n"
        "• **Any message** – I'll respond with general chat or technical analysis.\n\n"
        "For advanced analysis, use keywords like *signal*, *analysis*, *setup*, *long*, *short*.\n"
        "I automatically switch to my powerful reasoning model for those requests.\n\n"
        "💬 I understand English and Roman Urdu."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force a trade signal for a given symbol (default BTCUSDT)."""
    args = context.args
    if args:
        raw = args[0].upper()
        symbol = COIN_MAP.get(raw, raw + "USDT" if not raw.endswith("USDT") else raw)
    else:
        symbol = "BTCUSDT"
    await update.message.reply_text(f"🔍 Generating advanced signal for {symbol}...")
    await generate_and_send_signal(update, context, symbol)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main text router – general chat vs. signal analysis."""
    user_text = update.message.text
    if not user_text:
        return

    if is_signal_request(user_text):
        symbol = extract_symbol(user_text)
        await update.message.reply_text(f"🔍 Analyzing {symbol} – please wait...")
        await generate_and_send_signal(update, context, symbol, user_text)
    else:
        await update.message.chat.send_action("typing")
        reply = await call_deepseek(
            model=DEEPSEEK_CHAT_MODEL,
            system_prompt=SYSTEM_PROMPT_GENERAL,
            user_message=user_text
        )
        await send_long_message(update, reply)

async def generate_and_send_signal(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                   symbol: str, user_query: str = None):
    """Fetch market data, compute indicators, request a signal from DeepSeek reasoner."""
    await update.message.chat.send_action("typing")
    market_data = await get_market_data(symbol)

    if market_data["price"] is None:
        await update.message.reply_text(f"❌ Could not fetch data for {symbol}. Check symbol or try again later.")
        return

    data_block = json.dumps(market_data, indent=2)
    prompt = (
        f"User request: {user_query if user_query else 'Generate a trading signal'}\n\n"
        f"Real‑time market data for {symbol}:\n{data_block}\n\n"
        "Based on this data, your deep technical knowledge, and the required signal format, "
        "provide a complete trade signal with clear reasoning."
    )
    reply = await call_deepseek(
        model=DEEPSEEK_REASONER_MODEL,
        system_prompt=SYSTEM_PROMPT_SIGNAL,
        user_message=prompt
    )
    await send_long_message(update, reply)

# ---------- UTILITY ----------
async def send_long_message(update: Update, text: str):
    """Split long messages to respect Telegram's 4096 char limit."""
    max_len = 4000
    if len(text) <= max_len:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    else:
        for i in range(0, len(text), max_len):
            await update.message.reply_text(text[i:i+max_len], parse_mode=ParseMode.MARKDOWN)

# ---------- MAIN ENTRY POINT ----------
def main():
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("signal", signal_command))

    # Message handler for all non‑command text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting in fully async mode...")
    application.run_polling()

if __name__ == "__main__":
    main()