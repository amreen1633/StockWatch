
import os
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DATA_FILE = Path(app.root_path) / "stock_history.csv"
CACHE_SECONDS = 60
REQUEST_GAP_SECONDS = 12

cache = {}
cache_lock = threading.Lock()
api_lock = threading.Lock()
last_api_request = 0.0

watchlist = ["AAPL", "MSFT", "GOOGL"]
watchlist_lock = threading.Lock()

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def valid_symbol(symbol):
    symbol = str(symbol or "").strip().upper()
    return bool(symbol) and len(symbol) <= 20 and all(
        char.isalnum() or char in ".-" for char in symbol
    )


def read_history(symbol=None, limit=500):
    records = []
    if not DATA_FILE.exists():
        return records

    try:
        with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if symbol and row.get("symbol", "").upper() != symbol.upper():
                    continue
                try:
                    records.append({
                        "time": row["time"],
                        "symbol": row["symbol"].upper(),
                        "price": float(row["price"]),
                        "change": float(row.get("change", 0)),
                        "percent": row.get("percent", ""),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        app.logger.exception("Could not read CSV history")

    return records[-limit:]


def save_history(symbol, data):
    fields = ["time", "symbol", "price", "change", "percent", "high", "low"]
    exists = DATA_FILE.exists() and DATA_FILE.stat().st_size > 0

    try:
        with DATA_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "time": data["updated_at"],
                "symbol": symbol,
                "price": data["price"],
                "change": data["change"],
                "percent": data["change_percent"],
                "high": data["high"],
                "low": data["low"],
            })
    except OSError:
        app.logger.exception("Could not save CSV history")


def fetch_quote_from_api(symbol):
    global last_api_request

    if not ALPHA_VANTAGE_API_KEY:
        return None, "Alpha Vantage API key is missing."

    # Serialize API calls and space them apart to reduce rate-limit errors.
    with api_lock:
        wait = REQUEST_GAP_SECONDS - (time.monotonic() - last_api_request)
        if wait > 0:
            time.sleep(wait)

        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
                timeout=20,
            )
            last_api_request = time.monotonic()
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError):
            return None, "Could not retrieve data from Alpha Vantage."

    if "Note" in result or "Information" in result:
        return None, "Alpha Vantage rate limit reached. Wait before retrying."

    quote = result.get("Global Quote", {})
    if not quote:
        return None, "No quote returned. Check the ticker or API availability."

    try:
        data = {
            "symbol": symbol,
            "price": float(quote["05. price"]),
            "change": float(quote["09. change"]),
            "change_percent": quote["10. change percent"],
            "high": float(quote["03. high"]),
            "low": float(quote["04. low"]),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except (KeyError, TypeError, ValueError):
        return None, "The provider returned incomplete quote data."

    with cache_lock:
        cache[symbol] = {"saved_at": time.time(), "data": data}

    save_history(symbol, data)
    return data, None


def get_stock(symbol, force=False):
    symbol = str(symbol or "").strip().upper()

    if not valid_symbol(symbol):
        return None, "Please enter a valid stock ticker."

    with cache_lock:
        item = cache.get(symbol)
        if item and not force and time.time() - item["saved_at"] < CACHE_SECONDS:
            return item["data"], None

    return fetch_quote_from_api(symbol)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "StockWatch"})


@app.route("/api/config")
def public_config():
    # Never send API keys to the browser.
    return jsonify({
        "alpha_vantage_configured": bool(ALPHA_VANTAGE_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "cache_seconds": CACHE_SECONDS,
        "request_gap_seconds": REQUEST_GAP_SECONDS,
    })


@app.route("/api/watchlist", methods=["GET", "POST"])
def manage_watchlist():
    if request.method == "GET":
        with watchlist_lock:
            return jsonify({"symbols": list(watchlist)})

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip().upper()

    if not valid_symbol(symbol):
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    with watchlist_lock:
        if symbol not in watchlist:
            watchlist.append(symbol)

    return jsonify({"symbols": list(watchlist), "added": symbol})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    symbol = symbol.strip().upper()

    with watchlist_lock:
        if symbol in watchlist:
            watchlist.remove(symbol)
        remaining = list(watchlist)

    return jsonify({"symbols": remaining})


@app.route("/api/stock/<symbol>")
def stock_quote(symbol):
    force = request.args.get("refresh", "").lower() in ("1", "true", "yes")
    data, error = get_stock(symbol, force=force)

    if error:
        status = 429 if "limit" in error.lower() else 503
        return jsonify({"error": error, "symbol": symbol.upper()}), status

    return jsonify(data)


@app.route("/api/history/<symbol>")
def stock_history(symbol):
    symbol = symbol.strip().upper()
    if not valid_symbol(symbol):
        return jsonify({"error": "Invalid ticker."}), 400

    return jsonify(read_history(symbol, limit=500))


@app.route("/api/summary")
def summary():
    records = read_history(limit=10000)
    symbols = set(row["symbol"] for row in records)
    return jsonify({
        "watchlist_count": len(watchlist),
        "saved_snapshots": len(records),
        "symbols_with_history": len(symbols),
        "last_saved": records[-1]["time"] if records else None,
    })


@app.route("/api/ai-insights", methods=["POST"])
def ai_insights():
    if gemini_client is None:
        return jsonify({
            "error": "Gemini is not configured. Add GEMINI_API_KEY to Render environment variables."
        }), 503

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip().upper()

    if not valid_symbol(symbol):
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    records = read_history(symbol, limit=20)
    latest = None

    with cache_lock:
        cached = cache.get(symbol)
        if cached:
            latest = cached["data"]

    if not records and not latest:
        return jsonify({
            "error": "No saved quote data for this stock yet. Try loading a quote first."
        }), 404

    prompt = f"""
You are the educational analysis assistant for StockWatch.
Ticker: {symbol}
Saved historical snapshots, oldest first: {records}
Latest cached quote if available: {latest}

Write a concise report with:
1. What the supplied data actually shows.
2. Whether the snapshots suggest rising, falling, or mixed prices.
3. Limitations and missing data.
4. Two questions a learner could investigate next.

Never invent prices, news, or company facts. If there are too few
snapshots, say so. These are saved snapshots, not necessarily continuous
historical market data. Do not promise returns or give personalized
investment advice. State that this is educational information, not advice.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = response.text or ""
        if not text.strip():
            return jsonify({"error": "Gemini returned an empty report."}), 502

        return jsonify({
            "symbol": symbol,
            "insights": text,
            "data_points": len(records),
            "source": "StockWatch saved snapshots and available cached quote",
        })
    except Exception as error:
        app.logger.error("Gemini request failed: %s", type(error).__name__)
        return jsonify({
            "error": "Gemini could not generate insights. Check the Render key, model access, and quota."
        }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
