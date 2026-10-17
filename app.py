
import os
import csv
import time
from datetime import datetime
from threading import Lock

import requests
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

DATA_FILE = os.path.join(app.root_path, "stock_history.csv")

# Longer cache reduces requests to Alpha Vantage's free API.
CACHE_SECONDS = 300
cache = {}
cache_lock = Lock()
file_lock = Lock()

watchlist = ["AAPL", "MSFT", "GOOGL"]

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = "gemini-3.8-flash"

ALPHA_URL = "https://www.alphavantage.co/query"


def valid_symbol(symbol):
    symbol = str(symbol).strip().upper()
    return (
        bool(symbol)
        and len(symbol) <= 15
        and symbol.replace(".", "").replace("-", "").isalnum()
    )


def save_history(symbol, price, change, percent, high, low):
    with file_lock:
        exists = os.path.exists(DATA_FILE)

        with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "time", "symbol", "price",
                    "change", "percent", "high", "low"
                ]
            )

            if not exists or os.path.getsize(DATA_FILE) == 0:
                writer.writeheader()

            writer.writerow({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "price": price,
                "change": change,
                "percent": percent,
                "high": high,
                "low": low
            })


def get_stock(symbol):
    symbol = str(symbol).strip().upper()

    if not valid_symbol(symbol):
        return None, "Please enter a valid stock ticker."

    # Return the last successful quote while it is fresh.
    with cache_lock:
        cached = cache.get(symbol)
        if cached and time.time() - cached["saved_at"] < CACHE_SECONDS:
            return cached["data"], None

    if not ALPHA_VANTAGE_API_KEY:
        return None, "Alpha Vantage API key is missing."

    try:
        response = requests.get(
            ALPHA_URL,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()

        if "Note" in result or "Information" in result:
            with cache_lock:
                cached = cache.get(symbol)
            if cached:
                return cached["data"], (
                    "Showing the last saved quote; Alpha Vantage "
                    "has temporarily limited requests."
                )
            return None, (
                "Alpha Vantage request limit reached. Try again later."
            )

        quote = result.get("Global Quote", {})
        if not quote:
            return None, (
                "No quote returned. Check the ticker, API access, "
                "or provider response."
            )

        price = float(quote["05. price"])
        change = float(quote["09. change"])
        percent = quote["10. change percent"]
        high = float(quote["03. high"])
        low = float(quote["04. low"])

        data = {
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": percent,
            "high": high,
            "low": low,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with cache_lock:
            cache[symbol] = {"saved_at": time.time(), "data": data}

        save_history(symbol, price, change, percent, high, low)
        return data, None

    except requests.RequestException:
        return None, "Could not connect to the stock data provider."
    except (KeyError, TypeError, ValueError):
        return None, "The provider returned incomplete quote data."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/watchlist", methods=["GET", "POST"])
def manage_watchlist():
    if request.method == "GET":
        return jsonify({"symbols": watchlist})

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip().upper()

    if not valid_symbol(symbol):
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    if symbol not in watchlist:
        watchlist.append(symbol)

    return jsonify({"symbols": watchlist, "added": symbol})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    symbol = symbol.strip().upper()

    if symbol in watchlist:
        watchlist.remove(symbol)

    return jsonify({"symbols": watchlist})


@app.route("/api/stock/<symbol>")
def stock_quote(symbol):
    data, error = get_stock(symbol)

    if data:
        result = dict(data)
        if error:
            result["notice"] = error
        return jsonify(result)

    status = 429 if error and "limit" in error.lower() else 400
    return jsonify({"error": error, "symbol": symbol.upper()}), status


@app.route("/api/history/<symbol>")
def stock_history(symbol):
    symbol = symbol.strip().upper()
    records = []

    if os.path.exists(DATA_FILE):
        with file_lock:
            with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if row.get("symbol", "").upper() == symbol:
                        try:
                            records.append({
                                "time": row["time"],
                                "price": float(row["price"])
                            })
                        except (ValueError, KeyError):
                            continue

    return jsonify(records[-100:])


@app.route("/api/ai-insights", methods=["POST"])
def ai_insights():
    if gemini_client is None:
        return jsonify({
            "error": "Gemini API key is missing. Configure GEMINI_API_KEY."
        }), 503

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip().upper()

    if not valid_symbol(symbol):
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    records = []

    if os.path.exists(DATA_FILE):
        with file_lock:
            with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if row.get("symbol", "").upper() == symbol:
                        try:
                            records.append({
                                "time": row.get("time", ""),
                                "price": float(row["price"]),
                                "change": float(row.get("change", 0)),
                                "percent": row.get("percent", ""),
                                "high": float(row.get("high", 0)),
                                "low": float(row.get("low", 0))
                            })
                        except (ValueError, TypeError):
                            continue

    with cache_lock:
        cached = cache.get(symbol)
        latest_quote = cached["data"] if cached else None

    if not records and not latest_quote:
        return jsonify({
            "error": (
                f"No saved price data is available for {symbol} yet. "
                "Try again after a quote has been successfully retrieved."
            )
        }), 404

    prompt = f"""
You are StockWatch's educational stock-data assistant.

Analyze only the supplied data for ticker {symbol}.

Saved snapshots, oldest to newest:
{records[-20:]}

Latest cached quote, if available:
{latest_quote}

Write a concise report covering:
1. What the supplied data shows.
2. Whether saved prices rise, fall, or look mixed.
3. Missing information and limitations.
4. Two useful things to investigate further.

Never invent prices, news, or financial facts. Say when there are
too few observations to identify a trend. These may be sparse snapshots,
not continuous historical prices. Do not claim to have fetched live data.
Do not provide personalized investment advice or promise returns.
This is educational information, not financial advice.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        insight_text = response.text
        if not insight_text:
            return jsonify({"error": "Gemini returned an empty response."}), 502

        return jsonify({
            "symbol": symbol,
            "insights": insight_text,
            "data_points": len(records),
            "source": "StockWatch saved snapshots and cached quote"
        })

    except Exception as error:
        print("Gemini request failed:", type(error).__name__, str(error))
        return jsonify({
            "error": (
                "Gemini could not generate insights. Check the model name, "
                "API key, access, internet connection, and quota."
            )
        }), 502


if __name__ == "__main__":
    # Render supplies PORT. Disable debug mode in production.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
