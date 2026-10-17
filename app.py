
import os
import csv
import time
from datetime import datetime
from threading import Lock

import requests
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai

# Load API keys from .env
load_dotenv()

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

DATA_FILE = os.path.join(app.root_path, "stock_history.csv")

CACHE_SECONDS = 60
cache = {}
cache_lock = Lock()

watchlist = ["AAPL", "MSFT", "GOOGL"]

# Initialize Gemini only when a key is configured
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = "gemini-3.8-flash"


def save_history(symbol, price, change, percent, high, low):
    file_exists = os.path.exists(DATA_FILE)

    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time", "symbol", "price",
                "change", "percent", "high", "low"
            ]
        )

        if not file_exists or os.path.getsize(DATA_FILE) == 0:
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
    symbol = symbol.strip().upper()

    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        return None, "Please enter a valid stock ticker."

    with cache_lock:
        cached = cache.get(symbol)
        if cached and time.time() - cached["saved_at"] < CACHE_SECONDS:
            return cached["data"], None

    if not ALPHA_VANTAGE_API_KEY:
        return None, "Alpha Vantage API key is missing."

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
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
            return None, (
                "Alpha Vantage request limit reached. "
                "Please try again later. Saved history is still available."
            )

        quote = result.get("Global Quote", {})
        if not quote:
            return None, "No quote returned. Check the ticker or API availability."

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
            cache[symbol] = {
                "saved_at": time.time(),
                "data": data
            }

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

    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
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

    if error:
        status = 429 if "limit" in error.lower() else 400
        return jsonify({"error": error, "symbol": symbol.upper()}), status

    return jsonify(data)


@app.route("/api/history/<symbol>")
def stock_history(symbol):
    symbol = symbol.strip().upper()
    records = []

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
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
            "error": "Gemini API key is missing. Add GEMINI_API_KEY to .env."
        }), 503

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip().upper()

    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    # Read saved records; do not invent missing market prices.
    records = []

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
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

    # Use current in-memory cached quote if available.
    latest_quote = None
    with cache_lock:
        cached = cache.get(symbol)
        if cached:
            latest_quote = cached["data"]

    if not records and not latest_quote:
        return jsonify({
            "error": (
                f"No saved price data is available for {symbol} yet. "
                "AI insights need a saved quote first."
            )
        }), 404

    prompt = f"""
You are the AI insights assistant inside a student stock-monitoring
project called StockWatch.

Analyze the supplied data for ticker {symbol}.

Saved price records (oldest to newest):
{records[-20:]}

Latest cached quote, if available:
{latest_quote}

Write a concise, beginner-friendly report with:
1. What the supplied data shows.
2. Whether the saved prices appear to rise, fall, or remain mixed.
3. Important limitations or missing information.
4. Two things a person could investigate further.

Rules:
- Do not invent prices, company news, or financial facts.
- State clearly when the dataset is too small to show a trend.
- These records may be sparse snapshots, not continuous historical prices.
- Do not claim that you have fetched live market data.
- Do not give personalized investment advice or promise returns.
- This is educational information, not financial advice.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        insight_text = response.text

        if not insight_text:
            return jsonify({
                "error": "Gemini returned an empty response. Please try again."
            }), 502

        return jsonify({
            "symbol": symbol,
            "insights": insight_text,
            "data_points": len(records),
            "source": "StockWatch saved snapshots and available cached quote"
        })

    except Exception as error:
        # Avoid returning API keys or full internal error details to browsers.
        print("Gemini request failed:", type(error).__name__, str(error))
        return jsonify({
            "error": (
                "Gemini could not generate insights. Check your API key, "
                "model access, internet connection, and API quota."
            )
        }), 502


if __name__ == "__main__":
    app.run(debug=True)
