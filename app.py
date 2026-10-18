
import os
import csv
import time
from datetime import datetime
from threading import Lock

import requests
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv(override=False)

app = Flask(__name__)

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Keep the existing Gemini model setting from your working project.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DATA_FILE = os.path.join(app.root_path, "stock_history.csv")

CACHE_SECONDS = 300
cache = {}
cache_lock = Lock()
file_lock = Lock()

# The dashboard starts with these stocks.
watchlist = ["AAPL", "MSFT", "GOOGL"]

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

session = requests.Session()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def valid_symbol(symbol):
    symbol = str(symbol or "").strip().upper()

    if not symbol or len(symbol) > 20:
        return None

    if not symbol.replace(".", "").replace("-", "").isalnum():
        return None

    return symbol


def unique_symbols(symbols):
    """Return valid symbols once each, preserving their order."""
    result = []

    for item in symbols:
        symbol = valid_symbol(item)

        if symbol and symbol not in result:
            result.append(symbol)

    return result


def save_history(data):
    """Save a quote snapshot to CSV."""
    fields = [
        "time", "symbol", "price",
        "change", "percent", "high", "low"
    ]

    with file_lock:
        file_exists = (
            os.path.exists(DATA_FILE)
            and os.path.getsize(DATA_FILE) > 0
        )

        with open(
            DATA_FILE, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "time": data["updated_at"],
                "symbol": data["symbol"],
                "price": data["price"],
                "change": data["change"],
                "percent": data["change_percent"],
                "high": data["high"],
                "low": data["low"]
            })


def read_history(symbol=None, limit=None):
    """Read saved snapshots, optionally for one symbol."""
    records = []

    if not os.path.exists(DATA_FILE):
        return records

    with file_lock:
        try:
            with open(
                DATA_FILE, "r", newline="", encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)

                for row in reader:
                    if symbol and (
                        row.get("symbol", "").upper()
                        != symbol.upper()
                    ):
                        continue

                    try:
                        records.append({
                            "time": row.get("time", ""),
                            "symbol": row.get("symbol", "").upper(),
                            "price": float(row["price"]),
                            "change": float(row.get("change", 0)),
                            "percent": row.get("percent", ""),
                            "high": float(row.get("high", 0)),
                            "low": float(row.get("low", 0))
                        })
                    except (ValueError, TypeError, KeyError):
                        continue
        except OSError:
            app.logger.exception("Could not read stock history")

    if limit is not None:
        return records[-limit:]

    return records


def fetch_twelve_data(symbol):
    """Fetch one quote and normalize its fields for the dashboard."""
    response = session.get(
        "https://api.twelvedata.com/quote",
        params={
            "symbol": symbol,
            "apikey": TWELVE_DATA_API_KEY
        },
        timeout=15
    )

    try:
        result = response.json()
    except ValueError:
        result = {}

    # Twelve Data may return an API error inside a JSON response.
    if response.status_code == 429:
        raise RuntimeError(
            "Twelve Data rate limit reached. Wait before refreshing."
        )

    if response.status_code >= 400:
        message = result.get("message", "Stock provider request failed.")
        raise RuntimeError(str(message))

    if result.get("status") == "error" or result.get("code"):
        message = result.get("message", "Stock data is unavailable.")
        raise RuntimeError(str(message))

    try:
        price = float(result["close"])
        change = float(result.get("change") or 0)
        percent = result.get("percent_change", "0")
        high = float(result.get("high") or price)
        low = float(result.get("low") or price)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            "Twelve Data returned incomplete quote data. "
            "Check the symbol and your plan's market-data access."
        )

    return {
        "symbol": symbol,
        "price": price,
        "change": change,
        "change_percent": str(percent),
        "high": high,
        "low": low,
        "previous_close": result.get("previous_close"),
        "open": result.get("open"),
        "volume": result.get("volume"),
        "currency": result.get("currency", "USD"),
        "exchange": result.get("exchange", ""),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Twelve Data"
    }


def get_stock(symbol):
    """Return a cached quote when possible to limit API usage."""
    symbol = valid_symbol(symbol)

    if not symbol:
        return None, "Please enter a valid stock ticker."

    with cache_lock:
        cached = cache.get(symbol)

        if cached and time.time() - cached["saved_at"] < CACHE_SECONDS:
            return cached["data"], None

    if not TWELVE_DATA_API_KEY:
        return None, (
            "Twelve Data API key is missing. "
            "Set TWELVE_DATA_API_KEY in your environment."
        )

    try:
        data = fetch_twelve_data(symbol)

        with cache_lock:
            cache[symbol] = {
                "saved_at": time.time(),
                "data": data
            }

        try:
            save_history(data)
        except OSError:
            app.logger.exception("Could not save stock snapshot")

        return data, None

    except requests.RequestException:
        app.logger.exception("Twelve Data connection failed")
        return None, "Could not connect to Twelve Data."

    except RuntimeError as error:
        message = str(error)

        # If the provider is temporarily unavailable, use a cached
        # quote if one exists rather than inventing a new price.
        with cache_lock:
            cached = cache.get(symbol)

        if cached:
            stale = dict(cached["data"])
            stale["stale"] = True
            return stale, None

        return None, message


# --------------------------------------------------
# Website and status routes
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/config")
def api_config():
    """Expose configuration status only; never expose secret keys."""
    return jsonify({
        "stock_provider": "Twelve Data",
        "twelve_data_configured": bool(TWELVE_DATA_API_KEY),
        "alpha_vantage_configured": False,
        "alpha_configured": False,
        "gemini_configured": bool(GEMINI_API_KEY),
        "ai_configured": bool(GEMINI_API_KEY),
        "cache_seconds": CACHE_SECONDS
    })


@app.route("/api/stats")
def api_stats():
    records = read_history()

    return jsonify({
        "watchlist_count": len(unique_symbols(watchlist)),
        "saved_snapshots": len(records),
        "provider": "Twelve Data"
    })


# --------------------------------------------------
# Watchlist routes
# --------------------------------------------------

@app.route("/api/watchlist", methods=["GET", "POST"])
def manage_watchlist():
    global watchlist

    if request.method == "GET":
        watchlist = unique_symbols(watchlist)
        return jsonify({"symbols": watchlist})

    body = request.get_json(silent=True) or {}
    symbol = valid_symbol(body.get("symbol"))

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    if symbol not in watchlist:
        watchlist.append(symbol)

    watchlist = unique_symbols(watchlist)

    return jsonify({
        "symbols": watchlist,
        "added": symbol
    })


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    global watchlist

    symbol = valid_symbol(symbol)

    if symbol:
        watchlist = [
            item for item in unique_symbols(watchlist)
            if item != symbol
        ]

    return jsonify({"symbols": watchlist})


# --------------------------------------------------
# Quote and history routes
# --------------------------------------------------

@app.route("/api/stock/<symbol>")
def stock_quote(symbol):
    data, error = get_stock(symbol)

    if error:
        status = 429 if "rate limit" in error.lower() else 400
        return jsonify({
            "error": error,
            "symbol": str(symbol).upper(),
            "source": "Twelve Data"
        }), status

    return jsonify(data)


@app.route("/api/history/<symbol>")
def stock_history(symbol):
    symbol = valid_symbol(symbol)

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    records = read_history(symbol=symbol, limit=100)

    return jsonify([
        {"time": item["time"], "price": item["price"]}
        for item in records
    ])


# --------------------------------------------------
# Gemini AI insights route
# --------------------------------------------------

@app.route("/api/ai-insights", methods=["POST"])
def ai_insights():
    if gemini_client is None:
        return jsonify({
            "error": "Gemini API key is missing. Configure GEMINI_API_KEY."
        }), 503

    body = request.get_json(silent=True) or {}
    symbol = valid_symbol(body.get("symbol"))

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    records = read_history(symbol=symbol, limit=20)

    with cache_lock:
        cached = cache.get(symbol)
        latest_quote = cached["data"] if cached else None

    if not records and not latest_quote:
        return jsonify({
            "error": (
                f"No saved price data is available for {symbol} yet. "
                "Load a quote successfully before generating AI insights."
            )
        }), 404

    prompt = f"""
You are the educational AI assistant inside StockWatch,
a student stock-market monitoring project.

Analyze only the supplied information for ticker {symbol}.

Saved price snapshots, oldest to newest:
{records}

Latest cached quote, if available:
{latest_quote}

Write a clear report with these sections:
1. Summary of the supplied data.
2. Whether the recorded prices rise, fall, or look mixed.
3. Important limitations and missing information.
4. Two questions a learner could investigate next.

Rules:
- Do not invent prices, financial facts, or company news.
- Explain if the dataset is too small to identify a trend.
- Snapshots are not necessarily continuous historical data.
- Do not claim to have fetched live data during this AI request.
- Do not provide personalized investment advice or promise returns.
- State that the report is educational, not financial advice.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        insight_text = response.text

        if not insight_text:
            return jsonify({
                "error": "Gemini returned an empty response. Try again."
            }), 502

        return jsonify({
            "symbol": symbol,
            "insights": insight_text,
            "data_points": len(records),
            "source": "StockWatch saved snapshots and cached quote"
        })

    except Exception as error:
        app.logger.error(
            "Gemini request failed: %s: %s",
            type(error).__name__,
            str(error)
        )

        return jsonify({
            "error": (
                "Gemini could not generate insights. Check the Gemini "
                "key, model access, internet connection, and API quota."
            )
        }), 502


# --------------------------------------------------
# Run locally or on Render
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
