
import matplotlib.pyplot as plt
import os
import csv
import json
import time
from datetime import datetime
from threading import Lock
from io import BytesIO

import requests
from flask import Flask, render_template, jsonify, request, send_file
from dotenv import load_dotenv
from google import genai
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Optional plotting libraries for the Matplotlib history chart.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

load_dotenv(override=False)

app = Flask(__name__)

# API configuration
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Data files
DATA_FILE = os.path.join(app.root_path, "stock_history.csv")
WATCHLIST_FILE = os.path.join(app.root_path, "watchlist.json")

CACHE_SECONDS = 300

cache = {}
cache_lock = Lock()
file_lock = Lock()
watchlist_lock = Lock()

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL"]

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY else None
)

session = requests.Session()


# -------------------- SYMBOL HELPERS --------------------

def valid_symbol(value):
    symbol = str(value or "").strip().upper()

    if not symbol or len(symbol) > 20:
        return None

    if not symbol.replace(".", "").replace("-", "").isalnum():
        return None

    return symbol


def unique_symbols(values):
    result = []

    for value in values:
        symbol = valid_symbol(value)

        if symbol and symbol not in result:
            result.append(symbol)

    return result


# -------------------- WATCHLIST STORAGE --------------------

def load_watchlist():
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                return unique_symbols(data)

    except (OSError, ValueError):
        app.logger.exception("Could not load watchlist")

    return DEFAULT_WATCHLIST.copy()


def save_watchlist(symbols):
    temporary_file = WATCHLIST_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(unique_symbols(symbols), file, indent=2)

    os.replace(temporary_file, WATCHLIST_FILE)


watchlist = load_watchlist()


# -------------------- PRICE HISTORY STORAGE --------------------

def save_history(data):
    fields = [
        "time",
        "symbol",
        "price",
        "change",
        "percent",
        "high",
        "low",
    ]

    with file_lock:
        exists = (
            os.path.exists(DATA_FILE)
            and os.path.getsize(DATA_FILE) > 0
        )

        with open(
            DATA_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)

            if not exists:
                writer.writeheader()

            writer.writerow({
                "time": data["updated_at"],
                "symbol": data["symbol"],
                "price": data["price"],
                "change": data["change"],
                "percent": data["change_percent"],
                "high": data["high"],
                "low": data["low"],
            })


def read_history(symbol=None, limit=None):
    records = []

    if not os.path.exists(DATA_FILE):
        return records

    symbol = valid_symbol(symbol) if symbol else None

    with file_lock:
        try:
            with open(
                DATA_FILE,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                for row in csv.DictReader(file):
                    row_symbol = row.get("symbol", "").upper()

                    if symbol and row_symbol != symbol:
                        continue

                    try:
                        records.append({
                            "time": row.get("time", ""),
                            "symbol": row_symbol,
                            "price": float(row["price"]),
                            "change": float(row.get("change", 0)),
                            "percent": row.get("percent", ""),
                            "high": float(row.get("high", 0)),
                            "low": float(row.get("low", 0)),
                        })

                    except (ValueError, TypeError, KeyError):
                        continue

        except OSError:
            app.logger.exception("Could not read stock history")

    return records[-limit:] if limit else records


# -------------------- TWELVE DATA API --------------------

def fetch_twelve_data(symbol):
    response = session.get(
        "https://api.twelvedata.com/quote",
        params={
            "symbol": symbol,
            "apikey": TWELVE_DATA_API_KEY,
        },
        timeout=(5, 10),
    )

    try:
        result = response.json()
    except ValueError:
        result = {}

    if response.status_code == 429:
        raise RuntimeError(
            "Twelve Data rate limit reached. "
            "Please wait before refreshing."
        )

    if response.status_code >= 400:
        raise RuntimeError(
            str(result.get("message", "Twelve Data request failed."))
        )

    if result.get("status") == "error" or result.get("code"):
        raise RuntimeError(
            str(result.get("message", "Stock data is unavailable."))
        )

    try:
        price = float(result["close"])
        change = float(result.get("change") or 0)
        percent = result.get("percent_change", "0")
        high = float(result.get("high") or price)
        low = float(result.get("low") or price)

    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            "Twelve Data returned incomplete quote data. "
            "Check the ticker and market-data access."
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
        "source": "Twelve Data",
    }


def get_stock(symbol, force=False):
    symbol = valid_symbol(symbol)

    if not symbol:
        return None, "Please enter a valid stock ticker."

    with cache_lock:
        cached = cache.get(symbol)

        if (
            not force
            and cached
            and time.time() - cached["saved_at"] < CACHE_SECONDS
        ):
            return cached["data"], None

    if not TWELVE_DATA_API_KEY:
        return None, (
            "Twelve Data API key is missing. "
            "Configure TWELVE_DATA_API_KEY."
        )

    try:
        data = fetch_twelve_data(symbol)

        with cache_lock:
            cache[symbol] = {
                "saved_at": time.time(),
                "data": data,
            }

        try:
            save_history(data)
        except OSError:
            app.logger.exception("Could not save stock snapshot")

        return data, None

    except requests.Timeout:
        return None, "Twelve Data timed out. Please try again shortly."

    except requests.RequestException:
        app.logger.exception("Twelve Data connection failed")
        return None, "Could not connect to Twelve Data."

    except RuntimeError as error:
        return None, str(error)


# -------------------- MAIN DASHBOARD --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# -------------------- CONFIGURATION --------------------

@app.route("/api/config")
def api_config():
    return jsonify({
        "stock_provider": "Twelve Data",
        "twelve_data_configured": bool(TWELVE_DATA_API_KEY),
        "twelvedata_configured": bool(TWELVE_DATA_API_KEY),
        "alpha_vantage_configured": False,
        "gemini_configured": bool(GEMINI_API_KEY),
        "ai_configured": bool(GEMINI_API_KEY),
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
        "cache_seconds": CACHE_SECONDS,
    })


# -------------------- DASHBOARD SUMMARY --------------------

@app.route("/api/stats")
@app.route("/api/summary")
def api_stats():
    with watchlist_lock:
        count = len(watchlist)

    return jsonify({
        "watchlist_count": count,
        "saved_snapshots": len(read_history()),
        "provider": "Twelve Data",
    })


# -------------------- WATCHLIST ROUTES --------------------

@app.route("/api/watchlist", methods=["GET", "POST"])
def manage_watchlist():
    global watchlist

    if request.method == "GET":
        with watchlist_lock:
            watchlist = unique_symbols(watchlist)
            return jsonify({"symbols": watchlist.copy()})

    body = request.get_json(silent=True) or {}
    symbol = valid_symbol(body.get("symbol"))

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    with watchlist_lock:
        watchlist = unique_symbols(watchlist)

        if symbol in watchlist:
            return jsonify({
                "error": f"{symbol} is already in your watchlist.",
                "symbols": watchlist.copy(),
            }), 409

        watchlist.append(symbol)

        try:
            save_watchlist(watchlist)

        except OSError:
            watchlist.remove(symbol)
            app.logger.exception("Could not save watchlist")

            return jsonify({
                "error": "Could not save the watchlist on this server."
            }), 500

        return jsonify({
            "symbols": watchlist.copy(),
            "added": symbol,
        }), 201


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    global watchlist

    symbol = valid_symbol(symbol)

    if not symbol:
        return jsonify({"error": "Invalid stock ticker."}), 400

    with watchlist_lock:
        watchlist = [
            item for item in unique_symbols(watchlist)
            if item != symbol
        ]

        try:
            save_watchlist(watchlist)

        except OSError:
            app.logger.exception("Could not save watchlist")

            return jsonify({
                "error": "Could not save the updated watchlist."
            }), 500

        return jsonify({"symbols": watchlist.copy()})


# -------------------- STOCK QUOTE ROUTE --------------------

@app.route("/api/stock/<symbol>")
def stock_quote(symbol):
    force = request.args.get(
        "refresh", ""
    ).lower() in ("1", "true", "yes")

    data, error = get_stock(symbol, force=force)

    if error:
        status = 429 if "rate limit" in error.lower() else 400

        return jsonify({
            "error": error,
            "symbol": str(symbol).upper(),
            "source": "Twelve Data",
        }), status

    return jsonify(data)


# -------------------- PRICE HISTORY API --------------------

@app.route("/api/history/<symbol>", methods=["GET"])
def stock_history(symbol):
    symbol = valid_symbol(symbol)

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    records = read_history(symbol=symbol, limit=100)

    return jsonify([
        {
            "time": item["time"],
            "price": item["price"],
        }
        for item in records
    ])


# -------------------- MATPLOTLIB HISTORY CHART --------------------

@app.route("/api/history-chart/<symbol>", methods=["GET"])
def history_chart(symbol):
    symbol = valid_symbol(symbol)

    if not symbol:
        return jsonify({"error": "Enter a valid stock ticker."}), 400

    if not MATPLOTLIB_AVAILABLE:
        return jsonify({
            "error": (
                "Matplotlib is not installed. "
                "Run: pip install matplotlib"
            )
        }), 503

    records = read_history(symbol=symbol, limit=100)

    if not records:
        return jsonify({
            "error": (
                f"No saved snapshots for {symbol}. "
                "Load the stock quote first."
            )
        }), 404

    times = [item["time"] for item in records]
    prices = [item["price"] for item in records]

    figure = None

    try:
        figure, axis = plt.subplots(figsize=(12, 5))

        axis.plot(
            times,
            prices,
            marker="o",
            linewidth=2,
            label=f"{symbol} price",
        )

        axis.set_title(f"{symbol} Saved Price History")
        axis.set_xlabel("Recorded time")
        axis.set_ylabel("Price (USD)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()

        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=140)
        image_buffer.seek(0)

        return send_file(
            image_buffer,
            mimetype="image/png",
            download_name=f"{symbol}_history.png",
        )

    except Exception:
        app.logger.exception("Could not generate history chart")

        return jsonify({
            "error": "Could not generate the history chart."
        }), 500

    finally:
        if figure is not None:
            plt.close(figure)


# -------------------- GEMINI AI INSIGHTS --------------------

@app.route("/api/ai-insights", methods=["POST"])
def ai_insights():
    if gemini_client is None:
        return jsonify({
            "error": (
                "Gemini API key is missing. "
                "Configure GEMINI_API_KEY."
            )
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
You are the educational AI assistant inside StockWatch.

Analyze only the supplied information for ticker {symbol}.

Saved price snapshots:
{records}

Latest cached quote, if available:
{latest_quote}

Write a beginner-friendly report covering:
1. Summary of the supplied data.
2. Whether recorded prices rise, fall, or look mixed.
3. Important limitations and missing information.
4. Two questions a learner could investigate next.

Do not invent prices, company news, or financial facts.
Explain when the data is too small to identify a trend.
Snapshots may be sparse and are not continuous historical data.
Do not claim to have fetched live data during this AI request.
Do not give personalized investment advice or promise returns.
State that the report is educational, not financial advice.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        insight_text = response.text

        if not insight_text:
            return jsonify({
                "error": (
                    "Gemini returned an empty response. Please try again."
                )
            }), 502

        return jsonify({
            "symbol": symbol,
            "insights": insight_text,
            "data_points": len(records),
            "source": "StockWatch saved snapshots and cached quote",
        })

    except Exception as error:
        app.logger.exception("Gemini API request failed")

        return jsonify({
            "error": (
                "Gemini API request failed. Check the Render logs "
                "for the detailed error message."
            ),
            "details": str(error)[:1000],
            "model": GEMINI_MODEL
        }), 502


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
