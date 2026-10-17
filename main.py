
import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

import requests
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

API_URL = "https://www.alphavantage.co/query"
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
CSV_FILE = "stock_history.csv"

console = Console()

FIELDS = [
    "time", "symbol", "price", "change",
    "change_percent", "high", "low"
]


def get_quote(symbol):
    """Fetch and validate a stock quote."""

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": API_KEY
    }

    try:
        response = requests.get(
            API_URL, params=params, timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "Note" in data or "Information" in data:
            message = data.get("Note", data.get("Information"))
            console.print(
                f"API limit/info for {symbol}: {message}",
                style="yellow"
            )
            return None

        if "Error Message" in data:
            console.print(
                f"API error for {symbol}: "
                f"{data['Error Message']}",
                style="red"
            )
            return None

        quote = data.get("Global Quote", {})

        if not quote or "05. price" not in quote:
            console.print(
                f"No quote available for {symbol}.",
                style="yellow"
            )
            return None

        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": quote.get("01. symbol", symbol),
            "price": float(quote["05. price"]),
            "change": float(quote.get("09. change", 0)),
            "change_percent": float(
                quote.get("10. change percent", "0%")
                .replace("%", "")
            ),
            "high": quote.get("03. high", "N/A"),
            "low": quote.get("04. low", "N/A")
        }

    except requests.exceptions.RequestException as error:
        console.print(f"Network error: {error}", style="red")
    except (ValueError, TypeError) as error:
        console.print(f"Data error: {error}", style="red")

    return None


def save_quote(quote):
    """Append a quote to the existing CSV history."""

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()

        writer.writerow(quote)


def display_quotes(quotes, alert_threshold):
    """Display quotes with colors and trigger alerts."""

    table = Table(
        title="REAL-TIME STOCK MARKET WATCHDOG",
        box=box.ROUNDED
    )

    for heading in [
        "Symbol", "Price", "Change", "Change %", "Day High", "Day Low"
    ]:
        table.add_column(heading, justify="right")

    for quote in quotes:
        change_pct = quote["change_percent"]

        if change_pct > 0:
            style = "green"
        elif change_pct < 0:
            style = "red"
        else:
            style = "white"

        table.add_row(
            quote["symbol"],
            f"{quote['price']:.2f}",
            f"{quote['change']:+.2f}",
            f"{change_pct:+.4f}%",
            str(quote["high"]),
            str(quote["low"]),
            style=style
        )

        if abs(change_pct) >= alert_threshold:
            console.print(
                f"ALERT: {quote['symbol']} daily change is "
                f"{change_pct:+.4f}% "
                f"(threshold: {alert_threshold:.2f}%)",
                style="bold yellow"
            )

    console.print(table)


def plot_history():
    """Plot saved prices from the CSV file."""

    if not os.path.exists(CSV_FILE):
        console.print("No CSV history found yet.", style="yellow")
        return

    history = {}

    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                symbol = row.get("symbol", "").strip()
                if not symbol:
                    continue

                try:
                    price = float(row["price"])
                    stamp = row["time"]
                except (ValueError, KeyError):
                    continue

                history.setdefault(symbol, []).append((stamp, price))

    except OSError as error:
        console.print(f"Could not read CSV: {error}", style="red")
        return

    if not history:
        console.print("No valid price history to graph.", style="yellow")
        return

    for symbol, records in history.items():
        records.sort(key=lambda item: item[0])

        times = [
            datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S")
            for item in records
        ]
        prices = [item[1] for item in records]

        plt.plot(
            times,
            prices,
            marker="o",
            linewidth=2,
            label=symbol
        )

    plt.title("Stock Price History")
    plt.xlabel("Recorded time")
    plt.ylabel("Price (API currency)")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    if not API_KEY:
        console.print(
            "API key missing. Set ALPHA_VANTAGE_API_KEY first.",
            style="red"
        )
        return

    symbols_input = input(
        "Enter stock symbols separated by commas (e.g. AAPL,MSFT,GOOGL): "
    ).strip().upper()

    symbols = list(dict.fromkeys(
        item.strip()
        for item in symbols_input.split(",")
        if item.strip()
    ))

    if not symbols:
        console.print("Enter at least one stock symbol.", style="red")
        return

    try:
        interval = int(input(
            "Seconds between monitoring cycles (minimum 60): "
        ))
        alert_threshold = float(input(
            "Alert threshold for daily change, in percent (e.g. 2): "
        ))

        if interval < 60 or alert_threshold < 0:
            raise ValueError

    except ValueError:
        console.print(
            "Enter an interval of at least 60 and a non-negative threshold.",
            style="red"
        )
        return

    save_choice = input("Save quotes to CSV? (y/n): ").strip().lower()
    graph_choice = input(
        "Show price graph when stopped? (y/n): ").strip().lower()

    console.print(
        "\nMonitoring started. Press Ctrl+C to stop.",
        style="bold cyan"
    )

    try:
        while True:
            quotes = []

            for symbol in symbols:
                quote = get_quote(symbol)

                if quote:
                    quotes.append(quote)
                    if save_choice == "y":
                        save_quote(quote)

            if quotes:
                display_quotes(quotes, alert_threshold)
            else:
                console.print(
                    "No quotes retrieved this cycle.",
                    style="yellow"
                )

            console.print(
                f"Next check in {interval} seconds...",
                style="cyan"
            )
            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\nMonitoring stopped.", style="bold yellow")

    if graph_choice == "y":
        plot_history()


if __name__ == "__main__":
    main()
