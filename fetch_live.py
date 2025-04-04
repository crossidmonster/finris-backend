import yfinance as yf
import sqlite3
import logging
import time
from datetime import datetime
import pytz

# Setup logging
logging.basicConfig(
    filename='fetch_live.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# PST timezone
PST = pytz.timezone('America/Los_Angeles')

# Database paths
MASTER_DB_PATH = "C:\\FINRIS\\master_ticker.db"
CURRENT_DAY_DB_PATH = "C:\\FINRIS\\current_day.db"

def get_tickers():
    """Fetch unique tickers from master_ticker.db."""
    try:
        conn = sqlite3.connect(MASTER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ticker_data")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        logging.info(f"Fetched {len(tickers)} tickers from master_ticker.db")
        return tickers
    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        return []

def init_db():
    """Initialize current_day.db with ticker_data table if it doesn't exist."""
    try:
        conn = sqlite3.connect(CURRENT_DAY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_data (
                ticker TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, datetime)
            )
        """)
        conn.commit()
        conn.close()
        logging.info("Initialized current_day.db")
    except Exception as e:
        logging.error(f"Error initializing current_day.db: {e}")

def fetch_2m_data(tickers):
    """Fetch the latest 2m interval data for today for all tickers."""
    data = {}
    for ticker in tickers:
        for attempt in range(3):  # Retry up to 3 times
            try:
                stock = yf.Ticker(ticker)
                # Fetch today's data with pre/post market included
                df = stock.history(interval="2m", period="1d", prepost=True)
                if not df.empty:
                    # Take only the last row (latest 2m candle)
                    data[ticker] = df.tail(1)
                    logging.info(f"Fetched latest row for {ticker}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < 2:
                    time.sleep(5)  # Wait before retrying
                else:
                    logging.error(f"Max retries reached for {ticker}")
    return data

def save_to_db(data):
    """Save fetched data to current_day.db."""
    try:
        conn = sqlite3.connect(CURRENT_DAY_DB_PATH)
        cursor = conn.cursor()
        for ticker, df in data.items():
            for index, row in df.iterrows():
                dt = index.astimezone(PST).strftime('%Y-%m-%d %H:%M:%S-07:00')
                cursor.execute("""
                    INSERT OR IGNORE INTO ticker_data (ticker, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ticker, dt, row['Open'], row['High'], row['Low'], row['Close'], int(row['Volume'])))
        conn.commit()
        conn.close()
        logging.info(f"Saved data for {len(data)} tickers to current_day.db")
    except Exception as e:
        logging.error(f"Error saving to current_day.db: {e}")

def is_trading_time():
    """Check if current time is within trading hours (1:00 AM - 5:00 PM PST)."""
    now = datetime.now(PST)
    start_time = now.replace(hour=1, minute=0, second=0, microsecond=0)    # 1:00 AM PST
    end_time = now.replace(hour=17, minute=0, second=0, microsecond=0)     # 5:00 PM PST
    return start_time <= now < end_time

def main():
    logging.info("Starting live fetch script")
    init_db()  # Set up the database once at start
    tickers = get_tickers()
    if not tickers:
        logging.error("No tickers found, exiting")
        return

    logging.info(f"Monitoring {len(tickers)} tickers")
    while True:
        now = datetime.now(PST)
        if not is_trading_time():
            logging.info(f"Outside trading hours ({now.strftime('%H:%M:%S')}), waiting...")
            time.sleep(60)  # Check every minute outside trading hours
            continue

        logging.info(f"Fetching live 2m data at {now.strftime('%Y-%m-%d %H:%M:%S-07:00')}")
        data = fetch_2m_data(tickers)
        if data:
            save_to_db(data)
        time.sleep(120)  # Wait 2 minutes (120 seconds) before next fetch

if __name__ == "__main__":
    main()