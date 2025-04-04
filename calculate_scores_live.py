import sqlite3
import requests
import time
from datetime import datetime, timedelta
import pytz
import logging
from backend.IndicatorFormulas import calculate_indicators
from backend.FINRISFormulas import calculate_finris_indicators

# Setup logging
logging.basicConfig(
    filename='C:\\FINRIS\\calculate_scores_live.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PT = pytz.timezone('America/Los_Angeles')
CURRENT_DAY_DB_PATH = "C:\\FINRIS\\current_day.db"
TRAJECTORY_DB_PATH = "C:\\FINRIS\\trajectory_data.db"
FORMULA_SETTINGS_DB_PATH = "C:\\FINRIS\\formula_settings.db"
TEMP_SCORES_DB_PATH = "C:\\FINRIS\\temp_scores.db"  # Temp DB for live scores
API_BASE_URL = "http://192.168.1.127:8000"

def init_temp_db():
    """Initialize temp_scores.db with live_scores table and ensure opportunity_icon column exists."""
    with sqlite3.connect(TEMP_SCORES_DB_PATH) as conn:
        cursor = conn.cursor()
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_scores (
                ticker TEXT,
                time TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                base_score TEXT,
                finris_score REAL,
                roc_2d REAL,
                opportunity_icon TEXT,  -- Added for Buy/Sell icons
                PRIMARY KEY (ticker, time)
            )
        """)
        # Check if opportunity_icon column exists, add it if not
        cursor.execute("PRAGMA table_info(live_scores)")
        columns = [col[1] for col in cursor.fetchall()]
        if "opportunity_icon" not in columns:
            cursor.execute("ALTER TABLE live_scores ADD COLUMN opportunity_icon TEXT")
            logging.info("Added opportunity_icon column to live_scores table")
        conn.commit()
    logging.info("Initialized temp_scores.db with live_scores table including opportunity_icon")

def load_settings():
    """Load formula settings from formula_settings.db."""
    with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT formula, parameter, value FROM settings")
        settings_raw = cursor.fetchall()
    base_settings = {formula: {} for formula, _, _ in settings_raw if not formula.startswith('finris_')}
    finris_settings = {formula.replace('finris_', ''): {} for formula, _, _ in settings_raw if formula.startswith('finris_')}
    for formula, param, value in settings_raw:
        if formula.startswith('finris_'):
            finris_settings[formula.replace('finris_', '')][param] = value
        else:
            base_settings[formula][param] = value
    return base_settings, finris_settings

def get_banner_tickers():
    """Fetch current banner tickers from /green_tickers."""
    try:
        response = requests.get(f"{API_BASE_URL}/green_tickers")
        response.raise_for_status()
        return response.json().get("tickers", [])
    except Exception as e:
        logging.error(f"Error fetching banner tickers: {e}")
        return []

def calculate_ticker_scores(ticker, base_settings, finris_settings, close_2d_ago):
    """Calculate scores for a ticker's 2-minute rows, including opportunity icons."""
    current_date = datetime.now(PT).strftime("%Y-%m-%d")
    with sqlite3.connect(CURRENT_DAY_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT datetime, open, high, low, close, volume
            FROM ticker_data
            WHERE ticker = ? AND datetime LIKE ?
            ORDER BY datetime ASC
        """, (ticker, f"{current_date}%"))
        data = cursor.fetchall()
    
    if not data:
        logging.info(f"No data for {ticker} in current_day.db")
        return []
    
    rows = []
    for i, (dt, o, h, l, c, v) in enumerate(data):
        closes = [float(r[4]) for r in data[max(0, i-19):i+1]]
        highs = [float(r[2]) for r in data[max(0, i-19):i+1]]
        lows = [float(r[3]) for r in data[max(0, i-19):i+1]]
        timestamps = [r[0] for r in data[max(0, i-19):i+1]]
        volumes = [float(r[5]) for r in data[max(0, i-19):i+1]]
        
        if len(closes) < 20:
            base_score = 0
            direction = "neutral"
            finris_score = 0
        else:
            base_score, direction = calculate_indicators(closes, highs, lows, c, timestamps, volumes, base_settings)
            finris_score_with_direction = calculate_finris_indicators(closes, highs, lows, c, timestamps, volumes, finris_settings)
            _, finris_score = finris_score_with_direction.split(" ", 1)
            finris_score = float(finris_score)
        
        roc_2d = ((c - close_2d_ago) / close_2d_ago) * 100 if close_2d_ago and close_2d_ago != 0 else None
        
        # Filter for eligibility first
        is_green_or_yellow = roc_2d is not None and (roc_2d >= 2 or (0.6 <= roc_2d <= 1.9))
        if finris_score >= 50 and is_green_or_yellow and direction == "buy":
            row = {
                "time": dt.split(" ")[1][:5],
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(v),
                "base_score": f"{direction:<7} {base_score:>5}",
                "finris_score": finris_score,
                "roc_2d": roc_2d,
                "opportunity_icon": ""  # Default, updated below
            }
            
            # Calculate opportunity icon using 8-row window
            if roc_2d is not None:
                start_idx = max(0, i - 7)  # Last 8 rows or all if less
                recent_rows = data[start_idx:i+1]
                roc_values = [
                    ((float(r[4]) - close_2d_ago) / close_2d_ago) * 100 
                    for r in recent_rows 
                    if close_2d_ago and close_2d_ago != 0
                ]
                if roc_values:
                    min_roc = min(roc_values)
                    max_roc = max(roc_values)
                    range_roc = max_roc - min_roc or 1  # Avoid division by 0
                    base_score_value = float(row["base_score"].split()[1]) if len(row["base_score"].split()) > 1 else 0
                    
                    if (roc_2d <= min_roc + 0.1 * range_roc and 
                        direction == "buy" and 
                        base_score_value > 20 and 
                        finris_score >= 60):
                        row["opportunity_icon"] = "▲"  # Buy icon
                    elif (roc_2d >= max_roc - 0.1 * range_roc and 
                          direction == "sell" and 
                          base_score_value > 20):
                        row["opportunity_icon"] = "■"  # Sell icon
            
            rows.append(row)
            logging.info(f"Row for {ticker} at {row['time']} - roc_2d: {roc_2d}, icon: {row['opportunity_icon']}")
    
    return rows

def main():
    logging.info("Starting calculate_scores_live.py")
    init_temp_db()  # Initialize temp DB at start
    base_settings, finris_settings = load_settings()
    
    while True:
        start_time = time.time()
        banner_tickers = get_banner_tickers()
        logging.info(f"Banner tickers: {banner_tickers}")
        
        if not banner_tickers:
            logging.warning("No banner tickers found, skipping cycle")
            time.sleep(60)
            continue
        
        # Fetch close_2d_ago for all banner tickers
        two_days_ago = (datetime.now(PT) - timedelta(days=2)).strftime("%Y-%m-%d")
        close_2d_ago_dict = {}
        with sqlite3.connect(TRAJECTORY_DB_PATH) as conn_traj:
            cursor_traj = conn_traj.cursor()
            for ticker in banner_tickers:
                cursor_traj.execute("""
                    SELECT close 
                    FROM daily_closes 
                    WHERE ticker = ? AND date = ?
                """, (ticker, two_days_ago))
                result = cursor_traj.fetchone()
                close_2d_ago_dict[ticker] = result[0] if result else None
        
        # Calculate scores and write to temp DB
        with sqlite3.connect(TEMP_SCORES_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM live_scores")  # Clear previous scores
            live_scores = {}
            for ticker in banner_tickers:
                close_2d_ago = close_2d_ago_dict.get(ticker)
                if close_2d_ago is None:
                    logging.warning(f"No close_2d_ago for {ticker}, skipping")
                    continue
                scores = calculate_ticker_scores(ticker, base_settings, finris_settings, close_2d_ago)
                if scores:
                    live_scores[ticker] = scores
                    for row in scores:
                        cursor.execute("""
                            INSERT OR REPLACE INTO live_scores (ticker, time, open, high, low, close, volume, base_score, finris_score, roc_2d, opportunity_icon)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (ticker, row["time"], row["open"], row["high"], row["low"], row["close"],
                              row["volume"], row["base_score"], row["finris_score"], row["roc_2d"], row["opportunity_icon"]))
                        logging.info(f"Inserted row for {ticker} at {row['time']}")
            conn.commit()
        
        logging.info(f"Live scores calculated and saved: {len(live_scores)} tickers with eligible rows")
        
        # Sleep to maintain 1-minute cycle
        elapsed = time.time() - start_time
        sleep_time = max(60 - elapsed, 0)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()