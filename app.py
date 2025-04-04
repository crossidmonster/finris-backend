import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pytz
from datetime import datetime, timedelta
import subprocess
from IndicatorFormulas import calculate_indicators
from FINRISFormulas import calculate_finris_indicators
from trajectory import get_trajectory, save_trajectory
from typing import List
import logging
import yfinance as yf
import os  # Added for environment variables
import uvicorn  # Added for running server

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
PT = pytz.timezone('America/Los_Angeles')

# Database paths with environment variables, defaulting to local paths
DB_PATH = os.getenv("DB_PATH", "C:\\FINRIS\\stock_monitor.db")
MASTER_DB_PATH = os.getenv("MASTER_DB_PATH", "C:\\FINRIS\\master_ticker.db")
SCORES_DB_PATH = os.getenv("SCORES_DB_PATH", "C:\\FINRIS\\scores.db")
FORMULA_SETTINGS_DB_PATH = os.getenv("FORMULA_SETTINGS_DB_PATH", "C:\\FINRIS\\formula_settings.db")
CURRENT_DAY_DB_PATH = os.getenv("CURRENT_DAY_DB_PATH", "C:\\FINRIS\\current_day.db")
TRAJECTORY_DB_PATH = os.getenv("TRAJECTORY_DB_PATH", "C:\\FINRIS\\trajectory_data.db")
TEMP_SCORES_DB_PATH = os.getenv("TEMP_SCORES_DB_PATH", "C:\\FINRIS\\temp_scores.db")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Schema update for ticker_scores (run once on startup)
with sqlite3.connect(SCORES_DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ticker_scores)")
    columns = [col[1] for col in cursor.fetchall()]
    if "trajectory" not in columns:
        cursor.execute("ALTER TABLE ticker_scores ADD COLUMN trajectory TEXT")
    if "trajectory_score" not in columns:
        cursor.execute("ALTER TABLE ticker_scores ADD COLUMN trajectory_score REAL")
    conn.commit()

@app.get("/stocks")
async def get_stocks():
    print(f"/stocks called at {datetime.now(PT).strftime('%Y-%m-%d %H:%M:%S-07:00')}")
    with sqlite3.connect(MASTER_DB_PATH) as conn_master:
        with sqlite3.connect(SCORES_DB_PATH) as conn_scores:
            cursor_master = conn_master.cursor()
            cursor_scores = conn_scores.cursor()
            
            cursor_master.execute("SELECT DISTINCT ticker FROM ticker_data")
            tickers = [row[0] for row in cursor_master.fetchall()]
            stocks = []
            for ticker in tickers:
                try:
                    cursor_master.execute("""
                        SELECT close 
                        FROM ticker_data 
                        WHERE ticker=? 
                        ORDER BY datetime DESC 
                        LIMIT 1
                    """, (ticker,))
                    price = cursor_master.fetchone()[0]
                    
                    cursor_scores.execute("""
                        SELECT base_score, direction 
                        FROM ticker_scores 
                        WHERE ticker=? 
                        ORDER BY datetime DESC 
                        LIMIT 1
                    """, (ticker,))
                    score_data = cursor_scores.fetchone()
                    base_score = score_data[0] if score_data else "N/A"
                    direction = score_data[1] if score_data else "N/A"
                    
                    stocks.append({
                        "ticker": ticker,
                        "price": f"${price:.4f}",
                        "score": base_score if base_score != "N/A" else "N/A",
                        "direction": direction
                    })
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    stocks.append({"ticker": ticker, "price": "N/A", "score": "N/A", "direction": "N/A"})
    
    print(f"Returning {len(stocks)} stocks")
    return {"stocks": stocks}

@app.get("/stock_data")
async def get_stock_data(ticker: str, date: str):
    with sqlite3.connect(MASTER_DB_PATH) as conn_master:
        with sqlite3.connect(SCORES_DB_PATH) as conn_scores:
            with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn_settings:
                with sqlite3.connect(TRAJECTORY_DB_PATH) as conn_traj:
                    cursor_master = conn_master.cursor()
                    cursor_scores = conn_scores.cursor()
                    c_settings = conn_settings.cursor()
                    cursor_traj = conn_traj.cursor()
                    
                    cursor_master.execute("""
                        SELECT datetime, open, high, low, close, volume 
                        FROM ticker_data 
                        WHERE ticker=? AND datetime LIKE ? 
                        ORDER BY datetime ASC
                    """, (ticker, f"{date}%"))
                    data = cursor_master.fetchall()
                    
                    # Fetch yesterday's roc_5d from trajectory_data.db
                    view_date = datetime.strptime(date, "%Y-%m-%d")
                    yesterday = (view_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    cursor_traj.execute("""
                        SELECT roc_5d 
                        FROM daily_closes 
                        WHERE ticker = ? AND date = ?
                    """, (ticker, yesterday))
                    roc_result = cursor_traj.fetchone()
                    roc_5d_yesterday = roc_result[0] if roc_result else None
                    logger.info(f"Ticker: {ticker}, View Date: {date}, Yesterday: {yesterday}, ROC_5d_Yesterday: {roc_5d_yesterday}")
                    
                    # Fetch close from 2 days ago for roc_2d
                    two_days_ago = (view_date - timedelta(days=2)).strftime("%Y-%m-%d")
                    cursor_traj.execute("""
                        SELECT close 
                        FROM daily_closes 
                        WHERE ticker = ? AND date = ?
                    """, (ticker, two_days_ago))
                    two_days_ago_result = cursor_traj.fetchone()
                    close_2d_ago = two_days_ago_result[0] if two_days_ago_result else None
                    logger.info(f"Ticker: {ticker}, View Date: {date}, Two Days Ago: {two_days_ago}, Close_2d_Ago: {close_2d_ago}")
                    
                    if not data:
                        return {"ticker": ticker, "date": date, "data": [], "roc_5d_yesterday": roc_5d_yesterday, "close_2d_ago": close_2d_ago}
                    
                    c_settings.execute("SELECT formula, parameter, value FROM settings")
                    settings_raw = c_settings.fetchall()
                    base_settings = {formula: {} for formula, _, _ in settings_raw if not formula.startswith('finris_')}
                    finris_settings = {formula.replace('finris_', ''): {} for formula, _, _ in settings_raw if formula.startswith('finris_')}
                    trajectory_settings = {formula: {} for formula, _, _ in settings_raw if formula.startswith('trajectory_')}
                    for formula, param, value in settings_raw:
                        if formula.startswith('finris_'):
                            finris_settings[formula.replace('finris_', '')][param] = value
                        elif formula.startswith('trajectory_'):
                            trajectory_settings[formula][param] = value
                        else:
                            base_settings[formula][param] = value
                    logger.info(f"Base settings for {ticker}: {base_settings}")
                    logger.info(f"FINRIS settings for {ticker}: {finris_settings}")
                    logger.info(f"Trajectory settings for {ticker}: {trajectory_settings}")
                    
                    rows = []
                    for i, (dt, o, h, l, c, v) in enumerate(data):
                        closes = [float(r[4]) for r in data[max(0, i-19):i+1]]
                        highs = [float(r[2]) for r in data[max(0, i-19):i+1]]
                        lows = [float(r[3]) for r in data[max(0, i-19):i+1]]
                        timestamps = [r[0] for r in data[max(0, i-19):i+1]]
                        volumes = [float(r[5]) for r in data[max(0, i-19):i+1]]
                        
                        logger.info(f"Ticker: {ticker}, Date: {date}, Index: {i}, Lengths - closes: {len(closes)}, highs: {len(highs)}, lows: {len(lows)}, timestamps: {len(timestamps)}, volumes: {len(volumes)}")
                        
                        if len(closes) < 20:
                            base_score = 0
                            direction = "neutral"
                            finris_score_with_direction = "neutral 0"
                        else:
                            base_score, direction = calculate_indicators(closes, highs, lows, c, timestamps, volumes, base_settings)
                            logger.info(f"Calling FINRIS with settings: {finris_settings}")
                            finris_score_with_direction = calculate_finris_indicators(closes, highs, lows, c, timestamps, volumes, finris_settings)
                            logger.info(f"FINRIS result: {finris_score_with_direction}")
                            if "07:32" in dt:
                                print(f"07:32 Raw - Base: {base_score}, FINRIS: {finris_score_with_direction}")
                            
                        # Parse FINRIS score for DB storage and frontend
                        finris_direction, finris_score = finris_score_with_direction.split(" ", 1)
                        cursor_scores.execute("""
                            INSERT OR REPLACE INTO ticker_scores (ticker, datetime, base_score, direction, finris_score, finris_direction)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (ticker, dt, base_score, direction, finris_score, finris_direction))
                        
                        base_score_with_direction = f"{direction:<7} {base_score:>5}"
                        rows.append({
                            "time": dt.split(" ")[1][:5],
                            "open": float(o),
                            "high": float(h),
                            "low": float(l),
                            "close": float(c),
                            "volume": int(v),
                            "base_score": base_score_with_direction,
                            "direction": direction,
                            "finris_score": finris_score
                        })
                    
                    conn_scores.commit()
                    return {"ticker": ticker, "date": date, "data": rows, "roc_5d_yesterday": roc_5d_yesterday, "close_2d_ago": close_2d_ago}

@app.get("/trajectory")
async def get_trajectory_data(ticker: str, date: str):
    with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn_settings:
        c_settings = conn_settings.cursor()
        c_settings.execute("SELECT formula, parameter, value FROM settings WHERE formula LIKE 'trajectory_%'")
        settings_raw = c_settings.fetchall()
        trajectory_settings = {}
        for formula, param, value in settings_raw:
            trajectory_settings[param] = value
        
        trend, score = get_trajectory(ticker, date)
        
        with sqlite3.connect(MASTER_DB_PATH) as conn_master:
            cursor_master = conn_master.cursor()
            cursor_master.execute("""
                SELECT MAX(datetime) 
                FROM ticker_data 
                WHERE ticker=? AND datetime LIKE ?
            """, (ticker, f"{date}%"))
            last_datetime = cursor_master.fetchone()[0]
            if last_datetime:
                save_trajectory(ticker, last_datetime, trend, score)
        
        return {"ticker": ticker, "date": date, "trajectory": trend, "score": score}

@app.post("/run_fetch")
async def run_fetch(date: str):
    if date == "export":
        subprocess.run(["python", "C:\\FINRIS\\export_to_master.py"])
        return {"message": "Export triggered"}
    elif date == "full":
        subprocess.run(["python", "C:\\FINRIS\\populate_master.py", "--full"])
        return {"message": "Full history fetch triggered"}
    elif date == "today":
        current_date = datetime.now(PT).strftime("%Y-%m-%d")
        subprocess.run(["python", "C:\\FINRIS\\populate_master.py", "--date", current_date])
        return {"message": f"Fetch for {current_date} triggered"}
    else:
        subprocess.run(["python", "C:\\FINRIS\\populate_master.py", "--date", date])
        return {"message": f"Fetch for {date} triggered"}

@app.get("/formula_settings")
async def get_formula_settings():
    with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT formula, parameter, value, description FROM settings")
        settings = [{"formula": row[0], "parameter": row[1], "value": row[2], "description": row[3]} for row in cursor.fetchall()]
        return {"settings": settings}

@app.post("/update_formula")
async def update_formula(updates: List[dict]):
    with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn:
        cursor = conn.cursor()
        for update in updates:
            formula = update['formula']
            parameter = update['parameter']
            value = update['value']
            description = None
            if formula == "adx" and parameter == "weight" and value is not None:
                description = "Weight applied to the ADX multiplier in the final score calculation"
            
            cursor.execute("""
                SELECT COUNT(*) FROM settings WHERE formula=? AND parameter=?
            """, (formula, parameter))
            exists = cursor.fetchone()[0] > 0
            if exists:
                cursor.execute("""
                    UPDATE settings
                    SET value=?
                    WHERE formula=? AND parameter=?
                """, (value, formula, parameter))
            else:
                cursor.execute("""
                    INSERT INTO settings (formula, parameter, value, description)
                    VALUES (?, ?, ?, ?)
                """, (formula, parameter, value, description))
        conn.commit()
        return {"message": "Formula settings updated successfully"}

@app.get("/live_data")
async def get_live_data(ticker: str):
    """Fetch all 2m data for the specified ticker from current_day.db for the current day."""
    current_date = datetime.now(PT).strftime("%Y-%m-%d")
    with sqlite3.connect(CURRENT_DAY_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT datetime, open, high, low, close, volume
            FROM ticker_data
            WHERE ticker=? AND datetime LIKE ?
            ORDER BY datetime ASC
        """, (ticker, f"{current_date}%"))
        data = cursor.fetchall()
        
        with sqlite3.connect(TRAJECTORY_DB_PATH) as conn_traj:
            cursor_traj = conn_traj.cursor()
            # Fetch yesterday's roc_5d from trajectory_data.db
            yesterday = (datetime.now(PT) - timedelta(days=1)).strftime("%Y-%m-%d")
            cursor_traj.execute("""
                SELECT roc_5d 
                FROM daily_closes 
                WHERE ticker = ? AND date = ?
            """, (ticker, yesterday))
            roc_result = cursor_traj.fetchone()
            roc_5d_yesterday = roc_result[0] if roc_result else None
            logger.info(f"Ticker: {ticker}, Current Date: {current_date}, Yesterday: {yesterday}, ROC_5d_Yesterday: {roc_5d_yesterday}")
            
            # Fetch close from 2 days ago for roc_2d
            two_days_ago = (datetime.now(PT) - timedelta(days=2)).strftime("%Y-%m-%d")
            cursor_traj.execute("""
                SELECT close 
                FROM daily_closes 
                WHERE ticker = ? AND date = ?
            """, (ticker, two_days_ago))
            two_days_ago_result = cursor_traj.fetchone()
            close_2d_ago = two_days_ago_result[0] if two_days_ago_result else None
            logger.info(f"Ticker: {ticker}, Current Date: {current_date}, Two Days Ago: {two_days_ago}, Close_2d_Ago: {close_2d_ago}")
        
        if not data:
            return {"ticker": ticker, "date": current_date, "data": [], "roc_5d_yesterday": roc_5d_yesterday, "close_2d_ago": close_2d_ago}
        
        with sqlite3.connect(FORMULA_SETTINGS_DB_PATH) as conn_settings:
            c_settings = conn_settings.cursor()
            c_settings.execute("SELECT formula, parameter, value FROM settings")
            settings_raw = c_settings.fetchall()
            base_settings = {formula: {} for formula, _, _ in settings_raw if not formula.startswith('finris_')}
            finris_settings = {formula.replace('finris_', ''): {} for formula, _, _ in settings_raw if formula.startswith('finris_')}
            for formula, param, value in settings_raw:
                if formula.startswith('finris_'):
                    finris_settings[formula.replace('finris_', '')][param] = value
                else:
                    base_settings[formula][param] = value
            logger.info(f"Base settings for {ticker}: {base_settings}")
            logger.info(f"FINRIS settings for {ticker}: {finris_settings}")
            
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
                    finris_score_with_direction = "neutral 0"
                else:
                    base_score, direction = calculate_indicators(closes, highs, lows, c, timestamps, volumes, base_settings)
                    finris_score_with_direction = calculate_finris_indicators(closes, highs, lows, c, timestamps, volumes, finris_settings)
                
                # Parse FINRIS score for frontend
                finris_direction, finris_score = finris_score_with_direction.split(" ", 1)
                base_score_with_direction = f"{direction:<7} {base_score:>5}"
                rows.append({
                    "time": dt.split(" ")[1][:5],
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": int(v),
                    "base_score": base_score_with_direction,
                    "direction": direction,
                    "finris_score": finris_score
                })
        
        logger.info(f"Returning live data for {ticker}: {len(rows)} rows")
        return {"ticker": ticker, "date": current_date, "data": rows, "roc_5d_yesterday": roc_5d_yesterday, "close_2d_ago": close_2d_ago}

@app.post("/add_ticker")
async def add_ticker(ticker: dict):
    """Add a new ticker to master_ticker.db and fetch its 50-day 2m historical data."""
    ticker_symbol = ticker.get("ticker")
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        return {"message": "Invalid ticker provided"}, 400
    
    with sqlite3.connect(MASTER_DB_PATH) as conn:
        cursor = conn.cursor()
        # Check if ticker already exists
        cursor.execute("SELECT COUNT(*) FROM ticker_data WHERE ticker = ?", (ticker_symbol,))
        exists = cursor.fetchone()[0] > 0
        if exists:
            return {"message": f"Ticker {ticker_symbol} already exists"}
        
        # Fetch 50 days of 2m data using yfinance
        try:
            stock = yf.Ticker(ticker_symbol)
            start_date = (datetime.now(PT) - timedelta(days=50)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now(PT).replace(hour=0, minute=0, second=0, microsecond=0)
            df = stock.history(start=start_date, end=end_date, interval="2m", prepost=True)
            
            if df.empty:
                return {"message": f"No data available for ticker {ticker_symbol}"}, 404
            
            # Insert data into master_ticker.db
            for index, row in df.iterrows():
                dt_pt = index.astimezone(PT).strftime("%Y-%m-%d %H:%M:%S-07:00")
                cursor.execute("""
                    INSERT OR IGNORE INTO ticker_data (ticker, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ticker_symbol, dt_pt, row["Open"], row["High"], row["Low"], row["Close"], int(row["Volume"])))
            conn.commit()
            logger.info(f"Added ticker {ticker_symbol} with {len(df)} rows of 2m data")
            return {"message": f"Ticker {ticker_symbol} added with 50-day historical data"}
        
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            # Add a placeholder entry if fetch fails, so the ticker is still tracked
            placeholder_datetime = "1970-01-01 00:00:00-07:00"
            cursor.execute("""
                INSERT OR IGNORE INTO ticker_data (ticker, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker_symbol, placeholder_datetime, None, None, None, None, 0))
            conn.commit()
            return {"message": f"Ticker {ticker_symbol} added, but data fetch failed: {str(e)}"}, 500

@app.post("/remove_ticker")
async def remove_ticker(ticker: dict):
    """Remove a ticker and all its data from master_ticker.db."""
    ticker_symbol = ticker.get("ticker")
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        return {"message": "Invalid ticker provided"}, 400
    
    with sqlite3.connect(MASTER_DB_PATH) as conn:
        cursor = conn.cursor()
        # Check if ticker exists
        cursor.execute("SELECT COUNT(*) FROM ticker_data WHERE ticker = ?", (ticker_symbol,))
        exists = cursor.fetchone()[0] > 0
        if not exists:
            return {"message": f"Ticker {ticker_symbol} not found"}, 404
        
        # Delete all entries for the ticker
        cursor.execute("DELETE FROM ticker_data WHERE ticker = ?", (ticker_symbol,))
        conn.commit()
        logger.info(f"Removed ticker {ticker_symbol} from master_ticker.db")
        return {"message": f"Ticker {ticker_symbol} removed successfully"}

@app.get("/green_tickers")
async def get_green_tickers():
    """Fetch tickers with roc_5d >= 2 from trajectory_data.db for yesterday."""
    yesterday = (datetime.now(PT) - timedelta(days=1)).strftime("%Y-%m-%d")
    with sqlite3.connect(TRAJECTORY_DB_PATH) as conn_traj:
        cursor_traj = conn_traj.cursor()
        cursor_traj.execute("""
            SELECT ticker 
            FROM daily_closes 
            WHERE date = ? AND roc_5d >= 2
        """, (yesterday,))
        green_tickers = [row[0] for row in cursor_traj.fetchall()]
        logger.info(f"Green tickers for {yesterday}: {green_tickers}")
        return {"tickers": green_tickers}

@app.get("/live_scores")
async def get_live_scores():
    """Fetch live scores calculated by calculate_scores_live.py from temp_scores.db."""
    try:
        with sqlite3.connect(TEMP_SCORES_DB_PATH) as conn:
            cursor = conn.cursor()
            # Fetch all eligible rows, grouped by ticker, including opportunity_icon
            cursor.execute("""
                SELECT ticker, time, open, high, low, close, volume, base_score, finris_score, roc_2d, opportunity_icon
                FROM live_scores
                ORDER BY ticker, time ASC
            """)
            rows = cursor.fetchall()
        
        if not rows:
            logger.info("No live scores found in temp_scores.db")
            return {"scores": {}}
        
        # Group rows by ticker
        scores = {}
        for row in rows:
            ticker = row[0]
            if ticker not in scores:
                scores[ticker] = []
            scores[ticker].append({
                "time": row[1],
                "open": row[2],
                "high": row[3],
                "low": row[4],
                "close": row[5],
                "volume": row[6],
                "base_score": row[7],
                "finris_score": row[8],
                "roc_2d": row[9],
                "opportunity_icon": row[10]
            })
        
        logger.info(f"Returning live scores for {len(scores)} tickers")
        return {"scores": scores}
    except Exception as e:
        logger.error(f"Error fetching live scores: {e}")
        return {"scores": {}, "error": str(e)}

@app.get("/nasdaq_roc")
async def get_nasdaq_roc():
    """Fetch daily % change from yfinance info and 15-min ROC direction for NASDAQ (^IXIC)."""
    try:
        nasdaq = yf.Ticker("^IXIC")
        today = datetime.now(PT)
        
        # Fetch daily % change directly from info
        daily = nasdaq.info.get("regularMarketChangePercent", None)
        
        # Fetch 1-minute data for 15-min ROC direction
        end_time = today
        start_time = end_time - timedelta(minutes=16)  # 16 mins to ensure 15-min span
        hist = nasdaq.history(start=start_time, end=end_time, interval="1m")
        
        if hist.empty or len(hist) < 15:
            logger.info("Insufficient NASDAQ data for ROC direction")
            roc_direction = None
        else:
            latest_close = hist['Close'].iloc[-1]
            past_close = hist['Close'].iloc[-15]  # 15 mins back
            roc = latest_close - past_close
            roc_direction = "up" if roc >= 0 else "down"
        
        logger.info(f"NASDAQ Daily: {daily}%, 15-min ROC direction: {roc_direction}")
        return {"daily": daily, "roc_direction": roc_direction}
    except Exception as e:
        logger.error(f"Error fetching NASDAQ data: {e}")
        return {"daily": None, "roc_direction": None}

# Run the server locally if executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 locally, use PORT env var on Render
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)