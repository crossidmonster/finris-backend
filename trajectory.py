import sqlite3

MASTER_TICKER_DB_PATH = "C:\\FINRIS\\master_ticker.db"
FORMULA_SETTINGS_DB_PATH = "C:\\FINRIS\\formula_settings.db"
SCORES_DB_PATH = "C:\\FINRIS\\scores.db"
TRAJECTORY_DB_PATH = "C:\\FINRIS\\trajectory_data.db"

def load_trajectory_settings():
    """Load Trajectory settings from formula_settings.db with defaults."""
    conn = sqlite3.connect(FORMULA_SETTINGS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT parameter, value FROM settings WHERE formula = 'trajectory'")
    rows = cursor.fetchall()
    conn.close()

    # Convert DB rows to a dictionary
    settings = {}
    for param, value in rows:
        settings[param] = value

    # Convert to appropriate types with defaults
    result = {
        "trajectory_fast_ema": int(settings.get("fast_ema", 12)),
        "trajectory_slow_ema": int(settings.get("slow_ema", 26)),
        "trajectory_signal_ema": int(settings.get("signal_ema", 9)),
        "trajectory_stagnant_threshold": float(settings.get("stagnant_threshold", 0.005))
    }
    return result

def fetch_trading_days(ticker, end_date):
    """Fetch up to 30 trading days of closing prices from trajectory_data.db."""
    conn = sqlite3.connect(TRAJECTORY_DB_PATH)
    cursor = conn.cursor()
    query = """
        SELECT date, close
        FROM historical_data
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 30
    """
    cursor.execute(query, (ticker, end_date[:10]))
    rows = cursor.fetchall()
    conn.close()
    print(f"Ticker: {ticker}, End Date: {end_date}, Fetched Days: {len(rows)}")  # Debug
    return [[row[0], row[1]] for row in rows][::-1]

def calc_macd_trajectory(closes, settings):
    """Calculate MACD over the data, return trend components for available days."""
    fast_ema = int(settings.get('trajectory_fast_ema', 12))
    slow_ema = int(settings.get('trajectory_slow_ema', 26))
    signal_ema = int(settings.get('trajectory_signal_ema', 9))
    n = len(closes)

    if n < 5:  # Lowered to 5
        return [], 0, "insufficient data"

    def ema(prices, period):
        if len(prices) < 2:  # Only average if less than 2 prices
            return sum(prices) / len(prices) if prices else 0
        alpha = 2 / (period + 1)
        # Start with the average of initial period or all available data
        ema_val = sum(prices[:min(period, len(prices))]) / min(period, len(prices))
        # Apply EMA from the first available point
        for p in prices[min(period, len(prices)):]:
            ema_val = p * alpha + ema_val * (1 - alpha)
        return ema_val

    macd_line = []
    # Adjust start index if n < slow_ema
    start_idx = max(0, min(slow_ema - 1, n - 1))
    for i in range(start_idx, n):
        ema_fast = ema(closes[:i + 1], fast_ema)
        ema_slow = ema(closes[:i + 1], slow_ema)
        macd_line.append(ema_fast - ema_slow)

    signal_line = []
    start_signal_idx = max(0, min(signal_ema - 1, len(macd_line) - 1))
    for i in range(start_signal_idx, len(macd_line)):
        signal_line.append(ema(macd_line[:i + 1], signal_ema))

    window = len(macd_line) - start_signal_idx
    if window <= 0:
        return [], 0, "insufficient data"
    histogram = [
        macd_line[i + start_signal_idx] - signal_line[i]
        for i in range(-window, 0)
    ]

    last_5 = min(5, len(histogram))
    crossovers = 0
    for i in range(-last_5 + 1, 0):
        if histogram[i - 1] * histogram[i] < 0:
            crossovers += 1

    return histogram, crossovers, None

def get_trajectory(ticker, date):
    """Determine momentum trajectory and score with available days."""
    settings = load_trajectory_settings()  # Load settings from DB
    trading_days = fetch_trading_days(ticker, date)
    if not trading_days:
        return "no data", 0

    closes = [row[1] for row in trading_days]
    n = len(closes)
    if n < 5:  # Lowered to 5
        return "insufficient data", 0

    histogram, crossovers, _ = calc_macd_trajectory(closes, settings)
    if not histogram:
        return "insufficient data", 0

    window = len(histogram)
    hist_window = histogram[-window:]
    avg_histogram = sum(hist_window) / len(hist_window)
    print(f"Ticker: {ticker}, Days: {n}, Window: {window}, Avg Histogram: {avg_histogram}, Crossovers: {crossovers}")  # Debug

    positive_days = sum(1 for h in hist_window if h > 0)
    negative_days = sum(1 for h in hist_window if h < 0)

    if crossovers >= 4 or abs(positive_days - negative_days) < max(1, window // 10):  # Adjusted for tiny windows
        trend = "stagnant"
    elif positive_days > negative_days:
        trend = "positive"
        if window > 5 and any(h < 0 for h in hist_window[:max(1, window//2)]) and sum(1 for h in hist_window[-min(15, window):] if h > settings.get('trajectory_stagnant_threshold', 0.005)) >= min(2, window//5):
            trend = "strong positive"
    elif negative_days > positive_days:
        trend = "negative"
        if window > 5 and any(h > 0 for h in hist_window[:max(1, window//2)]) and sum(1 for h in hist_window[-min(15, window):] if h < -settings.get('trajectory_stagnant_threshold', 0.005)) >= min(2, window//5):
            trend = "strong negative"
    else:
        trend = "stagnant"

    score = min(100, abs(avg_histogram) * 50)
    score = max(0, score - 5 * crossovers)
    score = round(score * 10)
    score = min(100, score)
    print(f"Trend: {trend}, Score: {score}, Pos Days: {positive_days}, Neg Days: {negative_days}")  # Debug

    return trend, score

def save_trajectory(ticker, datetime, trend, score):
    """Save Trajectory result to scores.db."""
    conn = sqlite3.connect(SCORES_DB_PATH)
    cursor = conn.cursor()
    query = """
        INSERT OR REPLACE INTO ticker_scores 
        (ticker, datetime, trajectory, trajectory_score) 
        VALUES (?, ?, ?, ?)
    """
    cursor.execute(query, (ticker, datetime, trend, score))
    conn.commit()
    conn.close()