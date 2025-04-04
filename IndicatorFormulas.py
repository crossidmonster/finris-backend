import sqlite3

FORMULA_SETTINGS_DB_PATH = "C:\\FINRIS\\formula_settings.db"

def calc_vwap(closes, highs, lows, current_price, timestamps, volumes, settings):
    """Calculate VWAP score and direction, using PST times."""
    vwap_settings = settings.get('vwap', {})
    start_time = vwap_settings.get('start_time', '06:30')  # 06:30 AM PST
    end_time = vwap_settings.get('end_time', '13:00')     # 01:00 PM PST
    n = len(closes)
    vwap = None
    vwap_score = 0
    direction = "neutral"
    if n == len(highs) == len(lows) == len(volumes) == len(timestamps):
        current_time = timestamps[-1]
        print(f"Base VWAP Raw Current Time: {current_time}")
        try:
            time_part = current_time.split(" ")[1].split("-")[0] if " " in current_time else current_time
            print(f"Base VWAP Time Part: {time_part}")
            hour, minute = map(int, time_part[:5].split(":"))
            current_time_minutes = hour * 60 + minute
            try:
                start_hour, start_min = map(int, start_time.split(':'))
                end_hour, end_min = map(int, end_time.split(':'))
                start_time_minutes = start_hour * 60 + start_min
                end_time_minutes = end_hour * 60 + end_min
            except ValueError:
                start_time_minutes = int(start_time)
                end_time_minutes = int(end_time)
            within_trading_hours = start_time_minutes <= current_time_minutes <= end_time_minutes
            print(f"Base VWAP Time Check: {start_time_minutes} <= {current_time_minutes} <= {end_time_minutes} = {within_trading_hours}")
        except (ValueError, IndexError) as e:
            print(f"Base VWAP Time Parse Error: {e}")
            within_trading_hours = False
        if within_trading_hours:
            typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(n)]
            cumulative_pv = 0
            cumulative_volume = 0
            for i in range(n):
                try:
                    ts = timestamps[i]
                    ts_part = ts.split(" ")[1].split("-")[0] if " " in ts else ts
                    hour, minute = map(int, ts_part[:5].split(":"))
                    time_minutes = hour * 60 + minute
                    if start_time_minutes <= time_minutes <= current_time_minutes:
                        cumulative_pv += typical_prices[i] * volumes[i]
                        cumulative_volume += volumes[i]
                except (ValueError, IndexError) as e:
                    print(f"Base VWAP Timestamp {i} Error: {e}")
                    continue
            print(f"Base VWAP Cumulative - PV: {cumulative_pv}, Volume: {cumulative_volume}")
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                print(f"Base VWAP: {vwap}, Price: {current_price}")
                if current_price > vwap:
                    vwap_score = min(100, (current_price - vwap) / vwap * 1000)
                    direction = "buy"
                elif current_price < vwap:
                    vwap_score = min(100, (vwap - current_price) / vwap * 1000)
                    direction = "sell"
    print(f"Base VWAP Debug: Score={vwap_score}, Direction={direction}, Current={current_price}, VWAP={vwap}")
    return vwap_score, direction

def calc_rsi(closes, settings):
    rsi_settings = settings.get('rsi', {})
    length = int(rsi_settings.get('length', 7))  # From formulasettings.db
    delta_offset = int(rsi_settings.get('delta_offset', 2))
    delta = int(rsi_settings.get('delta', 1))
    weight = float(rsi_settings.get('weight', 0.20))
    n = len(closes)
    rsi_score = 0
    direction = "neutral"
    
    if n >= length + 1:
        # Standard RSI calculation (SMA + smoothing)
        gains, losses = [], []
        for i in range(1, length + 1):
            diff = closes[i] - closes[i-1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains) / length if gains else 0
        avg_loss = sum(losses) / length if losses else 0
        for i in range(length + 1, n):
            current_gain = max(closes[i] - closes[i-1], 0)
            current_loss = max(closes[i-1] - closes[i], 0)
            avg_gain = (avg_gain * (length - 1) + current_gain) / length
            avg_loss = (avg_loss * (length - 1) + current_loss) / length
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs)) if rs != 100 else 100
        
        # Delta calculation (optional)
        rsi_prev = None
        if delta and n >= length + delta_offset + 1:
            prev_gains, prev_losses = [], []
            for i in range(delta_offset + 1, min(length + delta_offset + 1, n)):
                diff = closes[i] - closes[i - delta_offset - 1]
                prev_gains.append(max(diff, 0))
                prev_losses.append(max(-diff, 0))
            prev_avg_gain = sum(prev_gains) / length if prev_gains else 0
            prev_avg_loss = sum(prev_losses) / length if prev_losses else 0
            for i in range(length + delta_offset + 1, n):
                current_gain = max(closes[i] - closes[i-1], 0)
                current_loss = max(closes[i-1] - closes[i], 0)
                prev_avg_gain = (prev_avg_gain * (length - 1) + current_gain) / length
                prev_avg_loss = (prev_avg_loss * (length - 1) + current_loss) / length
            prev_rs = prev_avg_gain / prev_avg_loss if prev_avg_loss != 0 else 100
            rsi_prev = 100 - (100 / (1 + prev_rs)) if prev_rs != 100 else 100
            rsi_delta = rsi - rsi_prev if rsi_prev is not None else 0
        else:
            rsi_delta = 0
        
        # Debug print #1: After rsi_delta is set
        print(f"Base delta: {delta}, rsi: {rsi}, rsi_delta: {rsi_delta}")
        
        # Debug print #2: Before scoring
        print(f"Base score calc - rsi: {rsi}, rsi_delta: {rsi_delta}, delta: {delta}")
        
        # Swing/Day Trading Logic
        if rsi > 70 and rsi_delta < 0:
            rsi_score = min(100, (rsi - 70) * 2)  # Overbought, sell
            direction = "sell"
        elif rsi < 30 and rsi_delta > 0:
            rsi_score = min(100, (30 - rsi) * 2)  # Oversold, buy
            direction = "buy"
        elif rsi > 70:
            rsi_score = min(100, (rsi - 70) * 1.5)  # Overbought, weaker sell
            direction = "sell"
        elif rsi < 30:
            rsi_score = min(100, (30 - rsi) * 1.5)  # Oversold, weaker buy
            direction = "buy"
        else:
            rsi_score = 0  # Neutral zone (30–70)
            direction = "neutral"
    
    return rsi_score, direction

def calc_macd(closes, settings):
    macd_settings = settings.get('macd', {})
    fast_ema = int(macd_settings.get('fast_ema', 12))    # Common default
    slow_ema = int(macd_settings.get('slow_ema', 26))
    signal = int(macd_settings.get('signal', 9))
    weight = float(macd_settings.get('weight', 0.20))
    n = len(closes)
    macd_score = 0
    direction = "neutral"
    if n >= slow_ema:
        def ema(prices, period):
            if len(prices) < period: return 0
            alpha = 2 / (period + 1)
            ema_val = prices[-period]
            for p in prices[-period+1:]:
                ema_val = p * alpha + ema_val * (1 - alpha)
            return ema_val
        ema_fast = ema(closes, fast_ema)
        ema_slow = ema(closes, slow_ema)
        macd = ema_fast - ema_slow
        if n >= slow_ema + signal:
            macd_values = []
            for i in range(-signal - 1, 0):
                ema_fast_i = ema(closes[:i+1], fast_ema)
                ema_slow_i = ema(closes[:i+1], slow_ema)
                macd_values.append(ema_fast_i - ema_slow_i)
            macd_values.append(macd)
            signal_line = ema(macd_values, signal)
            signal_prev = ema(macd_values[:-1], signal)
            histogram = macd - signal_line
            histogram_prev = macd_values[-2] - signal_prev
            macd_score = min(100, abs(histogram) * 100)
            if macd > signal_line and histogram > histogram_prev:
                direction = "buy"
            elif macd < signal_line and histogram < histogram_prev:
                direction = "sell"
            else:
                direction = "neutral"
        else:
            histogram = macd
            macd_score = min(100, abs(histogram) * 100)
            if macd > 0:
                direction = "buy"
            elif macd < 0:
                direction = "sell"
    return macd_score, direction

def calc_adx(highs, lows, closes, settings):
    adx_settings = settings.get('adx', {})
    length = int(adx_settings.get('length', 14))         # Common default
    mult_high = float(adx_settings.get('multiplier_high', 1.25))
    mult_low = float(adx_settings.get('multiplier_low', 0.75))
    n = len(closes)
    adx_multiplier = 1.0
    if n >= length * 2:
        tr_list, plus_dm_list, minus_dm_list = [], [], []
        for i in range(1, n):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            plus_dm = highs[i] - highs[i-1] if highs[i] > highs[i-1] and highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
            minus_dm = lows[i-1] - lows[i] if lows[i] < lows[i-1] and lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
            tr_list.append(tr)
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        tr_len = sum(tr_list[-length:]) / length
        plus_dm_len = sum(plus_dm_list[-length:]) / length
        minus_dm_len = sum(minus_dm_list[-length:]) / length
        plus_di = (plus_dm_len / tr_len) * 100 if tr_len != 0 else 0
        minus_di = (minus_dm_len / tr_len) * 100 if tr_len != 0 else 0
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
        adx = dx
        if n >= length * 3:
            dx_list = []
            for j in range(-length * 2, 0):
                tr_len = sum(tr_list[j-length:j]) / length
                plus_dm_len = sum(plus_dm_list[j-length:j]) / length
                minus_dm_len = sum(minus_dm_list[j-length:j]) / length
                plus_di = (plus_dm_len / tr_len) * 100 if tr_len != 0 else 0
                minus_di = (minus_dm_len / tr_len) * 100 if tr_len != 0 else 0
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
                dx_list.append(dx)
            adx = sum(dx_list) / length
        if adx != adx:  # NaN check
            adx_multiplier = 1.0
        elif adx > 15:
            adx_multiplier = mult_high
        elif adx < 10:
            adx_multiplier = mult_low
    return adx_multiplier

def calc_ema(closes, period, weight):
    n = len(closes)
    ema_score = 0
    direction = "neutral"
    if n >= period:
        def ema(prices, p):
            if len(prices) < p: return 0
            alpha = 2 / (p + 1)
            ema_val = prices[-p]
            for price in prices[-p+1:]:
                ema_val = price * alpha + ema_val * (1 - alpha)
            return ema_val
        ema_val = ema(closes, period)
        current_price = closes[-1]
        if current_price > ema_val:
            ema_score = min(100, (current_price - ema_val) / ema_val * 1000)
            direction = "buy"
        elif current_price < ema_val:
            ema_score = min(100, (ema_val - current_price) / ema_val * 1000)
            direction = "sell"
    return ema_score, direction

def calc_roc(closes, settings):
    roc_settings = settings.get('roc', {})
    length = int(roc_settings.get('length', 5))          # Default from FINRIS
    weight = float(roc_settings.get('weight', 0.20))
    n = len(closes)
    roc_score = 0
    direction = "neutral"
    if n >= length + 1:
        price_n_ago = closes[-length - 1]
        current_price = closes[-1]
        roc = ((current_price - price_n_ago) / price_n_ago) * 100 if price_n_ago != 0 else 0
        roc_score = min(100, abs(roc) * 10)
        if roc > 0:
            direction = "buy"
        elif roc < 0:
            direction = "sell"
    return roc_score, direction

def calc_ema_crossover(closes, settings):
    ema_cross_settings = settings.get('ema_crossover', {})
    fast_length = int(ema_cross_settings.get('fast_length', 5))  # Example default
    slow_length = int(ema_cross_settings.get('slow_length', 13))
    weight = float(ema_cross_settings.get('weight', 0.20))
    n = len(closes)
    ema_score = 50
    direction = "neutral"
    if n >= slow_length:
        def ema(prices, period):
            if len(prices) < period: return 0
            alpha = 2 / (period + 1)
            ema_val = prices[-period]
            for p in prices[-period+1:]:
                ema_val = p * alpha + ema_val * (1 - alpha)
            return ema_val
        ema_fast = ema(closes, fast_length)
        ema_slow = ema(closes, slow_length)
        ema_score = 75 if ema_fast > ema_slow else (25 if ema_fast < ema_slow else 50)
        direction = "buy" if ema_fast > ema_slow else ("sell" if ema_fast < ema_slow else "neutral")
    return ema_score, direction

def calc_stochastic(highs, lows, closes, settings):
    stoch_settings = settings.get('stochastic', {})
    k_length = int(stoch_settings.get('k_length', 14))    # Common default
    smooth_k = int(stoch_settings.get('smooth_k', 3))
    smooth_d = int(stoch_settings.get('smooth_d', 3))
    weight = float(stoch_settings.get('weight', 0.20))
    n = len(closes)
    stochastic_score = 0
    direction = "neutral"
    # Validate inputs
    if not highs or not lows or not closes or n != len(highs) or n != len(lows):
        return 0, "neutral"
    if n >= k_length:
        # Calculate %K
        lowest_low_range = lows[-k_length:]
        highest_high_range = highs[-k_length:]
        if not lowest_low_range or not highest_high_range:
            return 0, "neutral"
        lowest_low = min(lowest_low_range)
        highest_high = max(highest_high_range)
        current_close = closes[-1]
        if highest_high != lowest_low:
            percent_k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
        else:
            percent_k = 50  # Default to neutral if no range
        # Smooth %K
        if n >= k_length + smooth_k - 1:
            k_values = []
            for i in range(-smooth_k, 0):
                low_range = lows[i-k_length+1:i+1]
                high_range = highs[i-k_length+1:i+1]
                if not low_range or not high_range:
                    k = 50  # Default to neutral if range is empty
                else:
                    low = min(low_range)
                    high = max(high_range)
                    close = closes[i]
                    if high != low:
                        k = 100 * (close - low) / (high - low)
                    else:
                        k = 50
                k_values.append(k)
            percent_k = sum(k_values) / smooth_k
        # Calculate %D (smoothing of %K)
        if n >= k_length + smooth_k + smooth_d - 1:
            d_values = []
            for i in range(-smooth_d, 0):
                k_vals = []
                for j in range(i-smooth_k+1, i+1):
                    low_range = lows[j-k_length+1:j+1]
                    high_range = highs[j-k_length+1:j+1]
                    if not low_range or not high_range:
                        k = 50  # Default to neutral if range is empty
                    else:
                        low = min(low_range)
                        high = max(high_range)
                        close = closes[j]
                        if high != low:
                            k = 100 * (close - low) / (high - low)
                        else:
                            k = 50
                    k_vals.append(k)
                d_values.append(sum(k_vals) / smooth_k)
            percent_d = sum(d_values) / smooth_d
        else:
            percent_d = percent_k  # If not enough data for %D, use %K
        # Scoring logic
        if percent_k > 80 and percent_k < percent_d:
            stochastic_score = 70  # Overbought, potential sell
            direction = "sell"
        elif percent_k < 20 and percent_k > percent_d:
            stochastic_score = 70  # Oversold, potential buy
            direction = "buy"
        elif percent_k > 50:
            stochastic_score = min(100, (percent_k - 50) * 2)  # Bullish
            direction = "buy"
        elif percent_k < 50:
            stochastic_score = min(100, (50 - percent_k) * 2)  # Bearish
            direction = "sell"
    return stochastic_score, direction

def calculate_indicators(closes, highs, lows, current_price, timestamps, volumes, settings):
    # Validate input lists
    if not closes or not highs or not lows or not timestamps or not volumes:
        return 0, "neutral"
    n = len(closes)
    if n < 20 or n != len(highs) or n != len(lows) or n != len(timestamps) or n != len(volumes):
        return 0, "neutral"
    
    vwap_score, vwap_dir = calc_vwap(closes, highs, lows, current_price, timestamps, volumes, settings)
    rsi_score, rsi_dir = calc_rsi(closes, settings)
    print(f"Base RSI Raw: {rsi_score}")
    macd_score, macd_dir = calc_macd(closes, settings)
    adx_multiplier = calc_adx(highs, lows, closes, settings)
    ema_5_score, ema_5_dir = calc_ema(closes, int(settings.get('ema_5', {}).get('length', 5)), float(settings.get('ema_5', {}).get('weight', 0.20)))
    ema_8_score, ema_8_dir = calc_ema(closes, int(settings.get('ema_8', {}).get('length', 8)), float(settings.get('ema_8', {}).get('weight', 0.20)))
    roc_score, roc_dir = calc_roc(closes, settings)
    ema_cross_score, ema_cross_dir = calc_ema_crossover(closes, settings)
    stochastic_score, stochastic_dir = calc_stochastic(highs, lows, closes, settings)
    
    # Replace NaN with 0 for each score, use defaults if settings missing
    scores = [
        vwap_score * float(settings.get('vwap', {}).get('weight', 0.20)),
        rsi_score * float(settings.get('rsi', {}).get('weight', 0.20)),
        macd_score * float(settings.get('macd', {}).get('weight', 0.20)),
        ema_5_score * float(settings.get('ema_5', {}).get('weight', 0.20)),
        ema_8_score * float(settings.get('ema_8', {}).get('weight', 0.20)),
        roc_score * float(settings.get('roc', {}).get('weight', 0.20)),
        ema_cross_score * float(settings.get('ema_crossover', {}).get('weight', 0.20)),
        stochastic_score * float(settings.get('stochastic', {}).get('weight', 0.20))
    ]
    score = sum(s if s == s else 0 for s in scores)  # NaN check: NaN != NaN
    score = min(max(score * adx_multiplier, 0), 100)
    
    # Weighted direction voting
    weighted_signals = [
        (vwap_dir, float(settings.get('vwap', {}).get('weight', 0.20))),
        (rsi_dir, float(settings.get('rsi', {}).get('weight', 0.20))),
        (macd_dir, float(settings.get('macd', {}).get('weight', 0.20))),
        (ema_5_dir, float(settings.get('ema_5', {}).get('weight', 0.20))),
        (ema_8_dir, float(settings.get('ema_8', {}).get('weight', 0.20))),
        (roc_dir, float(settings.get('roc', {}).get('weight', 0.20))),
        (ema_cross_dir, float(settings.get('ema_crossover', {}).get('weight', 0.20))),
        (stochastic_dir, float(settings.get('stochastic', {}).get('weight', 0.20)))
    ]
    buy_weight = sum(weight for direction, weight in weighted_signals if direction == "buy")
    sell_weight = sum(weight for direction, weight in weighted_signals if direction == "sell")
    neutral_weight = sum(weight for direction, weight in weighted_signals if direction == "neutral")
    
    if buy_weight > sell_weight and buy_weight > neutral_weight:
        direction = "buy"
    elif sell_weight > buy_weight and sell_weight > neutral_weight:
        direction = "sell"
    else:
        direction = "neutral"
    
    return round(score), direction