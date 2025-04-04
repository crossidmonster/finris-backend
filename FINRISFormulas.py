def calc_vwap(closes, highs, lows, current_price, timestamps, volumes, settings):
    """Calculate VWAP score and direction, using PST times."""
    vwap_settings = settings.get('vwap', {})  # Updated key
    start_time = vwap_settings.get('start_time', '390')
    end_time = vwap_settings.get('end_time', '780')
    n = len(closes)
    vwap = None
    vwap_score = 0
    direction = "neutral"
    if n == len(highs) == len(lows) == len(volumes) == len(timestamps):
        current_time = timestamps[-1]
        try:
            time_part = current_time.split(" ")[1].split("-")[0] if " " in current_time else current_time
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
            print(f"VWAP Time Check: {start_time_minutes} <= {current_time_minutes} <= {end_time_minutes} = {within_trading_hours}")
        except (ValueError, IndexError) as e:
            print(f"Time Parse Error: {e}")
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
                    print(f"Timestamp {i} Error: {e}")
                    continue
            print(f"Cumulative PV: {cumulative_pv}, Volume: {cumulative_volume}")
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                print(f"FINRIS VWAP: {vwap}, Price: {current_price}")
                if current_price > vwap:
                    vwap_score = min(100, (current_price - vwap) / vwap * 1000)
                    direction = "buy"
                elif current_price < vwap:
                    vwap_score = min(100, (vwap - current_price) / vwap * 1000)
                    direction = "sell"
    return vwap_score, direction

def calc_roc(closes, settings):
    """Calculate ROC score and direction."""
    roc_settings = settings.get('roc', {})  # Updated key
    length = int(roc_settings.get('length', 5))
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

def calc_rsi(closes, settings):
    """Calculate RSI score and direction."""
    rsi_settings = settings.get('rsi', {})  # Updated key
    length = int(rsi_settings.get('length', 7))
    delta_offset = int(rsi_settings.get('delta_offset', 2))
    delta = int(rsi_settings.get('delta', 1))
    n = len(closes)
    rsi_score = 0
    direction = "neutral"
    
    if n >= length + 1:
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
        
        print(f"FINRIS delta: {delta}, rsi: {rsi}, rsi_delta: {rsi_delta}")
        if rsi > 70 and rsi_delta < 0:
            rsi_score = min(100, (rsi - 70) * 2)
            direction = "sell"
        elif rsi < 30 and rsi_delta > 0:
            rsi_score = min(100, (30 - rsi) * 2)
            direction = "buy"
        elif rsi > 70:
            rsi_score = min(100, (rsi - 70) * 1.5)
            direction = "sell"
        elif rsi < 30:
            rsi_score = min(100, (30 - rsi) * 1.5)
            direction = "buy"
    return rsi_score, direction

def calc_adx(highs, lows, closes, settings):
    """Calculate ADX multiplier."""
    adx_settings = settings.get('adx', {})
    length = int(adx_settings.get('length', 7))
    mult_high = float(adx_settings.get('multiplier_high', 1.5))
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

def calculate_finris_indicators(closes, highs, lows, current_price, timestamps, volumes, settings):
    """Calculate FINRIS Score and direction."""
    if len(closes) < 20:
        return "neutral 0"

    rsi_weight = float(settings.get('rsi', {}).get('weight', 0.20))  # Updated key
    roc_weight = float(settings.get('roc', {}).get('weight', 0.20))  # Updated key
    vwap_weight = float(settings.get('vwap', {}).get('weight', 0.20))  # Updated key

    rsi_score, rsi_direction = calc_rsi(closes, settings)
    print(f"FINRIS RSI: {rsi_score}, Weight: {rsi_weight}")
    roc_score, roc_direction = calc_roc(closes, settings)
    print(f"FINRIS ROC: {roc_score}, Weight: {roc_weight}")
    vwap_score, vwap_direction = calc_vwap(closes, highs, lows, current_price, timestamps, volumes, settings)
    print(f"FINRIS VWAP: {vwap_score}, Weight: {vwap_weight}")

    adx_multiplier = calc_adx(highs, lows, closes, settings)
    print(f"FINRIS ADX: {adx_multiplier}")

    finris_score = (
        rsi_score * rsi_weight +
        roc_score * roc_weight +
        vwap_score * vwap_weight
    )
    print(f"FINRIS Score Before ADX: {finris_score}")
    finris_score = finris_score * adx_multiplier if finris_score == finris_score else 0
    finris_score = round(max(min(finris_score, 100), 0))

    weighted_signals = [
        (rsi_direction, rsi_weight),
        (roc_direction, roc_weight),
        (vwap_direction, vwap_weight)
    ]
    buy_weight = sum(weight for direction, weight in weighted_signals if direction == "buy")
    sell_weight = sum(weight for direction, weight in weighted_signals if direction == "sell")
    neutral_weight = sum(weight for direction, weight in weighted_signals if direction == "neutral")
    
    if buy_weight > sell_weight and buy_weight > neutral_weight:
        finris_direction = "buy"
    elif sell_weight > buy_weight and sell_weight > neutral_weight:
        finris_direction = "sell"
    else:
        finris_direction = "neutral"

    return f"{finris_direction} {finris_score}"