import pandas as pd

def detect_double_top_bottom(df, lookback=30, tolerance=0.015):
    if df is None or len(df) < lookback:
        return None
    highs = df["high"].iloc[-lookback:]
    lows = df["low"].iloc[-lookback:]
    half = lookback // 2
    peak1 = highs[:half].max()
    peak2 = highs[half:].max()
    low1 = lows[:half].min()
    low2 = lows[half:].min()
    if abs(peak1 - peak2) / peak1 <= tolerance:
        return "double_top"
    if abs(low1 - low2) / low1 <= tolerance:
        return "double_bottom"
    return None

def detect_bullish_engulfing(df):
    if len(df) < 2:
        return None
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    if c1["close"] < c1["open"] and c2["close"] > c2["open"]:
        if c2["close"] > c1["open"] and c2["open"] < c1["close"]:
            return "bullish_engulfing"
    return None

def detect_bearish_engulfing(df):
    if len(df) < 2:
        return None
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    if c1["close"] > c1["open"] and c2["close"] < c2["open"]:
        if c2["open"] > c1["close"] and c2["close"] < c1["open"]:
            return "bearish_engulfing"
    return None

def detect_head_and_shoulders(df, lookback=50):
    if len(df) < lookback:
        return None
    highs = df["high"].iloc[-lookback:]
    mid = lookback // 2
    left = highs[:mid]
    right = highs[mid:]
    left_shoulder = left.max()
    right_shoulder = right.max()
    head = highs[mid//2:mid+mid//2].max()
    if head > left_shoulder and head > right_shoulder:
        if abs(left_shoulder - right_shoulder) / head < 0.15:
            return "head_and_shoulders"
    return None

def detect_rising_wedge(df):
    if len(df) < 10:
        return None
    highs = df["high"].tail(10)
    lows = df["low"].tail(10)
    if highs.is_monotonic_increasing and lows.is_monotonic_increasing:
        if (highs.max() - highs.min()) > (lows.max() - lows.min()):
            return "rising_wedge"
    return None

def detect_falling_wedge(df):
    if len(df) < 10:
        return None
    highs = df["high"].tail(10)
    lows = df["low"].tail(10)
    if highs.is_monotonic_decreasing and lows.is_monotonic_decreasing:
        if (lows.max() - lows.min()) > (highs.max() - highs.min()):
            return "falling_wedge"
    return None

def detect_symmetrical_triangle(df):
    if len(df) < 20:
        return None
    highs = df["high"].tail(20)
    lows = df["low"].tail(20)
    if highs.iloc[-1] < highs.max() and lows.iloc[-1] > lows.min():
        return "symmetrical_triangle"
    return None

def detect_ascending_triangle(df):
    if len(df) < 20:
        return None
    highs = df["high"].tail(20)
    lows = df["low"].tail(20)
    resistance = highs.max()
    if all(abs(h - resistance) < 0.01 * resistance for h in highs) and lows.is_monotonic_increasing:
        return "ascending_triangle"
    return None

def detect_descending_triangle(df):
    if len(df) < 20:
        return None
    lows = df["low"].tail(20)
    highs = df["high"].tail(20)
    support = lows.min()
    if all(abs(l - support) < 0.01 * support for l in lows) and highs.is_monotonic_decreasing:
        return "descending_triangle"
    return None

def detect_triple_top(df):
    if len(df) < 60:
        return None
    highs = df["high"].tail(60)
    peaks = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    peaks = peaks[abs(peaks - peaks.mean()) < 0.01 * peaks.mean()]
    if len(peaks) >= 3:
        return "triple_top"
    return None

def detect_triple_bottom(df):
    if len(df) < 60:
        return None
    lows = df["low"].tail(60)
    troughs = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    troughs = troughs[abs(troughs - troughs.mean()) < 0.01 * troughs.mean()]
    if len(troughs) >= 3:
        return "triple_bottom"
    return None

def detect_hammer(df):
    if len(df) < 1:
        return None
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    lower_shadow = min(candle["open"], candle["close"]) - candle["low"]
    upper_shadow = candle["high"] - max(candle["open"], candle["close"])
    if lower_shadow > 2 * body and upper_shadow < body:
        return "hammer"
    return None

def detect_shooting_star(df):
    if len(df) < 1:
        return None
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    lower_shadow = min(candle["open"], candle["close"]) - candle["low"]
    upper_shadow = candle["high"] - max(candle["open"], candle["close"])
    if upper_shadow > 2 * body and lower_shadow < body:
        return "shooting_star"
    return None