import pandas as pd

def bollinger_bands_percentage(daily_prices, window=20, k=2):
    up = daily_prices.rolling(window=window).mean() + k * daily_prices.rolling(window=window).std()
    lb = daily_prices.rolling(window=window).mean() - k * daily_prices.rolling(window=window).std()
    
    return (daily_prices - lb) / (up - lb)

def relative_strength_index(daily_prices, k=14):
    delta = daily_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=k).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=k).mean()

    return 100 - (100 / (1 + gain / loss))

def simple_moving_average_golden_cross(daily_prices, short_window=50, long_window=200):
    sma_long = daily_prices.rolling(window=long_window).mean()
    sma_short = daily_prices.rolling(window=short_window).mean()
    
    signal = pd.Series(index=daily_prices.index, data=0.0)
    signal[sma_short > sma_long] = 1.0
    signal[sma_short < sma_long] = -1.0

    return signal

def rate_of_change(daily_prices, window=10):
    return (daily_prices / daily_prices.shift(window) - 1)

def moving_average_convergence_divergence(daily_prices, short_window=12, long_window=26, signal_window=9):
    macd = daily_prices.ewm(span=short_window, adjust=False).mean() - daily_prices.ewm(span=long_window, adjust=False).mean()
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    signals = pd.Series(index=macd.index, data=0)

    signals[macd < signal_line] = 1
    signals[macd >= signal_line] = -1

    return signals

def stochastic_oscillator(daily_prices, k=14):
    ll = daily_prices.rolling(window=k).min()
    hh = daily_prices.rolling(window=k).max()

    return 100 * (daily_prices - ll) / (hh - ll)
