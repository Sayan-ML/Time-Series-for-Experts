import numpy as np
import pandas as pd
from scipy.stats import boxcox, boxcox_normmax
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.api import OLS, add_constant

def log_transform(series):
    return np.log(series + 1e-9)

def boxcox_transform(series):
    if (series <= 0).any():
        raise ValueError("Box-Cox transform requires strictly positive values")
    lmbda = boxcox_normmax(series)
    transformed = boxcox(series, lmbda=lmbda)
    return transformed, lmbda

def first_order_difference(series):
    return series.diff().dropna()

def second_order_difference(series):
    return series.diff().diff().dropna()

def seasonal_difference(series, season_length):
    return series.diff(season_length).dropna()

def linear_detrend(series):
    """Remove linear trend using OLS regression"""
    x = np.arange(len(series))
    x = add_constant(x)
    model = OLS(series.values, x).fit()
    detrended = series.values - model.predict(x)
    return pd.Series(detrended, index=series.index), model.params

def sqrt_transform(series):
    if (series < 0).any():
        raise ValueError("Square root transform requires non-negative values")
    return np.sqrt(series)

def moving_average_transform(series, window=3):
    """Apply moving average smoothing with given window size"""
    if window < 1:
        raise ValueError("Window size must be at least 1")
    return series.rolling(window=window, center=False).mean().dropna()

def check_stationarity(series):
    adf_result = adfuller(series)
    kpss_result = kpss(series, regression='c', nlags='auto')
    # Return True if ADF p-value < 0.05 AND KPSS p-value > 0.05 (both tests agree on stationarity)
    return adf_result[1] < 0.05, kpss_result[1] > 0.05