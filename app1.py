import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend import (
    log_transform,
    boxcox_transform,
    first_order_difference,
    second_order_difference,
    seasonal_difference,
    check_stationarity,
    sqrt_transform,
    linear_detrend,
    moving_average_transform
)

import yfinance as yf 
st.set_page_config(layout="wide")
st.title("📈 Time Series AutoML Pipeline")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'log_transformed' not in st.session_state:
    st.session_state.log_transformed = False

uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Select Columns")
    date_col = st.selectbox("Select the Date Column", df.columns)
    value_col = st.selectbox("Select the Value Column", [col for col in df.columns if col != date_col])

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        df = df[[value_col]].dropna().rename(columns={value_col: "value"})
        st.session_state.df = df.copy()
    except Exception as e:
        st.error(f"Error processing date column: {e}")
        st.session_state.df = None

if st.session_state.df is not None:
    st.subheader("Loaded Time Series Data")
    st.dataframe(st.session_state.df)

if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df

    st.subheader("📊 Data Summary")
    st.write("Shape:", df.shape)
    st.write("Start Date:", df.index.min().date())
    st.write("End Date:", df.index.max().date())

    st.subheader("📟 Time Series Plot")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["value"])
    ax.set_title("Time Series")
    st.pyplot(fig)

    st.subheader("🔀 Rolling Standard Deviation")
    window = st.slider("Select Rolling Window Size", min_value=2, max_value=60, value=12)
    rolling_std = df["value"].rolling(window=window).std()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["value"], label='Original')
    ax.plot(rolling_std, label=f'Rolling Std (window={window})')
    # Place legend outside the right center
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.subplots_adjust(right=0.8)  # Make room on the right side for the legend
    st.pyplot(fig)

    st.subheader("🔁 Seasonality Check")
    try:
        stl = STL(df["value"].squeeze(), period=12, robust=True)
        result = stl.fit()
        seasonal_strength = result.seasonal.var() / result.resid.var()

        fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(result.observed); axs[0].set_title("Observed")
        axs[1].plot(result.trend); axs[1].set_title("Trend")
        axs[2].plot(result.seasonal); axs[2].set_title("Seasonal")
        axs[3].plot(result.resid); axs[3].set_title("Residual")
        st.pyplot(fig)

        if seasonal_strength > 0.6:
            st.success(f"✅ Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
        else:
            st.info(f"ℹ No Strong Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
    except Exception as e:
        st.error(f"Seasonality check failed: {e}")

    def display_stationarity(series):
        adf_stat, kpss_stat = check_stationarity(series.dropna())
        if adf_stat and kpss_stat:
            st.success("✅ Series is Stationary (ADF + KPSS tests passed)")
        else:
            st.warning("⚠ Series is NOT Stationary (ADF or KPSS test failed)")

    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        display_stationarity(df["value"])  # Display stationarity status to user

transformations = [
    "Log Transform",
    "Box-Cox Transform",
    "First-order Differencing",
    "Second-order Differencing",
    "Seasonal Differencing",
    "Linear Detrending",
    "Square Root Transform",
    "Moving Average Smoothing"
]

def apply_transformations(series, transformations, season_length=None, ma_window=3):
    transformed_series = series.copy()
    boxcox_lambda = None  # To keep lambda for inverse if needed
    linear_detrend_params = None  # To keep params for inverse linear detrend

    for transform in transformations:
        if transform == "Log Transform":
            transformed_series = log_transform(transformed_series)
        elif transform == "Box-Cox Transform":
            transformed_series, boxcox_lambda = boxcox_transform(transformed_series)
        elif transform == "Square Root Transform":
            transformed_series = sqrt_transform(transformed_series)
        elif transform == "Linear Detrending":
            transformed_series, linear_detrend_params = linear_detrend(transformed_series)
        elif transform == "Moving Average Smoothing":
            transformed_series = moving_average_transform(transformed_series, window=ma_window)
        elif transform == "First-order Differencing":
            transformed_series = first_order_difference(transformed_series)
        elif transform == "Second-order Differencing":
            transformed_series = second_order_difference(transformed_series)
        elif transform == "Seasonal Differencing":
            if season_length is None:
                raise ValueError("Season length must be provided for seasonal differencing")
            transformed_series = seasonal_difference(transformed_series, season_length)
        else:
            raise ValueError(f"Unknown transformation: {transform}")

    return transformed_series, boxcox_lambda, linear_detrend_params


selected_transforms = st.multiselect(
    "Choose transformations:",
    options=transformations,
    default=st.session_state.get("transform_multiselect", []),
    key="transform_multiselect"
)

season_length = None
if "Seasonal Differencing" in selected_transforms:
    season_length = st.number_input(
        "Enter season length for Seasonal Differencing",
        min_value=1,
        value=st.session_state.get("season_length_input", 12),
        key="season_length_input"
    )

ma_window = None
if "Moving Average Smoothing" in selected_transforms:
    ma_window = st.number_input(
        "Enter window size for Moving Average",
        min_value=1,
        max_value=30,
        value=3,
        key="ma_window_input"
    )

if st.button("Submit Transformations", key="submit_button"):
    if selected_transforms:
        try:
            transformed_series, boxcox_lambda, linear_detrend_params = apply_transformations(
                st.session_state.df['value'], selected_transforms, season_length=season_length, ma_window=ma_window
            )
            # Save transformed series in session state
            st.session_state['transformed_series'] = transformed_series.dropna()
            st.session_state['boxcox_lambda'] = boxcox_lambda
            st.success("Transformations applied successfully!")
        except Exception as e:
            st.error(f"Error applying transformations: {e}")
    else:
        st.write("No transformations selected")

# Outside button, render plots if transformed data exists
if 'transformed_series' in st.session_state and st.session_state['transformed_series'] is not None:
    df_t = st.session_state['transformed_series'].to_frame(name='value')

    st.subheader("📏 Stationarity Check of Transformed Series")
    display_stationarity(df_t["value"])

    st.subheader("📟 Time Series Plot (Transformed Data)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df_t["value"])
    ax.set_title("Time Series (Transformed)")
    st.pyplot(fig)

    st.subheader("🔀 Rolling Standard Deviation (Transformed Data)")
    window = st.slider("Select Rolling Window Size (Transformed)", min_value=2, max_value=60, value=12, key='rolling_window_transformed')
    rolling_std = df_t["value"].rolling(window=window).std()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df_t.index, df_t["value"], label='Transformed')
    ax.plot(rolling_std, label=f'Rolling Std (window={window})')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.subplots_adjust(right=0.8)
    st.pyplot(fig)

    st.subheader("🔁 Seasonality Check (Transformed Data)")
    try:
        stl = STL(df_t["value"].squeeze(), period=12, robust=True)
        result = stl.fit()
        seasonal_strength = result.seasonal.var() / result.resid.var()

        fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(result.observed); axs[0].set_title("Observed")
        axs[1].plot(result.trend); axs[1].set_title("Trend")
        axs[2].plot(result.seasonal); axs[2].set_title("Seasonal")
        axs[3].plot(result.resid); axs[3].set_title("Residual")
        st.pyplot(fig)

        if seasonal_strength > 0.6:
            st.success(f"✅ Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
        else:
            st.info(f"ℹ No Strong Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
    except Exception as e:
        st.error(f"Seasonality check failed: {e}")

    st.subheader("📊 ACF and PACF Plots (Transformed Data)")
    max_lags = st.slider("Select number of lags (Transformed)", min_value=10, max_value=60, value=40, key="acf_pacf_lags")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    acf_vals = acf(df_t["value"], nlags=max_lags)
    pacf_vals = pacf(df_t["value"], nlags=max_lags)
    conf = 1.96 / np.sqrt(len(df_t))

    ax[0].stem(range(len(acf_vals)), acf_vals, basefmt=" ", linefmt='b-', markerfmt='bo')
    ax[0].fill_between(range(len(acf_vals)), -conf, conf, color='blue', alpha=0.2)
    ax[0].axhline(0, linestyle='--', color='gray')
    ax[0].set_title("ACF")

    ax[1].stem(range(len(pacf_vals)), pacf_vals, basefmt=" ", linefmt='b-', markerfmt='bo')
    ax[1].fill_between(range(len(pacf_vals)), -conf, conf, color='blue', alpha=0.2)
    ax[1].axhline(0, linestyle='--', color='gray')
    ax[1].set_title("PACF")

    st.pyplot(fig)

    if "Moving Average Smoothing" in selected_transforms:
        ma_window = st.session_state.get("ma_window_input", 3)  # or default 3 if not set
        moving_avg = df_t["value"].rolling(window=ma_window).mean()

        st.subheader(f"📈 Moving Average Plot (window={ma_window})")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df_t.index, df_t["value"], label="Transformed")
        ax.plot(moving_avg, label=f"Moving Average (window={ma_window})", color='orange')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.subplots_adjust(right=0.8)
        st.pyplot(fig)

st.subheader("📈 ARIMA & SARIMA Modeling")

# Only proceed if original or transformed data is loaded
if ('transformed_series' in st.session_state and st.session_state.transformed_series is not None and not st.session_state.transformed_series.empty):
    modeling_series = st.session_state.transformed_series
elif (st.session_state.df is not None and not st.session_state.df.empty):
    modeling_series = st.session_state.df['value']
else:
    st.info("Please upload and transform data first before modeling.")
    modeling_series = None

def inverse_transformations(transformed_series, applied_transforms, boxcox_lambda=None, linear_detrend_params=None):
    inv_series = transformed_series.copy()

    # Inverse order of transformations applied
    for transform in reversed(applied_transforms):
        if transform == "Log Transform":
            inv_series = np.exp(inv_series)
        elif transform == "Box-Cox Transform":
            from scipy.special import inv_boxcox
            inv_series = inv_boxcox(inv_series, boxcox_lambda)
        elif transform == "Square Root Transform":
            inv_series = inv_series ** 2
        elif transform == "Linear Detrending":
            slope, intercept = linear_detrend_params
            time_idx = np.arange(len(inv_series))
            inv_series = inv_series + (slope * time_idx + intercept)
        elif transform == "Moving Average":
            pass
    return inv_series


if modeling_series is not None:
    test_start_date = '2022-12-01'
    original_series = st.session_state.df['value']

    train_original = original_series.loc[:test_start_date]
    test_original = original_series.loc[test_start_date:]

    # Remove differencing transforms before applying on train
    filtered_transforms = [
        t for t in selected_transforms
        if t not in ("First-order Differencing", "Second-order Differencing", "Seasonal Differencing")
    ]

    # Apply filtered transforms on train data only
    transformed_train, boxcox_lambda, linear_detrend_params = apply_transformations(train_original, filtered_transforms, season_length=season_length, ma_window=ma_window)

    st.subheader("Transformed Training Data Preview")
    st.write("Head:")
    st.write(transformed_train.head())
    st.write("Tail:")
    st.write(transformed_train.tail())

    model_type = st.selectbox("Select Model Type", ["ARIMA", "SARIMA"])
    p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1)
    d = st.number_input("d (Difference order)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1)

    if model_type == "SARIMA":
        seasonal_order = (
            st.number_input("P", min_value=0, max_value=5, value=1),
            st.number_input("D", min_value=0, max_value=2, value=1),
            st.number_input("Q", min_value=0, max_value=5, value=1),
            st.number_input("Seasonal Period (s)", min_value=1, max_value=60, value=12)
        )

    if st.button("Train Model"):
        try:
            # Use d and D as usual since differencing NOT applied manually here
            if model_type == "ARIMA":
                model = ARIMA(transformed_train, order=(p, d, q)).fit()
                forecast = model.predict(start=test_original.index[0], end=test_original.index[-1], typ='levels')
            else:
                model = SARIMAX(transformed_train, order=(p, d, q), seasonal_order=seasonal_order).fit()
                forecast = model.predict(start=test_original.index[0], end=test_original.index[-1], dynamic=True)


            st.session_state['trained_model'] = model
            st.session_state['model_type'] = model_type
            st.session_state['filtered_transforms'] = filtered_transforms
            st.session_state['boxcox_lambda'] = boxcox_lambda
            st.session_state['linear_detrend_params'] = linear_detrend_params

            # Inverse transform forecast back to original scale
            inv_forecast = inverse_transformations(forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)

            st.session_state['inv_forecast'] = inv_forecast
            st.session_state['test_index'] = test_original.index

            # Calculate metrics on original test data and inverse transformed forecast
            mae = mean_absolute_error(test_original, inv_forecast)
            rmse = np.sqrt(mean_squared_error(test_original, inv_forecast))
            r2 = r2_score(test_original, inv_forecast)
            mape = np.mean(np.abs((test_original - inv_forecast) / test_original)) * 100

            st.session_state['mae'] = mae
            st.session_state['rmse'] = rmse
            st.session_state['mape'] = mape
            st.session_state['r2'] = r2

            st.success("Model trained successfully!")

        except Exception as e:
            st.error(f"Model training or forecasting failed: {e}")

if 'inv_forecast' in st.session_state:
    st.write(f"*MAE*: {st.session_state['mae']:.2f}")
    st.write(f"*RMSE*: {st.session_state['rmse']:.2f}")
    st.write(f"*MAPE*: {st.session_state['mape']:.2f}%")
    st.write(f"*R² Score*: {st.session_state['r2']:.4f}")

    st.subheader("📉 Actual vs Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state['test_index'], test_original, label='Actual')
    ax.plot(st.session_state['test_index'], st.session_state['inv_forecast'], label='Forecast', linestyle='--')
    ax.legend()
    st.pyplot(fig)

forecast_months = st.number_input(
    "Enter number of months to forecast beyond test data",
    min_value=1,
    max_value=60,
    value=12,
    help="How many future months to forecast?"
)

# Generate Future Forecast button
if st.button("Generate Future Forecast"):
    try:
        if 'trained_model' not in st.session_state:
            st.error("Please train the model first before forecasting.")
        else:
            model = st.session_state['trained_model']
            model_type = st.session_state['model_type']
            filtered_transforms = st.session_state['filtered_transforms']
            boxcox_lambda = st.session_state['boxcox_lambda']
            linear_detrend_params = st.session_state['linear_detrend_params']

            last_test_date = test_original.index[-1]
            future_index = pd.date_range(start=last_test_date + pd.offsets.MonthBegin(1), periods=forecast_months, freq='MS')

            # Forecast future with ARIMA/SARIMA
            future_forecast_obj = model.get_forecast(steps=forecast_months)
            future_forecast = future_forecast_obj.predicted_mean
            future_forecast.index = future_index
            inv_future_forecast = inverse_transformations(future_forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)

            st.subheader(f"Forecast for next {forecast_months} months beyond test data (ARIMA/SARIMA)")
            st.line_chart(inv_future_forecast)

            # Holt-Winters forecast
            hw_model = ExponentialSmoothing(
                transformed_train,
                seasonal_periods=season_length,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            ).fit()

            hw_forecast = hw_model.forecast(forecast_months)
            hw_forecast.index = future_index
            inv_hw_forecast = inverse_transformations(hw_forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)

            st.subheader(f"Forecast for next {forecast_months} months beyond test data (Holt-Winters)")
            st.line_chart(inv_hw_forecast)

    except Exception as e:
        st.error(f"Future forecasting failed: {e}")