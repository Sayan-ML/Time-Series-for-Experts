import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta

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

# =========================
# 🎨 STREAMLIT UI STYLING
# =========================
st.set_page_config(layout="wide", page_title="📈 Time Series AutoML Pipeline")

st.markdown(
    """
    <style>
    /* Force black background for all elements */
    .main, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #111111 !important;
        color: white !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00E5FF !important;
    }

    /* Text elements */
    .stMarkdown, .stText, p{
        color:  white !important;
    }
    
    /* Text elements */
    div {
        color:  #A49797 !important;
    }
    
    /* DataFrame styling */
    .stDataFrame, .dataframe {
        color: black !important;
        background-color: #111111 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00E5FF;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FF4081;
        color: white;
    }
    
    /* Selectboxes and inputs */
    .stSelectbox > div > div {
        background-color: white;
        color: #A49797;
    }
    
    .stNumberInput > div > div > input {
        background-color: #333333;
        color: white;
        border: 1px solid #555555;
    }
    
    /* Sliders */
    .stSlider > div > div {
        color: black;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #111111;
        color: white;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #333333;
        color: black;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #222222;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    
    /* Success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: white !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #222222;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #222222;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 🎨 MATPLOTLIB DARK MODE
# =========================
matplotlib.rcParams.update({
    "axes.facecolor": "#000000",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "figure.facecolor": "#000000",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.color": "#ADA1A1",
    "text.color": "#FFFFFF",
    "legend.edgecolor": "#FFFFFF",
    "legend.facecolor": "#000000"
})

st.title("📈 Time Series AutoML Pipeline (Dark Mode)")

# =========================
# 📂 DATA SOURCE SELECTION
# =========================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'log_transformed' not in st.session_state:
    st.session_state.log_transformed = False

st.subheader("📊 Choose Data Source")
data_source = st.radio(
    "Select your data source:",
    ["Upload CSV File", "Yahoo Finance Data"],
    horizontal=True
)

# Popular stock symbols for dropdown
POPULAR_SYMBOLS = {
    "Technology": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "NFLX", "AMZN", 'AAL'],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "DHR"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC"],
    "Consumer": ["KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
    "Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
    "Commodities": ["GLD", "SLV", "USO", "DBA", "UNG"]
}

if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Select Columns")
        date_col = st.selectbox("Select the Date Column", df.columns)
        value_col = st.selectbox("Select the Value Column", [col for col in df.columns if col != date_col])

        try:
            # Convert to datetime & remove timezone
            df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df = df[[date_col, value_col]].dropna()
            df.columns = ["Date", "value"]  # Standard naming
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            st.session_state.df = df.copy()
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error processing date column: {e}")
            st.session_state.df = None

else:  # Yahoo Finance Data
    col1, col2 = st.columns([2, 1])

    with col1:
        category = st.selectbox("Select Category", list(POPULAR_SYMBOLS.keys()))
        symbol_options = POPULAR_SYMBOLS[category]
        selected_symbol = st.selectbox("Select Symbol", symbol_options)
        custom_symbol = st.text_input("Or enter custom symbol:", placeholder="e.g., AAPL, BTC-USD")
        final_symbol = custom_symbol.upper() if custom_symbol else selected_symbol

    with col2:
        end_date = st.date_input("End Date", datetime.now().date())
        start_date = st.date_input(
            "Start Date",
            datetime(2000, 1, 1).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime.now().date()
        )
        interval = st.selectbox("Data Interval", ["1d", "1wk", "1mo"])
        price_type = st.selectbox("Price Type", ["Close", "Open", "High", "Low", "Adj Close", "Volume"])

    if st.button("📥 Fetch Data from Yahoo Finance"):
        try:
            with st.spinner(f"Fetching {final_symbol} data..."):
                df_yf = yf.download(final_symbol, start=start_date, end=end_date, interval=interval)

                if df_yf.empty:
                    st.error(f"No data found for symbol: {final_symbol}")
                else:
                    # Keep only selected column
                    df = df_yf[[price_type]].reset_index()
                    df.columns = ["Date", "value"]
                    df.set_index("Date", inplace=True)
                    df.sort_index(inplace=True)

                    # Apply MinMax Scaling
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    df["value"] = scaler.fit_transform(df[["value"]])

                    st.session_state.df = df.copy()
                    st.session_state.symbol_info = {
                        'symbol': final_symbol,
                        'price_type': price_type,
                        'interval': interval,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                    st.success(f"✅ Data fetched & scaled successfully! Shape: {df.shape}")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.session_state.df = None


# =========================
# 📊 DATA SUMMARY & PLOTS
# =========================
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.subheader("📊 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📏 Shape</h4>
        <p>{df.shape[0]} rows × {df.shape[1]} cols</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📅 Start Date</h4>
        <p>{df.index.min().date()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📅 End Date</h4>
        <p>{df.index.max().date()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📊 Latest Value</h4>
        <p>{df['value'].iloc[-1]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Display symbol info if from Yahoo Finance
    if 'symbol_info' in st.session_state:
        info = st.session_state.symbol_info
        st.info(f"📈 **{info['symbol']}** - {info['price_type']} price ({info['interval']} interval)")

    st.subheader("📟 Time Series Plot")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["value"], color="#00E5FF", linewidth=2)
    ax.set_title("Time Series", color="#FFEB3B", fontsize=16)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("🔀 Rolling Standard Deviation")
    window = st.slider("Select Rolling Window Size", min_value=2, max_value=60, value=12)
    rolling_std = df["value"].rolling(window=window).std()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["value"], label='Original', color="#00E5FF", linewidth=2)
    ax.plot(rolling_std, label=f'Rolling Std (window={window})', color="#FF4081", linewidth=2)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(right=0.8)
    st.pyplot(fig)

    st.subheader("🔁 Seasonality Check")
    try:
        stl = STL(df["value"].squeeze(), period=12, robust=True)
        result = stl.fit()
        seasonal_strength = result.seasonal.var() / result.resid.var()
        
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axs[0].plot(result.observed, color="#00E5FF", linewidth=2)
        axs[0].set_title("Observed", color="#FFEB3B", fontsize=14)
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(result.trend, color="#FF4081", linewidth=2)
        axs[1].set_title("Trend", color="#FFEB3B", fontsize=14)
        axs[1].grid(True, alpha=0.3)
        
        axs[2].plot(result.seasonal, color="#8BC34A", linewidth=2)
        axs[2].set_title("Seasonal", color="#FFEB3B", fontsize=14)
        axs[2].grid(True, alpha=0.3)
        
        axs[3].plot(result.resid, color="#FFC107", linewidth=2)
        axs[3].set_title("Residual", color="#FFEB3B", fontsize=14)
        axs[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        if seasonal_strength > 0.6:
            st.success(f"✅ Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
        else:
            st.info(f"ℹ️ No Strong Seasonality Detected. Seasonal Strength = {seasonal_strength:.2f}")
    except Exception as e:
        st.error(f"Seasonality check failed: {e}")

    def display_stationarity(series):
        adf_stat, kpss_stat = check_stationarity(series.dropna())
        if adf_stat and kpss_stat:
            st.success("✅ Series is Stationary (ADF + KPSS tests passed)")
        else:
            st.warning("⚠️ Series is NOT Stationary (ADF or KPSS test failed)")

    st.subheader("📈 Stationarity Check")
    display_stationarity(df["value"])

# =========================
# 🔄 TRANSFORMATIONS
# =========================
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
    boxcox_lambda = None
    linear_detrend_params = None

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

if st.session_state.df is not None:
    st.subheader("🔄 Data Transformations")
    
    selected_transforms = st.multiselect(
        "Choose transformations:",
        options=transformations,
        default=st.session_state.get("transform_multiselect", []),
        key="transform_multiselect",
        help="Select transformations to apply to your time series data"
    )

    season_length = None
    if "Seasonal Differencing" in selected_transforms:
        season_length = st.number_input(
            "Enter season length for Seasonal Differencing",
            min_value=1,
            value=st.session_state.get("season_length_input", 12),
            key="season_length_input",
            help="Number of periods in a season (e.g., 12 for monthly data with yearly seasonality)"
        )

    ma_window = None
    if "Moving Average Smoothing" in selected_transforms:
        ma_window = st.number_input(
            "Enter window size for Moving Average",
            min_value=1,
            max_value=30,
            value=3,
            key="ma_window_input",
            help="Number of periods to include in moving average"
        )

    if st.button("🔄 Apply Transformations", key="submit_button"):
        if selected_transforms:
            try:
                transformed_series, boxcox_lambda, linear_detrend_params = apply_transformations(
                    st.session_state.df['value'], selected_transforms, season_length=season_length, ma_window=ma_window
                )
                st.session_state['transformed_series'] = transformed_series.dropna()
                st.session_state['boxcox_lambda'] = boxcox_lambda
                st.session_state['applied_transforms'] = selected_transforms
                st.session_state['linear_detrend_params'] = linear_detrend_params
                st.success("✅ Transformations applied successfully!")
            except Exception as e:
                st.error(f"❌ Error applying transformations: {e}")
        else:
            st.warning("⚠️ No transformations selected")

    # Display transformed data analysis
    if 'transformed_series' in st.session_state and st.session_state['transformed_series'] is not None:
        df_t = st.session_state['transformed_series'].to_frame(name='value')

        st.subheader("📏 Stationarity Check of Transformed Series")
        display_stationarity(df_t["value"])

        st.subheader("📟 Time Series Plot (Transformed Data)")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_t["value"], color="#FF4081", linewidth=2)
        ax.set_title("Time Series (Transformed)", color="#FFEB3B", fontsize=16)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("🔀 Rolling Standard Deviation (Transformed Data)")
        window = st.slider("Select Rolling Window Size (Transformed)", min_value=2, max_value=60, value=12, key='rolling_window_transformed')
        rolling_std = df_t["value"].rolling(window=window).std()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_t.index, df_t["value"], label='Transformed', color="#FF4081", linewidth=2)
        ax.plot(rolling_std, label=f'Rolling Std (window={window})', color="#00E5FF", linewidth=2)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)
        plt.subplots_adjust(right=0.8)
        st.pyplot(fig)

        st.subheader("📊 ACF and PACF Plots (Transformed Data)")
        max_lags = st.slider("Select number of lags (Transformed)", min_value=10, max_value=60, value=40, key="acf_pacf_lags")

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        acf_vals = acf(df_t["value"], nlags=max_lags)
        pacf_vals = pacf(df_t["value"], nlags=max_lags)
        conf = 1.96 / np.sqrt(len(df_t))

        ax[0].stem(range(len(acf_vals)), acf_vals, basefmt=" ", linefmt='#00E5FF', markerfmt='o')
        ax[0].fill_between(range(len(acf_vals)), -conf, conf, color='#00E5FF', alpha=0.2)
        ax[0].axhline(0, linestyle='--', color='gray')
        ax[0].set_title("ACF", color="#FFEB3B", fontsize=14)
        ax[0].grid(True, alpha=0.3)

        ax[1].stem(range(len(pacf_vals)), pacf_vals, basefmt=" ", linefmt='#FF4081', markerfmt='o')
        ax[1].fill_between(range(len(pacf_vals)), -conf, conf, color='#FF4081', alpha=0.2)
        ax[1].axhline(0, linestyle='--', color='gray')
        ax[1].set_title("PACF", color="#FFEB3B", fontsize=14)
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if "Moving Average Smoothing" in selected_transforms:
            ma_window = st.session_state.get("ma_window_input", 3)
            moving_avg = df_t["value"].rolling(window=ma_window).mean()

            st.subheader(f"📈 Moving Average Plot (window={ma_window})")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df_t.index, df_t["value"], label="Transformed", color="#FF4081", linewidth=2)
            ax.plot(moving_avg, label=f"Moving Average (window={ma_window})", color='#FFC107', linewidth=2)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.grid(True, alpha=0.3)
            plt.subplots_adjust(right=0.8)
            st.pyplot(fig)

# =========================
# 🤖 MODELING SECTION
# =========================
st.subheader("🤖 ARIMA & SARIMA Modeling")

if ('transformed_series' in st.session_state and st.session_state.transformed_series is not None and not st.session_state.transformed_series.empty):
    modeling_series = st.session_state.transformed_series
elif (st.session_state.df is not None and not st.session_state.df.empty):
    modeling_series = st.session_state.df['value']
else:
    st.info("📥 Please upload and transform data first before modeling.")
    modeling_series = None

def inverse_transformations(transformed_series, applied_transforms, boxcox_lambda=None, linear_detrend_params=None):
    inv_series = transformed_series.copy()

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
    return inv_series

if modeling_series is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        if data_source == "Upload CSV File" and st.session_state.df is not None:

            df = st.session_state.df
            end_date = df.index[-1].date()
            min_start_date = end_date - timedelta(days=730)

            test_start_date = st.date_input(
                "📅 Test Set Start Date",
                value=min_start_date,
                min_value=min_start_date,
                max_value=end_date,
                help="Select the date from which test data begins"
            )

            test_start_ts = pd.Timestamp(test_start_date)

            if test_start_ts in df.index:
                train_start_date = test_start_ts.date()
            else:
                # Find next available date after selected date
                future_dates = df.index[df.index > test_start_ts]

                if len(future_dates) == 0:
                    st.warning("No dates in data after selected test start date!")
                    train_start_date = None
                else:
                    train_start_date = future_dates[0].date()
                    st.write(f"Adjusted test start date to nearest data date: **{train_start_date}**")

        else:
            min_start_date = end_date - timedelta(days=730)

            test_start_date = st.date_input(
                "📅 Test Set Start Date",
                value=min_start_date,
                min_value=min_start_date,
                max_value=end_date,
                help="Select the date from which test data begins"
            )
            
            if st.session_state.df is not None:

                df = st.session_state.df
                test_start_ts = pd.Timestamp(test_start_date)

                if test_start_ts in df.index:
                    train_start_date = test_start_ts.date()
                else:
                    future_dates = df.index[df.index > test_start_ts]

                    if len(future_dates) == 0:
                        st.warning("No dates in data after selected test start date!")
                        train_start_date = None
                    else:
                        train_start_date = future_dates[0].date()
                        st.write(f"Adjusted test start date to nearest data date: **{train_start_date}**")
            else:
                train_start_date = test_start_date
        
        model_type = st.selectbox("🔧 Select Model Type", ["ARIMA", "SARIMA"])
        
    with col2:
        st.markdown("**Model Parameters:**")
        p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1)
        d = st.number_input("d (Difference order)", min_value=0, max_value=2, value=1)
        q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1)

    if model_type == "SARIMA":
        st.markdown("**Seasonal Parameters:**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            P = st.number_input("P", min_value=0, max_value=5, value=1)
        with col4:
            D = st.number_input("D", min_value=0, max_value=2, value=1)
        with col5:
            Q = st.number_input("Q", min_value=0, max_value=5, value=1)
        with col6:
            s = st.number_input("s (Seasonal Period)", min_value=1, max_value=60, value=12)
        
        seasonal_order = (P, D, Q, s)

    if st.button("🚀 Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                test_start_date_str = train_start_date.strftime('%Y-%m-%d')
                original_series = st.session_state.df['value']

                train_original = original_series.loc[:test_start_date_str]
                test_original = original_series.loc[test_start_date_str:]

                # Filter out differencing transforms for training
                filtered_transforms = [
                    t for t in st.session_state.get('applied_transforms', [])
                    if t not in ("First-order Differencing", "Second-order Differencing", "Seasonal Differencing")
                ]

                # Apply filtered transforms on train data only
                if filtered_transforms:
                    transformed_train, boxcox_lambda, linear_detrend_params = apply_transformations(
                        train_original, filtered_transforms, 
                        season_length=season_length, ma_window=ma_window
                    )
                else:
                    transformed_train = train_original
                    boxcox_lambda = None
                    linear_detrend_params = None

                # --- Forecasting ---
                if data_source == "Upload CSV File":

                    if model_type == "ARIMA":
                        model = ARIMA(transformed_train, order=(p, d, q)).fit()

                        # Display AIC and BIC
                        st.write(f"📊 Model Information — AIC: **{model.aic:.2f}**, BIC: **{model.bic:.2f}**")

                        # Map test start date to integer position in training index
                        start_pos = transformed_train.index.get_loc(test_original.index[0])
                        end_pos = start_pos + len(test_original) - 1

                        # Predict using integer positions
                        forecast_values = model.predict(
                            start=start_pos,
                            end=end_pos,
                            typ='levels'
                        )

                    elif model_type == "SARIMA":
                        model = SARIMAX(transformed_train, order=(p, d, q), seasonal_order=seasonal_order).fit()

                        # Display AIC and BIC
                        st.write(f"📊 Model Information — AIC: **{model.aic:.2f}**, BIC: **{model.bic:.2f}**")

                        start_pos = transformed_train.index.get_loc(test_original.index[0])
                        end_pos = start_pos + len(test_original) - 1

                        forecast_values = model.predict(
                            start=start_pos,
                            end=end_pos,
                            dynamic=True
                        )

                elif data_source == "Yahoo Finance Data":

                    history = transformed_train.tolist()
                    predictions = []
                    actuals = []

                    step = 10
                    n = len(test_original)

                    for i in range(0, n, step):
                        try:
                            if model_type == "ARIMA":
                                model = ARIMA(history, order=(p, d, q)).fit()
                            elif model_type == "SARIMA":
                                model = SARIMAX(history, order=(p, d, q), seasonal_order=seasonal_order).fit()

                            remaining = n - i
                            current_step = step if remaining >= step else remaining

                            forecast = model.forecast(steps=current_step)
                            predictions.extend(forecast.tolist())

                            actual_chunk = test_original[i:i+current_step].tolist()
                            actuals.extend(actual_chunk)

                            history.extend(actual_chunk)

                        except Exception as e:
                            print(f"Forecast failed at step {i}: {e}")
                            current_step = step if remaining >= step else remaining
                            predictions.extend([np.nan]*current_step)
                            actuals.extend([np.nan]*current_step)

                    forecast_values = predictions

                ## Ensure forecast_values is always a Series
                if not isinstance(forecast_values, pd.Series):
                    forecast = pd.Series(forecast_values, index=test_original.index)
                else:
                    forecast = forecast_values.reindex(test_original.index)


                # Store model and parameters
                st.session_state['trained_model'] = model
                st.session_state['model_type'] = model_type
                st.session_state['filtered_transforms'] = filtered_transforms
                st.session_state['boxcox_lambda'] = boxcox_lambda
                st.session_state['linear_detrend_params'] = linear_detrend_params
                st.session_state['test_original'] = test_original
                st.session_state['train_original'] = train_original

                # Inverse transform forecast
                if filtered_transforms:
                    inv_forecast = inverse_transformations(forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)
                else:
                    inv_forecast = forecast

                st.session_state['inv_forecast'] = inv_forecast
                st.session_state['test_index'] = test_original.index

                # Calculate metrics
                mae = mean_absolute_error(test_original, inv_forecast)
                rmse = np.sqrt(mean_squared_error(test_original, inv_forecast))
                r2 = r2_score(test_original, inv_forecast)
                mape = np.mean(np.abs((test_original - inv_forecast) / test_original)) * 100

                st.session_state['mae'] = mae
                st.session_state['rmse'] = rmse
                st.session_state['mape'] = mape
                st.session_state['r2'] = r2

                st.success("✅ Model trained successfully!")

        except Exception as e:
            st.error(f"❌ Model training failed:")
            st.exception(e)

    # Display results if model is trained
    if 'inv_forecast' in st.session_state:
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
            <h4>📏 MAE</h4>
            <p>{st.session_state['mae']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
            <h4>📊 RMSE</h4>
            <p>{st.session_state['rmse']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
            <h4>📈 MAPE</h4>
            <p>{st.session_state['mape']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
            <h4>🎯 R² Score</h4>
            <p>{st.session_state['r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📉 Actual vs Forecast Comparison")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot actual values
        ax.plot(st.session_state['test_index'], st.session_state['test_original'], 
                label='Actual', color='#00E5FF', linewidth=2, marker='o', markersize=4)
        
        # Plot forecast values
        ax.plot(st.session_state['test_index'], st.session_state['inv_forecast'], 
                label='Forecast', color='#FF4081', linewidth=2, linestyle='--', marker='s', markersize=4)
        
        ax.set_title("Actual vs Forecast Comparison", color="#FFEB3B", fontsize=16)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)
        plt.subplots_adjust(right=0.85)
        st.pyplot(fig)

# =========================
# 🔮 FUTURE FORECASTING
# =========================
st.subheader("🔮 Future Forecasting")

if 'trained_model' in st.session_state:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        forecast_months = st.number_input(
            "📅 Number of periods to forecast beyond test data",
            min_value=1,
            max_value=60,
            value=12,
            help="How many future periods to forecast?"
        )
    
    with col2:
        include_holt_winters = st.checkbox("Include Holt-Winters Forecast", value=True)

    if st.button("🔮 Generate Future Forecast", type="primary"):
        try:
            with st.spinner("Generating forecasts..."):
                model = st.session_state['trained_model']
                model_type = st.session_state['model_type']
                filtered_transforms = st.session_state['filtered_transforms']
                boxcox_lambda = st.session_state['boxcox_lambda']
                linear_detrend_params = st.session_state['linear_detrend_params']
                test_original = st.session_state['test_original']

                # Create future index
                last_test_date = test_original.index[-1]
                freq = pd.infer_freq(st.session_state.df.index) or 'MS'
                future_index = pd.date_range(start=last_test_date + pd.DateOffset(1), periods=forecast_months, freq=freq)

                # ARIMA/SARIMA Forecast
                future_forecast_obj = model.get_forecast(steps=forecast_months)
                future_forecast = future_forecast_obj.predicted_mean
                print(type(future_forecast))
                future_forecast.index = future_index
                
                # Get confidence intervals
                conf_int = future_forecast_obj.conf_int()
                conf_int.index = future_index
                
                # Inverse transform forecast
                if filtered_transforms:
                    inv_future_forecast = inverse_transformations(future_forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)
                    inv_conf_lower = inverse_transformations(conf_int.iloc[:, 0], filtered_transforms, boxcox_lambda, linear_detrend_params)
                    inv_conf_upper = inverse_transformations(conf_int.iloc[:, 1], filtered_transforms, boxcox_lambda, linear_detrend_params)
                else:
                    inv_future_forecast = future_forecast
                    inv_conf_lower = conf_int.iloc[:, 0]
                    inv_conf_upper = conf_int.iloc[:, 1]

                # Store results
                st.session_state['future_forecast'] = inv_future_forecast
                st.session_state['future_index'] = future_index
                st.session_state['conf_lower'] = inv_conf_lower
                st.session_state['conf_upper'] = inv_conf_upper

                # Display ARIMA/SARIMA forecast
                st.subheader(f"📈 {model_type} Forecast for next {forecast_months} periods")
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Plot historical data (last 50 points for context)
                historical_data = st.session_state.df['value'].tail(50)
                ax.plot(historical_data.index, historical_data, label='Historical Data', 
                       color='#00E5FF', linewidth=2, alpha=0.7)
                
                # Plot test data
                ax.plot(test_original.index, test_original, label='Test Data', 
                       color='#FF4081', linewidth=2)
                
                # Plot forecast
                ax.plot(future_index, inv_future_forecast, label=f'{model_type} Forecast', 
                       color='#8BC34A', linewidth=3, marker='o', markersize=4)
                
                # Plot confidence interval
                ax.fill_between(future_index, inv_conf_lower, inv_conf_upper, 
                               color='#8BC34A', alpha=0.2, label='Confidence Interval')
                
                ax.set_title(f"{model_type} Future Forecast", color="#FFEB3B", fontsize=16)
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
                ax.grid(True, alpha=0.3)
                plt.subplots_adjust(right=0.8)
                st.pyplot(fig)

                # Holt-Winters forecast if requested
                if include_holt_winters:
                    try:
                        # Get training data for Holt-Winters
                        if filtered_transforms:
                            hw_train_data = apply_transformations(
                                st.session_state.df['value'].loc[:test_original.index[0]], 
                                filtered_transforms, season_length=season_length, ma_window=ma_window
                            )[0]
                        else:
                            hw_train_data = st.session_state.df['value'].loc[:test_original.index[0]]

                        hw_model = ExponentialSmoothing(
                            hw_train_data,
                            seasonal_periods=season_length or 12,
                            trend='add',
                            seasonal='add',
                            initialization_method="estimated"
                        ).fit()

                        hw_forecast = hw_model.forecast(forecast_months)
                        hw_forecast.index = future_index
                        
                        # Inverse transform Holt-Winters forecast
                        if filtered_transforms:
                            inv_hw_forecast = inverse_transformations(hw_forecast, filtered_transforms, boxcox_lambda, linear_detrend_params)
                        else:
                            inv_hw_forecast = hw_forecast

                        st.subheader(f"📊 Holt-Winters Forecast for next {forecast_months} periods")
                        
                        fig, ax = plt.subplots(figsize=(14, 8))
                        
                        # Plot historical data
                        ax.plot(historical_data.index, historical_data, label='Historical Data', 
                               color='#00E5FF', linewidth=2, alpha=0.7)
                        
                        # Plot test data
                        ax.plot(test_original.index, test_original, label='Test Data', 
                               color='#FF4081', linewidth=2)
                        
                        # Plot Holt-Winters forecast
                        ax.plot(future_index, inv_hw_forecast, label='Holt-Winters Forecast', 
                               color='#FFC107', linewidth=3, marker='s', markersize=4)
                        
                        ax.set_title("Holt-Winters Future Forecast", color="#FFEB3B", fontsize=16)
                        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
                        ax.grid(True, alpha=0.3)
                        plt.subplots_adjust(right=0.8)
                        st.pyplot(fig)

                        # Comparison plot
                        st.subheader("🔄 Model Comparison")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        
                        # Plot historical and test data
                        ax.plot(historical_data.index, historical_data, label='Historical Data', 
                               color='#00E5FF', linewidth=2, alpha=0.7)
                        ax.plot(test_original.index, test_original, label='Test Data', 
                               color='#FF4081', linewidth=2)
                        
                        # Plot both forecasts
                        ax.plot(future_index, inv_future_forecast, label=f'{model_type} Forecast', 
                               color='#8BC34A', linewidth=2, marker='o', markersize=4)
                        ax.plot(future_index, inv_hw_forecast, label='Holt-Winters Forecast', 
                               color='#FFC107', linewidth=2, marker='s', markersize=4)
                        
                        ax.set_title("Forecast Comparison", color="#FFEB3B", fontsize=16)
                        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
                        ax.grid(True, alpha=0.3)
                        plt.subplots_adjust(right=0.8)
                        st.pyplot(fig)

                        st.session_state['hw_forecast'] = inv_hw_forecast

                    except Exception as e:
                        st.warning(f"⚠️ Holt-Winters forecasting failed: {e}")

                st.success("✅ Future forecasts generated successfully!")

        except Exception as e:
            st.error(f"❌ Future forecasting failed:")
            st.exception(e)

# =========================
# 📊 FORECAST SUMMARY TABLE
# =========================
if 'future_forecast' in st.session_state:
    st.subheader("📋 Forecast Summary Table")
    
    # Create summary dataframe
    summary_data = {
        'Date': st.session_state['future_index'].strftime('%Y-%m-%d'),
        f'{st.session_state["model_type"]} Forecast': st.session_state['future_forecast'].round(2),
        'Lower Bound': st.session_state['conf_lower'].round(2),
        'Upper Bound': st.session_state['conf_upper'].round(2)
    }
    
    if 'hw_forecast' in st.session_state:
        summary_data['Holt-Winters Forecast'] = st.session_state['hw_forecast'].round(2)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download button for forecast results
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Results as CSV",
        data=csv,
        file_name=f"time_series_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# =========================
# 📖 FOOTER AND INSTRUCTIONS
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #666;">
    <p>📈 Time Series AutoML Pipeline - Built with Streamlit & ❤️</p>
</div>
""", unsafe_allow_html=True)