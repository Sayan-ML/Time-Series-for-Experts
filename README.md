# Time Series Analysis & Modeling Automation

This Streamlit app provides an interactive environment for comprehensive time series analysis and forecasting. It includes various statistical tests, visualizations, and modeling options to help users understand their data and make informed modeling decisions.

## Features

- **Stationarity and Seasonality Tests**:  
  - Augmented Dickey-Fuller (ADF) test  
  - KPSS test  
  - Seasonal decomposition using STL  

- **Visual Diagnostics**:  
  - STL decomposition plot  
  - ACF and PACF plots  
  - Rolling statistics (mean and standard deviation)  

- **White Noise Detection**:  
  - Ljung-Box test to check if residuals are white noise  

- **Modeling**:  
  - Support for ARIMA and SARIMA models  
  - User-defined hyperparameters based on domain knowledge  

- **Stock Data Integration**:  
  - Connects to Yahoo Finance API via `yfinance`  
  - Users can select from various US stocks to download historical monthly data  
  - Perform the same analysis and forecasting on stock price data  

## How to Use

1. Upload your **monthly** time series data in the specified format or select a US stock ticker to fetch historical monthly data.  
2. Review the diagnostic plots and statistical test results to assess stationarity and seasonality.  
3. Use the insights along with your domain knowledge to select appropriate ARIMA/SARIMA hyperparameters (p, d, q, seasonal order).  
4. Run the model and evaluate forecasting performance.

## Requirements

See [requirements.txt](requirements.txt) for the list of required packages.

## Notes

- This app is designed to work with **monthly** data only.  
- It does not automatically tune ARIMA/SARIMA hyperparameters — user input is required.  
- The app focuses on transparency and user control during modeling.

## License

MIT License
