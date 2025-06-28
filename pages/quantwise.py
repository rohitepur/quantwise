# quantwise.py

import glob
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math, time
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential # type: ignore
#from keras.layers import LSTM, Dense, Dropout # type: ignore
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
import joblib

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

st.write("""
### Overview
This project predicts 5-day stock price movement using machine learning.  
We engineered technical indicators such as RSI, MACD, and rolling stats, and trained models (XGBoost, Random Forest, LSTM).  

### Problem Statement
How can we predict the short-term price movement of public stocks using historical price trends and momentum indicators?

### Approach
- Collected 1-year historical data using `yfinance`
- Engineered features from TA (technical analysis)
- Trained models per ticker and saved the best one based on R²
- Visualized forecasts with Streamlit

### Try it below ⬇️
""")

feature_defs = {
    'RSI': lambda df: RSIIndicator(df['Close']).rsi(),
    'MACD': lambda df: MACD(df['Close']).macd_diff(),
    'MA_10': lambda df: df['Close'].rolling(window=10).mean(),
    'Rolling_Volume_10': lambda df: df['Volume'].rolling(window=10).mean(),
    'Daily_Return': lambda df: df['Close'].pct_change(),
    'Log_Return': lambda df: np.log(df['Close'] / df['Close'].shift(1)),
    'Rolling_Std_10': lambda df: df['Close'].rolling(window=10).std(),
    'Lag_1': lambda df: df['Close'].shift(1),
    'Lag_5': lambda df: df['Close'].shift(5),
    'Lag_10': lambda df: df['Close'].shift(10),
}

target_feature = 'Future_5D_Log_Return'

#@st.cache_data

def detect_event_days_single(data, ticker, return_window=20, volume_multiplier=1.5, z_threshold=2.5):
    df = data[ticker][['Close', 'Volume']].copy().dropna()
    df['Ticker'] = ticker
    df['Return'] = df['Close'].pct_change()
    df['Return_Mean'] = df['Return'].rolling(return_window).mean()
    df['Return_Std'] = df['Return'].rolling(return_window).std()
    df['Z_Return'] = (df['Return'] - df['Return_Mean']) / df['Return_Std']
    df['Volume_SMA'] = df['Volume'].rolling(return_window).mean()
    df['Volume_Anomaly'] = df['Volume'] > volume_multiplier * df['Volume_SMA']
    df['Return_Anomaly'] = df['Z_Return'].abs() > z_threshold
    df['Event_Day'] = df['Return_Anomaly'] & df['Volume_Anomaly']
    return df

def prep_data(tickers):
    data = yf.download(tickers, start="2018-01-01", group_by='ticker')
    all_event_days = []
    for ticker in data.columns.levels[0]:
        try:
            df = detect_event_days_single(data, ticker)
            all_event_days.append(df[df['Event_Day']])
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    combined_event_days = pd.concat(all_event_days)
    data.index = pd.to_datetime(data.index)

    for ticker in data.columns.levels[0]:
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[(ticker, col)] = pd.to_numeric(data[(ticker, col)], errors='coerce')

    data = data.ffill()
    combined_event_days['Close'] = pd.to_datetime(combined_event_days['Close'])
    event_index = pd.MultiIndex.from_frame(combined_event_days[['Close', 'Ticker']])
    event_index.names = ['Date', 'Ticker']
    stacked = data.stack(level=0, future_stack=True)
    stacked_filtered = stacked[~stacked.index.isin(event_index)]
    data_cleaned = stacked_filtered.unstack(level=1)
    data_cleaned.columns = data_cleaned.columns.swaplevel(0, 1)
    data_cleaned = data_cleaned.sort_index(axis=1)
    data_cleaned.columns.names = ['Ticker', 'Price']
    return data_cleaned

def train_and_save_model(data_cleaned):

    def prepare_lstm_data(df, features, target, seq_len=10):
        data = df[features + [target]].dropna().copy()
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[features].iloc[i:i+seq_len].values)
            y.append(data[target].iloc[i+seq_len])
        return np.array(X), np.array(y)

 
    model_results = {}

    for ticker in data_cleaned.columns.levels[0]:
        try:
            df = data_cleaned[ticker].copy()
            for feat_name, func in feature_defs.items():
                df[feat_name] = func(df)

            df[target_feature] = np.log(df['Close'].shift(-5) / df['Close'])

            st.dataframe(df.tail(10))
            st.write("before feature engineering and dropna:", df.shape)

            #df.dropna(subset=list(feature_defs.keys()) + [target_feature], inplace=True)

            st.dataframe(data_cleaned.tail(10))
            st.write("after feature engineering and dropna:", df.shape)

            # Safety check after dropna
            if df.empty or df.shape[0] < 10:
                st.warning(f"⚠️ Skipping {ticker} — no rows left after feature engineering.")
                continue


            corr = df[list(feature_defs.keys()) + [target_feature]].corr()[target_feature].abs()
            top_features = corr.drop(target_feature).sort_values(ascending=False).head(3).index.tolist()
            if not top_features:
                st.write(f"No correlated features for {ticker}, skipping.")
                continue

            X_tabular = df[top_features].values
            y_tabular = df[target_feature].values
            X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(X_tabular, y_tabular, test_size=0.2, random_state=42)

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }

            model_results[ticker] = {}
            for name, model in models.items():
                model.fit(X_train_tab, y_train_tab)
                y_pred = model.predict(X_test_tab)
                sign_true = np.sign(np.diff(y_test_tab))
                sign_pred = np.sign(np.diff(y_pred))
                da = np.mean(sign_true == sign_pred)
                model_results[ticker][name] = {
                    'model': model,
                    'y_true': y_test_tab,
                    'y_pred': y_pred,
                    'mse': mean_squared_error(y_test_tab, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test_tab, y_pred)),
                    'r2': r2_score(y_test_tab, y_pred),
                    'mape': mean_absolute_percentage_error(y_test_tab, y_pred),
                    'da': da
                }

            if st.checkbox(f"LSTM model training disabled for {ticker}", value=False, key=f"lstm_checkbox_{ticker}"):

                # LSTM model
                scaler = MinMaxScaler()
                df[top_features] = scaler.fit_transform(df[top_features])
                X_lstm, y_lstm = prepare_lstm_data(df, top_features, target_feature, seq_len=10)
                split = int(len(X_lstm) * 0.8)
                X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
                y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
    
                lstm_model = Sequential()
                lstm_model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(1))
                lstm_model.compile(loss='mse', optimizer='adam')
                lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=32, verbose=0)
    
                y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
                sign_true_lstm = np.sign(np.diff(y_test_lstm))
                sign_pred_lstm = np.sign(np.diff(y_pred_lstm))
                da_lstm = np.mean(sign_true_lstm == sign_pred_lstm)
    
                model_results[ticker]['LSTM'] = {
                    'model': lstm_model,
                    'y_true': y_test_lstm,
                    'y_pred': y_pred_lstm,
                    'mse': mean_squared_error(y_test_lstm, y_pred_lstm),
                    'rmse': np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm)),
                    'r2': r2_score(y_test_lstm, y_pred_lstm),
                    'mape': mean_absolute_percentage_error(y_test_lstm, y_pred_lstm),
                    'da': da_lstm
                }

            st.write(f"{ticker}: All models trained using top features: {top_features}")

        except Exception as e:
            st.write(f"Skipping {ticker} due to error: {e}")

    return model_results

def save_best_models_from_results(model_results, output_dir='saved_models'):
    os.makedirs(output_dir, exist_ok=True)
    for ticker, models in model_results.items():
        best_model_name = max(models, key=lambda name: models[name].get('r2', float('-inf')))
        best_model_info = models[best_model_name]
        model = best_model_info['model']
        results = model_results[ticker][best_model_name]

        y_true = results['y_true']
        y_pred = results['y_pred']
        num_days=22
        test_size=0.2

        # Rebuild aligned date index after feature and target engineering
        df = data_cleaned[ticker].copy()
        for feat_name, func in feature_defs.items():
            df[feat_name] = func(df)
        df[target_feature] = np.log(df['Close'].shift(-5) / df['Close'])
        df.dropna(subset=list(feature_defs.keys()) + [target_feature], inplace=True)

        test_index = int(len(df) * (1 - test_size))
        test_dates = df.index[test_index:]
        test_dates = test_dates[-len(y_true):]

        # Limit to the last `num_days`
        y_true = y_true[-num_days:]
        y_pred = y_pred[-num_days:]
        test_dates = test_dates[-num_days:]

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(test_dates, y_true, label='Actual', marker='o')
        plt.plot(test_dates, y_pred, label='Predicted', marker='x')
        plt.title(f"{ticker} - {best_model_name} Prediction vs Actual (Last {num_days} Days)")
        plt.xlabel("Date")
        plt.ylabel("5-Day Log Return")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

        model_path = os.path.join(output_dir, f"{ticker}_{best_model_name}.joblib")
        try:
            joblib.dump(model, model_path)
            st.write(f"Saved {ticker} best model ({best_model_name}) to {model_path}")
        except Exception as e:
            st.write(f"Failed to save model for {ticker}: {e}")

def load_data(ticker):
    df = yf.download(ticker, period="36mo", auto_adjust=True)
    df['Ticker'] = ticker
    return df

def predict_next_5_days_best_model(ticker, model_results, df_raw, feature_defs):

    model_pattern = os.path.join("saved_models", f"{ticker}_*.joblib")
    model_files = glob.glob(model_pattern)

    if not model_files:
        st.warning(f"[{ticker}] No saved models found.")
        return None

    model_path = model_files[0]
    model_name = os.path.basename(model_path).replace(f"{ticker}_", "").replace(".joblib", "")

    try:
        model = joblib.load(model_path)
        st.write(f"✅ Loaded model `{model_name}` for {ticker}")
    except Exception as e:
        st.error(f"[{ticker}] Failed to load model: {e}")
        return None

    # Step 1: Feature Engineering
    df = df_raw.copy()
    for feat_name, func in feature_defs.items():
        try:
            df[feat_name] = func(df)
        except Exception as e:
            st.warning(f"[{ticker}] Feature '{feat_name}' failed: {e}")
            return None

    # Step 2: Correlation-based feature selection (no target needed at prediction time)
    df_corr = df[list(feature_defs.keys())].dropna()
    if df_corr.empty:
        st.warning(f"[{ticker}] Not enough data for correlation.")
        return None

    # Estimate target temporarily to compute correlation
    df_temp = df.copy()
    df_temp['Future_5D_Log_Return'] = np.log(df_temp['Close'].shift(-5) / df_temp['Close'])
    df_temp = df_temp[list(feature_defs.keys()) + ['Future_5D_Log_Return']].dropna()

    if df_temp.empty:
        st.warning(f"[{ticker}] Not enough data for correlation with target.")
        return None

    corr = df_temp.corr()['Future_5D_Log_Return'].abs()
    top_features = corr.drop('Future_5D_Log_Return').sort_values(ascending=False).head(3).index.tolist()

    # Step 3: Use the **latest row** with complete feature data
    df_pred = df[top_features].dropna()
    if df_pred.empty:
        st.warning(f"[{ticker}] No recent row with complete top features.")
        return None

    latest_features = df_pred.iloc[-1:].values

    # Step 4: Predict and extrapolate prices
    try:
        log_return_pred = model.predict(latest_features)[0]
    except Exception as e:
        st.error(f"[{ticker}] Model prediction failed: {e}")
        return None

    last_close = df['Close'].iloc[-1]
    future_prices = [last_close * np.exp(log_return_pred * (i + 1) / 5) for i in range(5)]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')

    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices})

# Normalize close prices for weekly trend
def plot_weekly_normalized(data_dict):
    st.subheader("Normalized Weekly Closing Price (Last 3 years)")
    plt.figure(figsize=(14,7))
    for ticker, df in data_dict.items():
        weekly_close = df['Close'].resample('W').last()
        normalized = weekly_close / weekly_close.iloc[0] * 100
        plt.plot(normalized, label=ticker)
        plt.text(normalized.index[-1], normalized.iloc[-1], ticker,
            fontsize=10, fontweight='light', va='center', ha='left')
    #plt.legend()
    # Format x-axis to show quarterly ticks
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Risk vs return analysis
def analyze_and_plot_risk(data_dict):
    st.subheader("60-Day Risk vs Return")
    returns = []
    for ticker, df in data_dict.items():
        # 2. Compute daily return
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        # 3. Risk and distribution metrics
        mean_return = df['Return'].mean()
        vol_risk = df['Return'].std()
        VaR_95 = -np.percentile(df['Return'], 5)
        CVaR_95 = -df[df['Return'] < -VaR_95]['Return'].mean()
        skew = df['Return'].skew()
        kurt = df['Return'].kurtosis()

        # 4. Print summary
        st.write(f"##### {ticker} - Risk Metrics (Last 60 Trading Days)")

        st.markdown(f"""
        - **Mean Return** *(expected return per $1 invested daily)*: `{mean_return:.2%}`  
        - **Std Dev** *(average daily swing in return)*: `{vol_risk:.2%}`  
        - **95% VaR** *(95% confidence you won’t lose more than this per day)*: `{VaR_95:.2%}`  
        - **Conditional VaR** *(average loss on very bad days)*: `{CVaR_95:.2%}`  
        - **Skewness** *(negative = more downside risk)*: `{skew:.2f}`  
        - **Kurtosis** *(fat tails = bigger surprise losses)*: `{kurt:.2f}`  
        ---
        **→ Risk Summary**: ±`{vol_risk:.2%}` per $1 | **VaR**: `{VaR_95:.2%}` | **CVaR**: `{CVaR_95:.2%}`
        """)
    
        # 5. Plot histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Return'], bins=40, kde=True, color='lightblue', edgecolor='black')
        plt.axvline(mean_return, color='blue', linestyle='--', label=f'Mean: {mean_return:.2%}')
        plt.axvline(mean_return + vol_risk, color='green', linestyle='--', label=f'+1 SD: {mean_return + vol_risk:.2%}')
        plt.axvline(mean_return - vol_risk, color='red', linestyle='--', label=f'-1 SD: {mean_return - vol_risk:.2%}')
        plt.axvline(-VaR_95, color='purple', linestyle='-', label=f'VaR 95%: {-VaR_95:.2%}')
        plt.title(f'{ticker} - Return Distribution & Risk (Last 60 Days)')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

# Title
st.title("QuantWise: Multi-Ticker Stock Analyzer")
st.write("Data-driven wisdom for stock Market Investors!")

# User input
user_input = st.text_input("Enter tickers separated by commas (e.g. AAPL,MSFT,GOOG)")
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
data_dict = {ticker: load_data(ticker) for ticker in tickers}

# ---- Tabs for Layout ----
tab1, tab2, tab3 = st.tabs(["📂 Risk v/s Return", "📈 5 day Forecasts", "📊 Model Performance"])

if tickers:
    with tab3:
        data_cleaned = prep_data(tickers)
        model_results = train_and_save_model(data_cleaned)
        save_best_models_from_results(model_results)

        test_path = os.path.join(SAVE_DIR, "test_write.txt")

        try:
            with open(test_path, "w") as f:
                f.write("Streamlit can write to this directory!")
            st.success(f"✅ Write test succeeded: {test_path}")
        except Exception as e:
            st.error(f"❌ Write test failed: {e}")

        st.write("Current directory:", os.getcwd())
        st.write("Files in saved_models:", os.listdir("saved_models"))


        rows = []
        for stock, models in model_results.items():
                for model_name, metrics in models.items():
                    mse = metrics['mse']
                    rmse = metrics['rmse']
                    r2 = metrics['r2']
                    mape = metrics['mape']
                    da = metrics['da']
                    rows.append([stock, model_name, mse, rmse, r2, mape, da])
            
        df_summary = pd.DataFrame(rows, columns=["Stock", "Model", "MSE", "RMSE", "R2", "MAPE", "DA"])

        # Find best model per stock (highest R²)
        best_models = df_summary.loc[df_summary.groupby("Stock")["R2"].idxmax()]

        #Display in Streamlit
        st.subheader("Model Performance Summary")
        st.dataframe(df_summary.style.format({
                "MSE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R2": "{:.4f}",
                "MAPE": "{:.4f}",
                "DA": "{:.4f}"
            }))

        st.subheader("Best Models by Stock (based on R²)")
        st.table(best_models.style.format({
                "MSE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R2": "{:.4f}",
                "MAPE": "{:.4f}",
                "DA": "{:.4f}"
            }))

    with tab2:       
        n_days = 22  # How many past days to show
        for ticker in tickers:
            try:
      
                    df_orig = data_cleaned[ticker][['Close']].copy()

                    prediction_df = predict_next_5_days_best_model(
                        ticker=ticker,
                        model_results=model_results,
                        df_raw=data_cleaned[ticker],
                        feature_defs=feature_defs
                    )

                    if prediction_df is None or prediction_df.empty:
                        st.write(f"Skipping {ticker}: prediction not available.")
                        continue

                    prediction_df.set_index('Date', inplace=True)

                    plt.figure(figsize=(10, 5))
                    plt.plot(df_orig['Close'].iloc[-n_days:], label=f'{ticker} - Actual Close (Last {n_days} Days)')
                    plt.plot(prediction_df.index, prediction_df['Predicted_Close'], marker='o', label='Predicted Close (Next 5 Days)')
                    plt.title(f'{ticker} - Stock Price Forecast')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    st.pyplot(plt)
                    st.write(prediction_df)
            except Exception as e:
                st.write(f"Error processing {ticker}: {e}")
    with tab1:
        plot_weekly_normalized(data_dict)
        analyze_and_plot_risk(data_dict)
else:
    st.info("Please enter at least one ticker symbol above.")
