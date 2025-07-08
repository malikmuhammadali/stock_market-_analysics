import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import os
from datetime import timedelta
from io import BytesIO  # <-- Added for download button

# Set page configuration
st.set_page_config(page_title="Effects of political Events on Stock Market", layout="wide")

# Load data and model
@st.cache_data
def load_data_and_model():
    df_pak = pd.read_csv("Pakistan Stock Exchange Stock Price History (1).csv")
    df_recent = pd.read_csv('stock_pak.csv')
    
    df_pak['Date'] = pd.to_datetime(df_pak['Date'], format='%m/%d/%Y', errors='coerce')
    df_recent['Date'] = pd.to_datetime(df_recent['Date'], format='%d-%b-%y', errors='coerce')
    
    df_pak = df_pak.rename(columns={'Price': 'Close', 'Vol.': 'Volume', 'Change %': 'Change'})
    
    df_full = pd.concat([df_pak, df_recent], ignore_index=True)
    
    def clean_numeric(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.replace('K', 'e3', regex=False)
            .str.replace('M', 'e6', regex=False),
            errors='coerce'
        )
    
    for col in ['Open', 'High', 'Low', 'Close', 'Change', 'Volume']:
        if col in df_full.columns:
            df_full[col] = clean_numeric(df_full[col])
    
    df_full = df_full.sort_values('Date')
    
    model = load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return df_full, model, scaler

df_full, model, scaler = load_data_and_model()

event_dict = {
    'Elections': {
        'dates': pd.to_datetime(['2013-05-11', '2018-07-25', '2024-02-08']),
        'label': 'Election Day'
    },
    'Budget': {
        'dates': pd.to_datetime([
            '2009-06-13', '2010-06-05', '2011-06-03', '2012-06-01',
            '2013-06-12', '2014-06-03', '2015-06-05', '2016-06-03',
            '2017-05-26', '2018-04-27', '2019-06-11', '2020-06-12',
            '2021-06-11', '2022-06-10', '2023-06-09', '2024-06-07'
        ]),
        'label': 'Budget Day'
    },
    'Geopolitical Events': {
        'dates': pd.to_datetime(['2016-09-29', '2019-02-27', '2021-11-01', '2023-05-09']),
        'comments': {
            '2016-09-29': "India conducted 'surgical strike' after Uri attack",
            '2019-02-27': "Pakistan shot down Indian jet after Balakot airstrike",
            '2021-11-01': "Crackdown on TLP protests by Pakistan govt",
            '2023-05-09': "Imran Khan arrested – massive unrest in Pakistan"
        },
        'label': 'Event Day'
    },
    'IMF Events': {
        'dates': pd.to_datetime([
            '2008-11-24', '2013-09-04', '2019-07-03', '2022-08-29',
            '2023-06-30', '2024-01-11', '2024-04-29'
        ]),
        'comments': {
            '2008-11-24': "IMF approved $7.6B SBA",
            '2013-09-04': "IMF approved $6.6B EFF",
            '2019-07-03': "IMF approved $6B EFF",
            '2022-08-29': "Completion of 7th & 8th EFF reviews, $1.1B disbursed",
            '2023-06-30': "IMF approved $3B SBA",
            '2024-01-11': "IMF completed SBA review, $700M disbursed",
            '2024-04-29': "IMF completed SBA review, $1.1B disbursed"
        },
        'label': 'IMF Event Day'
    }
}

prediction_events = {
    '2028 Election': {'date': pd.to_datetime('2028-08-12'), 'window_days': 30},
    '2025 Budget': {'date': pd.to_datetime('2025-06-07'), 'window_days': 60},
    '2025 IMF Event': {'date': pd.to_datetime('2025-05-09'), 'window_days': 60},
    '2025 Pak-India War': {'date': pd.to_datetime('2025-05-10'), 'window_days': 60}
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Historical Analysis", "Predictions"])

if page == "Historical Analysis":
    st.title("Effects of political Events on Stock Market")
    
    event_type = st.selectbox("Select Event Type", list(event_dict.keys()))
    
    available_years = sorted(set(event_dict[event_type]['dates'].year))
    selected_year = st.selectbox("Select Year", available_years)
        
    selected_dates = event_dict[event_type]['dates'][event_dict[event_type]['dates'].year == selected_year]
        
    if selected_dates.empty:
        st.warning(f"No {event_type} data available for {selected_year}.")
    else:
        for date in selected_dates:
            window_days = 60 if event_type in ['Budget', 'Geopolitical Events', 'IMF Events'] else 180
            start = date - timedelta(days=window_days)
            end = date + timedelta(days=window_days)
            
            window = df_full[(df_full['Date'] >= start) & (df_full['Date'] <= end)].copy()
            
            if len(window) >= 40:
                window = window.set_index('Date').reindex(pd.date_range(start, end), method='nearest')
                window['Close'] = window['Close'].interpolate().bfill().ffill()
                
                if window['Close'].isnull().sum() == 0:
                    close_vals = window['Close'].values
                    normalized = close_vals / close_vals[0] * 100
                    days = np.arange(-window_days, window_days + 1)
                        
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(days, normalized, color='blue', label=f'{event_type} Pattern')
                    ax.axvline(x=0, color='red', linestyle='--', label=event_dict[event_type]['label'])
                    
                    title = f"Stock Price Movement Around {event_type} ({date.date()})"
                    if event_type in ['Geopolitical Events', 'IMF Events'] and str(date.date()) in event_dict[event_type]['comments']:
                        title += f"\n{event_dict[event_type]['comments'][str(date.date())]}"
                    
                    ax.set_title(title, fontsize=14)
                    ax.set_xlabel('Days from Event')
                    ax.set_ylabel('Normalized Stock Price (%)')
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf.getvalue(),
                        file_name="historical_plot.png",
                        mime="image/png"
                    )
                else:
                    st.warning(f"Skipped {date.date()} — too many missing Close values.")
            else:
                st.warning(f"Skipped {date.date()} — insufficient data points in ±{window_days} day window.")
    
else:
    st.title("Pakistan Stock Market Predictions")
    
    prediction_event = st.selectbox("Select Prediction Event", list(prediction_events.keys()))
    
    event_info = prediction_events[prediction_event]
    event_date = event_info['date']
    window_days = event_info['window_days']
    
    if prediction_event == '2028 Election':
        historical_dates = event_dict['Elections']['dates']
        patterns = []
        for date in historical_dates:
            start = date - timedelta(days=window_days)
            end = date + timedelta(days=window_days)
            window = df_full[(df_full['Date'] >= start) & (df_full['Date'] <= end)].copy()
            if len(window) >= 20:
                window = window.set_index('Date').reindex(pd.date_range(start, end), method='nearest')
                close_vals = window['Close'].interpolate().bfill().ffill().values
                if len(close_vals) == 2 * window_days + 1:
                    normalized = close_vals / close_vals[0] * 100
                    patterns.append(normalized)
        
        if patterns:
            avg_pattern = np.mean(patterns, axis=0)
            baseline_price = df_full['Close'].iloc[-1]
            predicted_prices = baseline_price * avg_pattern / 100
            dates = pd.date_range(start=event_date - timedelta(days=window_days), periods=2*window_days + 1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, predicted_prices, color='blue', marker='o', label='Simulated Election Pattern')
            ax.axvline(event_date, color='green', linestyle='--', label='2028 Election Day')
            ax.set_title('Simulated Stock Behavior Around 2028 Election (±30 Days)', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Close Price')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download Plot as PNG",
                data=buf.getvalue(),
                file_name="election_prediction.png",
                mime="image/png"
            )
        else:
            st.warning("Not enough clean past election data to generate prediction.")
    
    elif prediction_event == '2025 Budget':
        historical_dates = event_dict['Budget']['dates']
        patterns = []
        for date in historical_dates:
            start = date - timedelta(days=window_days)
            end = date + timedelta(days=window_days)
            window = df_full[(df_full['Date'] >= start) & (df_full['Date'] <= end)].copy()
            if len(window) >= 40:
                window = window.set_index('Date').reindex(pd.date_range(start, end), method='nearest')
                close_vals = window['Close'].interpolate().bfill().ffill().values
                if len(close_vals) == 2 * window_days + 1:
                    normalized = close_vals / close_vals[0] * 100
                    patterns.append(normalized)
        
        if patterns:
            avg_pattern = np.mean(patterns, axis=0)
            dates = pd.date_range(event_date - timedelta(days=window_days), event_date + timedelta(days=window_days))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, avg_pattern, color='purple', label='Predicted Budget 2025 Impact')
            ax.axvline(event_date, color='green', linestyle='--', label='Budget Day')
            ax.set_title("Predicted Stock Market Reaction to Pakistan Budget 2025", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Stock Price (%)")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download Plot as PNG",
                data=buf.getvalue(),
                file_name="budget_prediction.png",
                mime="image/png"
            )
        else:
            st.warning("Not enough clean past budget data to generate prediction.")
    
    elif prediction_event == '2025 IMF Event':
        historical_dates = event_dict['IMF Events']['dates']
        patterns = []
        for date in historical_dates:
            start = date - timedelta(days=window_days)
            end = date + timedelta(days=window_days)
            window = df_full[(df_full['Date'] >= start) & (df_full['Date'] <= end)].copy()
            if len(window) >= 40:
                window = window.set_index('Date').reindex(pd.date_range(start, end), method='nearest')
                close_vals = window['Close'].interpolate().bfill().ffill().values
                normalized = close_vals / close_vals[0] * 100
                patterns.append(normalized)
        
        if patterns:
            avg_pattern = np.mean(patterns, axis=0)
            days = np.arange(-window_days, window_days + 1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(days, avg_pattern, color='blue', label='IMF Pattern-Based Prediction (2025-05-09)')
            ax.axvline(x=0, color='red', linestyle='--', label='IMF Event Day')
            ax.set_title("Predicted Stock Market Impact of IMF Event (Based on Past IMF Patterns)", fontsize=14)
            ax.set_xlabel("Days from Event")
            ax.set_ylabel("Normalized Stock Price (%)")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download Plot as PNG",
                data=buf.getvalue(),
                file_name="imf_prediction.png",
                mime="image/png"
            )
        else:
            st.warning("Not enough clean past IMF data to generate prediction.")
    
    elif prediction_event == '2025 Pak-India War':
        pre_war_window = df_full[(df_full['Date'] < event_date)].tail(60)
        if len(pre_war_window) == 60:
            scaled_close = scaler.transform(pre_war_window[['Close']])
            X_input = np.array(scaled_close).reshape(1, 60, 1)
            predicted = []
            current_input = X_input.copy()
            
            for _ in range(60):
                pred = model.predict(current_input, verbose=0)
                predicted.append(pred[0, 0])
                current_input = np.append(current_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)
            
            predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
            combined = np.concatenate([pre_war_window['Close'].values, predicted_prices])
            normalized = combined / combined[0] * 100
            days = np.arange(-60, 60)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(days, normalized, color='darkred', label="Predicted War Impact (May 10, 2025)")
            ax.axvline(x=0, color='black', linestyle='--', label='War Event Day')
            ax.set_title("Predicted Pakistan Stock Market Impact of India-Pakistan War", fontsize=14)
            ax.set_xlabel("Days from Event")
            ax.set_ylabel("Normalized Stock Price (%)")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download Plot as PNG",
                data=buf.getvalue(),
                file_name="war_prediction.png",
                mime="image/png"
            )
        else:
            st.warning("Not enough pre-event data to simulate the war prediction.")
