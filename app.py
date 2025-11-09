import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, timedelta, timezone
from src.utils import create_features, convert_pm25_to_aqi, add_forecast_aggregates, ISB_LAT, ISB_LON

# --- Configuration ---
API_KEY = os.getenv("OPENWEATHER_API_KEY", "3e5573c559d066b9120b40bc0c08617d")
MODEL_PATH = 'models/aqi_model.joblib'
SCALER_PATH = 'models/scaler.joblib'

# --- Load Artifacts ---
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model or scaler not found. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
        return None, None

model, scaler = load_model_and_scaler()

# --- API Fetching ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_api_data():
    """Fetch all required data from OpenWeatherMap APIs."""
    # Use UTC time to match API expectations
    current_utc = datetime.now(timezone.utc)
    
    # 1. Current data (actual current pollution - not forecast)
    current_pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}"
    
    # 2. Forecast Data (Next 72 hours)
    pollution_forecast_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}"
    weather_forecast_url = f"http://api.openweathermap.org/data/2.5/forecast/hourly?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}&cnt=72"
    
    # 3. Historical Data (Last 25 hours for lags)
    hist_end_ts = int(current_utc.timestamp())
    hist_start_ts = int((current_utc - timedelta(hours=25)).timestamp())
    
    pollution_history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={hist_start_ts}&end={hist_end_ts}&appid={API_KEY}"
    weather_history_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={hist_start_ts}&end={hist_end_ts}&appid={API_KEY}"

    try:
        current_pol_data = requests.get(current_pollution_url).json()
        pol_forecast_data = requests.get(pollution_forecast_url).json()
        wea_forecast_data = requests.get(weather_forecast_url).json()
        pol_history_data = requests.get(pollution_history_url).json()
        wea_history_data = requests.get(weather_history_url).json()
        return current_pol_data, pol_forecast_data, wea_forecast_data, pol_history_data, wea_history_data
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, None, None, None, None

# --- Data Processing ---
def process_data_for_prediction(current_pol, pol_forecast, wea_forecast, pol_history, wea_history):
    """Combine and process API data to create the feature row for prediction."""
    # Historical Data (last 25 hours)
    hist_pol_df = pd.DataFrame(pol_history['list'])
    hist_pol_df['pm2_5'] = hist_pol_df['components'].apply(lambda x: x.get('pm2_5'))
    
    hist_wea_df = pd.DataFrame(wea_history['list'])
    hist_wea_df['temp'] = hist_wea_df['main'].apply(lambda x: x.get('temp'))
    hist_wea_df['humidity'] = hist_wea_df['main'].apply(lambda x: x.get('humidity'))
    hist_wea_df['wind_speed'] = hist_wea_df['wind'].apply(lambda x: x.get('speed'))
    
    historical_df = pd.merge_asof(
        hist_pol_df[['dt', 'pm2_5']].sort_values('dt'),
        hist_wea_df[['dt', 'temp', 'humidity', 'wind_speed']].sort_values('dt'),
        on='dt', direction='nearest'
    )
    
    # Current actual data (not forecast)
    current_pol_data = current_pol['list'][0]
    current_pm25 = current_pol_data['components']['pm2_5']
    current_dt = current_pol_data['dt']
    
    # Get current weather from forecast (first item is current/very near)
    current_wea = wea_forecast['list'][0]
    current_row = pd.DataFrame([{
        'dt': current_dt,
        'pm2_5': current_pm25,
        'temp': current_wea['main']['temp'],
        'humidity': current_wea['main']['humidity'],
        'wind_speed': current_wea['wind']['speed']
    }])

    # Forecast Data (for the next 72 hours)
    fore_pol_df = pd.DataFrame(pol_forecast['list'])
    fore_pol_df['pm2_5'] = fore_pol_df['components'].apply(lambda x: x.get('pm2_5'))
    
    fore_wea_df = pd.DataFrame(wea_forecast['list'])
    fore_wea_df['temp'] = fore_wea_df['main'].apply(lambda x: x.get('temp'))
    fore_wea_df['humidity'] = fore_wea_df['main'].apply(lambda x: x.get('humidity'))
    fore_wea_df['wind_speed'] = fore_wea_df['wind'].apply(lambda x: x.get('speed'))

    forecast_df = pd.merge_asof(
        fore_pol_df[['dt', 'pm2_5']].sort_values('dt'),
        fore_wea_df[['dt', 'temp', 'humidity', 'wind_speed']].sort_values('dt'),
        on='dt', direction='nearest'
    )

    # Combine: historical + current + forecast
    combined_df = pd.concat([historical_df, current_row, forecast_df], ignore_index=True)
    features_df = create_features(combined_df, is_training=False)
    
    # Use the row right after historical data (which is the current moment with proper lags)
    prediction_row_index = len(historical_df)
    X_pred = features_df.iloc[prediction_row_index]
    
    # Add forecast aggregate features to the prediction row
    # Use only weather forecast (not pollution forecast) for aggregates
    forecast_weather_df = fore_wea_df[['temp', 'humidity', 'wind_speed']].copy()
    X_pred = add_forecast_aggregates(X_pred, forecast_weather_df)
    
    # Convert back to DataFrame for prediction
    X_pred = pd.DataFrame([X_pred])
    
    # Current AQI from actual current data
    current_aqi = convert_pm25_to_aqi(current_pm25)

    return X_pred, current_aqi, forecast_df

# --- Streamlit App ---
st.set_page_config(page_title="AQI Predictor", layout="wide")
st.title('AQI 72-Hour Forecast for Islamabad')

if model is not None and scaler is not None:
    current_pol, pol_forecast, wea_forecast, pol_history, wea_history = fetch_api_data()

    if all([current_pol, pol_forecast, wea_forecast, pol_history, wea_history]):
        X_pred, current_aqi, forecast_df = process_data_for_prediction(
            current_pol, pol_forecast, wea_forecast, pol_history, wea_history
        )
        
        # Align columns with the training columns
        training_cols = scaler.get_feature_names_out()
        X_pred_aligned = X_pred[training_cols]

        # Scale the input
        X_pred_scaled = scaler.transform(X_pred_aligned)

        # Make Prediction
        y_pred_pm25 = model.predict(X_pred_scaled)[0] # Get the single row of 72 predictions

        # Convert to AQI
        predicted_aqi = [convert_pm25_to_aqi(p) for p in y_pred_pm25]

        # --- Display Results ---
        st.metric("Current AQI in Islamabad", value=f"{current_aqi}")

        if max(predicted_aqi) > 150:
            st.warning("High AQI levels expected in the next 72 hours!")

        # Create a DataFrame for hourly tiles
        # Convert UTC timestamps to local time (UTC+5 for Pakistan)
        future_timestamps_utc = pd.to_datetime(forecast_df['dt'].iloc[:72], unit='s')
        future_timestamps_local = future_timestamps_utc + timedelta(hours=5)
        
        # Get temperature data from forecast
        try:
            temperatures = forecast_df['temp'].iloc[:72].values
        except Exception as e:
            st.error(f"Error getting temperature data: {e}")
            st.write("Available columns:", forecast_df.columns.tolist())
            temperatures = [273.15] * 72  # Default to 0°C if error
        
        # Function to get AQI category and color
        def get_aqi_category(aqi):
            if aqi <= 50:
                return "Good", "#00e400"
            elif aqi <= 100:
                return "Moderate", "#ffff00"
            elif aqi <= 150:
                return "Unhealthy for Sensitive", "#ff7e00"
            elif aqi <= 200:
                return "Unhealthy", "#ff0000"
            elif aqi <= 300:
                return "Very Unhealthy", "#8f3f97"
            else:
                return "Hazardous", "#7e0023"
        
        st.subheader("72-Hour Hourly AQI Forecast (Pakistan Time - UTC+5)")
        
        # Build HTML for horizontal scrolling tiles
        html_content = """
        <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            overflow-y: hidden;
            gap: 15px;
            padding: 20px 10px;
            scroll-behavior: smooth;
        }
        .scroll-container::-webkit-scrollbar {
            height: 12px;
        }
        .scroll-container::-webkit-scrollbar-track {
            background: #2b2b2b;
            border-radius: 10px;
        }
        .scroll-container::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 10px;
        }
        .scroll-container::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        .aqi-tile {
            min-width: 135px;
            flex-shrink: 0;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        .aqi-tile:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .tile-time {
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #000;
            line-height: 1.3;
        }
        .tile-aqi {
            font-size: 40px;
            font-weight: bold;
            margin: 10px 0;
            color: #000;
        }
        .tile-temp {
            font-size: 17px;
            margin: 10px 0;
            color: #000;
            font-weight: 500;
        }
        .tile-category {
            font-size: 11px;
            margin-top: 10px;
            color: #000;
            font-weight: 600;
            text-transform: uppercase;
        }
        </style>
        <div class="scroll-container">
        """
        
        for i in range(72):
            timestamp = future_timestamps_local.iloc[i] if hasattr(future_timestamps_local, 'iloc') else future_timestamps_local[i]
            aqi = int(predicted_aqi[i])
            temp_kelvin = float(temperatures[i])
            temp_celsius = temp_kelvin - 273.15
            category, color = get_aqi_category(aqi)
            
            # Format time display
            hour_str = timestamp.strftime("%I %p")
            day_str = timestamp.strftime("%b %d")
            
            html_content += f"""
            <div class="aqi-tile" style="background-color: {color};">
                <div class="tile-time">{hour_str}<br>{day_str}</div>
                <div class="tile-aqi">{aqi}</div>
                <div class="tile-temp">{temp_celsius:.0f}°C</div>
                <div class="tile-category">{category}</div>
            </div>
            """
        
        html_content += "</div>"
        
        # Render with proper height
        components.html(html_content, height=250, scrolling=False)

        with st.expander("View Raw Prediction Data"):
            raw_df = pd.DataFrame({
                'Timestamp': future_timestamps_local,
                'AQI': predicted_aqi,
                'Temperature (°C)': (temperatures - 273.15).round(1),
                'Category': [get_aqi_category(aqi)[0] for aqi in predicted_aqi]
            })
            st.dataframe(raw_df)
    else:
        st.warning("Could not retrieve data for prediction. Please check API key and network.")
else:
    st.info("Model is loading or not available.")
