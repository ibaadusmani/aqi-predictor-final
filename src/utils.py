import pandas as pd
import numpy as np

ISB_LAT = 33.7380
ISB_LON = 73.0845

def create_features(df, is_training=True):
    """
    Create time-series features from a DataFrame.
    """
    # Create readable timestamp column
    df['timestamp'] = pd.to_datetime(df['dt'], unit='s')
    
    # Set index for time series operations but keep timestamp as column too
    df_indexed = df.set_index('timestamp')
    
    df_indexed['hour'] = df_indexed.index.hour
    df_indexed['day_of_week'] = df_indexed.index.dayofweek
    df_indexed['month'] = df_indexed.index.month

    # Cyclical features for hour
    df_indexed['hour_sin'] = np.sin(2 * np.pi * df_indexed['hour'] / 24)
    df_indexed['hour_cos'] = np.cos(2 * np.pi * df_indexed['hour'] / 24)
    
    # Cyclical features for day of week
    df_indexed['day_of_week_sin'] = np.sin(2 * np.pi * df_indexed['day_of_week'] / 7)
    df_indexed['day_of_week_cos'] = np.cos(2 * np.pi * df_indexed['day_of_week'] / 7)
    
    # Weekend indicator
    df_indexed['is_weekend'] = (df_indexed['day_of_week'] >= 5).astype(int)

    # === PM2.5 Features ===
    # Lag features for pm2.5
    df_indexed['pm25_lag_1hr'] = df_indexed['pm2_5'].shift(1)
    df_indexed['pm25_lag_2hr'] = df_indexed['pm2_5'].shift(2)
    df_indexed['pm25_lag_3hr'] = df_indexed['pm2_5'].shift(3)
    df_indexed['pm25_lag_6hr'] = df_indexed['pm2_5'].shift(6)
    df_indexed['pm25_lag_12hr'] = df_indexed['pm2_5'].shift(12)
    df_indexed['pm25_lag_24hr'] = df_indexed['pm2_5'].shift(24)

    # Rolling averages for pm2.5
    df_indexed['pm25_rolling_avg_3hr'] = df_indexed['pm2_5'].rolling(window=3).mean()
    df_indexed['pm25_rolling_avg_6hr'] = df_indexed['pm2_5'].rolling(window=6).mean()
    df_indexed['pm25_rolling_avg_12hr'] = df_indexed['pm2_5'].rolling(window=12).mean()
    df_indexed['pm25_rolling_avg_24hr'] = df_indexed['pm2_5'].rolling(window=24).mean()
    
    # Rolling standard deviation for pm2.5 variability
    df_indexed['pm25_rolling_std_6hr'] = df_indexed['pm2_5'].rolling(window=6).std()
    df_indexed['pm25_rolling_std_24hr'] = df_indexed['pm2_5'].rolling(window=24).std()
    
    # Rolling extremes for pm2.5
    df_indexed['pm25_rolling_min_6hr'] = df_indexed['pm2_5'].rolling(window=6).min()
    df_indexed['pm25_rolling_max_6hr'] = df_indexed['pm2_5'].rolling(window=6).max()
    df_indexed['pm25_rolling_min_24hr'] = df_indexed['pm2_5'].rolling(window=24).min()
    df_indexed['pm25_rolling_max_24hr'] = df_indexed['pm2_5'].rolling(window=24).max()
    
    # PM2.5 change features
    df_indexed['pm25_change_1hr'] = df_indexed['pm2_5'] - df_indexed['pm25_lag_1hr']
    df_indexed['pm25_change_6hr'] = df_indexed['pm2_5'] - df_indexed['pm25_lag_6hr']
    df_indexed['pm25_change_12hr'] = df_indexed['pm2_5'] - df_indexed['pm25_lag_12hr']
    
    # PM2.5 trend (simple linear slope over 6hr window)
    df_indexed['pm25_trend_6hr'] = (df_indexed['pm2_5'] - df_indexed['pm25_lag_6hr']) / 6

    # === Temperature Features ===
    # Lags - understand temperature trends and seasonal context
    df_indexed['temp_lag_1hr'] = df_indexed['temp'].shift(1)
    df_indexed['temp_lag_2hr'] = df_indexed['temp'].shift(2)
    df_indexed['temp_lag_3hr'] = df_indexed['temp'].shift(3)
    df_indexed['temp_lag_6hr'] = df_indexed['temp'].shift(6)
    df_indexed['temp_lag_12hr'] = df_indexed['temp'].shift(12)
    df_indexed['temp_lag_24hr'] = df_indexed['temp'].shift(24)
    
    # Rolling averages - detect heating/cooling trends
    df_indexed['temp_roll_mean_3hr'] = df_indexed['temp'].rolling(window=3).mean()
    df_indexed['temp_roll_mean_6hr'] = df_indexed['temp'].rolling(window=6).mean()
    df_indexed['temp_roll_mean_12hr'] = df_indexed['temp'].rolling(window=12).mean()
    df_indexed['temp_roll_mean_24hr'] = df_indexed['temp'].rolling(window=24).mean()
    
    # Temperature variability
    df_indexed['temp_roll_std_6hr'] = df_indexed['temp'].rolling(window=6).std()
    df_indexed['temp_roll_std_24hr'] = df_indexed['temp'].rolling(window=24).std()
    
    # Temperature extremes
    df_indexed['temp_roll_min_6hr'] = df_indexed['temp'].rolling(window=6).min()
    df_indexed['temp_roll_max_6hr'] = df_indexed['temp'].rolling(window=6).max()
    df_indexed['temp_roll_min_24hr'] = df_indexed['temp'].rolling(window=24).min()
    df_indexed['temp_roll_max_24hr'] = df_indexed['temp'].rolling(window=24).max()
    
    # Temperature change rate - useful for ozone prediction
    df_indexed['temp_change_3hr'] = df_indexed['temp'] - df_indexed['temp_lag_3hr']
    df_indexed['temp_change_24hr'] = df_indexed['temp'] - df_indexed['temp_lag_24hr']

    # === Humidity Features ===
    # Lags - humidity patterns affect particle behavior
    df_indexed['humidity_lag_1hr'] = df_indexed['humidity'].shift(1)
    df_indexed['humidity_lag_2hr'] = df_indexed['humidity'].shift(2)
    df_indexed['humidity_lag_3hr'] = df_indexed['humidity'].shift(3)
    df_indexed['humidity_lag_6hr'] = df_indexed['humidity'].shift(6)
    df_indexed['humidity_lag_12hr'] = df_indexed['humidity'].shift(12)
    df_indexed['humidity_lag_24hr'] = df_indexed['humidity'].shift(24)
    
    # Rolling averages - detect moisture trends
    df_indexed['humidity_roll_mean_3hr'] = df_indexed['humidity'].rolling(window=3).mean()
    df_indexed['humidity_roll_mean_6hr'] = df_indexed['humidity'].rolling(window=6).mean()
    df_indexed['humidity_roll_mean_12hr'] = df_indexed['humidity'].rolling(window=12).mean()
    df_indexed['humidity_roll_mean_24hr'] = df_indexed['humidity'].rolling(window=24).mean()
    
    # Humidity variability
    df_indexed['humidity_roll_std_6hr'] = df_indexed['humidity'].rolling(window=6).std()
    df_indexed['humidity_roll_std_24hr'] = df_indexed['humidity'].rolling(window=24).std()
    
    # Humidity extremes
    df_indexed['humidity_roll_min_6hr'] = df_indexed['humidity'].rolling(window=6).min()
    df_indexed['humidity_roll_max_6hr'] = df_indexed['humidity'].rolling(window=6).max()
    df_indexed['humidity_roll_min_24hr'] = df_indexed['humidity'].rolling(window=24).min()
    df_indexed['humidity_roll_max_24hr'] = df_indexed['humidity'].rolling(window=24).max()

    # === Wind Speed Features ===
    # Lags - understand wind persistence patterns
    df_indexed['wind_speed_lag_1hr'] = df_indexed['wind_speed'].shift(1)
    df_indexed['wind_speed_lag_2hr'] = df_indexed['wind_speed'].shift(2)
    df_indexed['wind_speed_lag_3hr'] = df_indexed['wind_speed'].shift(3)
    df_indexed['wind_speed_lag_6hr'] = df_indexed['wind_speed'].shift(6)
    df_indexed['wind_speed_lag_12hr'] = df_indexed['wind_speed'].shift(12)
    df_indexed['wind_speed_lag_24hr'] = df_indexed['wind_speed'].shift(24)
    
    # Rolling averages - detect "washout" events (sustained high winds)
    df_indexed['wind_speed_roll_mean_3hr'] = df_indexed['wind_speed'].rolling(window=3).mean()
    df_indexed['wind_speed_roll_mean_6hr'] = df_indexed['wind_speed'].rolling(window=6).mean()
    df_indexed['wind_speed_roll_mean_12hr'] = df_indexed['wind_speed'].rolling(window=12).mean()
    df_indexed['wind_speed_roll_mean_24hr'] = df_indexed['wind_speed'].rolling(window=24).mean()
    
    # Wind speed variability - gusty vs steady winds
    df_indexed['wind_speed_roll_std_6hr'] = df_indexed['wind_speed'].rolling(window=6).std()
    df_indexed['wind_speed_roll_std_24hr'] = df_indexed['wind_speed'].rolling(window=24).std()
    
    # Wind speed extremes
    df_indexed['wind_speed_roll_min_6hr'] = df_indexed['wind_speed'].rolling(window=6).min()
    df_indexed['wind_speed_roll_max_6hr'] = df_indexed['wind_speed'].rolling(window=6).max()
    df_indexed['wind_speed_roll_min_24hr'] = df_indexed['wind_speed'].rolling(window=24).min()
    df_indexed['wind_speed_roll_max_24hr'] = df_indexed['wind_speed'].rolling(window=24).max()

    # === Interaction Features ===
    # Temperature-Wind interaction (convection strength)
    df_indexed['temp_wind_interaction'] = df_indexed['temp'] * df_indexed['wind_speed']
    
    # Humidity-Temperature interaction (atmospheric stability)
    df_indexed['humidity_temp_interaction'] = df_indexed['humidity'] * df_indexed['temp']
    
    # Low wind + high humidity = stagnant conditions
    df_indexed['stagnation_index'] = df_indexed['humidity'] / (df_indexed['wind_speed'] + 0.1)
    
    # PM2.5 interactions with weather
    df_indexed['temp_pm25_interaction'] = df_indexed['temp'] * df_indexed['pm2_5']
    df_indexed['humidity_pm25_interaction'] = df_indexed['humidity'] * df_indexed['pm2_5']
    df_indexed['wind_pm25_interaction'] = df_indexed['wind_speed'] * df_indexed['pm2_5']
    
    # Drop only hour column
    df_indexed = df_indexed.drop(columns=['hour'])
    
    # Reset index to get timestamp back as a column
    df_indexed = df_indexed.reset_index()

    if is_training:
        # === Create Forecast Aggregate Features using ACTUAL future weather ===
        # For TRAINING ONLY: Calculate forecast aggregates from actual future weather
        # This simulates what a perfect forecast would have shown
        print("  Creating forecast aggregate features from actual future weather...")
        
        # Calculate forward-looking aggregates using explicit loops
        n = len(df_indexed)
        
        # Initialize arrays
        temp_forecast_24h_avg = np.full(n, np.nan)
        temp_forecast_72h_max = np.full(n, np.nan)
        temp_forecast_72h_min = np.full(n, np.nan)
        
        wind_speed_forecast_24h_avg = np.full(n, np.nan)
        wind_speed_forecast_72h_max = np.full(n, np.nan)
        wind_speed_forecast_24h_min = np.full(n, np.nan)
        
        humidity_forecast_24h_avg = np.full(n, np.nan)
        humidity_forecast_72h_max = np.full(n, np.nan)
        humidity_forecast_72h_min = np.full(n, np.nan)
        
        stagnation_forecast_24h_avg = np.full(n, np.nan)
        
        # Get numpy arrays for faster computation
        temp_arr = df_indexed['temp'].values
        wind_speed_arr = df_indexed['wind_speed'].values
        humidity_arr = df_indexed['humidity'].values
        stagnation_arr = df_indexed['stagnation_index'].values
        
        # Calculate forecast aggregates for each row
        for i in range(n - 72):  # Only for rows that have 72 hours of future data
            # Temperature forecasts (next 24 and 72 hours)
            temp_forecast_24h_avg[i] = np.mean(temp_arr[i+1:i+25])
            temp_forecast_72h_max[i] = np.max(temp_arr[i+1:i+73])
            temp_forecast_72h_min[i] = np.min(temp_arr[i+1:i+73])
            
            # Wind speed forecasts
            wind_speed_forecast_24h_avg[i] = np.mean(wind_speed_arr[i+1:i+25])
            wind_speed_forecast_72h_max[i] = np.max(wind_speed_arr[i+1:i+73])
            wind_speed_forecast_24h_min[i] = np.min(wind_speed_arr[i+1:i+25])
            
            # Humidity forecasts
            humidity_forecast_24h_avg[i] = np.mean(humidity_arr[i+1:i+25])
            humidity_forecast_72h_max[i] = np.max(humidity_arr[i+1:i+73])
            humidity_forecast_72h_min[i] = np.min(humidity_arr[i+1:i+73])
            
            # Stagnation index forecast
            stagnation_forecast_24h_avg[i] = np.mean(stagnation_arr[i+1:i+25])
        
        # Assign to dataframe
        df_indexed['temp_forecast_24h_avg'] = temp_forecast_24h_avg
        df_indexed['temp_forecast_72h_max'] = temp_forecast_72h_max
        df_indexed['temp_forecast_72h_min'] = temp_forecast_72h_min
        df_indexed['temp_forecast_change_24h'] = temp_forecast_24h_avg - temp_arr
        
        df_indexed['wind_speed_forecast_24h_avg'] = wind_speed_forecast_24h_avg
        df_indexed['wind_speed_forecast_72h_max'] = wind_speed_forecast_72h_max
        df_indexed['wind_speed_forecast_24h_min'] = wind_speed_forecast_24h_min
        
        df_indexed['humidity_forecast_24h_avg'] = humidity_forecast_24h_avg
        df_indexed['humidity_forecast_72h_max'] = humidity_forecast_72h_max
        df_indexed['humidity_forecast_72h_min'] = humidity_forecast_72h_min
        
        df_indexed['stagnation_forecast_24h_avg'] = stagnation_forecast_24h_avg
        
        # Create target variables
        for i in range(1, 73):
            df_indexed[f'pm25_t+{i}'] = df_indexed['pm2_5'].shift(-i)
        
        # Drop rows with NaN in feature columns (lag/rolling features at the start)
        # Also drops last 72 rows which don't have complete future data for forecast aggregates
        feature_cols = [col for col in df_indexed.columns if not col.startswith('pm25_t+')]
        df_indexed = df_indexed.dropna(subset=feature_cols)
    
    else:
        # === For INFERENCE: Add placeholder forecast aggregate columns ===
        # These will be filled by add_forecast_aggregates() using actual API forecast data
        # We add them as NaN here to keep column consistency with training data
        forecast_feature_names = [
            'temp_forecast_24h_avg', 'temp_forecast_72h_max', 'temp_forecast_72h_min',
            'temp_forecast_change_24h', 'wind_speed_forecast_24h_avg', 
            'wind_speed_forecast_72h_max', 'wind_speed_forecast_24h_min',
            'humidity_forecast_24h_avg', 'humidity_forecast_72h_max',
            'humidity_forecast_72h_min', 'stagnation_forecast_24h_avg'
        ]
        for col_name in forecast_feature_names:
            df_indexed[col_name] = np.nan
        
        # For inference, we DON'T drop rows - keep all historical data!
        # The last row will be used for prediction with forecast aggregates added separately

    return df_indexed

def add_forecast_aggregates(feature_row, forecast_df):
    """
    Add aggregate forecast weather features to the prediction row.
    
    This function takes forecasted weather data for the next 72 hours
    and creates summary statistics (mean, max, min, sum, change) that
    help the model understand future weather patterns.
    
    Parameters:
    -----------
    feature_row : pd.Series or pd.DataFrame (single row)
        The feature row for the current moment (time t)
    forecast_df : pd.DataFrame
        DataFrame with columns ['temp', 'humidity', 'wind_speed']
        containing forecasted weather for the next 72 hours
        
    Returns:
    --------
    pd.Series or pd.DataFrame
        The feature row with added forecast aggregate features
    """
    # Convert to Series if it's a single-row DataFrame
    if isinstance(feature_row, pd.DataFrame):
        feature_row = feature_row.iloc[0].copy()
    else:
        feature_row = feature_row.copy()
    
    # Ensure we have forecast data
    if forecast_df is None or len(forecast_df) == 0:
        # If no forecast data, add NaN placeholders
        forecast_feature_names = [
            'temp_forecast_24h_avg', 'temp_forecast_72h_max', 'temp_forecast_72h_min',
            'temp_forecast_change_24h', 'wind_speed_forecast_24h_avg', 
            'wind_speed_forecast_72h_max', 'wind_speed_forecast_24h_min',
            'humidity_forecast_24h_avg', 'humidity_forecast_72h_max',
            'humidity_forecast_72h_min', 'stagnation_forecast_24h_avg'
        ]
        for name in forecast_feature_names:
            feature_row[name] = np.nan
        return feature_row
    
    # === Temperature Forecast Aggregates ===
    # Average temperature in next 24 and 72 hours
    feature_row['temp_forecast_24h_avg'] = forecast_df['temp'][:24].mean() if len(forecast_df) >= 24 else forecast_df['temp'].mean()
    feature_row['temp_forecast_72h_max'] = forecast_df['temp'][:72].max()
    feature_row['temp_forecast_72h_min'] = forecast_df['temp'][:72].min()
    
    # Temperature change (current to +24h average) - indicates heating/cooling trend
    current_temp = feature_row.get('temp', forecast_df['temp'].iloc[0])
    feature_row['temp_forecast_change_24h'] = feature_row['temp_forecast_24h_avg'] - current_temp
    
    # === Wind Speed Forecast Aggregates ===
    # Average and extreme wind speeds - detect washout events
    feature_row['wind_speed_forecast_24h_avg'] = forecast_df['wind_speed'][:24].mean() if len(forecast_df) >= 24 else forecast_df['wind_speed'].mean()
    feature_row['wind_speed_forecast_72h_max'] = forecast_df['wind_speed'][:72].max()
    feature_row['wind_speed_forecast_24h_min'] = forecast_df['wind_speed'][:24].min() if len(forecast_df) >= 24 else forecast_df['wind_speed'].min()
    
    # === Humidity Forecast Aggregates ===
    # Humidity patterns - affect particle behavior
    feature_row['humidity_forecast_24h_avg'] = forecast_df['humidity'][:24].mean() if len(forecast_df) >= 24 else forecast_df['humidity'].mean()
    feature_row['humidity_forecast_72h_max'] = forecast_df['humidity'][:72].max()
    feature_row['humidity_forecast_72h_min'] = forecast_df['humidity'][:72].min()
    
    # === Forecast Stagnation Index ===
    # Low wind + high humidity in forecast = pollution accumulation risk
    forecast_df_24h = forecast_df[:24] if len(forecast_df) >= 24 else forecast_df
    stagnation_forecast = forecast_df_24h['humidity'] / (forecast_df_24h['wind_speed'] + 0.1)
    feature_row['stagnation_forecast_24h_avg'] = stagnation_forecast.mean()
    
    return feature_row

def convert_pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration to AQI using EPA standard breakpoints.
    """
    if pm25 is None:
        return None

    # EPA AQI breakpoints for PM2.5 (24-hour average)
    # Using < with next range start to avoid gaps (e.g., 55.43 falls between 55.4 and 55.5)
    if pm25 < 12.1:
        return _linear_conversion(pm25, 0, 50, 0, 12.0)
    elif pm25 < 35.5:
        return _linear_conversion(pm25, 51, 100, 12.1, 35.4)
    elif pm25 < 55.5:
        return _linear_conversion(pm25, 101, 150, 35.5, 55.4)
    elif pm25 < 150.5:
        return _linear_conversion(pm25, 151, 200, 55.5, 150.4)
    elif pm25 < 250.5:
        return _linear_conversion(pm25, 201, 300, 150.5, 250.4)
    elif pm25 < 350.5:
        return _linear_conversion(pm25, 301, 400, 250.5, 350.4)
    elif pm25 < 500.5:
        return _linear_conversion(pm25, 401, 500, 350.5, 500.4)
    else:
        # Beyond 500.4 is considered Hazardous (500+)
        return 500

def _linear_conversion(val, i_low, i_high, c_low, c_high):
    return round(((val - c_low) / (c_high - c_low)) * (i_high - i_low) + i_low)
