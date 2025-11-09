"""
Build Feature Store Script - Run Once
======================================
This script builds the complete historical feature store from scratch.
It fetches 350 days of data and creates all features.

Usage: python src/build_feature_store.py
"""
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ISB_LAT, ISB_LON, create_features
import numpy as np

# Load API key from environment variable or use the one from the plan
API_KEY = os.getenv("OPENWEATHER_API_KEY", "3e5573c559d066b9120b40bc0c08617d")

def fetch_data(url):
    """Fetches data from a given URL and returns the JSON response."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def main():
    """Main function to build the complete feature store from scratch."""
    print("=" * 60)
    print("BUILDING COMPLETE FEATURE STORE FROM SCRATCH")
    print("=" * 60)
    
    # 1. Define time range - Use UTC time to match API expectations
    current_date_utc = datetime.now(timezone.utc)
    start_date = current_date_utc - timedelta(days=350)
    
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(current_date_utc.timestamp())
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d %H:%M UTC')} to {current_date_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    # 2. Fetch Air Pollution Data (all 350 days up to now)
    print("\nFetching air pollution data...")
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    pollution_data = fetch_data(pollution_url)
    
    if not pollution_data or 'list' not in pollution_data:
        print("Could not fetch or parse pollution data. Exiting.")
        return

    pollution_df = pd.DataFrame(pollution_data['list'])
    pollution_df['pm2_5'] = pollution_df['components'].apply(lambda x: x.get('pm2_5'))
    pollution_df = pollution_df[['dt', 'pm2_5']]
    print(f"  ✓ Fetched {len(pollution_df)} pollution records")

    # 3. Fetch Weather Data in chunks (7 days at a time - API limit)
    print("\nFetching weather data in chunks...")
    weather_list = []
    chunk_days = 7
    
    current_chunk_end = end_timestamp
    chunks_fetched = 0
    
    while current_chunk_end > start_timestamp:
        current_chunk_start = max(start_timestamp, current_chunk_end - (chunk_days * 24 * 3600))
        
        weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={current_chunk_start}&end={current_chunk_end}&appid={API_KEY}"
        weather_data = fetch_data(weather_url)
        
        if weather_data and 'list' in weather_data:
            weather_list.extend(weather_data['list'])
            chunks_fetched += 1
            chunk_start_date = datetime.fromtimestamp(current_chunk_start, tz=timezone.utc)
            chunk_end_date = datetime.fromtimestamp(current_chunk_end, tz=timezone.utc)
            print(f"  Chunk {chunks_fetched}: Fetched {len(weather_data['list'])} records ({chunk_start_date.strftime('%Y-%m-%d')} to {chunk_end_date.strftime('%Y-%m-%d')})")
        else:
            print(f"  Warning: Failed to fetch weather data for chunk")
            break
        
        current_chunk_end = current_chunk_start - 1
        time.sleep(0.5)  # Small delay to avoid rate limits
    
    if not weather_list:
        print("Could not fetch weather data. Exiting.")
        return

    weather_df = pd.DataFrame(weather_list)
    weather_df['temp'] = weather_df['main'].apply(lambda x: x.get('temp'))
    weather_df['humidity'] = weather_df['main'].apply(lambda x: x.get('humidity'))
    weather_df['wind_speed'] = weather_df['wind'].apply(lambda x: x.get('speed'))
    weather_df = weather_df[['dt', 'temp', 'humidity', 'wind_speed']]
    
    # Remove duplicates that might occur at chunk boundaries
    weather_df = weather_df.drop_duplicates(subset=['dt']).sort_values('dt')
    print(f"  ✓ Total weather records: {len(weather_df)}")

    # 4. Merge pollution and weather data
    print("\nMerging pollution and weather data...")
    df = pd.merge(pollution_df, weather_df, on='dt', how='inner')
    print(f"  ✓ Merged dataset has {len(df)} records")

    # 5. Create features using the utility function
    print("\nCreating features (this may take a moment)...")
    features_df = create_features(df, is_training=True)
    print(f"  ✓ Created {len(features_df)} feature records with {len(features_df.columns)} columns")

    # 6. Save to parquet and CSV
    os.makedirs("data/processed", exist_ok=True)
    features_df.to_parquet("data/processed/features.parquet", index=False)
    features_df.to_csv("data/processed/features.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ FEATURE STORE BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total feature records: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"Data date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    print(f"Latest data timestamp (UTC): {features_df['timestamp'].max()}")
    print(f"\nSaved to:")
    print(f"  - data/processed/features.parquet")
    print(f"  - data/processed/features.csv")

if __name__ == "__main__":
    main()
