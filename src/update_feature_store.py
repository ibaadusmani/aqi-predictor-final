"""
Update Feature Store Script - Run Hourly/Daily
===============================================
This script efficiently updates the feature store by only fetching new data
since the last update. It appends new rows to the existing features.parquet.

IMPORTANT: Training data will always be ~72 hours behind current time because
we need 72 hours of future actual weather data to create forecast aggregate 
features. This is EXPECTED and CORRECT for training.

For real-time predictions, the inference pipeline (app.py) uses actual API 
forecasts instead, so it can predict from the current moment.

Usage: python src/update_feature_store.py
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

# Load API key from environment variable
API_KEY = os.getenv("OPENWEATHER_API_KEY", "3e5573c559d066b9120b40bc0c08617d")

FEATURE_STORE_PATH = "data/processed/features.parquet"

def fetch_data(url):
    """Fetches data from a given URL and returns the JSON response."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def main():
    """Main function to update the feature store with new data."""
    print("=" * 60)
    print("UPDATING FEATURE STORE (APPEND-ONLY)")
    print("=" * 60)
    
    # 1. Check if feature store exists
    if not os.path.exists(FEATURE_STORE_PATH):
        print(f"\n❌ Error: Feature store not found at {FEATURE_STORE_PATH}")
        print("Please run 'python src/build_feature_store.py' first to create the initial feature store.")
        return
    
    # 2. Load existing feature store and find last timestamp
    print(f"\nLoading existing feature store...")
    existing_features = pd.read_parquet(FEATURE_STORE_PATH)
    
    # Get the last timestamp from dt column (unix timestamp)
    last_dt = existing_features['dt'].max()
    last_timestamp_utc = datetime.fromtimestamp(last_dt, tz=timezone.utc)
    
    print(f"  ✓ Loaded {len(existing_features)} existing records")
    print(f"  Last timestamp: {last_timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # 3. Determine new data range
    current_utc = datetime.now(timezone.utc)
    
    # Start fetching from 26 hours before last timestamp to ensure we have enough for lag features
    # This creates overlap, but we'll deduplicate later
    fetch_start_utc = last_timestamp_utc - timedelta(hours=26)
    
    start_timestamp = int(fetch_start_utc.timestamp())
    end_timestamp = int(current_utc.timestamp())
    
    hours_to_fetch = (current_utc - fetch_start_utc).total_seconds() / 3600
    
    print(f"\nFetching new data:")
    print(f"  From: {fetch_start_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  To:   {current_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Duration: {hours_to_fetch:.1f} hours")
    
    # 4. Check if update is needed
    if hours_to_fetch <= 26:
        print("\n✓ Feature store is already up to date! No new data to fetch.")
        return
    
    # 5. Fetch new pollution data
    print("\nFetching new air pollution data...")
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    pollution_data = fetch_data(pollution_url)
    
    if not pollution_data or 'list' not in pollution_data:
        print("❌ Could not fetch pollution data. Exiting.")
        return
    
    pollution_df = pd.DataFrame(pollution_data['list'])
    pollution_df['pm2_5'] = pollution_df['components'].apply(lambda x: x.get('pm2_5'))
    pollution_df = pollution_df[['dt', 'pm2_5']]
    print(f"  ✓ Fetched {len(pollution_df)} pollution records")
    
    # 6. Fetch new weather data
    print("\nFetching new weather data...")
    
    # If less than 7 days, fetch in one call. Otherwise, use chunks
    days_to_fetch = hours_to_fetch / 24
    
    if days_to_fetch <= 7:
        weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        weather_data = fetch_data(weather_url)
        
        if not weather_data or 'list' not in weather_data:
            print("❌ Could not fetch weather data. Exiting.")
            return
        
        weather_list = weather_data['list']
        print(f"  ✓ Fetched {len(weather_list)} weather records")
    else:
        # Fetch in 7-day chunks
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
            time.sleep(0.5)
        
        print(f"  ✓ Total weather records: {len(weather_list)}")
    
    if not weather_list:
        print("❌ Could not fetch weather data. Exiting.")
        return
    
    # 7. Process weather data
    weather_df = pd.DataFrame(weather_list)
    weather_df['temp'] = weather_df['main'].apply(lambda x: x.get('temp'))
    weather_df['humidity'] = weather_df['main'].apply(lambda x: x.get('humidity'))
    weather_df['wind_speed'] = weather_df['wind'].apply(lambda x: x.get('speed'))
    weather_df = weather_df[['dt', 'temp', 'humidity', 'wind_speed']]
    weather_df = weather_df.drop_duplicates(subset=['dt']).sort_values('dt')
    
    # 8. Merge new data
    print("\nMerging new pollution and weather data...")
    new_df = pd.merge(pollution_df, weather_df, on='dt', how='inner')
    print(f"  ✓ Merged {len(new_df)} new records")
    
    # 9. Create features for new data
    print("\nCreating features for new data...")
    print("  Note: Only rows with 72 hours of future data can be processed (training requirement)")
    new_features_df = create_features(new_df, is_training=True)
    print(f"  ✓ Created {len(new_features_df)} new feature records")
    
    # Debug: Show date range of new features
    if len(new_features_df) > 0:
        new_features_df['timestamp'] = pd.to_datetime(new_features_df['dt'], unit='s')
        print(f"  New features date range: {new_features_df['timestamp'].min()} to {new_features_df['timestamp'].max()}")
    else:
        print("  ℹ️  No new complete records yet (need 72 hours of future data for each row)")
    
    # 10. Combine with existing features and remove duplicates
    print("\nCombining with existing feature store...")
    combined_features = pd.concat([existing_features, new_features_df], ignore_index=True)
    
    # Remove duplicates based on dt (keeping the latest)
    combined_features = combined_features.drop_duplicates(subset=['dt'], keep='last')
    combined_features = combined_features.sort_values('dt').reset_index(drop=True)
    
    new_records_added = len(combined_features) - len(existing_features)
    
    print(f"  ✓ Total records after merge: {len(combined_features)}")
    print(f"  ✓ New records added: {new_records_added}")
    
    # 11. Save updated feature store
    print("\nSaving updated feature store...")
    combined_features.to_parquet(FEATURE_STORE_PATH, index=False)
    combined_features.to_csv("data/processed/features.csv", index=False)
    
    # 12. Print summary
    print("\n" + "=" * 60)
    print("✓ FEATURE STORE UPDATE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Previous records: {len(existing_features)}")
    print(f"New records added: {new_records_added}")
    print(f"Total records now: {len(combined_features)}")
    print(f"Total features: {len(combined_features.columns)}")
    print(f"Data date range: {combined_features['timestamp'].min()} to {combined_features['timestamp'].max()}")
    print(f"Latest data timestamp (UTC): {combined_features['timestamp'].max()}")
    
    if new_records_added > 0:
        print(f"\n✅ Successfully added {new_records_added} new hourly records")
    else:
        print(f"\n✅ No new records added (already up to date)")

if __name__ == "__main__":
    main()
