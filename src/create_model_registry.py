import joblib
import pandas as pd
from datetime import datetime

def create_model_registry():
    """Create a human-readable model registry file."""
    
    # Load the model and scaler
    try:
        model = joblib.load('models/aqi_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
    except FileNotFoundError:
        print("Error: Model or scaler not found. Please run the training pipeline first.")
        return
    
    # Load the features to get column information
    try:
        features_df = pd.read_parquet('data/processed/features.parquet')
    except FileNotFoundError:
        print("Error: features.parquet not found.")
        return
    
    # Create the registry content
    registry_content = []
    registry_content.append("=" * 80)
    registry_content.append("AQI PREDICTOR - MODEL REGISTRY")
    registry_content.append("=" * 80)
    registry_content.append("")
    
    # Timestamp
    registry_content.append(f"Registry Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    registry_content.append("")
    
    # Model Information
    registry_content.append("-" * 80)
    registry_content.append("MODEL INFORMATION")
    registry_content.append("-" * 80)
    registry_content.append(f"Model Type: {type(model).__name__}")
    registry_content.append(f"Model Parameters:")
    for param, value in model.get_params().items():
        registry_content.append(f"  - {param}: {value}")
    registry_content.append("")
    
    # Feature Information
    registry_content.append("-" * 80)
    registry_content.append("FEATURE INFORMATION")
    registry_content.append("-" * 80)
    
    # Get target columns
    target_cols = [col for col in features_df.columns if col.startswith('pm25_t+')]
    feature_cols = [col for col in features_df.columns if col not in target_cols and col != 'pm2_5']
    
    registry_content.append(f"Number of Features: {len(feature_cols)}")
    registry_content.append(f"Number of Target Variables: {len(target_cols)}")
    registry_content.append(f"Total Training Samples: {len(features_df)}")
    registry_content.append("")
    registry_content.append("Feature Names:")
    for i, feature in enumerate(feature_cols, 1):
        registry_content.append(f"  {i}. {feature}")
    registry_content.append("")
    
    # Scaler Information
    registry_content.append("-" * 80)
    registry_content.append("DATA PREPROCESSING")
    registry_content.append("-" * 80)
    registry_content.append(f"Scaler Type: {type(scaler).__name__}")
    registry_content.append(f"Features Scaled: {len(feature_cols)}")
    registry_content.append("")
    
    # Model Performance (Placeholder - will be updated after training)
    registry_content.append("-" * 80)
    registry_content.append("MODEL PERFORMANCE METRICS")
    registry_content.append("-" * 80)
    registry_content.append("Note: Run the training pipeline to see actual performance metrics.")
    registry_content.append("Metrics are calculated on the hold-out test set (20% of data).")
    registry_content.append("")
    registry_content.append("Expected Metrics Format:")
    registry_content.append("  - Horizon t+1:  RMSE = [value], MAE = [value]")
    registry_content.append("  - Horizon t+24: RMSE = [value], MAE = [value]")
    registry_content.append("  - Horizon t+72: RMSE = [value], MAE = [value]")
    registry_content.append("")
    
    # Usage Information
    registry_content.append("-" * 80)
    registry_content.append("USAGE")
    registry_content.append("-" * 80)
    registry_content.append("To load this model in Python:")
    registry_content.append("  import joblib")
    registry_content.append("  model = joblib.load('models/aqi_model.joblib')")
    registry_content.append("  scaler = joblib.load('models/scaler.joblib')")
    registry_content.append("")
    registry_content.append("To make predictions:")
    registry_content.append("  X_scaled = scaler.transform(X)")
    registry_content.append("  predictions = model.predict(X_scaled)")
    registry_content.append("")
    registry_content.append("=" * 80)
    
    # Write to file
    output_path = 'models/model_registry.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(registry_content))
    
    print(f"Model registry saved to {output_path}")

if __name__ == "__main__":
    create_model_registry()
