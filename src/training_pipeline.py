import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def main():
    """Main function to run the training pipeline."""
    # 1. Load Data
    print("Loading processed data...")
    try:
        df = pd.read_parquet('data/processed/features.parquet')
    except FileNotFoundError:
        print("Error: features.parquet not found. Please run the feature pipeline first.")
        return

    # 2. Prepare Data for Training
    print("Preparing data for training...")
    target_cols = [f'pm25_t+{i}' for i in range(1, 73)]
    # Ensure all target columns exist, in case the feature engineering changed
    target_cols = [col for col in target_cols if col in df.columns]
    
    # Exclude non-numeric columns (timestamp, dt) and target columns
    exclude_cols = target_cols + ['pm2_5', 'timestamp', 'dt']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    y = df[target_cols]
    X = df[feature_cols]
    
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Targets: {len(target_cols)} columns")
    
    # Check for data quality issues
    print("\nData Quality Check:")
    print(f"  Missing values in features: {X.isnull().sum().sum()}")
    print(f"  Missing values in targets: {y.isnull().sum().sum()}")
    print(f"  Feature value ranges:")
    print(f"    Min: {X.min().min():.2f}")
    print(f"    Max: {X.max().max():.2f}")
    print(f"    Mean: {X.mean().mean():.2f}")
    print(f"  Target (PM2.5) statistics:")
    print(f"    Min: {y.min().min():.2f} µg/m³")
    print(f"    Max: {y.max().max():.2f} µg/m³")
    print(f"    Mean: {y.mean().mean():.2f} µg/m³")
    print(f"    Std: {y.std().mean():.2f} µg/m³")

    # 3. Split Data (Time-series split, no shuffle)
    print("Splitting data...")
    # First, remove rows where ALL target columns are NaN (last 72 hours)
    valid_rows = y.notna().any(axis=1)
    X_valid = X[valid_rows]
    y_valid = y[valid_rows]
    
    print(f"  Total rows: {len(X)}")
    print(f"  Valid rows (with at least one target): {len(X_valid)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, shuffle=False)

    # 4. Scale Data
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = 'models/scaler.joblib'
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)

    # 5. Train Model
    print("Training RandomForestRegressor model...")
    print("  Using enhanced hyperparameters for 94 features...")
    model = RandomForestRegressor(
        n_estimators=200,           # More trees for better averaging
        max_depth=25,                # Deeper trees for 94 features
        min_samples_split=5,         # Less restrictive
        min_samples_leaf=2,          # Less restrictive
        max_features='sqrt',         # Good default for many features
        n_jobs=-1, 
        random_state=42,
        verbose=0                    # Suppress progress logs
    )
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics for specific horizons
    def calc_metrics_for_horizon(y_true_col, y_pred_col, horizon_name):
        """Calculate comprehensive metrics for a specific horizon, ignoring NaN values."""
        valid_mask = y_true_col.notna()
        if valid_mask.sum() == 0:
            return None, None, None, None
        y_true_valid = y_true_col[valid_mask]
        y_pred_valid = y_pred_col[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        # Calculate mean absolute percentage error (MAPE)
        # Avoid division by zero
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (y_true_valid + 1e-10))) * 100
        
        return rmse, mae, r2, mape
    
    # Calculate metrics for key horizons
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    horizons_to_evaluate = [
        (0, 1, "t+1 (1 hour ahead)"),
        (5, 6, "t+6 (6 hours ahead)"),
        (11, 12, "t+12 (12 hours ahead)"),
        (23, 24, "t+24 (24 hours ahead)"),
        (47, 48, "t+48 (48 hours ahead)"),
        (71, 72, "t+72 (72 hours ahead)")
    ]
    
    for idx, horizon_num, horizon_label in horizons_to_evaluate:
        rmse, mae, r2, mape = calc_metrics_for_horizon(
            y_test.iloc[:, idx], 
            y_pred[:, idx], 
            horizon_label
        )
        
        if rmse is not None:
            print(f"\n{horizon_label}:")
            print(f"  RMSE:  {rmse:>8.4f} µg/m³")
            print(f"  MAE:   {mae:>8.4f} µg/m³")
            # Suppressing R² score as per the change request
            # print(f"  R²:    {r2:>8.4f}")
            print(f"  MAPE:  {mape:>8.2f}%")
    
    # Calculate average metrics across all horizons
    print("\n" + "-"*70)
    print("AVERAGE METRICS ACROSS ALL HORIZONS")
    print("-"*70)
    
    all_rmse = []
    all_mae = []
    all_r2 = []
    all_mape = []
    
    for i in range(72):
        rmse, mae, r2, mape = calc_metrics_for_horizon(
            y_test.iloc[:, i], 
            y_pred[:, i], 
            f"t+{i+1}"
        )
        if rmse is not None:
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_r2.append(r2)
            all_mape.append(mape)
    
    print(f"\nAverage RMSE:  {np.mean(all_rmse):>8.4f} µg/m³")
    print(f"Average MAE:   {np.mean(all_mae):>8.4f} µg/m³")
    print(f"Average MAPE:  {np.mean(all_mape):>8.2f}%")
    
    print("\n" + "="*70)
    print()

    # 6b. Feature Importance Analysis
    print("Analyzing feature importance...")
    # Get average importance across all target outputs
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Remove printing of most and least important features
    # print("\nTop 20 Most Important Features:")
    # print("-" * 50)
    # for idx, row in feature_importance_df.head(20).iterrows():
    #     print(f"  {row['feature']:<35} {row['importance']:.4f}")
    # print("\nBottom 10 Least Important Features:")
    # print("-" * 50)
    # for idx, row in feature_importance_df.tail(10).iterrows():
    #     print(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to models/feature_importance.csv")
    print()

    # 7. Store Model
    model_path = 'models/aqi_model.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
