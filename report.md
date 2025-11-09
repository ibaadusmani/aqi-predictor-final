# Project Report: 72-Hour AQI Predictor for Islamabad

## 1. Project Overview

### Problem
Air pollution is a significant environmental and health concern in many urban areas, including Islamabad. The ability to forecast air quality, specifically the Air Quality Index (AQI), provides valuable information for public health advisories, personal planning, and environmental policy.

### Goal
The primary objective of this project is to design, build, and deploy an end-to-end data science application that predicts the AQI for Islamabad for the next 72 hours. The project follows a simplified MLOps approach, emphasizing automation and reproducibility, using a "serverless" stack suitable for rapid development and deployment.

### Core Objectives:
* **72-Hour Forecast:** Predict hourly PM2.5 values for the next 72 hours and convert them to the standard AQI.
* **Automated Data Pipeline:** An automated feature pipeline runs hourly to fetch the latest data and update the feature store.
* **Automated Retraining:** A model training pipeline runs daily to retrain the model on the freshest data.
* **Simple Stack:** The project intentionally avoids complex MLOps platforms, instead using files within the GitHub repo as a simple "feature store" (`features.parquet`) and "model registry" (`aqi_model.joblib`).

### Technology Stack
The project's dependencies are defined in `requirements.txt`:
* **Data Handling:** `pandas`, `pyarrow`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Web Application:** `streamlit`
* **API & Utilities:** `requests`, `joblib`, `plotly`

---

## 2. System Architecture & MLOps

The project follows a 4-step MLOps workflow: **Data -> Features -> Training -> App**. This entire workflow is automated using GitHub Actions.

### 1. Automated Feature Pipeline (Hourly)
* **Trigger:** Runs hourly (via `cron: '0 */1 * * *'`) or on pushes to `main`.
* **Workflow:** `.github/workflows/1_feature_pipeline.yml`.
* **Script:** Executes `src/feature_pipeline.py`. This script acts as a wrapper:
    * If `data/processed/features.parquet` exists, it runs `src/update_feature_store.py`.
    * If not, it runs `src/build_feature_store.py` to create it from scratch.
* **Logic (`src/update_feature_store.py`):**
    1.  Loads the existing `features.parquet` file.
    2.  Finds the last timestamp and fetches new API data (pollution and weather) from 26 hours *prior* to that timestamp up to the current moment. This overlap is necessary to build lag features.
    3.  Applies feature engineering from `src/utils.py` to the new raw data.
    4.  Appends new feature rows, removes duplicates based on the `dt` (timestamp), and overwrites the `features.parquet` file.
* **Commit:** The GitHub Action commits and pushes the updated `data/processed/features.parquet` and `data/processed/features.csv` back to the repository.

### 2. Automated Training Pipeline (Daily)
* **Trigger:** Runs daily at midnight UTC (via `cron: '0 0 * * *'`) or manually (`workflow_dispatch`).
* **Workflow:** `.github/workflows/2_training_pipeline.yml`.
* **Script:** Executes `src/training_pipeline.py`.
* **Logic (`src/training_pipeline.py`):**
    1.  Loads the complete `data/processed/features.parquet` dataset.
    2.  Defines 72 target columns (`pm25_t+1` to `pm25_t+72`) and selects all other relevant columns as features.
    3.  Splits the data into training and test sets (`test_size=0.2`) with **`shuffle=False`**. This chronological split is critical for time-series data to prevent leakage.
    4.  Initializes a `StandardScaler`, fits it on the training features (`X_train`), and saves it to `models/scaler.joblib`.
    5.  Trains a `RandomForestRegressor` model on the scaled training data.
    6.  Saves the trained model to `models/aqi_model.joblib`.
* **Commit:** The GitHub Action commits and pushes the updated `models/aqi_model.joblib` and `models/scaler.joblib` files back to the repository.

### 3. Inference Application (Streamlit)
* **Script:** `app.py`.
* **Logic:**
    1.  At startup, it loads `models/aqi_model.joblib` and `models/scaler.joblib` using `@st.cache_resource`.
    2.  It performs its *own* set of API calls (cached for 1 hour) to fetch the data needed for a real-time prediction:
        * Current Pollution (for the "Current AQI" metric).
        * Historical Pollution & Weather (for building lag features).
        * Forecast Pollution & Weather (for building weather features).
    3.  This data is processed by the `create_features` function from `src/utils.py` to create a single feature vector for the current time.
    4.  This vector is scaled using the loaded `scaler`.
    5.  The scaled vector is passed to the `model.predict()` function, which returns an array of 72 predicted PM2.5 values.
    6.  These values are converted to AQI using `convert_pm25_to_aqi` and displayed in a custom HTML scrolling component.

---

## 3. Data Sourcing & Feature Engineering

### Data Source
* **Provider:** OpenWeatherMap Student Plan.
* **API Key:** A default key (`3e5573c559d066b9120b40bc0c08617d`) is present in the scripts, intended to be replaced by a GitHub Secret (`OPENWEATHER_API_KEY`) in the automated workflows.
* **Location:** The project is hardcoded for Islamabad, Pakistan, using coordinates (Lat: `33.7380`, Lon: `73.0845`) stored in `src/utils.py`.
* **Endpoints Used:**
    *
