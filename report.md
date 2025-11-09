# AQI Predictor Project Report

## 1. Introduction

### Problem
Air pollution is a significant environmental and health concern in many urban areas, including Islamabad. The ability to forecast air quality, specifically the Air Quality Index (AQI), provides valuable information for public health advisories, personal planning, and environmental policy. This project addresses the need for an accessible and accurate short-term AQI forecast.

### Goal
The primary objective of this project is to design, build, and deploy an end-to-end data science application that predicts the AQI for Islamabad for the next 72 hours. The project follows a simplified MLOps approach, emphasizing automation and reproducibility, using a "serverless" stack suitable for rapid development and deployment.

## 2. System Architecture

The project follows a 4-step MLOps workflow: **Data -> Features -> Training -> App**.

![System Architecture](https://i.imgur.com/your-diagram.png) <!-- Replace with your diagram -->

### Data Flow
1.  **Data Sourcing**: Historical and forecast data for air pollution (PM2.5) and weather (temperature, humidity, wind) are fetched from the **OpenWeatherMap Student Plan APIs**.
2.  **Feature Pipeline (`feature_pipeline.py`)**: An automated script runs hourly via a GitHub Action. It fetches the last 12 months of data, engineers a rich set of features (lags, rolling averages, cyclical time features), and stores the result in a Parquet file (`data/processed/features.parquet`), which acts as our simple feature store.
3.  **Training Pipeline (`training_pipeline.py`)**: Another automated script runs daily. It loads the processed features, splits the data chronologically, trains a `RandomForestRegressor` model to predict 72 future values of PM2.5, and saves the trained model (`aqi_model.joblib`) and data scaler (`scaler.joblib`) to the `models/` directory, our simple model registry.
4.  **Inference App (`app.py`)**: A Streamlit web application loads the latest model and scaler. To make a prediction, it fetches the required recent historical and forecast data, engineers the same features the model was trained on, and predicts the next 72 hours of AQI. The results are displayed in an interactive chart.

### Automation
*   **GitHub Actions** are used to automate the entire backend pipeline.
    *   The feature pipeline runs hourly to keep the dataset fresh.
    *   The training pipeline runs daily to ensure the model adapts to new data patterns.
    *   Changes to the feature store and model registry are automatically committed back to the Git repository.

## 3. Data Sourcing & EDA

### Data Source
The project exclusively uses the **OpenWeatherMap Student Plan**, which provides access to several crucial API endpoints:
*   **Air Pollution API** (Historical & Forecast)
*   **History API** (Weather)
*   **Hourly Forecast 4 days API** (Weather)

### EDA Findings
The initial data exploration was conducted in the `notebooks/01_data_exploration.ipynb` notebook. Key findings include:
*   **PM2.5 as the Target**: PM2.5 was confirmed as the most critical pollutant and the primary target for our prediction model.
*   **Seasonality and Trends**: A plot of PM2.5 over time revealed clear patterns, with higher concentrations often observed during specific seasons (e.g., winter) and times of day.
*   **Correlation with Weather**: A correlation matrix showed a noticeable relationship between PM2.5 and weather variables. For instance, higher wind speeds often correlated with lower PM2.5 concentrations, likely due to the dispersion of pollutants. Temperature and humidity also showed complex interactions with air quality.

*(Example Plot from EDA)*
![PM2.5 Time Series](https://i.imgur.com/your-plot.png) <!-- Replace with a plot from your notebook -->

## 4. Methodology & Feature Engineering

### Feature Engineering
This was a critical part of the project. The function `create_features` in `src/utils.py` was designed to generate a comprehensive set of features to capture the temporal dynamics of air pollution.
*   **Time-based Features**: Cyclical features (`hour_sin`, `hour_cos`), `day_of_week`, and `month` were created to capture daily, weekly, and yearly patterns.
*   **Lag Features (Auto-regressive)**: `pm25_lag_1hr`, `pm25_lag_3hr`, and `pm25_lag_24hr` were created because the PM2.5 value at a given time is highly dependent on its recent past values.
*   **Rolling Averages**: `pm25_rolling_avg_6hr` and `pm25_rolling_avg_24hr` were used to smooth out short-term fluctuations and capture recent trends.
*   **Weather Features**: Raw values for `temp`, `humidity`, `wind_speed`, and `wind_dir` were included as direct predictors.
*   **Interaction Features**: A `temp * wind_speed` feature was created to capture the combined effect of temperature and wind on pollutant concentration.

### Model Selection
A **`RandomForestRegressor`** from Scikit-learn was chosen for this multi-output regression task.
*   **Rationale**: It is a powerful ensemble model that performs well out-of-the-box, is robust to outliers, and can capture non-linear relationships in the data. Its ability to handle multiple target variables simultaneously (predicting all 72 hours at once) simplifies the modeling process significantly compared to training 72 separate models.

### Evaluation
*   **Metrics**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) were used to evaluate the model's performance. These metrics provide a clear understanding of the model's prediction error in the original units of the target (μg/m³).
*   **Time-Series Split**: It was **critical** to split the training and testing data chronologically (`shuffle=False`). Shuffling time-series data would lead to data leakage, where the model is trained on future data and tested on past data, resulting in an unrealistically optimistic performance evaluation.

## 5. Results

The model was evaluated on a hold-out test set (the most recent 20% of the data). The performance metrics for key prediction horizons were as follows:

| Horizon | RMSE (μg/m³) | MAE (μg/m³) |
| :------ | :------------- | :------------ |
| **t+1**   | [Value]        | [Value]       |
| **t+24**  | [Value]        | [Value]       |
| **t+72**  | [Value]        | [Value]       |

*(Note: Replace `[Value]` with the actual numbers printed by your `training_pipeline.py` script after you run it.)*

### Interpretation
The results indicate that the model is most accurate for short-term forecasts (e.g., 1 hour ahead) and its error increases for longer-term horizons (e.g., 72 hours ahead), which is expected. The MAE for the 24-hour forecast indicates that, on average, the model's prediction for PM2.5 is off by approximately [Value] μg/m³.

## 6. Conclusion & Future Work

### Summary
This project successfully delivered an end-to-end, automated AQI forecasting system. It demonstrates a practical, simplified MLOps workflow using a serverless stack of Python scripts, GitHub Actions, and Streamlit. The final application provides a valuable 72-hour forecast for Islamabad, backed by an automated data and model pipeline.

### Challenges
The most complex part of the project was the **inference logic in `app.py`**. To predict for the *current* moment, the model requires a feature vector that itself depends on both past *and* future data (historical data for lags, forecast data for weather). Assembling this single row of data correctly required careful fetching, combining, and processing of data from multiple API endpoints.

### Future Work
*   **Advanced Models**: Experiment with more sophisticated time-series models like LSTMs or GRUs, which might capture long-term dependencies more effectively.
*   **Model Explainability**: Integrate SHAP (SHapley Additive exPlanations) to explain individual predictions, helping users understand *why* the AQI is predicted to be high or low.
*   **Expand to More Locations**: Generalize the application to allow users to select other cities, rather than hardcoding Islamabad.
*   **More Granular Error Metrics**: Display horizon-specific error estimates in the app to give users a better sense of the forecast's confidence over time.
