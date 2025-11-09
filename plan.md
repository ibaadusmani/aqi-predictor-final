Project Plan: 72-Hour AQI Forecast (Student Plan)

Goal: Build an end-to-end data science project to predict the Air Quality Index (AQI) for a specific location (e.g., Islamabad) for the next 72 hours, using the OpenWeatherMap Student Plan. This plan is designed to be a comprehensive context file for a development agent.

Project Principles:

Simplified MLOps: Follow the spirit of the 4-step diagram (Data, Features, Training, App) but use simpler tools.

"Serverless" Stack: Rely on Python scripts, GitHub Actions for automation, and Streamlit for the UI.

No Dedicated Feature Store: As requested, we will not use Hopsworks. We will use files within the GitHub repo (data/processed/features.parquet) as our simple "feature store" and "model registry".

Phase 0: Setup & Initial Data Exploration

Objective: Prepare the project environment, select data sources, and understand the data.

Project Structure:

aqi-predictor/
├── .github/
│   └── workflows/
│       ├── 1_feature_pipeline.yml
│       └── 2_training_pipeline.yml
├── data/
│   ├── raw/
│   └── processed/
│       └── features.parquet
├── models/
│   ├── scaler.joblib
│   └── aqi_model.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── utils.py
├── app.py
├── requirements.txt
├── report.md
└── README.md


Environment:

Set up a Python virtual environment: python -m venv venv

Create requirements.txt with initial libraries:

pandas
scikit-learn
requests
streamlit
plotly
pyarrow  # For parquet files
joblib


Data Source Selection & EDA:

Data Source: OpenWeatherMap Student Plan. This plan provides all necessary APIs.

API Key: Use your API key (3e5573c559d066b9120b40bc0c08617d). Store this as a GitHub Secret (e.g., OPENWEATHER_API_KEY) for GitHub Actions. For local development, it will be loaded from an environment variable.

Location:

We will hardcode the coordinates for Islamabad to simplify the project.

Store these lat (33.7380) and lon (73.0845) values as constants in src/utils.py to be used in all other API calls.

EDA (01_data_exploration.ipynb):

Fetch sample data for Islamabad using its lat/lon from:

Air Pollution API (historical)

History API (weather)

Hourly Forecast 4 days (weather)

Confirm you can merge them on their dt (timestamp).

Analyze the components (PM2.5, NO2, O3, etc.) and confirm PM2.5 as the target for prediction.

Phase 1: Feature Pipeline (Data -> Features)

Objective: Create a script to fetch historical data, engineer features, and save them.

File: src/feature_pipeline.py

Define Shared Logic:

The feature engineering logic will be complex. Create a function create_features(df) in src/utils.py so it can be reused by both feature_pipeline.py and app.py.

Fetch Data (Last 12 Months):

Load the API_KEY and ISB_LAT, ISB_LON constants.

Define a start and end timestamp (e.g., 12 months ago to today).

Pollution Data: Call the Air Pollution API (history endpoint).

http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}

Parse the list of hourly data. Key is list[0].components.pm2_5.

Weather Data: Call the History API.

https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_KEY}

Parse the list of hourly data. Keys are list[0].main.temp, main.humidity, wind.speed, wind.deg.

Merge: Combine both lists into a single Pandas DataFrame, using the dt (timestamp) as the index.

Feature Engineering (in src/utils.py):

Create the create_features(df, is_training=True) function.

Input: A DataFrame with pm25, temp, humidity, wind_speed, wind_dir.

Target Variables (if is_training=True):

df['pm25_t+1'] = df['pm25'].shift(-1)

...

df['pm25_t+72'] = df['pm25'].shift(-72)

Features to Create (for all data):

Time-based: hour_sin, hour_cos (cyclical), day_of_week, month.

Lag Features (Auto-regressive): pm25_lag_1hr, pm25_lag_3hr, pm25_lag_24hr.

Rolling Averages: pm25_rolling_avg_6hr, pm25_rolling_avg_24hr.

Weather Features: temp, humidity, wind_speed, wind_dir.

Interaction Features: e.g., temp * wind_speed.

Clean Data (if is_training=True):

df = df.dropna() (This drops rows with NaNs created by lags/shifts).

Return df.

Store Features:

In feature_pipeline.py, call df = create_features(merged_df, is_training=True).

Save the final DataFrame: df.to_parquet('data/processed/features.parquet').

Phase 2: Model Training Pipeline (Features -> Model)

Objective: Create a script that loads processed features, trains a model, and saves it.

File: src/training_pipeline.py

Load Data:

Read data/processed/features.parquet.

Prepare Data for Training:

Define X and y:

target_cols = [f'pm25_t+{i}' for i in range(1, 73)]

feature_cols = [col for col in df.columns if col not in target_cols]

y = df[target_cols]

X = df[feature_cols]

Split Data: CRITICAL: Do not shuffle. This is time-series data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

Scale Data:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

Save this scaler: joblib.dump(scaler, 'models/scaler.joblib').

Train Model:

model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

model.fit(X_train_scaled, y_train)

Evaluate Model:

y_pred = model.predict(X_test_scaled)

Calculate and print RMSE and MAE for key horizons (e.g., t+1, t+24, t+72).

Store Model:

Save the model: joblib.dump(model, 'models/aqi_model.joblib').

Phase 3: Automation (GitHub Actions)

Objective: Automate the feature and training pipelines.

File: .github/workflows/1_feature_pipeline.yml

Trigger: On a schedule (e.g., every hour: cron: '0 * * * *') and on push to main (for testing).

Steps:

Check out the code.

Set up Python and install requirements.txt.

Run the feature pipeline: python src/feature_pipeline.py (with the API key passed as a secret: env: OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}).

Commit and push the updated data/processed/features.parquet file.

File: .github/workflows/2_training_pipeline.yml

Trigger: On a schedule (e.g., daily at midnight: cron: '0 0 * * *') or manually (workflow_dispatch).

Steps:

Check out the code.

Set up Python and install requirements.txt.

Run the training pipeline: python src/training_pipeline.py.

Commit and push the updated models/aqi_model.joblib and models/scaler.joblib.

Phase 4: Inference App (Streamlit)

Objective: Create a simple web app to display the 72-hour forecast.

File: app.py

Title: st.title('AQI 72-Hour Forecast for Islamabad')

Load Artifacts (at startup):

model = joblib.load('models/aqi_model.joblib')

scaler = joblib.load('models/scaler.joblib')

from src.utils import create_features, convert_pm25_to_aqi (This function needs to be created).

Get Input Data for Prediction (The hard part):

This logic is complex. The model needs a single row of features, which requires historical data for lags and forecasted data for weather.

Step 1: Get Forecast Data (Next 72 hours)

Call Air Pollution API (forecast): .../air_pollution/forecast -> list[0...71].components.pm2_5

Call Hourly Forecast 4 days API: .../forecast/hourly -> list[0...71].main.temp, .humidity, etc.

Create a forecast_df with pm25, temp, etc. for the future 72 hours.

Step 2: Get Historical Data (Last 24+ hours for lags)

Call Air Pollution API (history) for the last 24 hours.

Call History API (weather) for the last 24 hours.

Create a historical_df with the past 24 hours.

Step 3: Combine and Build Feature Row

combined_df = pd.concat([historical_df, forecast_df])

Call features_df = create_features(combined_df, is_training=False).

Select the first row that corresponds to the current time: X_pred = features_df.iloc[len(historical_df)] (This row's lag features will be built from the past, and its weather features will be from the present).

Step 4: Scale the Input

X_pred_scaled = scaler.transform(X_pred.to_frame().T) (Must be a 2D array).

Make & Display Prediction:

y_pred_pm25 = model.predict(X_pred_scaled)

This will return an array [val1, val2, ..., val72].

Convert PM2.5 to AQI: Use convert_pm25_to_aqi(y_pred_pm25) (This function needs to be defined in src/utils.py based on standard AQI calculation tables).

Create a DataFrame for the plot: forecast_plot_df = pd.DataFrame({'Timestamp': future_timestamps, 'Predicted_AQI': predicted_aqi}).

Display:

Show the current AQI (from historical_df.iloc[-1]).

Show a line chart: st.line_chart(forecast_plot_df, x='Timestamp', y='Predicted_AQI').

Add alerts: if max(predicted_aqi) > 150: st.warning("Hazardous AQI levels expected!").

Phase 5: Shared Utilities

Objective: Store shared logic to avoid code duplication (DRY principle).

File: src/utils.py

ISB_LAT, ISB_LON: Constants for Islamabad's coordinates (e.g., ISB_LAT = 33.7380, ISB_LON = 73.0845).

create_features(df, is_training=True): The feature engineering function described in Phase 1.

convert_pm25_to_aqi(pm25_value): A function that takes a PM2.5 value (or array) and converts it to the standard Air Quality Index (AQI) value. This requires a lookup table (piecewise linear function).

Phase 6: Documentation & Reporting

Objective: Document the project for submission and public sharing.

README.md (The "Shop Window"):

Project Title & Badge: "72-Hour AQI Predictor for Islamabad".

Description: A 1-paragraph summary of the project's goal and the tech stack (Python, Streamlit, GitHub Actions).

Live App: A link to your deployed Streamlit Community Cloud app.

Screenshot: A screenshot of the final dashboard.

Features: Bullet points (e.g., "72-hour PM2.5 forecast", "Automated daily model retraining").

How to Run Locally:

git clone ...

pip install -r requirements.txt

streamlit run app.py

report.md (The Detailed Report - per your PDF):

This is the detailed report documenting your work.

Introduction:

Problem: The problem of air pollution in Islamabad and the need for accurate forecasts.

Goal: State the project's objective (72-hour forecast, end-to-end pipeline).

System Architecture:

Include the 4-step diagram from your PDF.

Explain the data flow: APIs -> feature_pipeline.py -> features.parquet -> training_pipeline.py -> aqi_model.joblib -> app.py.

Explain the automation (GitHub Actions).

Data Sourcing & EDA:

What APIs you used (OpenWeatherMap Student Plan).

Key findings from your EDA (01_data_exploration.ipynb). Show 1-2 plots (e.g., PM2.5 over time) and- explain any trends you found.

Methodology & Feature Engineering:

Feature Engineering: This is the most important part. List the features you created (lags, rolling averages, cyclical time features, weather) and why you created them.

Model Selection: Explain why you chose RandomForestRegressor (e.g., good baseline, handles multi-output).

Evaluation: Explain your evaluation metric (RMSE/MAE) and why shuffle=False was critical.

Results:

Show your final model's performance (the RMSE/MAE scores from training_pipeline.py).

Interpret the results (e.g., "The model is, on average, off by X μg/m³ for a 24-hour forecast").

(Optional but good) Include a plot of y_test vs. y_pred for one horizon (e.g., t+24).

Conclusion & Future Work:

Summarize what you accomplished.

Challenges: What was the hardest part? (e.g., "The inference logic in app.py was complex").

Future Work: What you would do next (e.g., "Use Deep Learning models like LSTMs", "Add SHAP explanations", "Add more cities").