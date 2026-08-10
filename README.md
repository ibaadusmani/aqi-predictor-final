# 72-Hour AQI Predictor for Islamabad

[![CI/CD](https://github.com/your-username/your-repo/actions/workflows/1_feature_pipeline.yml/badge.svg)](https://github.com/your-username/your-repo/actions)

A data science project to forecast the Air Quality Index (AQI) for Islamabad up to 72 hours in advance. This project uses a "serverless" stack with Python, Streamlit, and GitHub Actions for MLOps.
<!-- Replace with your actual screenshot -->

## 🚀 Live App

https://aqi-predictor-final.streamlit.app/)

## ✨ Features

*   **72-Hour PM2.5 Forecast:** Predicts hourly PM2.5 values, which are then converted to the standard AQI.
*   **Automated Feature Pipeline:** A GitHub Action runs hourly to fetch the latest weather and pollution data, process it, and update the feature set.
*   **Automated Model Retraining:** A GitHub Action runs daily to retrain the `RandomForestRegressor` model on the latest data, ensuring the model adapts to new patterns.
*   **Simple & Effective Stack:** Built with Python, Pandas, Scikit-learn, and deployed with Streamlit.

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set the API Key:**
    The app has a default key, but it's best to use your own.
    ```bash

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

```
aqi-predictor/
├── .github/workflows/   # GitHub Actions for automation
│   ├── 1_feature_pipeline.yml
│   └── 2_training_pipeline.yml
├── data/
│   ├── processed/features.parquet  # The "feature store"
│   └── raw/
├── models/              # The "model registry"
│   ├── aqi_model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/                 # Source code for pipelines and utils
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── utils.py
├── app.py               # The Streamlit web application
├── requirements.txt
├── report.md            # Detailed project report
└── README.md
```

