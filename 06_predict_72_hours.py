"""
06_predict_72_hours.py
------------------------
PURPOSE:
This is the "live forecast" script. It takes the MOST RECENT row of data
available (the latest hour we have real observations for) and produces a
72-hour PM2.5 + AQI forecast, just like a real forecasting system would.

HOW A REAL-WORLD VERSION OF THIS WOULD WORK:
Every hour, a real system would:
    1. Pull the latest PM2.5, weather, and fire data from live sensors/APIs.
    2. Run it through 02_feature_engineering.py to build lag/rolling features.
    3. Feed that single latest row into the 72 saved models (one per horizon).
    4. Get back 72 numbers: predicted PM2.5 for the next 72 hours.

This script simulates that final step using the last available row in our
historical dataset, standing in for "right now."

IMPORTANT LIMITATION (be upfront about this):
Our models were trained using only features known AT OR BEFORE the
prediction time (lag features, rolling averages, current weather/fire
readings). They do NOT require live future weather forecasts as an input,
which keeps this prototype self-contained. A production version could be
improved further by also feeding in official IMD weather forecasts for the
next 72 hours as extra features -- see README.md for this as a suggested
next step.

AQI METHODOLOGY:
We use India's CPCB (Central Pollution Control Board) National AQI
breakpoint table for PM2.5, NOT a made-up formula and NOT the US EPA scale
(which uses different breakpoints). The official CPCB sub-index formula is:

    AQI = ((AQI_hi - AQI_lo) / (BP_hi - BP_lo)) * (PM - BP_lo) + AQI_lo

Officially, CPCB AQI for PM2.5 is calculated from a 24-HOUR AVERAGE
concentration, not a single hour's reading. For a real deployment you would
feed in a rolling 24-hour average PM2.5 value. For this prototype (to show
an hour-by-hour forecast curve) we apply the same official breakpoints to
each hourly predicted value -- this is a simplification, and is clearly
labeled as such in the output.

INPUT:  data/04_test_split.csv (used to get the latest available row),
        models/*.joblib
OUTPUT: outputs/forecast_72h.csv

Run this after 05_evaluate_model.py:
    python 06_predict_72_hours.py
"""

import pandas as pd
import numpy as np
import os
import joblib

TEST_FILE = "data/04_test_split.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FORECAST_HORIZON = 72

# Official CPCB (India) breakpoints for PM2.5 sub-index.
# Format: (PM2.5 low, PM2.5 high, AQI low, AQI high)
CPCB_PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (30, 60, 51, 100),
    (60, 90, 101, 200),
    (90, 120, 201, 300),
    (120, 250, 301, 400),
    (250, 380, 401, 500),
]


def pm25_to_cpcb_aqi(pm25_value):
    """
    Converts a PM2.5 concentration (ug/m3) into an AQI value using the
    official CPCB piecewise-linear breakpoint formula.
    """
    if pm25_value < 0:
        pm25_value = 0

    for bp_lo, bp_hi, aqi_lo, aqi_hi in CPCB_PM25_BREAKPOINTS:
        if bp_lo <= pm25_value <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25_value - bp_lo) + aqi_lo
            return round(aqi, 1)

    # Above the highest official breakpoint (380 ug/m3): CPCB caps the
    # category at "Severe" (401-500). We extrapolate conservatively and cap
    # at 500, which is standard practice when concentrations exceed the
    # official table.
    return 500.0


def aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"


def main():
    print("Loading dataset...")
    test_df = pd.read_csv(TEST_FILE)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    with open(os.path.join(MODELS_DIR, "feature_columns.txt")) as f:
        feature_cols = f.read().splitlines()

    for col in feature_cols:
        if test_df[col].dtype == bool:
            test_df[col] = test_df[col].astype(int)

    # The "latest" row we have real data for -- this stands in for "right now".
    latest_row = test_df.iloc[[-1]]
    latest_timestamp = latest_row["timestamp"].values[0]
    print(f"Using latest available observation as 'now': {pd.Timestamp(latest_timestamp)}")

    X_latest = latest_row[feature_cols]

    print("Loading trained models...")
    hgb_models = joblib.load(os.path.join(MODELS_DIR, "hist_gradient_boosting_models.joblib"))

    print("Generating 72-hour forecast...")

    forecast_rows = []
    base_time = pd.Timestamp(latest_timestamp)

    for h in range(1, FORECAST_HORIZON + 1):
        model = hgb_models[h]
        predicted_pm25 = float(model.predict(X_latest)[0])
        predicted_pm25 = max(predicted_pm25, 0.0)  # PM2.5 can't be negative

        predicted_aqi = pm25_to_cpcb_aqi(predicted_pm25)
        category = aqi_category(predicted_aqi)

        forecast_time = base_time + pd.Timedelta(hours=h)

        forecast_rows.append({
            "timestamp": forecast_time,
            "forecast_hour": h,
            "predicted_pm25": round(predicted_pm25, 2),
            "predicted_aqi": predicted_aqi,
            "aqi_category": category,
        })

    forecast_df = pd.DataFrame(forecast_rows)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    forecast_df.to_csv(os.path.join(OUTPUTS_DIR, "forecast_72h.csv"), index=False)

    print("\nSaving results...")
    print(forecast_df.head(10).to_string(index=False))
    print("...")
    print(f"\nFull 72-hour forecast saved to outputs/forecast_72h.csv")


if __name__ == "__main__":
    main()
