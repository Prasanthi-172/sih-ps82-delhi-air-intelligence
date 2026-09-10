"""
08_plot_results.py
--------------------
PURPOSE:
Create the visualizations needed to explain and demonstrate the project.

CREATES 4 PLOTS:
    1. Actual vs Predicted PM2.5 (24h horizon, across the test period)
       -- shows how well the model tracks real pollution swings over time.
    2. Error comparison across 24h / 48h / 72h horizons, for all 3 models
       -- shows how forecast accuracy degrades the further ahead you look,
          and whether our models still beat the persistence baseline.
    3. Feature importance chart (re-plotted here for convenience;
       07_feature_importance.py also saves this).
    4. 72-hour forecast curve -- the actual "product" of this project,
       plotting outputs/forecast_72h.csv as a line chart with AQI category
       shading.

INPUT:  data/04_test_split.csv, models/*.joblib,
        outputs/05_metrics_by_horizon.csv, outputs/forecast_72h.csv
OUTPUT: plots/08_actual_vs_predicted.png
        plots/08_error_by_horizon.png
        plots/08_72hour_forecast_curve.png

Run this LAST, after 05, 06, and 07:
    python 08_plot_results.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TEST_FILE = "data/04_test_split.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
PLOTS_DIR = "plots"


def plot_actual_vs_predicted():
    print("Creating plot 1: Actual vs Predicted PM2.5 (24h horizon)...")

    test_df = pd.read_csv(TEST_FILE)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    with open(os.path.join(MODELS_DIR, "feature_columns.txt")) as f:
        feature_cols = f.read().splitlines()
    for col in feature_cols:
        if test_df[col].dtype == bool:
            test_df[col] = test_df[col].astype(int)

    hgb_models = joblib.load(os.path.join(MODELS_DIR, "hist_gradient_boosting_models.joblib"))
    model_24h = hgb_models[24]

    X_test = test_df[feature_cols]
    y_true = test_df["pm25_future_24h"]
    y_pred = model_24h.predict(X_test)

    # Plot the forecast-target timestamp (i.e. when the +24h prediction
    # actually applies), not the "now" timestamp, so the two lines line up.
    forecast_times = test_df["timestamp"] + pd.Timedelta(hours=24)

    plt.figure(figsize=(13, 5))
    plt.plot(forecast_times, y_true, label="Actual PM2.5", color="#2c3e50", linewidth=1)
    plt.plot(forecast_times, y_pred, label="Predicted PM2.5 (+24h)", color="#e74c3c",
              linewidth=1, alpha=0.8)
    plt.title("Actual vs Predicted PM2.5 -- 24-Hour-Ahead Forecast (Test Period)")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_actual_vs_predicted.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved to {path}")


def plot_error_by_horizon():
    print("Creating plot 2: Error comparison across 24h/48h/72h horizons...")

    metrics_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "05_metrics_by_horizon.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MAE across ALL 72 horizons, one line per model
    for model_name, color in [("Persistence", "#95a5a6"),
                               ("RandomForest", "#2980b9"),
                               ("HistGradientBoosting", "#c0392b")]:
        subset = metrics_df[metrics_df["model"] == model_name].sort_values("horizon_hours")
        axes[0].plot(subset["horizon_hours"], subset["MAE"], label=model_name, color=color)

    axes[0].set_xlabel("Forecast Horizon (hours)")
    axes[0].set_ylabel("MAE (ug/m3)")
    axes[0].set_title("MAE vs Forecast Horizon (1h to 72h)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: bar chart of MAE at exactly 24h/48h/72h for all 3 models
    headline = metrics_df[metrics_df["horizon_hours"].isin([24, 48, 72])]
    pivot = headline.pivot(index="horizon_hours", columns="model", values="MAE")
    pivot.plot(kind="bar", ax=axes[1], color=["#2980b9", "#c0392b", "#95a5a6"])
    axes[1].set_title("MAE at 24h / 48h / 72h")
    axes[1].set_ylabel("MAE (ug/m3)")
    axes[1].set_xlabel("Forecast Horizon (hours)")
    axes[1].legend(title="Model")
    axes[1].grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_error_by_horizon.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved to {path}")


def plot_72_hour_forecast():
    print("Creating plot 3: 72-hour forecast curve...")

    forecast_path = os.path.join(OUTPUTS_DIR, "forecast_72h.csv")
    if not os.path.exists(forecast_path):
        print("  SKIPPED: outputs/forecast_72h.csv not found. "
              "Run 06_predict_72_hours.py first.")
        return

    forecast_df = pd.read_csv(forecast_path)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(forecast_df["timestamp"], forecast_df["predicted_pm25"],
              color="#c0392b", linewidth=2, label="Predicted PM2.5")
    ax1.set_ylabel("Predicted PM2.5 (ug/m3)", color="#c0392b")
    ax1.set_xlabel("Forecast Time")
    ax1.set_title("72-Hour PM2.5 Forecast")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    plt.xticks(rotation=45)
    ax1.grid(alpha=0.3)

    # AQI category background shading for context
    category_colors = {
        "Good": "#a9dfbf", "Satisfactory": "#f9e79f", "Moderate": "#f5cba7",
        "Poor": "#edbb99", "Very Poor": "#e6b0aa", "Severe": "#cd6155",
    }
    for _, row in forecast_df.iterrows():
        ax1.axvspan(row["timestamp"] - pd.Timedelta(minutes=30),
                    row["timestamp"] + pd.Timedelta(minutes=30),
                    color=category_colors.get(row["aqi_category"], "white"), alpha=0.15)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_72hour_forecast_curve.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved to {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_actual_vs_predicted()
    plot_error_by_horizon()
    plot_72_hour_forecast()
    print("\nAll plots created in the plots/ folder.")


if __name__ == "__main__":
    main()
