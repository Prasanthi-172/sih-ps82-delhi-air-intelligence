"""
05_evaluate_model.py
---------------------
PURPOSE:
Measure how good our trained models actually are, and honestly compare them
against the simplest possible forecast (the "persistence baseline").

WHAT IS THE PERSISTENCE BASELINE?
It simply guesses: "PM2.5 at T+h will be the same as PM2.5 right now."
It requires no training at all. Any model we build MUST beat this baseline,
or it isn't worth using -- this script checks that honestly instead of
just assuming our fancier models are better.

METRICS EXPLAINED:
    MAE  (Mean Absolute Error)      -- average size of the error, in ug/m3.
                                        Easy to interpret: "on average we are
                                        off by X ug/m3."
    RMSE (Root Mean Squared Error)  -- similar to MAE but penalizes large
                                        errors more heavily. Useful for
                                        catching models that are usually
                                        fine but occasionally very wrong.
    R2   (R-squared)                -- how much of the variation in PM2.5
                                        the model explains. 1.0 = perfect,
                                        0.0 = no better than predicting the
                                        average every time, negative = worse
                                        than just guessing the average.

We report these separately for the 24h, 48h, and 72h forecast horizons
(the three headline numbers), AND for every single one of the 72 horizons
in a full table (so you can see how accuracy degrades the further out you
forecast -- which is expected and normal).

INPUT:  data/04_test_split.csv, models/*.joblib
OUTPUT: outputs/05_metrics_by_horizon.csv
        outputs/05_summary_24_48_72.csv

Run this after 04_train_model.py:
    python 05_evaluate_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_FILE = "data/04_test_split.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FORECAST_HORIZON = 72
HEADLINE_HORIZONS = [24, 48, 72]


def main():
    print("Loading dataset...")
    test_df = pd.read_csv(TEST_FILE)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    with open(os.path.join(MODELS_DIR, "feature_columns.txt")) as f:
        feature_cols = f.read().splitlines()

    for col in feature_cols:
        if test_df[col].dtype == bool:
            test_df[col] = test_df[col].astype(int)

    X_test = test_df[feature_cols]

    print("Loading trained models...")
    rf_models = joblib.load(os.path.join(MODELS_DIR, "random_forest_models.joblib"))
    hgb_models = joblib.load(os.path.join(MODELS_DIR, "hist_gradient_boosting_models.joblib"))

    print("Evaluating model...")

    rows = []

    for h in range(1, FORECAST_HORIZON + 1):
        y_true = test_df[f"pm25_future_{h}h"]

        # --- Persistence baseline: predict "same as right now" ---
        y_pred_persistence = test_df["pm2_5_ugm3"]

        # --- Random Forest ---
        y_pred_rf = rf_models[h].predict(X_test)

        # --- HistGradientBoosting ---
        y_pred_hgb = hgb_models[h].predict(X_test)

        for model_name, y_pred in [
            ("Persistence", y_pred_persistence),
            ("RandomForest", y_pred_rf),
            ("HistGradientBoosting", y_pred_hgb),
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            rows.append({
                "horizon_hours": h,
                "model": model_name,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(r2, 4),
            })

    results_df = pd.DataFrame(rows)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "05_metrics_by_horizon.csv"), index=False)

    # ------------------------------------------------------------------
    # Headline summary at 24h / 48h / 72h
    # ------------------------------------------------------------------
    summary = results_df[results_df["horizon_hours"].isin(HEADLINE_HORIZONS)]
    summary = summary.sort_values(["horizon_hours", "model"])
    summary.to_csv(os.path.join(OUTPUTS_DIR, "05_summary_24_48_72.csv"), index=False)

    print("\n" + "=" * 70)
    print("HEADLINE RESULTS (24h / 48h / 72h forecasts)")
    print("=" * 70)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Honest verdict: did our models actually beat persistence?
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HONEST COMPARISON vs PERSISTENCE BASELINE")
    print("=" * 70)
    for h in HEADLINE_HORIZONS:
        subset = results_df[results_df["horizon_hours"] == h]
        persistence_mae = subset[subset["model"] == "Persistence"]["MAE"].values[0]
        rf_mae = subset[subset["model"] == "RandomForest"]["MAE"].values[0]
        hgb_mae = subset[subset["model"] == "HistGradientBoosting"]["MAE"].values[0]

        print(f"\nHorizon +{h}h:")
        print(f"  Persistence MAE:          {persistence_mae:.2f} ug/m3")
        print(f"  Random Forest MAE:        {rf_mae:.2f} ug/m3  "
              f"({'BETTER' if rf_mae < persistence_mae else 'WORSE'} than persistence)")
        print(f"  HistGradientBoosting MAE: {hgb_mae:.2f} ug/m3  "
              f"({'BETTER' if hgb_mae < persistence_mae else 'WORSE'} than persistence)")

    print(f"\nFull 72-horizon metrics table saved to outputs/05_metrics_by_horizon.csv")
    print(f"Headline summary saved to outputs/05_summary_24_48_72.csv")


if __name__ == "__main__":
    main()
