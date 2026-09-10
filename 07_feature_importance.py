"""
07_feature_importance.py
--------------------------
PURPOSE:
Answer the question: "WHAT is the model actually using to make its
predictions?" This matters a lot for an SIH demo -- judges will want to see
that your model is using real physical signals (inversion, PBL height,
upwind fires, PM2.5 history) rather than something meaningless.

HOW IT WORKS:
Random Forest models can tell you, for each feature, how much it helped
reduce prediction error across all the decision trees inside the forest.
This is called "feature importance". We extract this for the 24h, 48h,
and 72h horizon models specifically (the three headline forecasts).

INPUT:  models/random_forest_models.joblib, models/feature_columns.txt
OUTPUT: outputs/07_feature_importance.csv
        plots/07_feature_importance.png

Run this after 04_train_model.py (does not require 05 or 06 first):
    python 07_feature_importance.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
PLOTS_DIR = "plots"
HORIZONS_TO_CHECK = [24, 48, 72]
TOP_N_FEATURES = 15

# Feature name fragments we specifically want to call out, as requested --
# this helps confirm the model is really using physical drivers.
FEATURES_OF_INTEREST = [
    "pm25_lag", "pm25_avg", "boundary_layer_height", "inversion",
    "wind_speed", "relative_humidity", "temperature_2m",
    "punjab_fire", "haryana_fire", "uttar_pradesh_fire",
    "delhi_ncr_fire", "upwind_stubble_fire",
]


def main():
    print("Loading trained models...")
    rf_models = joblib.load(os.path.join(MODELS_DIR, "random_forest_models.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_columns.txt")) as f:
        feature_cols = f.read().splitlines()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_importance_rows = []

    for h in HORIZONS_TO_CHECK:
        model = rf_models[h]
        importances = model.feature_importances_

        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        imp_df["horizon_hours"] = h
        all_importance_rows.append(imp_df)

        print(f"\nTop {TOP_N_FEATURES} features for +{h}h forecast:")
        print(imp_df.head(TOP_N_FEATURES)[["feature", "importance"]].to_string(index=False))

    combined = pd.concat(all_importance_rows, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUTS_DIR, "07_feature_importance.csv"), index=False)

    # ------------------------------------------------------------------
    # Check whether the "features of interest" show up prominently
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECKING KEY PHYSICAL DRIVERS (24h horizon model)")
    print("=" * 60)
    h24 = combined[combined["horizon_hours"] == 24].sort_values("importance", ascending=False)
    h24 = h24.reset_index(drop=True)
    h24["rank"] = h24.index + 1

    for keyword in FEATURES_OF_INTEREST:
        matches = h24[h24["feature"].str.contains(keyword, case=False)]
        if len(matches) > 0:
            best = matches.iloc[0]
            print(f"  '{keyword}' -> best match: {best['feature']} "
                  f"(rank {best['rank']} of {len(h24)}, importance={best['importance']:.4f})")
        else:
            print(f"  '{keyword}' -> no matching feature found")

    # ------------------------------------------------------------------
    # Plot: top features for the 24h horizon model
    # ------------------------------------------------------------------
    top_features = h24.head(TOP_N_FEATURES)

    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#2980b9")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {TOP_N_FEATURES} Features -- Random Forest, 24h Forecast")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "07_feature_importance.png")
    plt.savefig(plot_path, dpi=130)
    plt.close()

    print(f"\nSaving results...")
    print(f"Feature importance table saved to outputs/07_feature_importance.csv")
    print(f"Feature importance chart saved to {plot_path}")


if __name__ == "__main__":
    main()
