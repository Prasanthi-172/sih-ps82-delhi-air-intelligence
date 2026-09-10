"""
04_train_model.py
-----------------
PURPOSE:
Train Random Forest and HistGradientBoosting models for PM2.5 forecasting
from 1 hour ahead to 72 hours ahead.

The data is split chronologically:
    First 80%  -> training
    Last 20%   -> testing

NO RANDOM SHUFFLING is used because this is time-series forecasting.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/03_features_with_targets.csv"

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

FORECAST_HORIZON = 72

TRAIN_FRACTION = 0.8

# Increased from 30 to 100
RF_N_ESTIMATORS = 100


# ============================================================
# TARGET COLUMNS
# ============================================================

def get_target_columns():

    return [
        f"pm25_future_{h}h"
        for h in range(1, FORECAST_HORIZON + 1)
    ]


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):

    target_cols = get_target_columns()

    # Timestamp is not a numeric ML feature.
    #
    # IMPORTANT:
    # Current PM2.5 is intentionally kept as a feature because
    # it is known at the time we make a forecast.
    exclude = (
        set(target_cols)
        |
        {"timestamp"}
    )

    feature_cols = [
        c for c in df.columns
        if c not in exclude
    ]

    return feature_cols


# ============================================================
# LEAKAGE CHECKS
# ============================================================

def run_leakage_checks(
    df,
    feature_cols,
    target_cols
):

    print()
    print("Running data leakage checks...")

    messages = []

    all_passed = True


    # --------------------------------------------------------
    # 1. TARGETS MUST NOT BE FEATURES
    # --------------------------------------------------------

    overlap = (
        set(feature_cols)
        &
        set(target_cols)
    )

    if len(overlap) > 0:

        all_passed = False

        messages.append(
            "FAIL: target columns found inside features: "
            + str(overlap)
        )

    else:

        messages.append(
            "PASS: no target columns are present in features."
        )


    # --------------------------------------------------------
    # 2. MISSING VALUES
    # --------------------------------------------------------

    n_nan_features = (
        df[feature_cols]
        .isna()
        .sum()
        .sum()
    )

    n_nan_targets = (
        df[target_cols]
        .isna()
        .sum()
        .sum()
    )

    if (
        n_nan_features > 0
        or
        n_nan_targets > 0
    ):

        all_passed = False

        messages.append(
            f"FAIL: NaN values found "
            f"(features={n_nan_features}, "
            f"targets={n_nan_targets})"
        )

    else:

        messages.append(
            "PASS: no missing NaN values."
        )


    # --------------------------------------------------------
    # 3. INFINITE VALUES
    # --------------------------------------------------------

    numeric_feats = (
        df[feature_cols]
        .select_dtypes(
            include=[np.number]
        )
    )

    n_inf = (
        np.isinf(
            numeric_feats.to_numpy()
        ).sum()
    )

    if n_inf > 0:

        all_passed = False

        messages.append(
            f"FAIL: found {n_inf} infinite values."
        )

    else:

        messages.append(
            "PASS: no infinite values."
        )


    # --------------------------------------------------------
    # 4. SORTED TIMESTAMPS
    # --------------------------------------------------------

    ts = pd.to_datetime(
        df["timestamp"]
    )

    if not ts.is_monotonic_increasing:

        all_passed = False

        messages.append(
            "FAIL: timestamps are not sorted."
        )

    else:

        messages.append(
            "PASS: timestamps are sorted."
        )


    # --------------------------------------------------------
    # 5. DUPLICATE TIMESTAMPS
    # --------------------------------------------------------

    n_dupes = (
        ts.duplicated()
        .sum()
    )

    if n_dupes > 0:

        all_passed = False

        messages.append(
            f"FAIL: found {n_dupes} duplicate timestamps."
        )

    else:

        messages.append(
            "PASS: no duplicate timestamps."
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    for message in messages:

        print(
            "  " + message
        )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUTS_DIR,
        exist_ok=True
    )

    report_file = os.path.join(
        OUTPUTS_DIR,
        "04_leakage_check.txt"
    )

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "\n".join(messages)
        )

        f.write(
            "\n\nRESULT: "
        )

        if all_passed:

            f.write(
                "DATA LEAKAGE CHECK PASSED"
            )

        else:

            f.write(
                "DATA LEAKAGE CHECK FAILED"
            )


    # --------------------------------------------------------
    # STOP IF FAILED
    # --------------------------------------------------------

    if not all_passed:

        print()
        print(
            "DATA LEAKAGE CHECK FAILED."
        )

        print(
            "Fix the issues before training."
        )

        sys.exit(1)


    print()
    print(
        "DATA LEAKAGE CHECK PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading dataset..."
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):

        print(
            f"ERROR: {INPUT_FILE} not found."
        )

        return


    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # TARGETS / FEATURES
    # --------------------------------------------------------

    target_cols = (
        get_target_columns()
    )

    feature_cols = (
        get_feature_columns(df)
    )


    print()
    print(
        f"Total features: {len(feature_cols)}"
    )

    print(
        f"Total targets: {len(target_cols)}"
    )


    # --------------------------------------------------------
    # BOOLEAN -> INTEGER
    # --------------------------------------------------------

    for col in feature_cols:

        if df[col].dtype == bool:

            df[col] = (
                df[col]
                .astype(int)
            )


    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    run_leakage_checks(
        df,
        feature_cols,
        target_cols
    )


    # ========================================================
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # ========================================================

    split_index = int(
        len(df)
        *
        TRAIN_FRACTION
    )

    train_df = (
        df.iloc[:split_index]
        .copy()
    )

    test_df = (
        df.iloc[split_index:]
        .copy()
    )


    print()
    print(
        "=" * 60
    )

    print(
        "CHRONOLOGICAL SPLIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows: {len(test_df)}"
    )

    print()

    print(
        "Training period:"
    )

    print(
        train_df["timestamp"].min(),
        "to",
        train_df["timestamp"].max()
    )

    print()

    print(
        "Testing period:"
    )

    print(
        test_df["timestamp"].min(),
        "to",
        test_df["timestamp"].max()
    )


    # --------------------------------------------------------
    # X DATA
    # --------------------------------------------------------

    X_train = (
        train_df[feature_cols]
    )

    X_test = (
        test_df[feature_cols]
    )


    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    train_df.to_csv(
        "data/04_train_split.csv",
        index=False
    )

    test_df.to_csv(
        "data/04_test_split.csv",
        index=False
    )


    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    feature_file = os.path.join(
        MODELS_DIR,
        "feature_columns.txt"
    )

    with open(
        feature_file,
        "w"
    ) as f:

        f.write(
            "\n".join(feature_cols)
        )


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "TRAINING MODELS"
    )

    print(
        "=" * 60
    )

    print(
        "Random Forest trees:",
        RF_N_ESTIMATORS
    )

    print(
        "Forecast horizons:",
        FORECAST_HORIZON
    )

    print()


    rf_models = {}

    hgb_models = {}


    start_time = time.time()


    # --------------------------------------------------------
    # TRAIN 72 MODELS
    # --------------------------------------------------------

    for h in range(
        1,
        FORECAST_HORIZON + 1
    ):

        target_col = (
            f"pm25_future_{h}h"
        )

        y_train = (
            train_df[target_col]
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        rf = RandomForestRegressor(

            n_estimators=RF_N_ESTIMATORS,

            max_depth=10,

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1
        )


        rf.fit(
            X_train,
            y_train
        )


        rf_models[h] = rf


        # ====================================================
        # HIST GRADIENT BOOSTING
        # ====================================================

        hgb = HistGradientBoostingRegressor(

            max_iter=300,

            learning_rate=0.05,

            max_leaf_nodes=31,

            l2_regularization=1.0,

            random_state=42
        )


        hgb.fit(
            X_train,
            y_train
        )


        hgb_models[h] = hgb


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            h == 1
            or
            h % 12 == 0
        ):

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"Trained +{h}h models "
                f"(elapsed {elapsed:.1f}s)"
            )


    total_time = (
        time.time()
        -
        start_time
    )


    print()
    print(
        f"All models trained in "
        f"{total_time:.1f} seconds."
    )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    print()
    print(
        "Saving models..."
    )


    joblib.dump(

        rf_models,

        os.path.join(
            MODELS_DIR,
            "random_forest_models.joblib"
        )
    )


    joblib.dump(

        hgb_models,

        os.path.join(
            MODELS_DIR,
            "hist_gradient_boosting_models.joblib"
        )
    )


    print()
    print(
        "Models saved successfully."
    )

    print(
        f"Directory: {MODELS_DIR}/"
    )

    print()
    print(
        "Training complete."
    )

    print(
        "Next run:"
    )

    print(
        "python 05_evaluate_model.py"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()