# PS 82 — Air Pollution–Weather Coupled Forecasting System (Delhi NCR)

A 72-hour PM2.5 / AQI forecasting pipeline for Delhi NCR that uses weather,
pollution, and stubble-fire history as inputs. Built for Smart India
Hackathon Problem Statement 82.

---

## 1. Folder Structure

```
sih_ps82/
├── data/
│   ├── delhincr.csv                     <- your original raw file (NEVER modified)
│   ├── 02_features.csv                  <- created by script 2
│   ├── 03_features_with_targets.csv     <- created by script 3
│   ├── 04_train_split.csv               <- created by script 4
│   └── 04_test_split.csv                <- created by script 4
├── models/
│   ├── feature_columns.txt
│   ├── random_forest_models.joblib
│   └── hist_gradient_boosting_models.joblib
├── outputs/
│   ├── 01_validation_report.txt
│   ├── 04_leakage_check.txt
│   ├── 05_metrics_by_horizon.csv
│   ├── 05_summary_24_48_72.csv
│   ├── 07_feature_importance.csv
│   └── forecast_72h.csv
├── plots/
│   ├── 07_feature_importance.png
│   ├── 08_actual_vs_predicted.png
│   ├── 08_error_by_horizon.png
│   └── 08_72hour_forecast_curve.png
├── 01_validate_data.py
├── 02_feature_engineering.py
├── 03_create_targets.py
├── 04_train_model.py
├── 05_evaluate_model.py
├── 06_predict_72_hours.py
├── 07_feature_importance.py
├── 08_plot_results.py
├── requirements.txt
└── README.md
```

---

## 2. Setup (VS Code)

1. Open the `sih_ps82` folder in VS Code.
2. Make sure `data/delhincr.csv` is present (your raw dataset).
3. Open a terminal in VS Code (`Terminal` → `New Terminal`) and run:

```bash
pip install -r requirements.txt
```

If a package fails to install, the exact command needed is printed by pip.
The most common one you'll need manually is:

```bash
pip install scikit-learn
```

---

## 3. Exact Commands To Run (in order)

Run these one at a time, in this exact order, from the `sih_ps82` folder:

```bash
python 01_validate_data.py
python 02_feature_engineering.py
python 03_create_targets.py
python 04_train_model.py
python 05_evaluate_model.py
python 06_predict_72_hours.py
python 07_feature_importance.py
python 08_plot_results.py
```

**Note on runtime:** `04_train_model.py` trains 2 models (Random Forest +
HistGradientBoosting) for each of the 72 forecast horizons — that's 144
models total. On a typical laptop this takes roughly **5–10 minutes**. This
is normal; the script prints progress every 12 horizons so you can see it's
working.

---

## 4. What Each File Does

| File | What it does |
|---|---|
| `01_validate_data.py` | Checks the raw CSV for missing hours, duplicates, NaNs — never modifies it |
| `02_feature_engineering.py` | Builds lag features, rolling averages, fire history, time features, wind sin/cos encoding |
| `03_create_targets.py` | Creates the 72 `pm25_future_Nh` target columns by shifting PM2.5 forward |
| `04_train_model.py` | Chronological train/test split, data-leakage checks, trains Persistence/RF/HGB models per horizon |
| `05_evaluate_model.py` | Computes MAE/RMSE/R² per horizon, honestly compares against the persistence baseline |
| `06_predict_72_hours.py` | Produces a live-style 72-hour forecast from the latest available data, with CPCB AQI |
| `07_feature_importance.py` | Shows which features the model actually relies on |
| `08_plot_results.py` | Generates all 4 result plots |

---

## 5. Why Chronological Splitting (Not Random Shuffle)

If we randomly shuffled rows before splitting into train/test, some
training rows would end up occurring *after* some test rows in real time.
The model could then indirectly "see the future" through similar nearby
conditions, making test accuracy look artificially good — a problem called
**data leakage**.

Instead we split by date:
- **Training data:** 2 Jan 2025 – 17 Oct 2025 (first 80%)
- **Testing data:** 17 Oct 2025 – 28 Dec 2025 (last 20%)

This matches reality: a real forecasting system only ever has the past to
learn from, and must predict a future it has never seen. Conveniently, our
test period covers the stubble-burning season and the start of winter —
exactly the hardest, most important period to forecast correctly.

---

## 6. Actual Results (from this run)

### Why accuracy looked low initially — the real diagnosis

Before adding features, we checked *why* — and it's a data problem, not (primarily)
a modeling problem: the chronological test period (17 Oct – 28 Dec) has a
**49% higher mean PM2.5** (108 µg/m³) than the training period (72 µg/m³),
and far wilder swings (std 56 vs 39). The split accidentally put the entire
stubble-burning + winter-inversion season into the test set, while the
model trained mostly on calmer months. Three compounding causes:

1. **Distribution shift** — the model is graded on exactly the conditions
   it saw least of during training.
2. **No future weather as input** — at 48–72h the model has zero
   information about whether an inversion will form between now and
   then; it's genuinely forecasting blind on the atmospheric side.
3. **Tree models don't extrapolate** — Random Forest/HGB predict by
   averaging training examples, so when test PM2.5 hits 300+ and
   training rarely saw values that high, the model structurally
   under-predicts the extremes (visible in `08_actual_vs_predicted.png`).

### Feature engineering experiment

We added 21 new features to directly target the above (see
`02_feature_engineering.py`, sections 8–11):

| Feature type | Examples | Rationale |
|---|---|---|
| Trend/momentum | `pm25_diff_3h/6h/24h`, `pm25_ewm_12h/24h` | Captures *direction* of change, not just level |
| Anomaly | `pm25_anomaly_vs_72h_avg`, `pm25_ratio_vs_72h_avg` | Generalizes better to unseen extreme absolute values |
| Cyclical time | `hour_sin/cos`, `month_sin/cos` | Fixes false "cliff" between hour 23→0 and Dec→Jan |
| Physical interaction | `fire_x_inversion`, `calm_wind_flag`, `dew_point_spread_c` | Encodes known compound trapping mechanisms directly |
| Extended fire history | `*_fire_count_last_48h/72h` | Accounts for longer smoke transport time |
| Pressure trend | `pressure_trend_24h` | Proxy for incoming stable/calm conditions |

**Honest result: average MAE across all 72 horizons barely moved**
(Random Forest: 37.92 → 37.93 µg/m³, HistGradientBoosting: 38.00 → 38.32
µg/m³ — essentially flat, even marginally worse for HGB). Individually,
two new features *did* rank highly (`haryana_fire_count_last_72h` at
rank #4 for the 24h model, `pressure_trend_24h` at rank #10) — so they
aren't useless — but they were largely redundant with information the
model already captured indirectly through other correlated columns. This
confirms the diagnosis above: **the accuracy ceiling here is a data/
distribution problem, not a feature-richness problem.** More features
alone won't fix it. If you want to genuinely improve accuracy further,
prioritize (in order of likely impact):

1. **Feed in real future weather forecasts** (e.g. from IMD or another
   72h weather API) as additional inputs — the single biggest fix, since
   right now the model has no visibility into future atmospheric
   conditions at all.
2. **Train on multiple years** if you can obtain them, so the model
   actually sees extreme winter events during training, not just at
   test time.
3. **Log-transform the PM2.5 target** before training — compresses the
   extreme right-tail values so tree models are less penalized for
   missing them and can learn the shape of extreme events better.
4. **Season-stratified validation** — evaluate using a few different
   train/test splits (not just one chronological cut) to see how much
   the "hard test season" effect specifically is driving the numbers.

### Headline metrics (MAE, in µg/m³ — lower is better, after new features)

| Horizon | Persistence | Random Forest | HistGradientBoosting |
|---|---|---|---|
| 24h | **29.5** | 34.5 | 36.1 |
| 48h | 41.6 | 40.4 | **38.0** |
| 72h | 48.2 | 42.9 | **42.8** |

### Honest interpretation (this is important — read before your demo)

- **At 24 hours**, simple persistence ("tomorrow = today") actually **beats**
  both ML models. This makes physical sense: PM2.5 doesn't change chaotically
  hour-to-hour, so "no change" is a strong short-term guess, and it's hard
  for a model to beat with only one year of training data.
- **At 48 and 72 hours**, both ML models **clearly beat persistence** —
  Random Forest cuts error by ~13% at 72h. This is the real value of the
  coupled model: persistence has no way to "know" that an inversion is
  forming or that upwind fires are picking up, but the ML models pick up on
  these signals.
- **Do not claim in your presentation that your model beats persistence at
  every horizon** — it doesn't, and a judge who checks your metrics table
  will catch an overstated claim immediately. Instead, present this exact
  finding: *"our coupled model doesn't help much for tomorrow's forecast,
  but its advantage grows the further out you forecast — exactly where a
  naive approach struggles most."* That's a stronger, more credible story
  than a blanket "our model is better" claim.
- Full per-hour metrics for all 72 horizons are in
  `outputs/05_metrics_by_horizon.csv`.

### Feature importance (24h model, top drivers)

The model's top features are the current PM2.5 reading and its own recent
AQI estimate, followed by temperature, boundary layer height, and
inversion-related lags. Punjab/Haryana/upwind fire counts and
inversion strength do show up as meaningful contributors, confirming the
model is using the physical coupling mechanisms the problem statement asks
for — see `outputs/07_feature_importance.csv` for the full ranked list.

---

## 7. AQI Methodology

We use India's **CPCB (Central Pollution Control Board) National AQI**
breakpoint table for PM2.5 — not a generic or made-up formula, and not the
US EPA scale (which uses different breakpoints and would give different
numbers).

```
AQI = ((AQI_hi − AQI_lo) / (BP_hi − BP_lo)) × (PM − BP_lo) + AQI_lo
```

**Important limitation to mention to judges:** officially, CPCB AQI for
PM2.5 is calculated from a **24-hour average** concentration, not a single
hour's reading. This prototype applies the official breakpoints to each
hourly *predicted* value so you can show an hour-by-hour forecast curve —
this is a simplification, and it's clearly noted in the code and in
`outputs/forecast_72h.csv`. A production version should also output a
rolling 24-hour average AQI alongside the hourly curve.

---

## 8. How This Would Work As A Real, Live System

Right now, this project runs on one static historical CSV. In production,
you would put steps 02 → 03 (well, technically 02 → 06 for prediction only,
skip 03/target-creation since that's only for training) on a schedule:

1. **Every hour**, pull live data:
   - PM2.5/PM10/NO2/O3/SO2 from CPCB monitoring stations (public API/data portal)
   - Weather (temperature, wind, boundary layer height, pressure) from
     IMD or a weather API such as Open-Meteo
   - Active fire detections from NASA FIRMS for Punjab/Haryana/UP/Uttarakhand
2. **Append** the new hour to your historical database.
3. Run `02_feature_engineering.py`-style logic on just the newest rows
   (lag/rolling features only need recent history, not the whole year).
4. Feed the latest row into the 72 already-trained models (no retraining
   needed every hour) to get a fresh 72-hour forecast — this is exactly
   what `06_predict_72_hours.py` does, just using live data instead of a
   historical test row.
5. Periodically (e.g. weekly or monthly) **retrain** the models as more
   real data accumulates, so the models keep learning from recent seasons.

This design means **you do NOT need 2026 data in advance** — the same
trained models keep making rolling forecasts as new hourly data arrives;
you only retrain occasionally to keep the models current.

---

## 9. How To Demo This To SIH Judges

Suggested demo order:
1. **Open `plots/08_72hour_forecast_curve.png`** first — this is the
   product. Point out the AQI category shading and the diurnal (day/night)
   cycle the model correctly captures.
2. **Show `plots/08_error_by_horizon.png`** — this is your proof of rigor.
   Explain honestly that persistence wins at 24h but your model wins at
   48h/72h, and why that's the expected, credible outcome.
3. **Show `plots/07_feature_importance.png`** — prove the model is using
   real physical drivers (fires, inversion, PBL height), not just noise.
4. **Show `outputs/04_leakage_check.txt`** — proves you specifically
   checked for and avoided data leakage, a common flaw judges look for in
   ML projects.
5. Explain the "real-world deployment" story from Section 8 above — judges
   care a lot about whether a prototype could actually be operationalized.

---

## 10. Common Errors and Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `ModuleNotFoundError: No module named 'joblib'` | Run `pip install joblib` |
| `FileNotFoundError: data/delhincr.csv` | Make sure your raw CSV is placed inside the `data/` folder and named exactly `delhincr.csv` |
| `FileNotFoundError: models/...joblib` when running script 05/06/07 | You must run `04_train_model.py` first — scripts must run in order (01→08) |
| Training takes a long time | Normal — 144 models are being trained. Reduce `RF_N_ESTIMATORS` in `04_train_model.py` if you need it faster for testing, but note this may reduce accuracy |
| `DATA LEAKAGE CHECK FAILED` | Check `outputs/04_leakage_check.txt` for the specific failed check (usually leftover NaNs) — re-run `02` and `03` to regenerate clean files |

---

## 11. Honest Limitations (mention these proactively — it builds credibility)

- Single point-location weather/pollution data (not a full spatial NCR grid)
- One year of training data only — no cross-year validation possible yet
- Future weather forecasts are not fed into the model (it only uses
  information known at prediction time); a stronger version would pull in
  IMD's own 72-hour weather forecast as additional input features
- Fire data is regional counts, not plume-trajectory modeling — wind
  direction/speed features are used as a proxy for transport, but this is
  an approximation, not a true dispersion model
- CPCB AQI is applied per-hour here as a simplification of the official
  24-hour-average methodology
