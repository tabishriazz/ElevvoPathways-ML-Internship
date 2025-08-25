# =========================
# Sales Forecasting Pipeline (Safe Version)
# =========================
# Requirements:
#   pip install numpy pandas scikit-learn matplotlib statsmodels
#   (optional) pip install xgboost lightgbm

import sys, os

# Safety check: avoid shadowing numpy/pandas
for bad in ["pickle.py", "numpy.py", "pandas.py"]:
    if os.path.exists(bad):
        raise RuntimeError(
            f"⚠️ You have a file named {bad} in your project folder. "
            "Please rename it (e.g., my_pickle.py) because it conflicts with libraries."
        )

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "❌ Failed to import numpy/pandas. "
        "Run:\n    pip install --upgrade numpy pandas\n\n"
        f"Details: {e}"
    )

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Optional models
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Optional seasonal decomposition (bonus)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_DECOMP = True
except Exception:
    HAS_DECOMP = False


# =========================
# 1) CONFIG
# =========================
CSV_PATH = "features.csv"        # <<-- put your dataset path here
DATE_COL_CANDIDATES = ["date", "Date", "week", "Week"]
SALES_COL_CANDIDATES = ["weekly_sales", "sales", "Sales"]
STORE_COL_CANDIDATES = ["store", "Store"]
DEPT_COL_CANDIDATES = ["dept", "Dept", "department"]

# Focus on one store/dept or None for total aggregate
TARGET_GROUP = None   # e.g., {"store": 1, "dept": 5}

RESAMPLE_RULE = None   # e.g., 'W' for weekly, or None to keep original

LAGS = [1, 2, 3, 4, 7, 14, 28, 52]
ROLL_WINDOWS = [3, 7, 14, 28, 52]

N_SPLITS = 5
FORECAST_HORIZON = 1

RANDOM_STATE = 42


# =========================
# 2) LOAD + AUTO-COLUMN MAP
# =========================
def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)
    cols = df.columns.tolist()

    date_col = find_col(cols, DATE_COL_CANDIDATES)
    sales_col = find_col(cols, SALES_COL_CANDIDATES)
    store_col = find_col(cols, STORE_COL_CANDIDATES)
    dept_col = find_col(cols, DEPT_COL_CANDIDATES)

    if date_col is None or sales_col is None:
        raise ValueError(
            f"Could not find date/sales columns. Found {cols}. "
            f"Expected date in {DATE_COL_CANDIDATES}, sales in {SALES_COL_CANDIDATES}."
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Normalize column names
    df = df.rename(columns={date_col: "date", sales_col: "sales"})
    if store_col: df = df.rename(columns={store_col: "store"})
    if dept_col:  df = df.rename(columns={dept_col: "dept"})

    return df

df_raw = load_data(CSV_PATH)


# =========================
# 3) SELECT TARGET SERIES
# =========================
def filter_or_aggregate(df, target_group):
    if target_group:
        tmp = df.copy()
        for k, v in target_group.items():
            if k not in tmp.columns:
                raise ValueError(f"Group key '{k}' not in columns {tmp.columns.tolist()}")
            tmp = tmp[tmp[k] == v]
        if tmp.empty:
            raise ValueError(f"No rows found for TARGET_GROUP={target_group}")
        series_df = tmp.groupby("date", as_index=False)["sales"].sum()
        label = " - ".join([f"{k}={v}" for k, v in target_group.items()])
    else:
        series_df = df.groupby("date", as_index=False)["sales"].sum()
        label = "ALL"

    if RESAMPLE_RULE:
        series_df = (series_df.set_index("date")
                               .resample(RESAMPLE_RULE)["sales"].sum()
                               .reset_index())
    return series_df.sort_values("date").reset_index(drop=True), label

series_df, series_label = filter_or_aggregate(df_raw, TARGET_GROUP)


# =========================
# 4) (Optional) Seasonal Decomposition
# =========================
if HAS_DECOMP and len(series_df) >= 24:
    try:
        decomposition = seasonal_decompose(series_df.set_index("date")["sales"],
                                           model="additive", period=52)
        decomposition.plot()
        plt.suptitle(f"Seasonal Decomposition ({series_label})")
        plt.show()
    except Exception:
        pass


# =========================
# 5) FEATURE ENGINEERING
# =========================
def add_time_features(df):
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df

def add_lags_and_rolls(df, lags, windows):
    df = df.copy().sort_values("date")
    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df["sales"].shift(1).rolling(window=w, min_periods=max(1, w//2)).mean()
        df[f"roll_std_{w}"]  = df["sales"].shift(1).rolling(window=w, min_periods=max(1, w//2)).std()
    return df

feat_df = series_df.copy()
feat_df = add_time_features(feat_df)
feat_df = add_lags_and_rolls(feat_df, LAGS, ROLL_WINDOWS)
feat_df = feat_df.dropna().reset_index(drop=True)

FEATURE_COLS = [c for c in feat_df.columns if c not in ["date", "sales"]]


# =========================
# 6) TRAIN/VALIDATION SPLITS
# =========================
def ts_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))).mean() * 100.0
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape}

X = feat_df[FEATURE_COLS].values
y = feat_df["sales"].values
dates = feat_df["date"].values

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=600, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=RANDOM_STATE
    )
if HAS_LGBM:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=800, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

cv_scores = {name: [] for name in models}
print(f"TimeSeriesSplit CV (n_splits={N_SPLITS}) on series: {series_label}")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        cv_scores[name].append(ts_metrics(y_va, pred))

print("\n=== Cross-Validation Results ===")
summary_rows = []
for name, scores in cv_scores.items():
    mean_rmse = np.mean([s["RMSE"] for s in scores])
    mean_mae  = np.mean([s["MAE"] for s in scores])
    mean_mape = np.mean([s["MAPE%"] for s in scores])
    summary_rows.append((name, mean_rmse, mean_mae, mean_mape))
    print(f"{name:14s} | RMSE={mean_rmse:.2f} | MAE={mean_mae:.2f} | MAPE={mean_mape:.2f}%")

best_name, best_rmse = min(summary_rows, key=lambda x: x[1])[0:2]
best_model = models[best_name]
print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f})")


# =========================
# 7) HOLDOUT PLOT
# =========================
holdout_size = max(12, int(0.15 * len(X)))
X_train, X_hold = X[:-holdout_size], X[-holdout_size:]
y_train, y_hold = y[:-holdout_size], y[-holdout_size:]
dates_hold = dates[-holdout_size:]

best_model.fit(X_train, y_train)
y_pred_hold = best_model.predict(X_hold)
print("\nHoldout Metrics:", ts_metrics(y_hold, y_pred_hold))

plt.figure(figsize=(10,5))
plt.plot(dates_hold, y_hold, label="Actual")
plt.plot(dates_hold, y_pred_hold, label=f"Predicted ({best_name})")
plt.title(f"Holdout Forecast — {series_label}")
plt.legend(); plt.tight_layout(); plt.show()


# =========================
# 8) NEXT-PERIOD FORECAST
# =========================
best_model.fit(X, y)
last_date = feat_df["date"].iloc[-1]

if len(series_df) >= 3:
    diffs = np.diff(series_df["date"].values.astype("datetime64[D]")).astype(int)
    step = int(pd.Series(diffs).mode().iloc[0])
    freq_guess = timedelta(days=step)
else:
    freq_guess = timedelta(days=7)

next_date = pd.to_datetime(last_date) + pd.to_timedelta(freq_guess)

tmp = series_df.copy()
tmp = tmp.append({"date": next_date, "sales": np.nan}, ignore_index=True)
tmp = add_time_features(tmp)
tmp = add_lags_and_rolls(tmp, LAGS, ROLL_WINDOWS)
future_feat = tmp[tmp["date"] == next_date][FEATURE_COLS]
future_pred = best_model.predict(future_feat.values)[0]

print(f"\nNext-period forecast for {next_date.date()}: {future_pred:.2f}")

plt.figure(figsize=(10,5))
plt.plot(series_df.tail(40)["date"], series_df.tail(40)["sales"], label="Recent")
plt.scatter([next_date], [future_pred], color="red", marker="x", s=80, label="Forecast")
plt.title(f"Next-Period Forecast — {series_label}")
plt.legend(); plt.tight_layout(); plt.show()
