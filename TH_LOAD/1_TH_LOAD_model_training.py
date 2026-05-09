"""Train Random Forest models for EMS thermal-load forecasting targets.

This script mirrors the EL_LOAD training workflow, but for thermal targets:
- THERMAL_LOAD_kW
- DH_THERMAL_LOAD_kW

It reads configuration from TH_LOAD/.env and is intended to be run from the
external repository folder.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

TARGETS = ["THERMAL_LOAD_kW", "DH_THERMAL_LOAD_kW"]


def parse_datetime(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null", "no"}:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def parse_lag_hours(value: str) -> list[int]:
    lags = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not lags or any(lag <= 0 for lag in lags):
        raise argparse.ArgumentTypeError("Lag hours must be positive comma-separated integers.")
    return lags


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train thermal-load Random Forest models.")
    parser.add_argument("--dataset-path", default=os.getenv("THERMAL_DATASET_PATH", "TH_LOAD/dataset_thermal_clean.xlsx"))
    parser.add_argument("--models-dir", default=os.getenv("THERMAL_MODELS_DIR", "TH_LOAD/models"))
    parser.add_argument("--results-dir", default=os.getenv("THERMAL_RESULTS_DIR", "TH_LOAD/results"))
    parser.add_argument("--train-end", type=parse_datetime, default=parse_datetime(os.getenv("THERMAL_TRAIN_END", "2025-04-30T23:59:00")))
    parser.add_argument("--test-start", type=parse_datetime, default=parse_datetime(os.getenv("THERMAL_TEST_START", "2025-05-01T00:00:00")))
    parser.add_argument("--lag-hours", type=parse_lag_hours, default=parse_lag_hours(os.getenv("THERMAL_LAG_HOURS", "48,72,96,120,144,168")))
    parser.add_argument("--cv-splits", type=int, default=int(os.getenv("THERMAL_CV_SPLITS", "3")))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    return parser.parse_args()


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df_raw = pd.read_excel(dataset_path)
    if "datetime" not in df_raw.columns:
        raise ValueError("Column 'datetime' is required.")

    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce")
    df_raw = df_raw.dropna(subset=["datetime"])
    df_raw = df_raw.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    full_index = pd.date_range(df_raw["datetime"].min(), df_raw["datetime"].max(), freq="15min")
    return df_raw.set_index("datetime").reindex(full_index).rename_axis("datetime").reset_index()


def build_xy_for_target(df_continuous: pd.DataFrame, target: str, lag_hours: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df_continuous.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")

    df = df_continuous.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    other_targets = [t for t in TARGETS if t != target]
    drop_cols = ["datetime_italy", "month", "day_of_week", "quarter_hour", target] + other_targets
    drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=drop_cols).set_index("datetime")
    y = df.set_index("datetime")[target]

    for lag_h in lag_hours:
        X[f"{target}_lag{lag_h}h"] = y.reindex(X.index - pd.Timedelta(hours=lag_h)).values

    valid_mask = X.notna().all(axis=1) & y.notna() & (y != 0)
    return X.loc[valid_mask], y.loc[valid_mask]


def train_one_target(
    df_continuous: pd.DataFrame,
    target: str,
    args: argparse.Namespace,
    models_dir: Path,
    results_dir: Path,
) -> dict | None:
    if target not in df_continuous.columns:
        print(f"Warning: target '{target}' not found. Skipping.")
        return None

    print(f"\n=========================\nTraining target: {target}\n=========================")
    X, y = build_xy_for_target(df_continuous, target, args.lag_hours)

    X_train = X.loc[:args.train_end]
    y_train = y.loc[:args.train_end]
    X_test = X.loc[args.test_start:]
    y_test = y.loc[args.test_start:]

    if X_train.empty:
        raise ValueError(f"Training set is empty for {target}. Check THERMAL_TRAIN_END.")
    if X_test.empty:
        raise ValueError(f"Test set is empty for {target}. Check THERMAL_TEST_START.")

    print(f"Train period: {X_train.index.min()} -> {X_train.index.max()} ({len(X_train)} rows)")
    print(f"Test period:  {X_test.index.min()} -> {X_test.index.max()} ({len(X_test)} rows)")

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_features": ["log2", "sqrt", None],
        "min_samples_leaf": [2, 5, 10, 20, 30],
    }
    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    grid = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_squared_error", verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"Best params: {grid.best_params_}")
    print(f"Test RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    best_model.fit(X, y)
    model_path = models_dir / f"RF_{target}.joblib"
    dump(best_model, model_path)
    print(f"Saved model: {model_path}")

    verification = X.copy()
    verification[target] = y.loc[X.index]
    verification.to_csv(results_dir / f"verification_{target}.csv", index=True, index_label="datetime")

    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(results_dir / f"feature_importances_{target}.csv", header=["importance"])

    return {
        "target": target,
        "best_params": grid.best_params_,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "train_start": X_train.index.min(),
        "train_end": X_train.index.max(),
        "test_start": X_test.index.min(),
        "test_end": X_test.index.max(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main() -> None:
    args = parse_arguments()
    if args.cv_splits <= 1:
        raise ValueError("--cv-splits must be greater than 1.")

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Thermal training configuration ---")
    print(f"Dataset path:      {args.dataset_path}")
    print(f"Models directory:  {models_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Train end:         {args.train_end}")
    print(f"Test start:        {args.test_start}")
    print(f"Lag hours:         {args.lag_hours}")

    df = load_dataset(args.dataset_path)

    metrics = []
    for target in TARGETS:
        row = train_one_target(df, target, args, models_dir, results_dir)
        if row is not None:
            metrics.append(row)

    if not metrics:
        raise RuntimeError("No thermal targets were trained.")

    metrics_path = results_dir / "metrics_summary_thermal.xlsx"
    pd.DataFrame(metrics).to_excel(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    print("All thermal models trained successfully.")


if __name__ == "__main__":
    main()
