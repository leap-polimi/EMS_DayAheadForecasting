"""Train Random Forest models for EMS electrical load forecasting targets.

This script is designed for repository publication:
- Paths and dates are configurable from command-line arguments or a local .env file.
- The .env file is private and must not be committed to Git.
- The final trained models and metrics are saved in configurable output folders.

Example usage:
    python 1_EL_LOAD_creation_RFmodels.py
    python 1_EL_LOAD_creation_RFmodels.py --dataset-path dataset.xlsx
    python 1_EL_LOAD_creation_RFmodels.py --min-date 2024-06-08 --split-date 2025-06-08
    python 1_EL_LOAD_creation_RFmodels.py --bad-start 2025-02-07T21:45:00 --bad-end 2025-02-13T05:45:00
    python 1_EL_LOAD_creation_RFmodels.py --final-training-days 365

Environment variables can also be used:
    DATASET_PATH=dataset.xlsx
    MODELS_DIR=models
    RESULTS_DIR=results
    TRAIN_MIN_DATE=2024-06-08T08:00:00
    TEST_SPLIT_DATE=2025-06-08T23:59:00
    BAD_DATA_START=2025-02-07T21:45:00
    BAD_DATA_END=2025-02-13T05:45:00
    FINAL_TRAINING_DAYS=365
    LAG_HOURS=48,72,96,120,144,168
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Load local environment variables when a private .env file exists.
# The .env file is intentionally ignored by Git.
try:
    from dotenv import load_dotenv

    # Load the .env file located in the same folder as this script.
    # Example: EMS_DayAheadForecasting/EL_LOAD/.env
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv is optional; environment variables can also be set manually.
    pass


# ======================================================================
# Configuration helpers
# ======================================================================


def parse_datetime(value: str | None) -> pd.Timestamp | None:
    """Parse a date/datetime string into a timezone-naive pandas Timestamp.

    Accepted examples:
    - 2024-06-08
    - 2024-06-08T08:00:00
    - 2024-06-08 08:00:00
    - none, None, empty string -> None

    The dataset created by 0_EL_LOAD_creation_dataset.py stores datetime values
    as timezone-naive UTC timestamps, so this script also uses timezone-naive
    timestamps for filtering and splitting.
    """
    if value is None:
        return None

    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null", "no"}:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."
        ) from exc

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)

    return timestamp



def parse_lag_hours(value: str) -> list[int]:
    """Parse comma-separated lag hours, for example '48,72,96,120,144,168'."""
    try:
        lags = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid lag list. Use comma-separated integers, for example: 48,72,96."
        ) from exc

    if not lags:
        raise argparse.ArgumentTypeError("At least one lag hour must be provided.")

    if any(lag <= 0 for lag in lags):
        raise argparse.ArgumentTypeError("Lag hours must be positive integers.")

    return lags



def parse_arguments() -> argparse.Namespace:
    """Read command-line options used to customize model training."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest models for EMS electrical load forecasting."
    )

    parser.add_argument(
        "--dataset-path",
        default=os.getenv("DATASET_PATH", "dataset.xlsx"),
        help="Input dataset Excel file. Can also be set with DATASET_PATH.",
    )

    parser.add_argument(
        "--models-dir",
        default=os.getenv("MODELS_DIR", "models"),
        help="Directory where trained models are saved. Can also be set with MODELS_DIR.",
    )

    parser.add_argument(
        "--results-dir",
        default=os.getenv("RESULTS_DIR", "results"),
        help="Directory where metrics are saved. Can also be set with RESULTS_DIR.",
    )

    parser.add_argument(
        "--min-date",
        type=parse_datetime,
        default=parse_datetime(os.getenv("TRAIN_MIN_DATE", "2024-06-08T08:00:00")),
        help=(
            "Discard rows before this UTC datetime. "
            "Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS. "
            "Can also be set with TRAIN_MIN_DATE."
        ),
    )

    parser.add_argument(
        "--bad-start",
        type=parse_datetime,
        default=parse_datetime(os.getenv("BAD_DATA_START", "2025-02-07T21:45:00")),
        help=(
            "Start of known bad-data window to remove. "
            "Set to none to disable. Can also be set with BAD_DATA_START."
        ),
    )

    parser.add_argument(
        "--bad-end",
        type=parse_datetime,
        default=parse_datetime(os.getenv("BAD_DATA_END", "2025-02-13T05:45:00")),
        help=(
            "End of known bad-data window to remove. "
            "Set to none to disable. Can also be set with BAD_DATA_END."
        ),
    )

    parser.add_argument(
        "--split-date",
        type=parse_datetime,
        default=parse_datetime(os.getenv("TEST_SPLIT_DATE", "2025-06-08T23:59:00")),
        help=(
            "Last datetime included in the validation/training-tuning period. "
            "Rows after this timestamp are used as the test set. "
            "Can also be set with TEST_SPLIT_DATE."
        ),
    )

    parser.add_argument(
        "--final-training-days",
        type=int,
        default=int(os.getenv("FINAL_TRAINING_DAYS", "365")),
        help=(
            "Number of most recent days used to refit the final model after tuning. "
            "Can also be set with FINAL_TRAINING_DAYS. Default: 365."
        ),
    )

    parser.add_argument(
        "--lag-hours",
        type=parse_lag_hours,
        default=parse_lag_hours(os.getenv("LAG_HOURS", "48,72,96,120,144,168")),
        help=(
            "Comma-separated target lag hours. Can also be set with LAG_HOURS. "
            "Default: 48,72,96,120,144,168."
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=int(os.getenv("CV_FOLDS", "3")),
        help="Number of cross-validation folds for GridSearchCV. Default: 3.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
        help="Random seed used by RandomForestRegressor. Default: 42.",
    )

    return parser.parse_args()


# ======================================================================
# Data preparation
# ======================================================================


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load the training dataset and validate the required datetime column."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_excel(dataset_path)

    if "datetime" not in df.columns:
        raise ValueError("The dataset must contain a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)



def clean_dataset(
    df: pd.DataFrame,
    min_date: pd.Timestamp | None,
    bad_start: pd.Timestamp | None,
    bad_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Apply configurable date filters and remove rows with missing values."""
    mask_keep = pd.Series(True, index=df.index)

    if min_date is not None:
        mask_keep &= df["datetime"] >= min_date

    if bad_start is not None and bad_end is not None:
        if bad_end < bad_start:
            raise ValueError(f"bad_end {bad_end} must be after bad_start {bad_start}.")
        mask_bad = (df["datetime"] >= bad_start) & (df["datetime"] <= bad_end)
        mask_keep &= ~mask_bad

    df_clean = df.loc[mask_keep].copy().reset_index(drop=True)

    rows_before_dropna = len(df_clean)
    df_clean = df_clean.dropna().reset_index(drop=True)
    rows_after_dropna = len(df_clean)

    print("\n--- Dataset cleaning summary ---")
    print(f"Rows after date filtering: {rows_before_dropna}")
    print(f"Rows after dropping NaNs:  {rows_after_dropna}")
    print(f"Dropped because of NaNs:   {rows_before_dropna - rows_after_dropna}")

    if df_clean.empty:
        raise ValueError("No rows remain after cleaning. Check your date filters and dataset.")

    return df_clean



def get_targets(df: pd.DataFrame) -> list[str]:
    """Return all forecasting targets present in the dataset."""
    targets = [col for col in df.columns if col.startswith("gross_el_cons_")]

    for total_col in ["CONS_TOT_kW", "CONS_TOT_NET_kW"]:
        if total_col in df.columns:
            targets.append(total_col)

    if not targets:
        raise ValueError(
            "No targets found. Expected columns starting with 'gross_el_cons_' "
            "and/or total columns 'CONS_TOT_kW', 'CONS_TOT_NET_kW'."
        )

    return targets



def build_base_features(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Build the base feature matrix before adding target-specific lag columns."""
    drop_cols = ["datetime_italy", "month", "day_of_week", "quarter_hour"] + targets
    drop_cols = [col for col in drop_cols if col in df.columns]

    X_base = df.drop(columns=drop_cols).set_index("datetime")

    if X_base.empty:
        raise ValueError("No feature columns remain after dropping target and calendar columns.")

    return X_base


# ======================================================================
# Model training
# ======================================================================


def train_target_model(
    target: str,
    df: pd.DataFrame,
    X_base: pd.DataFrame,
    lag_hours: list[int],
    split_date: pd.Timestamp,
    final_training_days: int,
    models_dir: Path,
    cv_folds: int,
    random_state: int,
) -> dict:
    """Tune, evaluate, refit, and save one Random Forest model."""
    print(f"\n=========================\nTraining target: {target}\n=========================")

    y = df.set_index("datetime")[target].copy()
    X = X_base.copy()

    # Dataset resolution is 15 minutes, so 1 hour = 4 rows.
    for lag_h in lag_hours:
        X[f"{target}_lag{lag_h}"] = y.shift(lag_h * 4)

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if X.empty:
        raise ValueError(f"No valid rows remain for target {target} after creating lag features.")

    X_train = X.loc[:split_date]
    X_test = X.loc[split_date + pd.Timedelta(minutes=15):]
    y_train = y.loc[:split_date]
    y_test = y.loc[split_date + pd.Timedelta(minutes=15):]

    if X_train.empty:
        raise ValueError(
            f"Training set is empty for target {target}. "
            f"Check --split-date ({split_date}) and your dataset date range."
        )

    if X_test.empty:
        raise ValueError(
            f"Test set is empty for target {target}. "
            f"Choose an earlier --split-date than the last dataset timestamp."
        )

    print(f"Train period: {X_train.index.min()} -> {X_train.index.max()} ({len(X_train)} rows)")
    print(f"Test period:  {X_test.index.min()} -> {X_test.index.max()} ({len(X_test)} rows)")

    rf = RandomForestRegressor(random_state=random_state)
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_features": ["log2", "sqrt", None],
        "min_samples_leaf": [2, 5, 10, 20, 30],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"Best params: {grid.best_params_}")
    print(f"Test RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    # Refit final model on the most recent configurable training window.
    end_date = X.index.max()
    start_date = end_date - pd.Timedelta(days=final_training_days)
    X_recent = X.loc[start_date:end_date]
    y_recent = y.loc[start_date:end_date]

    if X_recent.empty:
        raise ValueError(
            f"Final training window is empty for target {target}. "
            "Check --final-training-days."
        )

    print(
        f"Retraining final model on recent window: {start_date.date()} -> {end_date.date()} "
        f"({len(X_recent)} samples)"
    )

    best_model.fit(X_recent, y_recent)

    model_path = models_dir / f"RF_{target}.joblib"
    dump(best_model, model_path)
    print(f"Saved model: {model_path}")

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
        "final_train_start": X_recent.index.min(),
        "final_train_end": X_recent.index.max(),
        "final_train_samples": len(X_recent),
    }


# ======================================================================
# Main entry point
# ======================================================================


def main() -> None:
    args = parse_arguments()

    if args.final_training_days <= 0:
        raise ValueError("--final-training-days must be a positive integer.")

    if args.cv_folds <= 1:
        raise ValueError("--cv-folds must be greater than 1.")

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Training configuration ---")
    print(f"Dataset path:        {args.dataset_path}")
    print(f"Models directory:    {models_dir}")
    print(f"Results directory:   {results_dir}")
    print(f"Minimum date:        {args.min_date}")
    print(f"Bad-data start:      {args.bad_start}")
    print(f"Bad-data end:        {args.bad_end}")
    print(f"Train/test split:    {args.split_date}")
    print(f"Final train days:    {args.final_training_days}")
    print(f"Lag hours:           {args.lag_hours}")
    print(f"CV folds:            {args.cv_folds}")
    print(f"Random state:        {args.random_state}")

    df_raw = load_dataset(args.dataset_path)
    df = clean_dataset(
        df=df_raw,
        min_date=args.min_date,
        bad_start=args.bad_start,
        bad_end=args.bad_end,
    )

    targets = get_targets(df)
    print("\n--- Targets ---")
    for target in targets:
        print(f"- {target}")

    X_base = build_base_features(df, targets)

    if args.split_date is None:
        raise ValueError("A split date is required. Set --split-date or TEST_SPLIT_DATE.")

    results = []
    for target in targets:
        result = train_target_model(
            target=target,
            df=df,
            X_base=X_base,
            lag_hours=args.lag_hours,
            split_date=args.split_date,
            final_training_days=args.final_training_days,
            models_dir=models_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        results.append(result)

    metrics_path = results_dir / "metrics_summary.xlsx"
    pd.DataFrame(results).to_excel(metrics_path, index=False)

    print(f"\nMetrics saved to: {metrics_path}")
    print("All models trained and saved successfully.")


if __name__ == "__main__":
    main()
