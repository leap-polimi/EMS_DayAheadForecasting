"""Append recent thermal-load data and retrain forecasting models monthly.

This script mirrors the EL_LOAD monthly retraining workflow for thermal targets.
It updates the thermal dataset, then retrains RF models using best parameters
saved by 1_TH_LOAD_model_training.py.

Typical usage from the external repository folder:
    python TH_LOAD/2_TH_LOAD_automatic_retraining.py

Only retrain on the existing dataset:
    python TH_LOAD/2_TH_LOAD_automatic_retraining.py --skip-dataset-update
"""

import argparse
import ast
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from joblib import dump
from OptimoApi import OptimoApi
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

TARGETS = ["THERMAL_LOAD_kW", "DH_THERMAL_LOAD_kW"]


def get_required_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}. Create TH_LOAD/.env first.")
    return value


def login_cloud_optimo() -> OptimoApi:
    return OptimoApi(
        api_key=get_required_env_var("OPTIMO_API_KEY"),
        app_id=get_required_env_var("OPTIMO_APP_ID"),
        app_secret=get_required_env_var("OPTIMO_APP_SECRET"),
    )


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_arguments() -> argparse.Namespace:
    results_dir_default = os.getenv("THERMAL_RESULTS_DIR", "TH_LOAD/results")
    parser = argparse.ArgumentParser(description="Update thermal dataset and retrain RF models.")
    parser.add_argument("--dataset-path", default=os.getenv("THERMAL_DATASET_PATH", "TH_LOAD/dataset_thermal_clean.xlsx"))
    parser.add_argument("--models-dir", default=os.getenv("THERMAL_MODELS_DIR", "TH_LOAD/models"))
    parser.add_argument("--results-dir", default=results_dir_default)
    parser.add_argument("--metrics-path", default=os.getenv("THERMAL_METRICS_PATH", str(Path(results_dir_default) / "metrics_summary_thermal.xlsx")))
    parser.add_argument("--latitude", type=float, default=float(os.getenv("LATITUDE", "45.4643")))
    parser.add_argument("--longitude", type=float, default=float(os.getenv("LONGITUDE", "9.1895")))
    parser.add_argument("--local-timezone", default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"))
    parser.add_argument("--thermal-var-id", default=os.getenv("THERMAL_LOAD_VAR_ID", "XVpGIF_pHutx0"))
    parser.add_argument("--dh-thermal-var-id", default=os.getenv("DH_THERMAL_LOAD_VAR_ID", "XVpGIF_Ai5Ip9"))
    parser.add_argument("--final-training-days", type=int, default=int(os.getenv("THERMAL_FINAL_TRAINING_DAYS", "365")))
    parser.add_argument("--initial-history-days", type=int, default=int(os.getenv("THERMAL_INITIAL_HISTORY_DAYS", "370")))
    parser.add_argument("--weather-data-delay-days", type=int, default=int(os.getenv("THERMAL_WEATHER_DATA_DELAY_DAYS", "2")))
    parser.add_argument("--lag-hours", default=os.getenv("THERMAL_LAG_HOURS", "48,72,96,120,144,168"))
    parser.add_argument("--upper-cap-kw", type=float, default=float(os.getenv("THERMAL_UPPER_CAP_KW", "13000")))
    parser.add_argument("--summer-zero-months", default=os.getenv("THERMAL_SUMMER_ZERO_MONTHS", "5,6,7,8,9"))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    parser.add_argument("--skip-dataset-update", action="store_true")
    args = parser.parse_args()
    args.lag_hours = parse_int_list(args.lag_hours)
    args.summer_zero_months = set(parse_int_list(args.summer_zero_months))
    return args


def ensure_utc_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df.sort_values(col).reset_index(drop=True)


def fetch_target_15min(api: OptimoApi, identifier: str, start_dt: dt.datetime, end_dt: dt.datetime, colname: str) -> pd.DataFrame:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    data = api.get_values_in_range([identifier], start_ms, end_ms, limit=1_000_000)
    records = data.get(identifier, []) or []
    if not records:
        return pd.DataFrame(columns=["datetime", colname])
    df = pd.DataFrame({"datetime": [r["timestamp"] for r in records], colname: [r["value"] for r in records]})
    df = ensure_utc_datetime_index(df, "datetime").set_index("datetime").resample("15min").mean().reset_index()
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def fetch_weather_15min(start_utc: dt.datetime, end_utc: dt.datetime, latitude: float, longitude: float) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_utc.date().isoformat(),
        "end_date": end_utc.date().isoformat(),
        "minutely_15": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "direct_normal_irradiance"],
    }
    responses = client.weather_api("https://historical-forecast-api.open-meteo.com/v1/forecast", params=params)
    m15 = responses[0].Minutely15()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=pd.to_datetime(m15.Time(), unit="s", utc=True),
                end=pd.to_datetime(m15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=m15.Interval()),
                inclusive="left",
            ),
            "temperature_2m": m15.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": m15.Variables(1).ValuesAsNumpy(),
            "dew_point_2m": m15.Variables(2).ValuesAsNumpy(),
            "direct_normal_irradiance": m15.Variables(3).ValuesAsNumpy(),
        }
    )
    df = df[(df["datetime"] >= start_utc) & (df["datetime"] <= end_utc)].copy()
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def add_calendar_features(df_input: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    df = df_input.copy()
    df["datetime_italy"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(local_tz)

    df["month"] = df["datetime_italy"].dt.month - 1
    df["day_of_week"] = df["datetime_italy"].dt.weekday
    df["quarter_hour"] = df["datetime_italy"].dt.hour * 4 + df["datetime_italy"].dt.minute // 15

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter_hour"] / 96)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter_hour"] / 96)

    italy_holidays = holidays.IT()

    def is_campus_closed(timestamp_local: pd.Timestamp) -> int:
        month = timestamp_local.month
        day = timestamp_local.day
        hour = timestamp_local.hour
        weekday = timestamp_local.weekday()
        week = timestamp_local.isocalendar().week
        if weekday == 6 or timestamp_local.date() in italy_holidays:
            return 1
        if week in [33, 34]:
            return 1
        if (month == 12 and day >= 23) or (month == 1 and day <= 3):
            return 1
        if weekday in range(0, 5):
            return 0 if 7 <= hour < 21 else 1
        if weekday == 5:
            return 0 if 7 <= hour < 20 else 1
        return 0

    df["campus_closed"] = df["datetime_italy"].apply(is_campus_closed)
    df["T_open"] = df["temperature_2m"] * (1 - df["campus_closed"])
    df["datetime_italy"] = df["datetime_italy"].dt.tz_localize(None)
    return df


def apply_thermal_cleaning(df: pd.DataFrame, upper_cap_kw: float, summer_months: set[int]) -> pd.DataFrame:
    out = df.copy()
    summer_mask = pd.to_datetime(out["datetime_italy"]).dt.month.isin(summer_months)
    for target in TARGETS:
        if target in out.columns:
            out[target] = pd.to_numeric(out[target], errors="coerce").clip(lower=0, upper=upper_cap_kw)
            out.loc[summer_mask, target] = 0.0
    return out


def build_block_dataset_like(api: OptimoApi, start_utc: dt.datetime, end_utc: dt.datetime, args: argparse.Namespace, local_tz: ZoneInfo) -> pd.DataFrame:
    print("\n--- Building new thermal dataset block ---")
    print(f"Block start UTC: {start_utc}")
    print(f"Block end UTC:   {end_utc}")

    df_weather = fetch_weather_15min(start_utc, end_utc, args.latitude, args.longitude)
    df_thermal = fetch_target_15min(api, args.thermal_var_id, start_utc, end_utc, "THERMAL_LOAD_kW")
    df_dh = fetch_target_15min(api, args.dh_thermal_var_id, start_utc, end_utc, "DH_THERMAL_LOAD_kW")

    merged = pd.merge(df_thermal.sort_values("datetime"), df_dh.sort_values("datetime"), on="datetime", how="outer")
    merged = pd.merge(merged, df_weather.sort_values("datetime"), on="datetime", how="outer")
    merged = add_calendar_features(merged, local_tz)
    merged = apply_thermal_cleaning(merged, args.upper_cap_kw, args.summer_zero_months)
    return merged


def load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.exists():
        df = pd.read_excel(dataset_path)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return pd.DataFrame()


def update_dataset_if_needed(dataset_path: Path, df_all: pd.DataFrame, api: OptimoApi, args: argparse.Namespace, local_tz: ZoneInfo) -> pd.DataFrame:
    if args.skip_dataset_update:
        print("Skipping dataset update because --skip-dataset-update was provided.")
        return df_all

    now_local = dt.datetime.now(local_tz)
    weather_cut_local = (now_local - dt.timedelta(days=args.weather_data_delay_days)).date()
    update_end_local = dt.datetime.combine(weather_cut_local, dt.time(23, 45), tzinfo=local_tz)
    update_end_utc = update_end_local.astimezone(dt.timezone.utc)

    if df_all.empty or df_all["datetime"].isna().all():
        start_local = (now_local - dt.timedelta(days=args.initial_history_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(dt.timezone.utc)
    else:
        last_ts = pd.to_datetime(df_all["datetime"].max())
        start_utc = (pd.Timestamp(last_ts).tz_localize("UTC") + pd.Timedelta(minutes=15)).to_pydatetime()

    if start_utc > update_end_utc:
        print("No new rows to append. Proceeding to retraining.")
        return df_all

    df_new = build_block_dataset_like(api, start_utc, update_end_utc, args, local_tz)
    df_all = pd.concat([df_all, df_new], axis=0, ignore_index=True)
    df_all = df_all.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_excel(dataset_path, index=False)
    print(f"Dataset updated: {dataset_path}")
    return df_all


def parse_best_params(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        try:
            return json.loads(value)
        except Exception:
            pass
    raise ValueError(f"Cannot parse best_params: {value}")


def retrain_models(df_all: pd.DataFrame, metrics_path: Path, models_dir: Path, results_dir: Path, args: argparse.Namespace) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot find {metrics_path}. Run 1_TH_LOAD_model_training.py first.")

    df_params = pd.read_excel(metrics_path)
    last_ts = pd.to_datetime(df_all["datetime"].max())
    first_ts = last_ts - pd.Timedelta(days=args.final_training_days)
    df = df_all[(df_all["datetime"] >= first_ts) & (df_all["datetime"] <= last_ts)].copy()

    full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="15min")
    df = df.set_index("datetime").reindex(full_index).rename_axis("datetime").reset_index()

    for target in TARGETS:
        if target not in df.columns:
            print(f"Skipping missing target: {target}")
            continue

        print(f"\n--- Retraining {target} ---")
        other_targets = [t for t in TARGETS if t != target]
        drop_cols = ["datetime_italy", "month", "day_of_week", "quarter_hour", target] + [c for c in other_targets if c in df.columns]
        drop_cols = [col for col in drop_cols if col in df.columns]

        X_base = df.drop(columns=drop_cols).set_index("datetime")
        y = pd.to_numeric(df.set_index("datetime")[target], errors="coerce")

        for lag_h in args.lag_hours:
            X_base[f"{target}_lag{lag_h}h"] = y.reindex(X_base.index - pd.Timedelta(hours=lag_h)).values

        valid_mask = X_base.notna().all(axis=1) & y.notna() & (y != 0)
        X_train = X_base.loc[valid_mask]
        y_train = y.loc[valid_mask]

        if X_train.empty:
            raise ValueError(f"No valid training rows remain for {target}.")

        row = df_params.loc[df_params["target"] == target]
        if row.empty:
            raise KeyError(f"No best_params found in {metrics_path} for target {target}.")

        best_params = parse_best_params(row.iloc[0]["best_params"])
        model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1, **best_params)
        model.fit(X_train, y_train)

        model_path = models_dir / f"RF_{target}.joblib"
        dump(model, model_path)
        print(f"Saved model: {model_path} | samples: {len(X_train)}")

        verification = X_train.copy()
        verification[target] = y_train
        verification.to_csv(results_dir / f"retrain_verification_{target}.csv", index=True, index_label="datetime")


def main() -> None:
    args = parse_arguments()
    local_tz = ZoneInfo(args.local_timezone)
    dataset_path = Path(args.dataset_path)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    metrics_path = Path(args.metrics_path)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Thermal monthly retraining configuration ---")
    print(f"Dataset path:       {dataset_path}")
    print(f"Metrics path:       {metrics_path}")
    print(f"Models directory:   {models_dir}")
    print(f"Skip update:        {args.skip_dataset_update}")

    df_all = load_or_create_dataset(dataset_path)
    api = None if args.skip_dataset_update else login_cloud_optimo()
    df_all = update_dataset_if_needed(dataset_path, df_all, api, args, local_tz)

    if df_all.empty:
        raise RuntimeError("Thermal dataset is empty. Run dataset creation first.")

    retrain_models(df_all, metrics_path, models_dir, results_dir, args)
    print("\nThermal monthly retraining complete.")


if __name__ == "__main__":
    main()
