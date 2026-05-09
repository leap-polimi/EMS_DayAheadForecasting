"""Run daily day-ahead forecasts for thermal-load targets and optionally upload to Optimo.

Typical usage from the external repository folder:
    python TH_LOAD/2_TH_LOAD_automatic_forecasting.py

Test locally without upload:
    python TH_LOAD/2_TH_LOAD_automatic_forecasting.py --no-upload
"""

import argparse
import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from joblib import load
from OptimoApi import OptimoApi
from retry_requests import retry

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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Use true/false, yes/no, or 1/0.")


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_forecast_date(value: str | None, local_tz: ZoneInfo) -> dt.date:
    if value is None or str(value).strip() == "":
        value = "tomorrow"
    value = str(value).strip().lower()
    now_local = dt.datetime.now(local_tz)
    if value == "tomorrow":
        return (now_local + dt.timedelta(days=1)).date()
    if value == "today":
        return now_local.date()
    return dt.date.fromisoformat(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run day-ahead thermal-load forecasting.")
    parser.add_argument("--forecast-date", default=os.getenv("THERMAL_FORECAST_DATE", "tomorrow"))
    parser.add_argument("--models-dir", default=os.getenv("THERMAL_MODELS_DIR", "TH_LOAD/models"))
    parser.add_argument("--forecasts-dir", default=os.getenv("THERMAL_FORECASTS_DIR", "TH_LOAD/forecasts"))
    parser.add_argument("--latitude", type=float, default=float(os.getenv("LATITUDE", "45.4643")))
    parser.add_argument("--longitude", type=float, default=float(os.getenv("LONGITUDE", "9.1895")))
    parser.add_argument("--local-timezone", default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"))
    parser.add_argument("--lag-hours", default=os.getenv("THERMAL_LAG_HOURS", "48,72,96,120,144,168"))
    parser.add_argument("--historical-lookback-days", type=int, default=int(os.getenv("THERMAL_HISTORICAL_LOOKBACK_DAYS", "8")))
    parser.add_argument("--min-lag-hours", type=int, default=int(os.getenv("THERMAL_MIN_LAG_HOURS", "48")))
    parser.add_argument("--lag-cap-kw", type=float, default=float(os.getenv("THERMAL_LAG_CAP_KW", "13000")))
    parser.add_argument("--summer-zero-months", default=os.getenv("THERMAL_SUMMER_ZERO_MONTHS", "5,6,7,8,9"))
    parser.add_argument("--thermal-var-id", default=os.getenv("THERMAL_LOAD_VAR_ID", "XVpGIF_pHutx0"))
    parser.add_argument("--dh-thermal-var-id", default=os.getenv("DH_THERMAL_LOAD_VAR_ID", "XVpGIF_Ai5Ip9"))
    parser.add_argument("--thermal-upload-variable-id", default=os.getenv("OPTIMO_THERMAL_FORECAST_VARIABLE_ID", ""))
    parser.add_argument("--dh-upload-variable-id", default=os.getenv("OPTIMO_DH_THERMAL_FORECAST_VARIABLE_ID", ""))
    parser.add_argument("--upload", dest="upload_to_optimo", action="store_true", default=parse_bool(os.getenv("UPLOAD_TO_OPTIMO", "true")))
    parser.add_argument("--no-upload", dest="upload_to_optimo", action="store_false")
    args = parser.parse_args()
    args.lag_hours = parse_int_list(args.lag_hours)
    args.summer_zero_months = set(parse_int_list(args.summer_zero_months))
    return args


def ensure_utc_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df.sort_values(col).reset_index(drop=True)


def fetch_weather_minutely15_for_day(target_date_local: dt.date, latitude: float, longitude: float, local_tz: ZoneInfo) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "direct_normal_irradiance"],
        "forecast_days": 2,
    }
    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
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

    start_local = dt.datetime.combine(target_date_local, dt.time(0, 0), tzinfo=local_tz)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(minutes=15)
    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    df = df[(df["datetime"] >= start_utc) & (df["datetime"] <= end_utc)].copy()
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.set_index("datetime").sort_index()


def add_calendar_features(df_base: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    df = df_base.copy()
    dt_local = df.index.tz_localize("UTC").tz_convert(local_tz)
    month = dt_local.month - 1
    day_of_week = dt_local.weekday
    quarter_hour = dt_local.hour * 4 + (dt_local.minute // 15)

    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)
    df["sin_day_of_week"] = np.sin(2 * np.pi * day_of_week / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * day_of_week / 7)
    df["sin_quarter"] = np.sin(2 * np.pi * quarter_hour / 96)
    df["cos_quarter"] = np.cos(2 * np.pi * quarter_hour / 96)

    italy_holidays = holidays.IT()

    def is_campus_closed(timestamp_local: pd.Timestamp) -> int:
        month_value = timestamp_local.month
        day = timestamp_local.day
        hour = timestamp_local.hour
        weekday = timestamp_local.weekday()
        week = timestamp_local.isocalendar().week
        if weekday == 6 or timestamp_local.date() in italy_holidays:
            return 1
        if week in [33, 34]:
            return 1
        if (month_value == 12 and day >= 23) or (month_value == 1 and day <= 3):
            return 1
        if weekday in range(0, 5):
            return 0 if 7 <= hour < 21 else 1
        if weekday == 5:
            return 0 if 7 <= hour < 20 else 1
        return 0

    df["campus_closed"] = [is_campus_closed(ts) for ts in dt_local]
    df["T_open"] = df["temperature_2m"] * (1 - df["campus_closed"])
    return df


def fetch_target_15min(api: OptimoApi, identifier: str, start_dt: dt.datetime, end_dt: dt.datetime, colname: str) -> pd.DataFrame:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    data = api.get_values_in_range([identifier], start_ms, end_ms, limit=1_000_000)
    records = data.get(identifier, []) or []
    if not records:
        return pd.DataFrame(columns=["datetime", colname])
    df = pd.DataFrame({"datetime": [r["timestamp"] for r in records], colname: [r["value"] for r in records]})
    df = ensure_utc_datetime_index(df, "datetime").set_index("datetime").resample("15min").mean().reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def build_hist_for_target(api: OptimoApi, identifier: str, start_dt: dt.datetime, end_dt: dt.datetime, target: str) -> pd.Series:
    df = fetch_target_15min(api, identifier, start_dt, end_dt, target)
    if df.empty:
        return pd.Series(dtype=float)
    series = df.set_index("datetime")[target].sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="15min")
    return series.reindex(full_index)


def build_lag_columns(target: str, forecast_index: pd.DatetimeIndex, hist_series: pd.Series, lag_hours: list[int]) -> pd.DataFrame:
    y = pd.to_numeric(hist_series, errors="coerce").sort_index()
    if y.empty:
        raise ValueError(f"Historical series for {target} is empty.")
    full_index = pd.date_range(y.index.min(), forecast_index.max(), freq="15min")
    y_full = y.reindex(full_index)
    data = {}
    for lag_h in lag_hours:
        data[f"{target}_lag{lag_h}h"] = y_full.shift(lag_h * 4).reindex(forecast_index).to_numpy()
    return pd.DataFrame(data, index=forecast_index)


def upload_to_optimo(out_df: pd.DataFrame, api: OptimoApi, value_col: str, variable_id: str) -> None:
    if not variable_id:
        print(f"Skipping {value_col} upload: upload variable ID is not set.")
        return
    df_up = out_df[["datetime", value_col]].copy().sort_values("datetime")
    samples = []
    missing_count = 0
    for ts_utc, value in zip(df_up["datetime"], df_up[value_col]):
        timestamp_ms = int(pd.Timestamp(ts_utc).tz_localize("UTC").timestamp() * 1000)
        if value is None or pd.isna(value):
            missing_count += 1
            value = 0.0
        samples.append({"timestamp": timestamp_ms, "value": float(value)})
    payload = [{"variable_id": variable_id, "samples": samples}]
    print(f"Uploading {value_col} -> {variable_id}: {len(samples)} samples; missing replaced: {missing_count}")
    response = api.injest_values(payload)
    print("Cloud Optimo ingest successful." if response == {} else f"Cloud Optimo ingest returned: {response}")


def forecast_one_target(
    api: OptimoApi,
    target: str,
    target_id: str,
    model_path: Path,
    X_base: pd.DataFrame,
    target_date_local: dt.date,
    local_tz: ZoneInfo,
    args: argparse.Namespace,
) -> pd.DataFrame:
    forecast_index = X_base.index

    if target_date_local.month in args.summer_zero_months:
        print(f"{target}: target day in summer-zero months -> forcing zero forecast.")
        out = pd.DataFrame({"datetime": forecast_index, target: np.zeros(len(forecast_index), dtype=float)})
        out["datetime_italy"] = out["datetime"].dt.tz_localize("UTC").dt.tz_convert(local_tz).dt.tz_localize(None)
        return out

    start_local = dt.datetime.combine(target_date_local, dt.time(0, 0), tzinfo=local_tz)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(minutes=15)
    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    hist_start = start_utc - dt.timedelta(days=args.historical_lookback_days)
    hist_end = end_utc - dt.timedelta(hours=args.min_lag_hours)

    hist_series = build_hist_for_target(api, target_id, hist_start, hist_end, target)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = load(model_path)
    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(f"Model {model_path} lacks feature_names_in_. Re-train with pandas DataFrame inputs.")

    feature_names = list(model.feature_names_in_)
    lag_cols = build_lag_columns(target, forecast_index, hist_series, args.lag_hours).clip(lower=0, upper=args.lag_cap_kw)
    base_needed = [col for col in X_base.columns if col in feature_names]
    X_target = pd.concat([X_base[base_needed], lag_cols], axis=1)

    missing = [col for col in feature_names if col not in X_target.columns]
    if missing:
        raise RuntimeError(f"[{target}] Missing required model features: {missing}")

    X_target = X_target.loc[:, feature_names]
    if X_target.isna().any().any():
        bad_cols = X_target.columns[X_target.isna().any()].tolist()
        X_target = X_target.interpolate(limit_direction="both", axis=0)
        if X_target.isna().any().any():
            raise RuntimeError(f"[{target}] NaNs remain after interpolation in columns: {bad_cols}")

    prediction = model.predict(X_target)
    out = pd.DataFrame({"datetime": forecast_index, target: prediction})
    out["datetime_italy"] = out["datetime"].dt.tz_localize("UTC").dt.tz_convert(local_tz).dt.tz_localize(None)
    return out


def main() -> None:
    args = parse_arguments()
    local_tz = ZoneInfo(args.local_timezone)
    target_date_local = parse_forecast_date(args.forecast_date, local_tz)
    models_dir = Path(args.models_dir)
    forecasts_dir = Path(args.forecasts_dir)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Thermal forecast configuration ---")
    print(f"Forecast local date: {target_date_local}")
    print(f"Models directory:    {models_dir}")
    print(f"Forecasts directory: {forecasts_dir}")
    print(f"Upload to Optimo:    {args.upload_to_optimo}")

    df_weather = fetch_weather_minutely15_for_day(target_date_local, args.latitude, args.longitude, local_tz)
    X_base = add_calendar_features(df_weather, local_tz)
    api = login_cloud_optimo()

    target_configs = {
        "THERMAL_LOAD_kW": {
            "id": args.thermal_var_id,
            "model": models_dir / "RF_THERMAL_LOAD_kW.joblib",
            "upload": args.thermal_upload_variable_id,
        },
        "DH_THERMAL_LOAD_kW": {
            "id": args.dh_thermal_var_id,
            "model": models_dir / "RF_DH_THERMAL_LOAD_kW.joblib",
            "upload": args.dh_upload_variable_id,
        },
    }

    for target, cfg in target_configs.items():
        print(f"\n=== Forecasting {target} ===")
        out = forecast_one_target(api, target, cfg["id"], cfg["model"], X_base, target_date_local, local_tz, args)
        out_file = forecasts_dir / f"forecast_{target}_{target_date_local.isoformat()}.xlsx"
        out.to_excel(out_file, index=False)
        print(f"Saved forecast -> {out_file}")

        if args.upload_to_optimo:
            upload_to_optimo(out, api, value_col=target, variable_id=cfg["upload"])
        else:
            print("Optimo upload disabled.")


if __name__ == "__main__":
    main()
