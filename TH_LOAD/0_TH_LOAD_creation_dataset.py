"""Build the historical training dataset for thermal-load day-ahead forecasting.

This script is GitHub-ready:
- credentials are read from a private .env file located in TH_LOAD/;
- dates, paths, coordinates, and Optimo variable IDs are configurable;
- no API keys or passwords are hard-coded in the source code.

Typical usage from the external repository folder:
    python TH_LOAD/0_TH_LOAD_creation_dataset.py

Custom date range:
    python TH_LOAD/0_TH_LOAD_creation_dataset.py --start-date 2024-12-03 --end-date 2025-09-30
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
from OptimoApi import OptimoApi
from retry_requests import retry

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def get_required_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. Create TH_LOAD/.env from TH_LOAD/.env.example."
        )
    return value


def login_cloud_optimo() -> OptimoApi:
    return OptimoApi(
        api_key=get_required_env_var("OPTIMO_API_KEY"),
        app_id=get_required_env_var("OPTIMO_APP_ID"),
        app_secret=get_required_env_var("OPTIMO_APP_SECRET"),
    )


def parse_utc_date(value: str, end_of_day: bool = False) -> dt.datetime:
    value = value.strip().lower()
    if value == "today":
        date_value = dt.datetime.now(dt.timezone.utc).date()
    else:
        try:
            date_value = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Use YYYY-MM-DD or 'today'.") from exc
    time_value = dt.time(23, 59, 59) if end_of_day else dt.time(0, 0, 0)
    return dt.datetime.combine(date_value, time_value, tzinfo=dt.timezone.utc)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create thermal-load forecasting dataset.")
    parser.add_argument("--start-date", default=os.getenv("THERMAL_DATASET_START_DATE", "2024-12-03"))
    parser.add_argument("--end-date", default=os.getenv("THERMAL_DATASET_END_DATE", "today"))
    parser.add_argument("--output-path", default=os.getenv("THERMAL_DATASET_PATH", "TH_LOAD/dataset_thermal_clean.xlsx"))
    parser.add_argument("--latitude", type=float, default=float(os.getenv("LATITUDE", "45.4643")))
    parser.add_argument("--longitude", type=float, default=float(os.getenv("LONGITUDE", "9.1895")))
    parser.add_argument("--local-timezone", default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"))
    parser.add_argument("--thermal-var-id", default=os.getenv("THERMAL_LOAD_VAR_ID", "XVpGIF_pHutx0"))
    parser.add_argument("--dh-thermal-var-id", default=os.getenv("DH_THERMAL_LOAD_VAR_ID", "XVpGIF_Ai5Ip9"))
    parser.add_argument("--upper-cap-kw", type=float, default=float(os.getenv("THERMAL_UPPER_CAP_KW", "13000")))
    parser.add_argument("--summer-zero-months", default=os.getenv("THERMAL_SUMMER_ZERO_MONTHS", "5,6,7,8,9"))
    return parser.parse_args()


def ensure_utc_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df.sort_values(col).reset_index(drop=True)


def fetch_thermal_15min(
    api: OptimoApi,
    identifier: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    column_name: str,
) -> pd.DataFrame:
    """Fetch one thermal-load channel and resample it to 15-minute mean power."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    print(f"[Thermal] fetching {column_name} ({identifier})")

    data = api.get_values_in_range([identifier], start_ms, end_ms, limit=1_000_000)
    records = data.get(identifier, []) or []
    if not records:
        print(f"  Warning: no data returned for {identifier}")
        return pd.DataFrame(columns=["datetime", column_name])

    df = pd.DataFrame(
        {
            "datetime": [record["timestamp"] for record in records],
            column_name: [record["value"] for record in records],
        }
    )
    df = ensure_utc_datetime_index(df, "datetime")
    df = df.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
    return df.resample("15min").mean().reset_index()


def fetch_weather_15min(
    latitude: float,
    longitude: float,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> pd.DataFrame:
    """Download 15-minute weather data from Open-Meteo Historical Forecast API."""
    print("\n--- Downloading 15-minute weather data from Open-Meteo ---")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "minutely_15": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "direct_normal_irradiance",
        ],
    }
    responses = openmeteo.weather_api("https://historical-forecast-api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    minutely_15 = response.Minutely15()

    df_weather = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left",
            ),
            "temperature_2m": minutely_15.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": minutely_15.Variables(1).ValuesAsNumpy(),
            "dew_point_2m": minutely_15.Variables(2).ValuesAsNumpy(),
            "direct_normal_irradiance": minutely_15.Variables(3).ValuesAsNumpy(),
        }
    )
    df_weather = df_weather[(df_weather["datetime"] >= start_dt) & (df_weather["datetime"] <= end_dt)]
    return df_weather.reset_index(drop=True)


def add_calendar_features(df_input: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    df = df_input.copy()
    df["datetime_italy"] = df["datetime"].dt.tz_convert(local_tz)

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
    return df


def apply_thermal_cleaning(df: pd.DataFrame, targets: list[str], upper_cap_kw: float, summer_months: set[int], local_tz: ZoneInfo) -> pd.DataFrame:
    """Clip thermal targets and set summer local months to zero."""
    out = df.copy()
    dt_local = out["datetime"].dt.tz_convert(local_tz)
    summer_mask = dt_local.dt.month.isin(summer_months)
    for target in targets:
        if target in out.columns:
            out[target] = pd.to_numeric(out[target], errors="coerce").clip(lower=0, upper=upper_cap_kw)
            out.loc[summer_mask, target] = 0.0
    return out


def build_full_dataset(args: argparse.Namespace) -> pd.DataFrame:
    local_tz = ZoneInfo(args.local_timezone)
    start_datetime = parse_utc_date(args.start_date, end_of_day=False)
    end_datetime = parse_utc_date(args.end_date, end_of_day=True)
    if end_datetime <= start_datetime:
        raise ValueError("End date must be after start date.")

    print("\n--- Thermal dataset configuration ---")
    print(f"Start datetime UTC: {start_datetime}")
    print(f"End datetime UTC:   {end_datetime}")
    print(f"Output path:        {args.output_path}")
    print(f"Latitude:           {args.latitude}")
    print(f"Longitude:          {args.longitude}")

    api = login_cloud_optimo()

    df_thermal = fetch_thermal_15min(api, args.thermal_var_id, start_datetime, end_datetime, "THERMAL_LOAD_kW")
    df_dh = fetch_thermal_15min(api, args.dh_thermal_var_id, start_datetime, end_datetime, "DH_THERMAL_LOAD_kW")
    df_weather = fetch_weather_15min(args.latitude, args.longitude, start_datetime, end_datetime)

    thermals = pd.merge(df_thermal.sort_values("datetime"), df_dh.sort_values("datetime"), on="datetime", how="outer")
    df_merged = pd.merge(thermals, df_weather.sort_values("datetime"), on="datetime", how="outer")
    df = add_calendar_features(df_merged, local_tz)

    summer_months = set(parse_int_list(args.summer_zero_months))
    df = apply_thermal_cleaning(
        df,
        targets=["THERMAL_LOAD_kW", "DH_THERMAL_LOAD_kW"],
        upper_cap_kw=args.upper_cap_kw,
        summer_months=summer_months,
        local_tz=local_tz,
    )

    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime_italy"] = df["datetime_italy"].dt.tz_localize(None)
    return df


def main() -> None:
    args = parse_arguments()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_full_dataset(args)
    print("\n--- Final NaN counts ---")
    print(df.isna().sum())
    df.to_excel(output_path, index=False)
    print(f"\nThermal dataset saved to {output_path}")


if __name__ == "__main__":
    main()
