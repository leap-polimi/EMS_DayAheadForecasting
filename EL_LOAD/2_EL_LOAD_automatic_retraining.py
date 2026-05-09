"""Append recent data and retrain EMS electrical-load forecasting models.

This script is designed for repository publication:
- Credentials are read from environment variables.
- Paths, dates, coordinates, lags, and retraining windows are configurable.
- No private API keys or passwords are hard-coded in the source code.

Typical usage from the external repository folder:
    python EL_LOAD/2_EL_LOAD_monthly_retraining.py

Run without appending new data, only retrain from the existing dataset:
    python EL_LOAD/2_EL_LOAD_monthly_retraining.py --skip-dataset-update

Use a custom initial dataset size if dataset.xlsx does not exist:
    python EL_LOAD/2_EL_LOAD_monthly_retraining.py --initial-history-days 500

Relevant .env variables:
    OPTIMO_API_KEY=...
    OPTIMO_APP_ID=...
    OPTIMO_APP_SECRET=...
    DATASET_PATH=EL_LOAD/dataset.xlsx
    MODELS_DIR=EL_LOAD/models
    RESULTS_DIR=EL_LOAD/results
    METRICS_PATH=EL_LOAD/results/metrics_summary.xlsx
    LATITUDE=45.4643
    LONGITUDE=9.1895
    LOCAL_TIMEZONE=Europe/Rome
    BAD_DATA_START=2025-02-07T21:45:00
    BAD_DATA_END=2025-02-13T05:45:00
    FINAL_TRAINING_DAYS=365
    INITIAL_HISTORY_DAYS=370
    WEATHER_DATA_DELAY_DAYS=2
    LAG_HOURS=48,72,96,120,144,168
    RANDOM_STATE=42
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


def get_required_env_var(name: str) -> str:
    """Return an environment variable or raise a clear configuration error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Create a local .env file from .env.example or export the variable "
            "before running this script."
        )
    return value



def login_cloud_optimo() -> OptimoApi:
    """Create an authenticated Optimo API client from environment variables."""
    return OptimoApi(
        api_key=get_required_env_var("OPTIMO_API_KEY"),
        app_id=get_required_env_var("OPTIMO_APP_ID"),
        app_secret=get_required_env_var("OPTIMO_APP_SECRET"),
    )



def parse_datetime(value: str | None) -> pd.Timestamp | None:
    """Parse a date/datetime string into a timezone-naive UTC pandas Timestamp.

    Accepted examples:
    - 2025-02-07
    - 2025-02-07T21:45:00
    - 2025-02-07 21:45:00
    - none, null, empty string -> None
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
    """Read command-line options used to customize monthly retraining."""
    default_results_dir = os.getenv("RESULTS_DIR", "EL_LOAD/results")

    parser = argparse.ArgumentParser(
        description="Append recent EMS electrical-load data and retrain Random Forest models."
    )

    parser.add_argument(
        "--dataset-path",
        default=os.getenv("DATASET_PATH", "EL_LOAD/dataset.xlsx"),
        help="Dataset Excel file to append/update. Can also be set with DATASET_PATH.",
    )

    parser.add_argument(
        "--models-dir",
        default=os.getenv("MODELS_DIR", "EL_LOAD/models"),
        help="Directory where trained models are saved. Can also be set with MODELS_DIR.",
    )

    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help="Directory where retraining outputs can be saved. Can also be set with RESULTS_DIR.",
    )

    parser.add_argument(
        "--metrics-path",
        default=os.getenv("METRICS_PATH", str(Path(default_results_dir) / "metrics_summary.xlsx")),
        help=(
            "Excel file containing best_params from initial model training. "
            "Can also be set with METRICS_PATH."
        ),
    )

    parser.add_argument(
        "--latitude",
        type=float,
        default=float(os.getenv("LATITUDE", "45.4643")),
        help="Site latitude. Can also be set with LATITUDE. Default: 45.4643.",
    )

    parser.add_argument(
        "--longitude",
        type=float,
        default=float(os.getenv("LONGITUDE", "9.1895")),
        help="Site longitude. Can also be set with LONGITUDE. Default: 9.1895.",
    )

    parser.add_argument(
        "--local-timezone",
        default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"),
        help="Local timezone used for calendar features. Default: Europe/Rome.",
    )

    parser.add_argument(
        "--bad-start",
        type=parse_datetime,
        default=parse_datetime(os.getenv("BAD_DATA_START", "2025-02-07T21:45:00")),
        help="Start of known bad-data window to remove. Set to none to disable.",
    )

    parser.add_argument(
        "--bad-end",
        type=parse_datetime,
        default=parse_datetime(os.getenv("BAD_DATA_END", "2025-02-13T05:45:00")),
        help="End of known bad-data window to remove. Set to none to disable.",
    )

    parser.add_argument(
        "--final-training-days",
        type=int,
        default=int(os.getenv("FINAL_TRAINING_DAYS", "365")),
        help="Number of recent days used for monthly retraining. Default: 365.",
    )

    parser.add_argument(
        "--initial-history-days",
        type=int,
        default=int(os.getenv("INITIAL_HISTORY_DAYS", "370")),
        help=(
            "Number of historical days to build if the dataset file does not exist. "
            "Default: 370."
        ),
    )

    parser.add_argument(
        "--weather-data-delay-days",
        type=int,
        default=int(os.getenv("WEATHER_DATA_DELAY_DAYS", "2")),
        help=(
            "Number of days to subtract from today because historical weather data are not "
            "fully available immediately. Default: 2."
        ),
    )

    parser.add_argument(
        "--lag-hours",
        type=parse_lag_hours,
        default=parse_lag_hours(os.getenv("LAG_HOURS", "48,72,96,120,144,168")),
        help="Comma-separated target lag hours. Can also be set with LAG_HOURS.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
        help="Random seed used by RandomForestRegressor. Default: 42.",
    )

    parser.add_argument(
        "--skip-dataset-update",
        action="store_true",
        help="Do not append new data; only retrain models using the existing dataset.",
    )

    return parser.parse_args()


# ======================================================================
# Static Cloud Optimo variable IDs
# ======================================================================

CABIN_IDS = {
    "C1": ["XVpGIF_wa4Kkr", "XVpGIF_CZZwk7", "XVpGIF_tYHam9"],
    "C2": ["XVpGIF_tdDW1q", "XVpGIF_K8mhUn", "XVpGIF_wrdfdl"],
    "C3": ["XVpGIF_OKhasJ", "XVpGIF_nEf6X9", "XVpGIF_ZUtNN3"],
    "C4": ["XVpGIF_QDVDHh", "XVpGIF_iIvC3b", "XVpGIF_cRAKdQ"],
    "C5": ["XVpGIF_SCXTiJ", "XVpGIF_tSDPAx", "XVpGIF_Vx6kuf"],
    "C6": ["XVpGIF_OKA8wi", "XVpGIF_WEWy3L", "XVpGIF_T20KiD"],
    "C8": ["XVpGIF_HHXDvh", "XVpGIF_beumkE"],
}

PV_IDS_MAIN = {
    "C1": ["XVpGIF_PlEgT8"],
    "C2": ["XVpGIF_yzWR0f"],
    "C3": ["XVpGIF_wwBKHP"],
    "C4": ["XVpGIF_TUV9xB"],
    "C5": ["XVpGIF_jvwCW0", "XVpGIF_dYthLo"],
    "C8": ["XVpGIF_mEIsFt", "XVpGIF_CDk2Gv"],
    "C9": ["XVpGIF_bImoB6", "XVpGIF_ot5CPh"],
}

PV_IDS_BACKUP = {
    "C1": ["s9BeRW_ZCe3ai"],
    "C2": ["s9BeRW_IZXLgt"],
    "C3": ["s9BeRW_K8AZky"],
    "C4": ["s9BeRW_HnPtSm"],
    "C5": ["s9BeRW_p0La2G", "s9BeRW_dfvjX9"],
    "C8": ["s9BeRW_qsWvJs", "XVpGIF_CDk2Gv"],
    "C9": ["s9BeRW_IhlsRK", "s9BeRW_OPTh3w"],
}

MCB_CHP_IDS = {
    "MCB1_P_kW": "XVpGIF_MVziRK",
    "MCB2_P_kW": "XVpGIF_LF3iDv",
    "CHP_P_kW": "XVpGIF_UjTIU5",
}


# ======================================================================
# Time-series helpers
# ======================================================================


def round_to_15_floor(timestamp: dt.datetime) -> dt.datetime:
    """Round a datetime down to the previous 15-minute boundary."""
    minute = (timestamp.minute // 15) * 15
    return timestamp.replace(minute=minute, second=0, microsecond=0)



def ensure_utc_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Convert an epoch-ms datetime column to timezone-aware UTC and sort it."""
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df.sort_values(col).reset_index(drop=True)



def to_utc_ms(timestamp: dt.datetime | pd.Timestamp) -> int:
    """Convert a naive/aware datetime to UTC epoch milliseconds."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


# ======================================================================
# Cloud fetch helpers
# ======================================================================


def fetch_cabin_net_15min(
    api: OptimoApi,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    cabin_id: str,
    identifiers: list[str],
) -> pd.DataFrame:
    """Fetch cabin net electrical consumption and resample to 15-minute means."""
    start_ms = to_utc_ms(start_dt)
    end_ms = to_utc_ms(end_dt)

    print(f"[Cabin {cabin_id}] fetching transformer powers...")
    data = api.get_values_in_range(identifiers, start_ms, end_ms, limit=1_000_000)

    parts = []
    for idx, identifier in enumerate(identifiers):
        records = data.get(identifier, []) or []
        if not records:
            print(f"  Warning: no data returned for {identifier}")
            continue

        part = pd.DataFrame(
            {
                "datetime": [record["timestamp"] for record in records],
                f"el_power{idx + 1}": [record["value"] for record in records],
            }
        )
        part = ensure_utc_datetime_index(part, "datetime")
        parts.append(part)

    if not parts:
        idx = pd.date_range(
            start=round_to_15_floor(start_dt),
            end=round_to_15_floor(end_dt),
            freq="15min",
            tz="UTC",
            inclusive="both",
        )
        return pd.DataFrame({"datetime": idx, f"net_el_cons_{cabin_id}": np.nan})

    df = parts[0]
    for part in parts[1:]:
        df = pd.merge(df, part, on="datetime", how="outer")

    df = df.sort_values("datetime").set_index("datetime")
    df = df.groupby(level=0).mean(numeric_only=True)

    power_cols = [col for col in df.columns if col.startswith("el_power")]
    df["net_kW"] = df[power_cols].sum(axis=1, min_count=1)

    output = df[["net_kW"]].resample("15min").mean().reset_index()
    output = output[(output["datetime"] >= start_dt) & (output["datetime"] <= end_dt)].reset_index(drop=True)
    output.rename(columns={"net_kW": f"net_el_cons_{cabin_id}"}, inplace=True)

    return output[["datetime", f"net_el_cons_{cabin_id}"]]



def fetch_1min_resample_15(
    api: OptimoApi,
    identifier: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    label: str,
) -> pd.DataFrame:
    """Fetch a 1-minute Cloud Optimo series and resample it to 15-minute means."""
    start_ms = to_utc_ms(start_dt)
    end_ms = to_utc_ms(end_dt)

    print(f"[1-min] fetching {label} ({identifier})...")
    data = api.get_values_in_range([identifier], start_ms, end_ms, limit=1_000_000)
    records = data.get(identifier, []) or []

    if not records:
        print(f"  Warning: no data returned for {identifier}")
        return pd.DataFrame(columns=["datetime", label])

    df = pd.DataFrame(
        {
            "datetime": [record["timestamp"] for record in records],
            label: [record["value"] for record in records],
        }
    )
    df = ensure_utc_datetime_index(df, "datetime")
    return df.set_index("datetime").resample("15min").mean().reset_index()



def fetch_pv_15min_sum(
    api: OptimoApi,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    cabin_id: str,
    ids_main: list[str],
    ids_backup: list[str],
) -> pd.DataFrame:
    """Fetch 15-minute PV power for one cabin from cumulative energy meters."""
    start_dt = round_to_15_floor(start_dt)
    end_dt = round_to_15_floor(end_dt)
    start_ms = to_utc_ms(start_dt)
    end_ms = to_utc_ms(end_dt)

    def read_energy_series(identifier: str | None) -> pd.DataFrame | None:
        if identifier is None:
            return None
        try:
            data = api.get_values_in_range([identifier], start_ms, end_ms, limit=1_000_000)
            records = data.get(identifier, []) or []
            if not records:
                return None
            df_energy = pd.DataFrame(
                {
                    "datetime": [record["timestamp"] for record in records],
                    "el_energy": [record["value"] for record in records],
                }
            )
            return ensure_utc_datetime_index(df_energy, "datetime")
        except Exception as exc:
            print(f"  Warning: fetch error for {identifier}: {exc}")
            return None

    def energy_to_power(df_energy: pd.DataFrame | None) -> pd.DataFrame | None:
        if df_energy is None or df_energy.empty:
            return None

        df = df_energy.copy().sort_values("datetime").drop_duplicates(subset="datetime")
        df["dt_h"] = df["datetime"].diff().dt.total_seconds() / 3600.0
        df["dE_kWh"] = df["el_energy"].diff()

        df.loc[df["dE_kWh"] < 0, "dE_kWh"] = np.nan
        df.loc[df["dt_h"] <= 0, "dt_h"] = np.nan

        df["pv_kW"] = df["dE_kWh"] / df["dt_h"]
        return df.set_index("datetime")["pv_kW"].resample("15min").mean().to_frame().reset_index()

    print(f"[PV {cabin_id}] computing 15-minute PV power...")
    parts = []
    ids_main = ids_main or []
    ids_backup = ids_backup or []

    for position in range(max(len(ids_main), len(ids_backup))):
        main_id = ids_main[position] if position < len(ids_main) else None
        backup_id = ids_backup[position] if position < len(ids_backup) else None

        power_main = energy_to_power(read_energy_series(main_id))
        power_backup = energy_to_power(read_energy_series(backup_id))

        if power_main is not None and power_backup is not None:
            merged = pd.merge(
                power_main,
                power_backup,
                on="datetime",
                how="outer",
                suffixes=("_main", "_backup"),
            )
            merged["pv_kW"] = merged["pv_kW_main"].combine_first(merged["pv_kW_backup"])
            parts.append(merged[["datetime", "pv_kW"]])
        elif power_main is not None:
            parts.append(power_main)
        elif power_backup is not None:
            parts.append(power_backup)

    if not parts:
        idx = pd.date_range(start_dt, end_dt, freq="15min", tz="UTC", inclusive="both")
        return pd.DataFrame({"datetime": idx, f"pv_power_{cabin_id}_kW": np.nan})

    df_pv = parts[0]
    for part in parts[1:]:
        df_pv = pd.merge(df_pv, part, on="datetime", how="outer", suffixes=("", "_dup"))

    pv_cols = [col for col in df_pv.columns if col.startswith("pv_kW")]
    df_pv["pv_kW_sum"] = df_pv[pv_cols].sum(axis=1, min_count=1)

    output = df_pv[["datetime", "pv_kW_sum"]].sort_values("datetime").reset_index(drop=True)
    output.rename(columns={"pv_kW_sum": f"pv_power_{cabin_id}_kW"}, inplace=True)
    return output


# ======================================================================
# Weather and feature engineering
# ======================================================================


def fetch_weather_15min(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Fetch native 15-minute historical weather data from Open-Meteo."""
    print("\n--- Downloading 15-minute historical weather data from Open-Meteo ---")

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_utc.date().isoformat(),
        "end_date": end_utc.date().isoformat(),
        "minutely_15": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "direct_normal_irradiance",
        ],
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    minutely_15 = response.Minutely15()

    df = pd.DataFrame(
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

    df = df[(df["datetime"] >= start_utc) & (df["datetime"] <= end_utc)].reset_index(drop=True)
    return df



def add_calendar_features(df_15: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    """Add calendar, cyclic, campus-closure, and derived features."""
    df = df_15.copy()
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


# ======================================================================
# Dataset update
# ======================================================================


def build_block_dataset_like(
    api: OptimoApi,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    latitude: float,
    longitude: float,
    local_tz: ZoneInfo,
) -> pd.DataFrame:
    """Reproduce the columns of dataset.xlsx for the given UTC-aware window."""
    print("\n--- Building new dataset block ---")
    print(f"Block start UTC: {start_utc}")
    print(f"Block end UTC:   {end_utc}")

    df_mcbchp = None
    for label, identifier in MCB_CHP_IDS.items():
        part = fetch_1min_resample_15(api, identifier, start_utc, end_utc, label)
        df_mcbchp = part if df_mcbchp is None else pd.merge(df_mcbchp, part, on="datetime", how="outer")

    df_net = None
    for cabin, identifiers in CABIN_IDS.items():
        part = fetch_cabin_net_15min(api, start_utc, end_utc, cabin, identifiers)
        df_net = part if df_net is None else pd.merge(df_net, part, on="datetime", how="outer")

    df_pv = None
    for cabin, identifiers in PV_IDS_MAIN.items():
        part = fetch_pv_15min_sum(api, start_utc, end_utc, cabin, identifiers, PV_IDS_BACKUP.get(cabin, []))
        df_pv = part if df_pv is None else pd.merge(df_pv, part, on="datetime", how="outer")

    df_base = df_mcbchp.copy()
    if df_net is not None:
        df_base = pd.merge(df_base, df_net, on="datetime", how="outer")
    if df_pv is not None:
        df_base = pd.merge(df_base, df_pv, on="datetime", how="outer")

    cabin_balance_cols = [
        f"net_el_cons_C{i}"
        for i in [1, 2, 3, 4, 5, 6, 8]
        if f"net_el_cons_C{i}" in df_base.columns
    ]
    sum_net_c1_c8 = df_base[cabin_balance_cols].sum(axis=1, min_count=1)
    df_base["net_el_cons_C10"] = (
        df_base[["CHP_P_kW", "MCB1_P_kW", "MCB2_P_kW"]].sum(axis=1, min_count=1)
        - sum_net_c1_c8
    )

    for cabin in list(CABIN_IDS.keys()) + ["C10"]:
        net_col = f"net_el_cons_{cabin}"
        pv_col = f"pv_power_{cabin}_kW"
        gross_col = f"gross_el_cons_{cabin}"

        if net_col not in df_base.columns:
            print(f"  Warning: missing {net_col}; skipping {gross_col}")
            continue

        pv_series = df_base[pv_col] if pv_col in df_base.columns else 0
        df_base[gross_col] = df_base[net_col] + pv_series

    gross_cols = [col for col in df_base.columns if col.startswith("gross_el_cons_")]
    df_final_cons = df_base[["datetime", "MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"] + gross_cols].copy()

    df_final_cons["CONS_TOT_kW"] = df_final_cons[gross_cols].sum(axis=1, min_count=1)
    df_final_cons["CONS_TOT_NET_kW"] = df_final_cons[["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"]].sum(
        axis=1,
        min_count=1,
    )
    df_final_cons.drop(["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"], axis=1, inplace=True)

    df_weather = fetch_weather_15min(start_utc, end_utc, latitude=latitude, longitude=longitude)

    df_merged = pd.merge(
        df_final_cons.sort_values("datetime"),
        df_weather.sort_values("datetime"),
        on="datetime",
        how="outer",
    )

    df_features = add_calendar_features(df_merged, local_tz=local_tz)

    df_features["datetime"] = df_features["datetime"].dt.tz_localize(None)
    df_features["datetime_italy"] = df_features["datetime_italy"].dt.tz_localize(None)

    return df_features



def load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the current dataset if it exists; otherwise return an empty skeleton."""
    if dataset_path.exists():
        df_all = pd.read_excel(dataset_path)
        df_all["datetime"] = pd.to_datetime(df_all["datetime"])
        return df_all.sort_values("datetime").reset_index(drop=True)

    return pd.DataFrame(
        columns=[
            "datetime",
            "datetime_italy",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "direct_normal_irradiance",
            "campus_closed",
            "T_open",
            "sin_month",
            "cos_month",
            "sin_day_of_week",
            "cos_day_of_week",
            "sin_quarter",
            "cos_quarter",
        ]
    )



def update_dataset_if_needed(
    dataset_path: Path,
    df_all: pd.DataFrame,
    api: OptimoApi,
    local_tz: ZoneInfo,
    latitude: float,
    longitude: float,
    initial_history_days: int,
    weather_data_delay_days: int,
    skip_dataset_update: bool,
) -> pd.DataFrame:
    """Append missing rows to the dataset, respecting Open-Meteo data delay."""
    if skip_dataset_update:
        print("Skipping dataset update because --skip-dataset-update was provided.")
        return df_all

    now_local = dt.datetime.now(local_tz)
    weather_cut_local = (now_local - dt.timedelta(days=weather_data_delay_days)).date()
    update_end_local = dt.datetime.combine(weather_cut_local, dt.time(23, 45), tzinfo=local_tz)
    update_end_utc = update_end_local.astimezone(dt.timezone.utc)

    if df_all.empty or df_all["datetime"].isna().all():
        start_local = (now_local - dt.timedelta(days=initial_history_days)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        start_utc = start_local.astimezone(dt.timezone.utc)
    else:
        last_timestamp = pd.to_datetime(df_all["datetime"].max())
        start_utc = (pd.Timestamp(last_timestamp).tz_localize("UTC") + pd.Timedelta(minutes=15)).to_pydatetime()

    if start_utc > update_end_utc:
        print("No new rows to append. Proceeding to retraining.")
        return df_all

    print(f"Appending rows from {start_utc} UTC to {update_end_utc} UTC.")
    df_new = build_block_dataset_like(
        api=api,
        start_utc=start_utc,
        end_utc=update_end_utc,
        latitude=latitude,
        longitude=longitude,
        local_tz=local_tz,
    )

    df_all = pd.concat([df_all, df_new], axis=0, ignore_index=True)
    df_all = df_all.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_excel(dataset_path, index=False)
    print(f"Dataset updated: {dataset_path}")
    print(f"Last row: {df_all['datetime'].max()}")

    return df_all


# ======================================================================
# Model retraining
# ======================================================================


def parse_best_params(value) -> dict:
    """Parse the best_params column saved by the initial training script."""
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



def prepare_training_frame(
    df_all: pd.DataFrame,
    bad_start: pd.Timestamp | None,
    bad_end: pd.Timestamp | None,
    final_training_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Clean dataset and keep only the final training window."""
    df = df_all.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if bad_start is not None and bad_end is not None:
        if bad_end < bad_start:
            raise ValueError(f"bad_end {bad_end} must be after bad_start {bad_start}.")
        bad_mask = (df["datetime"] >= bad_start) & (df["datetime"] <= bad_end)
        df = df.loc[~bad_mask]

    last_ts = pd.to_datetime(df["datetime"].max())
    if pd.isna(last_ts):
        raise ValueError("Dataset is empty or has no valid datetime values after cleaning.")

    first_ts = last_ts - pd.Timedelta(days=final_training_days)
    df = df[(df["datetime"] >= first_ts) & (df["datetime"] <= last_ts)].copy()

    rows_before_dropna = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after_dropna = len(df)

    print("\n--- Monthly retraining data summary ---")
    print(f"Training window start: {first_ts}")
    print(f"Training window end:   {last_ts}")
    print(f"Rows before dropna:    {rows_before_dropna}")
    print(f"Rows after dropna:     {rows_after_dropna}")
    print(f"Dropped NaN rows:      {rows_before_dropna - rows_after_dropna}")

    if df.empty:
        raise ValueError("No rows remain for retraining after cleaning. Check dataset and filters.")

    return df, first_ts, last_ts



def get_targets(df: pd.DataFrame) -> list[str]:
    """Return all forecasting targets present in the dataset."""
    targets = [col for col in df.columns if col.startswith("gross_el_cons_")]

    for total_col in ["CONS_TOT_kW", "CONS_TOT_NET_kW"]:
        if total_col in df.columns:
            targets.append(total_col)

    if not targets:
        raise ValueError(
            "No targets found. Expected columns starting with 'gross_el_cons_' "
            "and/or 'CONS_TOT_kW', 'CONS_TOT_NET_kW'."
        )

    return targets



def retrain_models(
    df: pd.DataFrame,
    metrics_path: Path,
    models_dir: Path,
    lag_hours: list[int],
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    random_state: int,
) -> None:
    """Retrain each model using saved best hyperparameters and overwrite joblib files."""
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot find {metrics_path}. Run the initial training script first to create metrics_summary.xlsx."
        )

    df_params = pd.read_excel(metrics_path)
    if "target" not in df_params.columns or "best_params" not in df_params.columns:
        raise ValueError(f"{metrics_path} must contain columns 'target' and 'best_params'.")

    targets = get_targets(df)

    drop_cols = ["datetime_italy", "month", "day_of_week", "quarter_hour"] + targets
    drop_cols = [col for col in drop_cols if col in df.columns]
    X_base = df.drop(columns=drop_cols).set_index("datetime")

    print(f"\nTraining base window: {X_base.index.min()} -> {X_base.index.max()} ({len(X_base)} rows)")

    models_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\n--- Retraining target: {target} ---")

        y = df.set_index("datetime")[target].copy()
        X = X_base.copy()

        for lag_h in lag_hours:
            X[f"{target}_lag{lag_h}"] = y.shift(lag_h * 4)

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_train = X.loc[valid_mask]
        y_train = y.loc[valid_mask]

        X_train = X_train.loc[first_ts:last_ts]
        y_train = y_train.loc[first_ts:last_ts]

        if X_train.empty:
            raise ValueError(
                f"No valid training rows remain for target {target} after lag creation. "
                "Check lag settings and dataset length."
            )

        row = df_params.loc[df_params["target"] == target]
        if row.empty:
            raise KeyError(f"No best_params found in {metrics_path} for target {target}.")

        best_params = parse_best_params(row.iloc[0]["best_params"])
        rf = RandomForestRegressor(random_state=random_state, **best_params)
        rf.fit(X_train, y_train)

        out_path = models_dir / f"RF_{target}.joblib"
        dump(rf, out_path)
        print(f"Saved model: {out_path} | samples: {len(X_train)}")

    print("\nMonthly retraining complete.")


# ======================================================================
# Main entry point
# ======================================================================


def main() -> None:
    args = parse_arguments()

    if args.final_training_days <= 0:
        raise ValueError("--final-training-days must be positive.")
    if args.initial_history_days <= 0:
        raise ValueError("--initial-history-days must be positive.")
    if args.weather_data_delay_days < 0:
        raise ValueError("--weather-data-delay-days cannot be negative.")

    dataset_path = Path(args.dataset_path)
    metrics_path = Path(args.metrics_path)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    local_tz = ZoneInfo(args.local_timezone)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Monthly retraining configuration ---")
    print(f"Dataset path:             {dataset_path}")
    print(f"Metrics path:             {metrics_path}")
    print(f"Models directory:         {models_dir}")
    print(f"Results directory:        {results_dir}")
    print(f"Local timezone:           {args.local_timezone}")
    print(f"Latitude:                 {args.latitude}")
    print(f"Longitude:                {args.longitude}")
    print(f"Bad-data start:           {args.bad_start}")
    print(f"Bad-data end:             {args.bad_end}")
    print(f"Final training days:      {args.final_training_days}")
    print(f"Initial history days:     {args.initial_history_days}")
    print(f"Weather data delay days:  {args.weather_data_delay_days}")
    print(f"Lag hours:                {args.lag_hours}")
    print(f"Random state:             {args.random_state}")
    print(f"Skip dataset update:      {args.skip_dataset_update}")

    df_all = load_or_create_dataset(dataset_path)

    api = None
    if not args.skip_dataset_update:
        api = login_cloud_optimo()

    df_all = update_dataset_if_needed(
        dataset_path=dataset_path,
        df_all=df_all,
        api=api,
        local_tz=local_tz,
        latitude=args.latitude,
        longitude=args.longitude,
        initial_history_days=args.initial_history_days,
        weather_data_delay_days=args.weather_data_delay_days,
        skip_dataset_update=args.skip_dataset_update,
    )

    df_train, first_ts, last_ts = prepare_training_frame(
        df_all=df_all,
        bad_start=args.bad_start,
        bad_end=args.bad_end,
        final_training_days=args.final_training_days,
    )

    retrain_models(
        df=df_train,
        metrics_path=metrics_path,
        models_dir=models_dir,
        lag_hours=args.lag_hours,
        first_ts=first_ts,
        last_ts=last_ts,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
