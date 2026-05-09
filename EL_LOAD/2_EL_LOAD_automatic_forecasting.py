"""Run daily day-ahead forecasts and optionally upload selected outputs to Optimo.

This script is designed for repository publication:
- Credentials are read from environment variables.
- Paths, dates, coordinates, lag hours, and upload behavior are configurable.
- No private API keys or passwords are hard-coded in the source code.

Typical usage from the external repository folder:
    python EL_LOAD/2_EL_LOAD_automatic_forecasting.py

Forecast a specific local date instead of tomorrow:
    python EL_LOAD/2_EL_LOAD_automatic_forecasting.py --forecast-date 2025-10-01

Disable Optimo upload and only save the Excel forecast:
    python EL_LOAD/2_EL_LOAD_automatic_forecasting.py --no-upload

Relevant .env variables:
    OPTIMO_API_KEY=...
    OPTIMO_APP_ID=...
    OPTIMO_APP_SECRET=...
    MODELS_DIR=EL_LOAD/models
    FORECASTS_DIR=EL_LOAD/forecasts
    LATITUDE=45.4643
    LONGITUDE=9.1895
    LAG_HOURS=48,72,96,120,144,168
    HISTORICAL_LOOKBACK_DAYS=8
    MIN_LAG_HOURS=48
    UPLOAD_TO_OPTIMO=true
    OPTIMO_FORECAST_NET_VARIABLE_ID=...
    OPTIMO_FORECAST_GROSS_VARIABLE_ID=...
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



def parse_bool(value: str | bool) -> bool:
    """Parse a boolean value from .env or command-line arguments."""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false, yes/no, or 1/0."
    )



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



def parse_forecast_date(value: str | None, local_tz: ZoneInfo) -> dt.date:
    """Parse the local forecast date.

    Accepted values:
    - tomorrow
    - today
    - YYYY-MM-DD
    """
    if value is None or str(value).strip() == "":
        value = "tomorrow"

    value = str(value).strip().lower()
    now_local = dt.datetime.now(local_tz)

    if value == "tomorrow":
        return (now_local + dt.timedelta(days=1)).date()
    if value == "today":
        return now_local.date()

    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid forecast date '{value}'. Use YYYY-MM-DD, today, or tomorrow."
        ) from exc



def parse_arguments() -> argparse.Namespace:
    """Read command-line options used to customize automatic forecasting."""
    parser = argparse.ArgumentParser(
        description="Run EMS electrical-load day-ahead forecasting."
    )

    parser.add_argument(
        "--forecast-date",
        default=os.getenv("FORECAST_DATE", "tomorrow"),
        help=(
            "Local date to forecast in Europe/Rome. Use YYYY-MM-DD, today, or tomorrow. "
            "Can also be set with FORECAST_DATE. Default: tomorrow."
        ),
    )

    parser.add_argument(
        "--models-dir",
        default=os.getenv("MODELS_DIR", "EL_LOAD/models"),
        help="Directory containing trained RF_*.joblib models. Can also be set with MODELS_DIR.",
    )

    parser.add_argument(
        "--forecasts-dir",
        default=os.getenv("FORECASTS_DIR", "EL_LOAD/forecasts"),
        help="Directory where forecast Excel files are saved. Can also be set with FORECASTS_DIR.",
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
        help="Local timezone used to define the forecast day. Default: Europe/Rome.",
    )

    parser.add_argument(
        "--lag-hours",
        type=parse_lag_hours,
        default=parse_lag_hours(os.getenv("LAG_HOURS", "48,72,96,120,144,168")),
        help="Comma-separated target lag hours. Can also be set with LAG_HOURS.",
    )

    parser.add_argument(
        "--historical-lookback-days",
        type=int,
        default=int(os.getenv("HISTORICAL_LOOKBACK_DAYS", "8")),
        help=(
            "Number of days before the forecast start to fetch for historical lag features. "
            "Can also be set with HISTORICAL_LOOKBACK_DAYS. Default: 8."
        ),
    )

    parser.add_argument(
        "--min-lag-hours",
        type=int,
        default=int(os.getenv("MIN_LAG_HOURS", "48")),
        help=(
            "Most recent historical lag allowed, in hours. Used to avoid using data too close "
            "to the forecast horizon. Can also be set with MIN_LAG_HOURS. Default: 48."
        ),
    )

    parser.add_argument(
        "--upload",
        dest="upload_to_optimo",
        action="store_true",
        default=parse_bool(os.getenv("UPLOAD_TO_OPTIMO", "true")),
        help="Upload selected forecast outputs to Optimo.",
    )

    parser.add_argument(
        "--no-upload",
        dest="upload_to_optimo",
        action="store_false",
        help="Do not upload forecasts to Optimo; only save the Excel file.",
    )

    parser.add_argument(
        "--net-upload-variable-id",
        default=os.getenv("OPTIMO_FORECAST_NET_VARIABLE_ID", ""),
        help="Optimo variable ID for CONS_TOT_NET_kW forecast upload.",
    )

    parser.add_argument(
        "--gross-upload-variable-id",
        default=os.getenv("OPTIMO_FORECAST_GROSS_VARIABLE_ID", ""),
        help="Optimo variable ID for CONS_TOT_kW forecast upload.",
    )

    return parser.parse_args()


# ======================================================================
# Static Cloud Optimo variable IDs
# ======================================================================

CABIN_DATA_CLOUD = {
    "C1": ["XVpGIF_wa4Kkr", "XVpGIF_CZZwk7", "XVpGIF_tYHam9"],
    "C2": ["XVpGIF_tdDW1q", "XVpGIF_K8mhUn", "XVpGIF_wrdfdl"],
    "C3": ["XVpGIF_OKhasJ", "XVpGIF_nEf6X9", "XVpGIF_ZUtNN3"],
    "C4": ["XVpGIF_QDVDHh", "XVpGIF_iIvC3b", "XVpGIF_cRAKdQ"],
    "C5": ["XVpGIF_SCXTiJ", "XVpGIF_tSDPAx", "XVpGIF_Vx6kuf"],
    "C6": ["XVpGIF_OKA8wi", "XVpGIF_WEWy3L", "XVpGIF_T20KiD"],
    "C8": ["XVpGIF_HHXDvh", "XVpGIF_beumkE"],
}

PV_DATA_CLOUD = {
    "C1": ["XVpGIF_PlEgT8"],
    "C2": ["XVpGIF_yzWR0f"],
    "C3": ["XVpGIF_wwBKHP"],
    "C4": ["XVpGIF_TUV9xB"],
    "C5": ["XVpGIF_jvwCW0", "XVpGIF_dYthLo"],
    "C8": ["XVpGIF_mEIsFt", "XVpGIF_CDk2Gv"],
    "C9": ["XVpGIF_bImoB6", "XVpGIF_ot5CPh"],
}

PV_DATA_CLOUD_BACKUP = {
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
# Cloud fetch functions
# ======================================================================


def fetch_cabin_net_15min(
    api: OptimoApi,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    cabin_id: str,
    identifiers: list[str],
) -> pd.DataFrame:
    """Fetch 15-minute net electrical power for one cabin."""
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
            pd.Timestamp(start_dt).tz_convert("UTC").floor("15min"),
            pd.Timestamp(end_dt).tz_convert("UTC").floor("15min"),
            freq="15min",
            inclusive="both",
        )
        return pd.DataFrame({"datetime": idx, f"net_el_cons_{cabin_id}": np.nan})

    df = parts[0]
    for part in parts[1:]:
        df = df.merge(part, on="datetime", how="outer")

    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    power_cols = [col for col in df.columns if col.startswith("el_power")]

    df_15min = df.set_index("datetime")[power_cols].resample("15min").mean()
    df_15min["net_kW"] = df_15min[power_cols].sum(axis=1, min_count=1)

    output = df_15min[["net_kW"]].reset_index()
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
    pv_parts = []

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
            pv_parts.append(merged[["datetime", "pv_kW"]])
        elif power_main is not None:
            pv_parts.append(power_main)
        elif power_backup is not None:
            pv_parts.append(power_backup)

    if not pv_parts:
        idx = pd.date_range(start_dt, end_dt, freq="15min", tz="UTC", inclusive="both")
        return pd.DataFrame({"datetime": idx, f"pv_power_{cabin_id}_kW": np.nan})

    df_pv = pv_parts[0]
    for part in pv_parts[1:]:
        df_pv = pd.merge(df_pv, part, on="datetime", how="outer", suffixes=("", "_dup"))

    pv_cols = [col for col in df_pv.columns if col.startswith("pv_kW")]
    df_pv["pv_kW_sum"] = df_pv[pv_cols].sum(axis=1, min_count=1)

    output = df_pv[["datetime", "pv_kW_sum"]].sort_values("datetime").reset_index(drop=True)
    output.rename(columns={"pv_kW_sum": f"pv_power_{cabin_id}_kW"}, inplace=True)
    return output


# ======================================================================
# Historical target assembly for lag features
# ======================================================================


def build_hist_targets(
    api: OptimoApi,
    start_dt_utc: dt.datetime,
    end_dt_utc: dt.datetime,
) -> pd.DataFrame:
    """Build historical target series needed for lagged forecast features."""
    print("\n--- Building historical target series for lag features ---")
    print(f"Historical start UTC: {start_dt_utc}")
    print(f"Historical end UTC:   {end_dt_utc}")

    df_net = None
    for cabin, identifiers in CABIN_DATA_CLOUD.items():
        part = fetch_cabin_net_15min(api, start_dt_utc, end_dt_utc, cabin, identifiers)
        df_net = part if df_net is None else pd.merge(df_net, part, on="datetime", how="outer")

    df_pv = None
    for cabin, identifiers in PV_DATA_CLOUD.items():
        part = fetch_pv_15min_sum(
            api,
            start_dt_utc,
            end_dt_utc,
            cabin,
            identifiers,
            PV_DATA_CLOUD_BACKUP.get(cabin, []),
        )
        df_pv = part if df_pv is None else pd.merge(df_pv, part, on="datetime", how="outer")

    df_base = df_net.copy()
    if df_pv is not None:
        df_base = pd.merge(df_base, df_pv, on="datetime", how="outer")

    df_mcbchp = None
    for label, identifier in MCB_CHP_IDS.items():
        part = fetch_1min_resample_15(api, identifier, start_dt_utc, end_dt_utc, label)
        df_mcbchp = part if df_mcbchp is None else pd.merge(df_mcbchp, part, on="datetime", how="outer")

    df_all = pd.merge(df_base, df_mcbchp, on="datetime", how="outer")

    cabin_balance_cols = [
        f"net_el_cons_C{i}"
        for i in [1, 2, 3, 4, 5, 6, 8]
        if f"net_el_cons_C{i}" in df_all.columns
    ]
    sum_net_c1_c8 = df_all[cabin_balance_cols].sum(axis=1, min_count=1)
    df_all["net_el_cons_C10"] = (
        df_all[["CHP_P_kW", "MCB1_P_kW", "MCB2_P_kW"]].sum(axis=1, min_count=1)
        - sum_net_c1_c8
    )

    for cabin in list(CABIN_DATA_CLOUD.keys()) + ["C10"]:
        net_col = f"net_el_cons_{cabin}"
        pv_col = f"pv_power_{cabin}_kW"
        gross_col = f"gross_el_cons_{cabin}"

        if net_col not in df_all.columns:
            print(f"  Warning: missing {net_col}; skipping {gross_col}")
            continue

        pv_series = df_all[pv_col] if pv_col in df_all.columns else 0
        df_all[gross_col] = df_all[net_col] + pv_series

    gross_cols = [col for col in df_all.columns if col.startswith("gross_el_cons_")]
    df_hist = df_all[["datetime"] + gross_cols + ["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"]].copy()

    df_hist["CONS_TOT_kW"] = df_hist[gross_cols].sum(axis=1, min_count=5)
    df_hist["CONS_TOT_NET_kW"] = df_hist[["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"]].sum(
        axis=1,
        min_count=2,
    )

    df_hist["datetime"] = pd.to_datetime(df_hist["datetime"], utc=True).dt.tz_localize(None)
    df_hist = df_hist.set_index("datetime").sort_index()
    df_hist = df_hist.drop(columns=["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"])

    return df_hist


# ======================================================================
# Weather forecast and feature engineering
# ======================================================================


def fetch_weather_minutely15_for_day(
    target_date_local: dt.date,
    latitude: float,
    longitude: float,
    local_tz: ZoneInfo,
) -> pd.DataFrame:
    """Return 15-minute weather forecast for the whole target local day."""
    print("\n--- Downloading 15-minute weather forecast from Open-Meteo ---")

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "direct_normal_irradiance",
        ],
        "forecast_days": 2,
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

    start_local = dt.datetime.combine(target_date_local, dt.time(0, 0), tzinfo=local_tz)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(minutes=15)
    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)

    df = df[(df["datetime"] >= start_utc) & (df["datetime"] <= end_utc)].copy()
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()

    if df.empty:
        raise RuntimeError(
            f"No weather forecast returned for local date {target_date_local}. "
            "Check Open-Meteo availability or forecast_days."
        )

    print(f"Weather forecast rows: {len(df)}")
    return df



def add_calendar_features(df_base: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    """Add calendar, cyclic, campus-closure, and derived features."""
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

    df["campus_closed"] = [is_campus_closed(timestamp) for timestamp in dt_local]
    df["T_open"] = df["temperature_2m"] * (1 - df["campus_closed"])

    return df


# ======================================================================
# Lag assembly and prediction
# ======================================================================


def build_target_lag_columns_for_model(
    target: str,
    forecast_index: pd.DatetimeIndex,
    hist_series: pd.Series,
    lag_hours: list[int],
) -> pd.DataFrame:
    """Build lag columns aligned to the forecast index for one target."""
    y = pd.to_numeric(hist_series, errors="coerce").sort_index()

    if y.empty:
        raise ValueError(f"Historical series for target {target} is empty.")

    full_start = y.index.min()
    full_end = forecast_index.max()
    full_index = pd.date_range(full_start, full_end, freq="15min")
    y_full = y.reindex(full_index)

    output = {}
    for lag_h in lag_hours:
        steps = lag_h * 4
        shifted = y_full.shift(steps)
        output[f"{target}_lag{lag_h}"] = shifted.reindex(forecast_index).to_numpy()

    return pd.DataFrame(output, index=forecast_index)



def predict_all_targets(
    models_dir: Path,
    X_base: pd.DataFrame,
    df_hist_targets: pd.DataFrame,
    lag_hours: list[int],
) -> pd.DataFrame:
    """Load all available trained models and produce target forecasts."""
    forecast_index = X_base.index
    predictions = pd.DataFrame(index=forecast_index)
    predictions.index.name = "datetime"

    model_files = sorted(models_dir.glob("RF_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No RF_*.joblib models found in {models_dir}")

    print("\n--- Forecasting targets ---")

    for model_path in model_files:
        target = model_path.stem.replace("RF_", "", 1)
        print(f"Target: {target}")

        if target not in df_hist_targets.columns:
            print(f"  Warning: historical target not available for {target}; skipping.")
            continue

        model = load(model_path)
        if not hasattr(model, "feature_names_in_"):
            raise RuntimeError(
                f"Model {model_path} lacks feature_names_in_. Re-train with pandas DataFrame inputs."
            )

        feature_names = list(model.feature_names_in_)
        base_needed = [col for col in X_base.columns if col in feature_names]
        X_target = X_base[base_needed].copy()

        y_hist = pd.to_numeric(df_hist_targets[target], errors="coerce").sort_index()

        if y_hist.isna().any():
            print(f"  Historical NaNs before repair: {int(y_hist.isna().sum())}")
            full_hist_idx = pd.date_range(y_hist.index.min(), y_hist.index.max(), freq="15min")
            y_hist = y_hist.reindex(full_hist_idx).interpolate(method="time", limit_direction="both")
            print(f"  Historical NaNs after repair:  {int(y_hist.isna().sum())}")

        lag_cols = build_target_lag_columns_for_model(
            target=target,
            forecast_index=forecast_index,
            hist_series=y_hist,
            lag_hours=lag_hours,
        )

        X_target = pd.concat([X_target, lag_cols], axis=1)

        missing = [col for col in feature_names if col not in X_target.columns]
        if missing:
            raise RuntimeError(f"[{target}] Missing required model features: {missing}")

        X_target = X_target.loc[:, feature_names]

        n_nans_before = int(X_target.isna().sum().sum())
        if n_nans_before > 0:
            print(f"  Found {n_nans_before} NaNs in features; interpolating.")
            X_target = X_target.interpolate(limit_direction="both", axis=0)

        if X_target.isna().any().any():
            bad_cols = X_target.columns[X_target.isna().any()].tolist()
            n_bad = int(X_target[bad_cols].isna().sum().sum())
            raise RuntimeError(
                f"[{target}] Found {n_bad} NaNs after interpolation in columns: {bad_cols}. "
                "Check historical data availability and lag settings."
            )

        predictions[target] = model.predict(X_target)

    if predictions.empty:
        raise RuntimeError("No forecasts were produced. Check available models and historical targets.")

    return predictions


# ======================================================================
# Optimo upload functions
# ======================================================================


def upload_forecast_column_to_optimo(
    out_df: pd.DataFrame,
    api: OptimoApi,
    column: str,
    variable_id: str,
) -> None:
    """Upload one forecast column to a Cloud Optimo timeseries variable."""
    if not variable_id:
        raise ValueError(f"Missing Optimo upload variable ID for {column}.")

    if "datetime" not in out_df.columns or column not in out_df.columns:
        raise ValueError(f"out_df must contain 'datetime' and '{column}' columns.")

    df_upload = out_df[["datetime", column]].copy()
    df_upload["datetime"] = pd.to_datetime(df_upload["datetime"])
    df_upload = df_upload.sort_values("datetime")

    samples = []
    missing_count = 0

    for timestamp_utc, value in zip(df_upload["datetime"], df_upload[column]):
        timestamp_ms = int(pd.Timestamp(timestamp_utc).tz_localize("UTC").timestamp() * 1000)

        if value is None or pd.isna(value):
            missing_count += 1
            value = 0.0

        samples.append({"timestamp": timestamp_ms, "value": float(value)})

    payload = [{"variable_id": variable_id, "samples": samples}]

    print(
        f"Uploading {column} -> {variable_id}: {len(samples)} samples "
        f"(missing replaced: {missing_count})"
    )
    response = api.injest_values(payload)

    if response is None:
        print("Ingest returned None: possible network/API issue.")
    elif response == {}:
        print("Cloud Optimo ingest successful.")
    else:
        print(f"Cloud Optimo ingest returned: {response}")


# ======================================================================
# Main entry point
# ======================================================================


def main() -> None:
    args = parse_arguments()
    local_tz = ZoneInfo(args.local_timezone)
    target_date_local = parse_forecast_date(args.forecast_date, local_tz)

    models_dir = Path(args.models_dir)
    forecasts_dir = Path(args.forecasts_dir)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    if args.historical_lookback_days <= 0:
        raise ValueError("--historical-lookback-days must be positive.")
    if args.min_lag_hours <= 0:
        raise ValueError("--min-lag-hours must be positive.")

    print("\n--- Forecast configuration ---")
    print(f"Forecast local date:       {target_date_local}")
    print(f"Local timezone:            {args.local_timezone}")
    print(f"Models directory:          {models_dir}")
    print(f"Forecasts directory:       {forecasts_dir}")
    print(f"Latitude:                  {args.latitude}")
    print(f"Longitude:                 {args.longitude}")
    print(f"Lag hours:                 {args.lag_hours}")
    print(f"Historical lookback days:  {args.historical_lookback_days}")
    print(f"Minimum lag hours:         {args.min_lag_hours}")
    print(f"Upload to Optimo:          {args.upload_to_optimo}")

    # Forecast horizon in local time and UTC.
    start_local = dt.datetime.combine(target_date_local, dt.time(0, 0), tzinfo=local_tz)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(minutes=15)
    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)

    # Weather forecast for the whole local forecast day.
    df_weather = fetch_weather_minutely15_for_day(
        target_date_local=target_date_local,
        latitude=args.latitude,
        longitude=args.longitude,
        local_tz=local_tz,
    )

    X_base = add_calendar_features(df_weather, local_tz=local_tz)

    # Historical window for target lags. The default reproduces the original logic:
    # start = forecast start - 8 days, end = forecast end - 48 hours.
    hist_start = start_utc - dt.timedelta(days=args.historical_lookback_days)
    hist_end = end_utc - dt.timedelta(hours=args.min_lag_hours)

    api = login_cloud_optimo()
    df_hist_targets = build_hist_targets(api, hist_start, hist_end)

    predictions = predict_all_targets(
        models_dir=models_dir,
        X_base=X_base,
        df_hist_targets=df_hist_targets,
        lag_hours=args.lag_hours,
    )

    out = predictions.reset_index()
    out["datetime_italy"] = (
        out["datetime"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(local_tz)
        .dt.tz_localize(None)
    )

    out_file = forecasts_dir / f"forecast_{target_date_local.isoformat()}.xlsx"
    out.to_excel(out_file, index=False)
    print(f"\nSaved day-ahead forecasts for {target_date_local} -> {out_file}")

    if args.upload_to_optimo:
        if args.net_upload_variable_id:
            upload_forecast_column_to_optimo(
                out_df=out,
                api=api,
                column="CONS_TOT_NET_kW",
                variable_id=args.net_upload_variable_id,
            )
        else:
            print("Skipping CONS_TOT_NET_kW upload: OPTIMO_FORECAST_NET_VARIABLE_ID is not set.")

        if args.gross_upload_variable_id:
            upload_forecast_column_to_optimo(
                out_df=out,
                api=api,
                column="CONS_TOT_kW",
                variable_id=args.gross_upload_variable_id,
            )
        else:
            print("Skipping CONS_TOT_kW upload: OPTIMO_FORECAST_GROSS_VARIABLE_ID is not set.")
    else:
        print("Optimo upload disabled. Forecast file saved locally only.")


if __name__ == "__main__":
    main()
