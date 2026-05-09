"""Build the historical training dataset for electrical load day-ahead forecasting.

This script is designed for repository publication:
- Credentials are read from environment variables.
- Dates, output path, and coordinates are configurable.
- No private API keys or passwords are hard-coded in the source code.

Example usage:
    python 0_creation_dataset.py --start-date 2024-05-30 --end-date 2025-09-30
    python 0_creation_dataset.py --start-date 2024-05-30 --end-date today
    python 0_creation_dataset.py --start-date 2024-05-30 --end-date 2025-09-30 --output-path data/dataset.xlsx

Environment variables can also be used:
    DATASET_START_DATE=2024-05-30
    DATASET_END_DATE=today
    DATASET_PATH=dataset.xlsx
    LATITUDE=45.4643
    LONGITUDE=9.1895
    OPTIMO_API_KEY=...
    OPTIMO_APP_ID=...
    OPTIMO_APP_SECRET=...
"""

import argparse
import datetime as dt
import os
from pathlib import Path

import holidays
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from OptimoApi import OptimoApi
from retry_requests import retry

# Load local environment variables when a private .env file exists.
# The .env file is intentionally ignored by Git.
try:
    from dotenv import load_dotenv

    # Load the .env file located in the same folder as this script.
    # Example:
    # EMS_DayAheadForecasting/EL_LOAD/.env
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv is optional; environment variables can also be set manually.
    pass


# ======================================================================
# Configuration helpers
# ======================================================================


def get_required_env_var(name: str) -> str:
    """Return an environment variable or raise a clear configuration error.

    Secrets must never be hard-coded in source code. Define them locally in a
    .env file, in your shell, or in the scheduler/CI environment that runs this
    script.
    """
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



def parse_utc_date(value: str, end_of_day: bool = False) -> dt.datetime:
    """Parse a date string and return a timezone-aware UTC datetime.

    Accepted formats:
    - YYYY-MM-DD
    - today

    Args:
        value: Date string provided by the user.
        end_of_day: If True, return 23:59:59. Otherwise return 00:00:00.
    """
    value = value.strip().lower()

    if value == "today":
        date_value = dt.datetime.now(dt.timezone.utc).date()
    else:
        try:
            date_value = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid date '{value}'. Use YYYY-MM-DD or 'today'."
            ) from exc

    time_value = dt.time(23, 59, 59) if end_of_day else dt.time(0, 0, 0)
    return dt.datetime.combine(date_value, time_value, tzinfo=dt.timezone.utc)



def parse_arguments() -> argparse.Namespace:
    """Read command-line options used to customize dataset creation."""
    parser = argparse.ArgumentParser(
        description="Create the historical dataset for day-ahead electrical load forecasting."
    )

    parser.add_argument(
        "--start-date",
        default=os.getenv("DATASET_START_DATE", "2024-05-30"),
        help=(
            "Dataset start date in UTC, format YYYY-MM-DD. "
            "Can also be set with DATASET_START_DATE. "
            "Default: 2024-05-30."
        ),
    )

    parser.add_argument(
        "--end-date",
        default=os.getenv("DATASET_END_DATE", "today"),
        help=(
            "Dataset end date in UTC, format YYYY-MM-DD or 'today'. "
            "Can also be set with DATASET_END_DATE. "
            "Default: today."
        ),
    )

    parser.add_argument(
        "--output-path",
        default=os.getenv("DATASET_PATH", "dataset.xlsx"),
        help=(
            "Output Excel file path. "
            "Can also be set with DATASET_PATH. "
            "Default: dataset.xlsx."
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

    return parser.parse_args()


# ======================================================================
# Time-series helpers
# ======================================================================


def round_to_15_floor(ts: dt.datetime) -> dt.datetime:
    """Round a datetime down to the previous 15-minute boundary."""
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)



def ensure_utc_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Convert an epoch-ms datetime column to timezone-aware UTC and sort it."""
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df.sort_values(col).reset_index(drop=True)


# ======================================================================
# Static Cloud Optimo variable IDs
# ======================================================================

# Cabin transformer power meters.
CABIN_DATA_CLOUD = {
    "C1": ["XVpGIF_wa4Kkr", "XVpGIF_CZZwk7", "XVpGIF_tYHam9"],
    "C2": ["XVpGIF_tdDW1q", "XVpGIF_K8mhUn", "XVpGIF_wrdfdl"],
    "C3": ["XVpGIF_OKhasJ", "XVpGIF_nEf6X9", "XVpGIF_ZUtNN3"],
    "C4": ["XVpGIF_QDVDHh", "XVpGIF_iIvC3b", "XVpGIF_cRAKdQ"],
    "C5": ["XVpGIF_SCXTiJ", "XVpGIF_tSDPAx", "XVpGIF_Vx6kuf"],
    "C6": ["XVpGIF_OKA8wi", "XVpGIF_WEWy3L", "XVpGIF_T20KiD"],
    "C8": ["XVpGIF_HHXDvh", "XVpGIF_beumkE"],  # TR1 missing
}

# Schneider PV meters.
PV_DATA_CLOUD = {
    "C1": ["XVpGIF_PlEgT8"],
    "C2": ["XVpGIF_yzWR0f"],
    "C3": ["XVpGIF_wwBKHP"],
    "C4": ["XVpGIF_TUV9xB"],
    "C5": ["XVpGIF_jvwCW0", "XVpGIF_dYthLo"],
    "C8": ["XVpGIF_mEIsFt", "XVpGIF_CDk2Gv"],
    "C9": ["XVpGIF_bImoB6", "XVpGIF_ot5CPh"],
}

# Fiscal PV meters used as backups where available.
PV_DATA_CLOUD_BACKUP = {
    "C1": ["s9BeRW_ZCe3ai"],
    "C2": ["s9BeRW_IZXLgt"],
    "C3": ["s9BeRW_K8AZky"],
    "C4": ["s9BeRW_HnPtSm"],
    "C5": ["s9BeRW_p0La2G", "s9BeRW_dfvjX9"],
    "C8": ["s9BeRW_qsWvJs", "XVpGIF_CDk2Gv"],
    "C9": ["s9BeRW_IhlsRK", "s9BeRW_OPTh3w"],
}

# Main grid/CHP power variables.
MCB_CHP_IDS = {
    "MCB1_P_kW": "XVpGIF_MVziRK",
    "MCB2_P_kW": "XVpGIF_LF3iDv",
    "CHP_P_kW": "XVpGIF_UjTIU5",
}


# ======================================================================
# Cloud data download functions
# ======================================================================


def fetch_cabin_net_15min(
    api: OptimoApi,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    cabin_id: str,
    identifiers: list[str],
) -> pd.DataFrame:
    """Fetch 15-minute net electrical power for one cabin.

    Each cabin may have multiple transformer meters. The function aligns all
    available series by timestamp, resamples them to 15-minute means, and sums
    them to obtain the cabin net load.
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

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

    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    power_cols = [col for col in df.columns if col.startswith("el_power")]

    df_15min = df.set_index("datetime")[power_cols].resample("15min").mean()
    df_15min["net_kW"] = df_15min[power_cols].sum(axis=1, min_count=1)

    output = df_15min[["net_kW"]].reset_index()
    output.rename(columns={"net_kW": f"net_el_cons_{cabin_id}"}, inplace=True)
    return output[["datetime", f"net_el_cons_{cabin_id}"]]



def fetch_pv_15min_sum(
    api: OptimoApi,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    cabin_id: str,
    ids_main: list[str],
    ids_backup: list[str],
) -> pd.DataFrame:
    """Fetch 15-minute PV power for one cabin.

    The Cloud Optimo PV variables are cumulative energy series. This function:
    1. Downloads main and backup energy meters.
    2. Converts energy differences into power values.
    3. Uses backup values where main values are missing.
    4. Sums all PV systems belonging to the cabin.
    """
    start_dt = round_to_15_floor(start_dt)
    end_dt = round_to_15_floor(end_dt)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

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

        # Remove physically impossible values caused by meter resets or duplicated timestamps.
        df.loc[df["dE_kWh"] < 0, "dE_kWh"] = np.nan
        df.loc[df["dt_h"] <= 0, "dt_h"] = np.nan

        df["pv_kW"] = df["dE_kWh"] / df["dt_h"]
        output = df.set_index("datetime")["pv_kW"].resample("15min").mean().to_frame()
        return output.reset_index()

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



def fetch_1min_resample_15(
    api: OptimoApi,
    identifier: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    label: str,
) -> pd.DataFrame:
    """Fetch a 1-minute Cloud Optimo series and resample it to 15-minute means."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

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


# ======================================================================
# Weather and feature engineering functions
# ======================================================================


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

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
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

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates: {response.Latitude()} deg N, {response.Longitude()} deg E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} s")

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

    # Keep exactly the requested UTC window.
    df_weather = df_weather[
        (df_weather["datetime"] >= start_dt) & (df_weather["datetime"] <= end_dt)
    ].reset_index(drop=True)

    print("--- Weather data downloaded successfully ---")
    return df_weather



def add_calendar_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Add local-time calendar, cyclic, campus-closure, and derived features."""
    df = df_input.copy()
    df["datetime_italy"] = df["datetime"].dt.tz_convert("Europe/Rome")

    df["month"] = df["datetime_italy"].dt.month - 1
    df["day_of_week"] = df["datetime_italy"].dt.weekday
    df["quarter_hour"] = df["datetime_italy"].dt.hour * 4 + df["datetime_italy"].dt.minute // 15

    # Cyclic encodings prevent the model from seeing artificial jumps such as
    # December -> January or 23:45 -> 00:00.
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter_hour"] / 96)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter_hour"] / 96)

    italy_holidays = holidays.IT()

    def is_campus_closed(timestamp_rome: pd.Timestamp) -> int:
        month = timestamp_rome.month
        day = timestamp_rome.day
        hour = timestamp_rome.hour
        weekday = timestamp_rome.weekday()
        week = timestamp_rome.isocalendar().week

        if weekday == 6 or timestamp_rome.date() in italy_holidays:
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

    # Temperature active only when the campus is open.
    df["T_open"] = df["temperature_2m"] * (1 - df["campus_closed"])

    return df


# ======================================================================
# Dataset assembly
# ======================================================================


def build_consumption_dataset(
    api: OptimoApi,
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
) -> pd.DataFrame:
    """Download electrical variables and build gross/net consumption targets."""
    print("\n--- Downloading 1-minute variables: MCB1, MCB2, CHP ---")

    df_mcbchp = None
    for label, identifier in MCB_CHP_IDS.items():
        part = fetch_1min_resample_15(api, identifier, start_datetime, end_datetime, label)
        df_mcbchp = part if df_mcbchp is None else pd.merge(df_mcbchp, part, on="datetime", how="outer")

    df_mcbchp = df_mcbchp.sort_values("datetime").reset_index(drop=True)
    df_mcbchp.interpolate(limit=2, inplace=True)

    print(df_mcbchp.head())
    print("\n--- Number of NaN values per MCB/CHP column ---")
    print(df_mcbchp.isna().sum())

    print("\n--- Downloading net electrical consumption per cabin ---")
    df_cabins_net = None
    for cabin, identifiers in CABIN_DATA_CLOUD.items():
        part = fetch_cabin_net_15min(api, start_datetime, end_datetime, cabin, identifiers)
        df_cabins_net = part if df_cabins_net is None else pd.merge(df_cabins_net, part, on="datetime", how="outer")

    print(df_cabins_net.head())
    print("\n--- Number of NaN values per cabin-net column ---")
    print(df_cabins_net.isna().sum())

    print("\n--- Downloading PV power per cabin ---")
    df_pv = None
    for cabin, identifiers in PV_DATA_CLOUD.items():
        part = fetch_pv_15min_sum(
            api,
            start_datetime,
            end_datetime,
            cabin,
            identifiers,
            PV_DATA_CLOUD_BACKUP.get(cabin, []),
        )
        df_pv = part if df_pv is None else pd.merge(df_pv, part, on="datetime", how="outer")

    print(df_pv.head())
    print("\n--- Number of NaN values per PV column ---")
    print(df_pv.isna().sum())

    print("\n--- Merging electrical datasets ---")
    df_base = df_mcbchp.copy()
    if df_cabins_net is not None:
        df_base = pd.merge(df_base, df_cabins_net, on="datetime", how="outer")
    if df_pv is not None:
        df_base = pd.merge(df_base, df_pv, on="datetime", how="outer")

    print(df_base.head())
    print("\n--- Number of NaN values after electrical merge ---")
    print(df_base.isna().sum())

    print("\n--- Computing C10 from electrical balance ---")
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

    print(df_base[["datetime", "net_el_cons_C10"]].head())

    print("\n--- Computing gross consumption as net consumption + PV ---")
    for cabin in list(CABIN_DATA_CLOUD.keys()) + ["C10"]:
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
    df_final_cons["CONS_TOT_NET_kW"] = df_final_cons[
        ["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"]
    ].sum(axis=1, min_count=1)

    df_final_cons.drop(["MCB1_P_kW", "MCB2_P_kW", "CHP_P_kW"], axis=1, inplace=True)

    print("\n--- 15-minute consumption totals preview ---")
    print(df_final_cons.head())

    return df_final_cons



def build_full_dataset(
    api: OptimoApi,
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Build the final dataset by merging consumption, weather, and calendar features."""
    df_consumption = build_consumption_dataset(api, start_datetime, end_datetime)
    df_weather = fetch_weather_15min(latitude, longitude, start_datetime, end_datetime)

    print("\n--- Merging consumption and weather datasets ---")
    df_merged = pd.merge(
        df_consumption.sort_values("datetime"),
        df_weather.sort_values("datetime"),
        on="datetime",
        how="outer",
    )

    print("\n--- Adding calendar and derived features ---")
    df = add_calendar_features(df_merged)

    # Export Excel with timezone-naive datetimes for compatibility with the
    # training and forecasting scripts.
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime_italy"] = df["datetime_italy"].dt.tz_localize(None)

    return df


# ======================================================================
# Main entry point
# ======================================================================


def main() -> None:
    args = parse_arguments()

    start_datetime = parse_utc_date(args.start_date, end_of_day=False)
    end_datetime = parse_utc_date(args.end_date, end_of_day=True)

    if end_datetime <= start_datetime:
        raise ValueError(
            f"Invalid date range: end datetime {end_datetime} must be after "
            f"start datetime {start_datetime}."
        )

    output_path = Path(args.output_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Dataset configuration ---")
    print(f"Start datetime UTC: {start_datetime}")
    print(f"End datetime UTC:   {end_datetime}")
    print(f"Output path:        {output_path}")
    print(f"Latitude:           {args.latitude}")
    print(f"Longitude:          {args.longitude}")

    api = login_cloud_optimo()

    # Lightweight check that credentials and permissions are valid.
    try:
        api.get_latest_value("XVpGIF_beumkE")
        print("\nLogin OK: data fetched successfully.")
    except Exception as exc:
        raise RuntimeError(f"Login or permission failed: {exc}") from exc

    df = build_full_dataset(
        api=api,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    df.to_excel(output_path, index=False)
    print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":
    main()
