"""Run day-ahead market price forecasts for PUN and NORD.

This script is intended for the automatic daily forecasting/retraining step of
the DAM price workflow.

Important:
Before running this script for the first time, the historical price datasets
must already be available in the `DAM_PRICES` folder with the following names:

    DAM_PRICES/PUN_dataset_final.csv
    DAM_PRICES/NORD_dataset_final.csv

These files are used as the historical input for the LEAR model. They must
contain at least the following columns:

    italian datetime
    datetime
    price

where `datetime` is the UTC timestamp and `price` is the historical market
price in EUR/MWh.

At each execution, the script:
1. checks the latest market dates available in the PUN and NORD CSV datasets;
2. downloads from GME only the missing market dates needed for the forecast;
3. appends the missing prices to the existing historical CSV datasets;
4. runs the LEAR model to forecast the selected day-ahead prices;
5. saves the forecast files locally;
6. optionally uploads the forecasts to Optimo Cloud.

If the CSV datasets already contain data up to the market day required for the
forecast, the GME API is not called.

The script is designed for repository publication:
- GME and Optimo credentials are read from environment variables.
- Paths, forecast date, calibration window, and upload behavior are configurable.
- No private usernames, passwords, API keys, or secrets are hard-coded.

Typical usage from the external repository folder:
    python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py

Forecast a specific local date instead of tomorrow:
    python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --forecast-date 2026-05-11

Run locally without uploading to Optimo:
    python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --no-upload

Force the script to avoid GME and use the existing CSV datasets only:
    python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --no-upload --skip-gme-update
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import time
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from OptimoApi import OptimoApi
from model_evaluation.lear import LEAR

# Load local environment variables when a private .env file exists.
# The .env file is intentionally ignored by Git.
try:
    from dotenv import load_dotenv

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
            "Create DAM_PRICES/.env from DAM_PRICES/.env.example or export the variable."
        )
    return value



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
    """Read command-line options used to customize DAM price forecasting."""
    parser = argparse.ArgumentParser(
        description="Run day-ahead PUN and NORD price forecasting using LEAR."
    )

    parser.add_argument(
        "--forecast-date",
        default=os.getenv("FORECAST_DATE", "tomorrow"),
        help=(
            "Local date to forecast in the configured timezone. "
            "Use YYYY-MM-DD, today, or tomorrow. Default: tomorrow."
        ),
    )

    parser.add_argument(
        "--local-timezone",
        default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"),
        help="Local timezone used for market-day timestamps. Default: Europe/Rome.",
    )

    parser.add_argument(
        "--gme-base-url",
        default=os.getenv("GME_BASE_URL", "https://api.mercatoelettrico.org/request"),
        help="Base URL of the GME API.",
    )

    parser.add_argument(
        "--price-granularity",
        default=os.getenv("PRICE_GRANULARITY", "PT60"),
        choices=["PT60", "PT15"],
        help="GME price granularity. Default: PT60 for hourly prices.",
    )

    parser.add_argument(
        "--pun-csv-path",
        default=os.getenv("PUN_CSV_PATH", "DAM_PRICES/PUN_dataset_final.csv"),
        help="CSV file used to store historical PUN prices.",
    )

    parser.add_argument(
        "--nord-csv-path",
        default=os.getenv("NORD_CSV_PATH", "DAM_PRICES/NORD_dataset_final.csv"),
        help="CSV file used to store historical NORD prices.",
    )

    parser.add_argument(
        "--forecasts-dir",
        default=os.getenv("FORECASTS_DIR", "DAM_PRICES/forecasts"),
        help="Directory where forecast CSV files are saved.",
    )

    parser.add_argument(
        "--calibration-window-days",
        type=int,
        default=int(os.getenv("CALIBRATION_WINDOW_DAYS", "364")),
        help="Number of historical days used by LEAR recalibration. Default: 364.",
    )

    parser.add_argument(
        "--gme-request-sleep-seconds",
        type=float,
        default=float(os.getenv("GME_REQUEST_SLEEP_SECONDS", "0.2")),
        help="Pause between multiple GME daily requests. Default: 0.2.",
    )

    parser.add_argument(
        "--skip-gme-update",
        action="store_true",
        help=(
            "Skip the GME API update step and run the forecast using the existing "
            "PUN_dataset_final.csv and NORD_dataset_final.csv files."
        ),
    )

    parser.add_argument(
        "--upload",
        dest="upload_to_optimo",
        action="store_true",
        default=parse_bool(os.getenv("UPLOAD_TO_OPTIMO", "true")),
        help="Upload the forecasts to Optimo.",
    )

    parser.add_argument(
        "--no-upload",
        dest="upload_to_optimo",
        action="store_false",
        help="Do not upload forecasts to Optimo; only save local CSV files.",
    )

    parser.add_argument(
        "--pun-upload-variable-id",
        default=os.getenv("OPTIMO_FORECAST_PUN_VARIABLE_ID", ""),
        help="Optimo variable ID for PUN forecast upload.",
    )

    parser.add_argument(
        "--nord-upload-variable-id",
        default=os.getenv("OPTIMO_FORECAST_NORD_VARIABLE_ID", ""),
        help="Optimo variable ID for NORD forecast upload.",
    )

    return parser.parse_args()


# ======================================================================
# Service login helpers
# ======================================================================


def login_cloud_optimo() -> OptimoApi:
    """Create an authenticated Optimo API client from environment variables."""
    return OptimoApi(
        api_key=get_required_env_var("OPTIMO_API_KEY"),
        app_id=get_required_env_var("OPTIMO_APP_ID"),
        app_secret=get_required_env_var("OPTIMO_APP_SECRET"),
    )



def get_gme_auth_token(base_url: str) -> str:
    """Authenticate to the GME API and return a bearer token."""
    url = f"{base_url}/api/v1/Auth"
    payload = {
        "Login": get_required_env_var("GME_USERNAME"),
        "Password": get_required_env_var("GME_PASSWORD"),
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"GME authentication failed: {data.get('Reason', data)}")

    print("GME authentication successful.")
    return data["token"]


# ======================================================================
# GME data download and local dataset update
# ======================================================================


def find_zone_price_records(json_data):
    """Recursively find records containing Zone and Price fields in decoded GME JSON."""
    if isinstance(json_data, list):
        if json_data and isinstance(json_data[0], dict):
            first_keys = set(json_data[0].keys())
            if {"Zone", "Price"}.issubset(first_keys):
                return json_data

        for item in json_data:
            found = find_zone_price_records(item)
            if found is not None:
                return found

    if isinstance(json_data, dict):
        keys = set(json_data.keys())
        if {"Zone", "Price"}.issubset(keys):
            return [json_data]

        for value in json_data.values():
            found = find_zone_price_records(value)
            if found is not None:
                return found

    return None



def request_me_zonal_prices(
    base_url: str,
    token: str,
    date_value: dt.date,
    granularity: str,
) -> pd.DataFrame:
    """Download GME ME_ZonalPrices for one market date."""
    date_str = date_value.strftime("%Y%m%d")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "Platform": "PublicMarketResults",
        "Segment": "MGP",
        "DataName": "ME_ZonalPrices",
        "IntervalStart": date_str,
        "IntervalEnd": date_str,
        "Attributes": {"GranularityType": granularity},
    }

    response = requests.post(
        f"{base_url}/api/v1/RequestData",
        json=payload,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    if "contentResponse" not in data:
        raise RuntimeError(f"GME RequestData failed or returned malformed response: {data}")

    zip_bytes = base64.b64decode(data["contentResponse"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zipped_file:
        json_filename = zipped_file.namelist()[0]
        json_data = json.loads(zipped_file.read(json_filename).decode("utf-8"))

    records = find_zone_price_records(json_data)
    if records is None:
        top_level_info = (
            list(json_data.keys()) if isinstance(json_data, dict) else f"type={type(json_data).__name__}"
        )
        raise RuntimeError(
            f"Could not find Zone/Price records for {date_str}. "
            f"Top-level JSON information: {top_level_info}"
        )

    df = pd.DataFrame(records)
    required_cols = {"Zone", "Price"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"GME response must contain columns {required_cols}. Got: {df.columns.tolist()}")

    return df



def build_local_utc_datetimes(
    base_date: dt.date,
    n_points: int,
    local_tz: ZoneInfo,
    step_minutes: int = 60,
) -> tuple[list[dt.datetime], list[dt.datetime]]:
    """Build local and UTC timestamps for one market day."""
    start_local = dt.datetime.combine(base_date, dt.time(0, 0), tzinfo=local_tz)
    local_datetimes = [start_local + dt.timedelta(minutes=step_minutes * i) for i in range(n_points)]
    utc_datetimes = [timestamp.astimezone(dt.timezone.utc) for timestamp in local_datetimes]
    return local_datetimes, utc_datetimes



def build_daily_zone_dataframe(
    base_date: dt.date,
    prices: pd.Series,
    local_tz: ZoneInfo,
) -> pd.DataFrame:
    """Create a daily price dataframe with local and UTC timestamps."""
    local_datetimes, utc_datetimes = build_local_utc_datetimes(
        base_date=base_date,
        n_points=len(prices),
        local_tz=local_tz,
        step_minutes=60,
    )

    return pd.DataFrame(
        {
            "italian datetime": local_datetimes,
            "datetime": utc_datetimes,
            "price": prices.values.astype(float),
        }
    )



def append_to_csv(csv_path: str | Path, new_data: pd.DataFrame) -> None:
    """Append new data to a CSV and deduplicate by UTC datetime."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data.copy()

    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["datetime"])
    combined = combined.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    combined.to_csv(csv_path, index=False)

    print(f"Saved/updated {csv_path} with {len(combined)} total rows.")



def get_available_market_dates(csv_path: str | Path, local_tz: ZoneInfo) -> set[dt.date]:
    """Return the local market dates already available in a historical price CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return set()

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'datetime' column.")

    timestamps_utc = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    timestamps_local = timestamps_utc.dt.tz_convert(local_tz)

    return set(timestamps_local.dt.date.dropna().unique())



def date_range_inclusive(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    """Return all dates in the inclusive interval [start_date, end_date]."""
    if end_date < start_date:
        return []

    return [
        start_date + dt.timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]



def get_missing_market_dates(
    pun_csv_path: str | Path,
    nord_csv_path: str | Path,
    target_market_date: dt.date,
    local_tz: ZoneInfo,
) -> list[dt.date]:
    """Find market dates missing in either the PUN or NORD dataset.

    The script needs historical prices up to target_market_date, which is
    usually the day before the forecast date.
    """
    pun_dates = get_available_market_dates(pun_csv_path, local_tz)
    nord_dates = get_available_market_dates(nord_csv_path, local_tz)

    available_common_dates = pun_dates.intersection(nord_dates)

    if not available_common_dates:
        print("Warning: no common dates found in PUN and NORD datasets.")
        print("The script will try to download only the required market date.")
        return [target_market_date]

    last_common_date = max(available_common_dates)

    print(f"Latest PUN market date available:  {max(pun_dates) if pun_dates else 'none'}")
    print(f"Latest NORD market date available: {max(nord_dates) if nord_dates else 'none'}")
    print(f"Latest common market date:         {last_common_date}")
    print(f"Market date required for forecast: {target_market_date}")

    if last_common_date >= target_market_date:
        return []

    candidate_dates = date_range_inclusive(
        start_date=last_common_date + dt.timedelta(days=1),
        end_date=target_market_date,
    )

    return [
        date_value
        for date_value in candidate_dates
        if date_value not in pun_dates or date_value not in nord_dates
    ]



def download_and_append_one_market_date(
    base_url: str,
    token: str,
    market_date: dt.date,
    granularity: str,
    pun_csv_path: str | Path,
    nord_csv_path: str | Path,
    local_tz: ZoneInfo,
) -> None:
    """Download one market date from GME and append PUN/NORD prices to the CSVs."""
    print(f"Downloading GME zonal prices for {market_date} with granularity {granularity}.")

    df_prices = request_me_zonal_prices(
        base_url=base_url,
        token=token,
        date_value=market_date,
        granularity=granularity,
    )

    pun_prices = (
        df_prices.loc[df_prices["Zone"] == "PUN", "Price"]
        .astype(float)
        .reset_index(drop=True)
    )
    nord_prices = (
        df_prices.loc[df_prices["Zone"] == "NORD", "Price"]
        .astype(float)
        .reset_index(drop=True)
    )

    if pun_prices.empty:
        raise RuntimeError(f"No PUN prices returned for {market_date}.")
    if nord_prices.empty:
        raise RuntimeError(f"No NORD prices returned for {market_date}.")

    print(f"Retrieved {len(pun_prices)} PUN prices and {len(nord_prices)} NORD prices.")

    append_to_csv(pun_csv_path, build_daily_zone_dataframe(market_date, pun_prices, local_tz))
    append_to_csv(nord_csv_path, build_daily_zone_dataframe(market_date, nord_prices, local_tz))



def update_missing_pun_nord_datasets(
    base_url: str,
    target_market_date: dt.date,
    granularity: str,
    pun_csv_path: str | Path,
    nord_csv_path: str | Path,
    local_tz: ZoneInfo,
    sleep_seconds: float = 0.2,
) -> None:
    """Download and append only the missing PUN/NORD market dates."""
    missing_dates = get_missing_market_dates(
        pun_csv_path=pun_csv_path,
        nord_csv_path=nord_csv_path,
        target_market_date=target_market_date,
        local_tz=local_tz,
    )

    if not missing_dates:
        print(
            f"PUN and NORD datasets already contain data up to {target_market_date}. "
            "Skipping GME update."
        )
        return

    print("Missing market dates to download from GME:")
    for date_value in missing_dates:
        print(f"- {date_value}")

    token = get_gme_auth_token(base_url)

    failed_dates = []
    for date_value in missing_dates:
        try:
            download_and_append_one_market_date(
                base_url=base_url,
                token=token,
                market_date=date_value,
                granularity=granularity,
                pun_csv_path=pun_csv_path,
                nord_csv_path=nord_csv_path,
                local_tz=local_tz,
            )
        except Exception as exc:
            print(f"Warning: failed to download/process {date_value}: {exc}")
            failed_dates.append(date_value)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if failed_dates:
        failed_as_text = ", ".join(date_value.isoformat() for date_value in failed_dates)
        print(f"Warning: these GME dates could not be updated: {failed_as_text}")
        print("The forecast will continue using the existing historical CSV datasets.")


# ======================================================================
# LEAR forecasting
# ======================================================================


def build_lear_dataframe_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a PUN/NORD CSV and build the dataframe expected by LEAR."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price history CSV not found: {csv_path}. "
            "Before running this script, provide the historical datasets "
            "DAM_PRICES/PUN_dataset_final.csv and DAM_PRICES/NORD_dataset_final.csv."
        )

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns or "price" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns 'datetime' and 'price'.")

    dt_utc = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.assign(dt_utc=dt_utc.dt.tz_localize(None))
    df = df.dropna(subset=["dt_utc"])
    df = df.sort_values("dt_utc").drop_duplicates(subset=["dt_utc"], keep="last")
    df = df.set_index("dt_utc")
    df.index.name = "Date"

    output = pd.DataFrame(index=df.index)
    output["Price"] = pd.to_numeric(df["price"], errors="coerce")
    output = output.dropna(subset=["Price"])

    if output.empty:
        raise ValueError(f"No valid price data found in {csv_path}.")

    return output



def choose_small_flag(df: pd.DataFrame, calibration_window: int) -> bool:
    """Choose whether LEAR should use small mode for limited datasets."""
    n_exogenous_inputs = len(df.columns) - 1
    calibration_days = df.iloc[-calibration_window * 24:].shape[0] / 24
    return calibration_days - 7 - 2 < 96 + 7 + n_exogenous_inputs * 72



def validate_history_coverage(
    df_hist: pd.DataFrame,
    forecast_date_local: dt.date,
    local_tz: ZoneInfo,
    csv_path: str | Path,
) -> None:
    """Warn if the historical dataset does not reach the day before the forecast."""
    required_market_date = forecast_date_local - dt.timedelta(days=1)
    latest_utc = df_hist.index.max().tz_localize("UTC")
    latest_local_date = latest_utc.tz_convert(local_tz).date()

    if latest_local_date < required_market_date:
        print(
            f"Warning: {csv_path} only contains data up to local market date "
            f"{latest_local_date}, but the forecast requires data up to "
            f"{required_market_date}. The LEAR forecast may fail or be less reliable."
        )



def lear_forecast_next_day_from_csv(
    csv_path: str | Path,
    forecast_date_local: dt.date,
    local_tz: ZoneInfo,
    calibration_window: int,
) -> pd.Series:
    """Run LEAR and return a 24-hour local-time forecast series."""
    df_hist = build_lear_dataframe_from_csv(csv_path).sort_index()
    validate_history_coverage(df_hist, forecast_date_local, local_tz, csv_path)

    next_day_start_local = dt.datetime.combine(forecast_date_local, dt.time(0, 0), tzinfo=local_tz)
    next_day_start_utc_naive = next_day_start_local.astimezone(dt.timezone.utc).replace(tzinfo=None)

    future_index_utc_naive = pd.date_range(start=next_day_start_utc_naive, periods=24, freq="h")
    df_future = pd.DataFrame(index=future_index_utc_naive, data={"Price": [np.nan] * 24})

    df_all = pd.concat([df_hist, df_future])
    df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()

    small = choose_small_flag(df_all, calibration_window)
    model = LEAR(calibration_window=calibration_window, small=small)

    y_pred = model.recalibrate_and_forecast_next_day(
        df=df_all,
        calibration_window=calibration_window,
        next_day_date=next_day_start_utc_naive,
        normalization="median",
        transformation="invariant",
    ).reshape(-1)

    future_index_local = pd.date_range(
        start=next_day_start_local.replace(tzinfo=None),
        periods=24,
        freq="h",
    )

    forecast = pd.Series(y_pred, index=future_index_local, name="Price_forecast")
    return forecast.clip(lower=0)



def save_forecast_csv(
    forecast: pd.Series,
    forecasts_dir: str | Path,
    zone_label: str,
    forecast_date: dt.date,
) -> Path:
    """Save a forecast series to CSV."""
    forecasts_dir = Path(forecasts_dir)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    output_path = forecasts_dir / f"{zone_label}_H_{forecast_date.isoformat()}.csv"
    forecast.to_csv(output_path, header=True)
    print(f"Saved {zone_label} forecast to {output_path}")
    return output_path


# ======================================================================
# Optimo upload
# ======================================================================


def upload_price_forecast_to_optimo(
    forecast: pd.Series,
    api: OptimoApi,
    variable_id: str,
    local_tz: ZoneInfo,
    label: str,
) -> None:
    """Upload a day-ahead hourly price forecast to Optimo."""
    if not variable_id:
        raise ValueError(f"Missing Optimo variable ID for {label}.")

    if forecast is None or forecast.empty:
        raise ValueError("Forecast series is empty or None.")

    df_upload = pd.DataFrame({"datetime": forecast.index, label: forecast.values})
    df_upload["datetime"] = pd.to_datetime(df_upload["datetime"])

    if df_upload["datetime"].dt.tz is not None:
        df_upload["datetime"] = df_upload["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        df_upload["datetime"] = (
            df_upload["datetime"]
            .dt.tz_localize(local_tz)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

    df_upload = df_upload.sort_values("datetime")

    samples = []
    missing_count = 0
    for timestamp_utc, value in zip(df_upload["datetime"], df_upload[label]):
        timestamp_ms = int(pd.Timestamp(timestamp_utc).tz_localize("UTC").timestamp() * 1000)

        if value is None or pd.isna(value):
            missing_count += 1
            value = 0.0

        samples.append({"timestamp": timestamp_ms, "value": float(value)})

    payload = [{"variable_id": variable_id, "samples": samples}]
    print(f"Uploading {label} -> {variable_id}: {len(samples)} samples (missing replaced: {missing_count})")

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

    if args.calibration_window_days <= 0:
        raise ValueError("--calibration-window-days must be positive.")

    local_tz = ZoneInfo(args.local_timezone)
    forecast_date = parse_forecast_date(args.forecast_date, local_tz)

    # Day-ahead prices for forecast_date are available from the market clearing
    # on the previous local day.
    market_data_date = forecast_date - dt.timedelta(days=1)

    print("\n--- DAM price forecasting configuration ---")
    print(f"Forecast local date:       {forecast_date}")
    print(f"Market data date required: {market_data_date}")
    print(f"Local timezone:            {args.local_timezone}")
    print(f"PUN CSV path:              {args.pun_csv_path}")
    print(f"NORD CSV path:             {args.nord_csv_path}")
    print(f"Forecasts directory:       {args.forecasts_dir}")
    print(f"Calibration window days:   {args.calibration_window_days}")
    print(f"Skip GME update:           {args.skip_gme_update}")
    print(f"Upload to Optimo:          {args.upload_to_optimo}")

    if args.skip_gme_update:
        print("Skipping GME update. Forecast will use the existing historical CSV datasets.")
    else:
        try:
            update_missing_pun_nord_datasets(
                base_url=args.gme_base_url,
                target_market_date=market_data_date,
                granularity=args.price_granularity,
                pun_csv_path=args.pun_csv_path,
                nord_csv_path=args.nord_csv_path,
                local_tz=local_tz,
                sleep_seconds=args.gme_request_sleep_seconds,
            )
        except requests.exceptions.RequestException as exc:
            print(f"Warning: GME update failed because of a network/API error: {exc}")
            print("Continuing with the existing historical CSV datasets.")
        except Exception as exc:
            print(f"Warning: GME update failed: {exc}")
            print("Continuing with the existing historical CSV datasets.")

    print(f"\nRunning LEAR forecasts for {forecast_date}.")
    pun_forecast = lear_forecast_next_day_from_csv(
        csv_path=args.pun_csv_path,
        forecast_date_local=forecast_date,
        local_tz=local_tz,
        calibration_window=args.calibration_window_days,
    )
    nord_forecast = lear_forecast_next_day_from_csv(
        csv_path=args.nord_csv_path,
        forecast_date_local=forecast_date,
        local_tz=local_tz,
        calibration_window=args.calibration_window_days,
    )

    save_forecast_csv(pun_forecast, args.forecasts_dir, "PUN", forecast_date)
    save_forecast_csv(nord_forecast, args.forecasts_dir, "NORD", forecast_date)

    print("\nPUN forecast preview:")
    print(pun_forecast.head())
    print("\nNORD forecast preview:")
    print(nord_forecast.head())

    if args.upload_to_optimo:
        api = login_cloud_optimo()

        if args.pun_upload_variable_id:
            upload_price_forecast_to_optimo(
                forecast=pun_forecast,
                api=api,
                variable_id=args.pun_upload_variable_id,
                local_tz=local_tz,
                label="Forecast_PUN_price_H",
            )
        else:
            print("Skipping PUN upload: OPTIMO_FORECAST_PUN_VARIABLE_ID is not set.")

        if args.nord_upload_variable_id:
            upload_price_forecast_to_optimo(
                forecast=nord_forecast,
                api=api,
                variable_id=args.nord_upload_variable_id,
                local_tz=local_tz,
                label="Forecast_NORD_price_H",
            )
        else:
            print("Skipping NORD upload: OPTIMO_FORECAST_NORD_VARIABLE_ID is not set.")
    else:
        print("Optimo upload disabled. Forecast files saved locally only.")


if __name__ == "__main__":
    main()
