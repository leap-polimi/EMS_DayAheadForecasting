"""Create historical DAM price datasets for PUN and NORD forecasting.

This script downloads historical day-ahead market prices from the GME API and
creates the CSV datasets required by the automatic LEAR forecasting script.

The script is designed for repository publication:
- GME credentials are read from environment variables.
- Paths and dates are configurable from command-line arguments or DAM_PRICES/.env.
- No private usernames, passwords, or API credentials are hard-coded.

Typical usage from the external repository folder:
    python DAM_PRICES/0_DAM_PRICES_creation_dataset.py --start-date 2024-01-01 --end-date 2025-09-30

Using dates from DAM_PRICES/.env:
    python DAM_PRICES/0_DAM_PRICES_creation_dataset.py

Relevant .env variables:
    GME_USERNAME=your_gme_username_here
    GME_PASSWORD=your_gme_password_here
    GME_BASE_URL=https://api.mercatoelettrico.org/request
    DAM_DATASET_START_DATE=2024-01-01
    DAM_DATASET_END_DATE=2025-09-30
    PUN_CSV_PATH=DAM_PRICES/PUN_dataset_final.csv
    NORD_CSV_PATH=DAM_PRICES/NORD_dataset_final.csv
    LOCAL_TIMEZONE=Europe/Rome
"""

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

import pandas as pd
import requests

# Load local environment variables when a private .env file exists.
# The .env file is intentionally ignored by Git.
try:
    from dotenv import load_dotenv

    # Load the .env file located in the same folder as this script.
    # Example: EMS_DayAheadForecasting/DAM_PRICES/.env
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



def parse_date(value: str) -> dt.date:
    """Parse a date string.

    Accepted formats:
    - YYYY-MM-DD
    - today
    - yesterday
    """
    value = str(value).strip().lower()

    if value == "today":
        return dt.date.today()
    if value == "yesterday":
        return dt.date.today() - dt.timedelta(days=1)

    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD, today, or yesterday."
        ) from exc



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



def parse_arguments() -> argparse.Namespace:
    """Read command-line options used to customize dataset creation."""
    parser = argparse.ArgumentParser(
        description="Create historical PUN and NORD DAM price CSV datasets from the GME API."
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=parse_date(os.getenv("DAM_DATASET_START_DATE", "2024-01-01")),
        help=(
            "Start date to download, format YYYY-MM-DD. "
            "Can also be set with DAM_DATASET_START_DATE. Default: 2024-01-01."
        ),
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=parse_date(os.getenv("DAM_DATASET_END_DATE", "yesterday")),
        help=(
            "End date to download, format YYYY-MM-DD, today, or yesterday. "
            "Can also be set with DAM_DATASET_END_DATE. Default: yesterday."
        ),
    )

    parser.add_argument(
        "--pun-csv-path",
        default=os.getenv("PUN_CSV_PATH", "DAM_PRICES/PUN_dataset_final.csv"),
        help="Output CSV path for the PUN dataset. Can also be set with PUN_CSV_PATH.",
    )

    parser.add_argument(
        "--nord-csv-path",
        default=os.getenv("NORD_CSV_PATH", "DAM_PRICES/NORD_dataset_final.csv"),
        help="Output CSV path for the NORD dataset. Can also be set with NORD_CSV_PATH.",
    )

    parser.add_argument(
        "--gme-base-url",
        default=os.getenv("GME_BASE_URL", "https://api.mercatoelettrico.org/request"),
        help="Base URL of the GME API. Can also be set with GME_BASE_URL.",
    )

    parser.add_argument(
        "--local-timezone",
        default=os.getenv("LOCAL_TIMEZONE", "Europe/Rome"),
        help="Local timezone used to create readable Italian timestamps. Default: Europe/Rome.",
    )

    parser.add_argument(
        "--granularity",
        default=os.getenv("DAM_PRICE_GRANULARITY", "PT60"),
        choices=["PT60", "PT15"],
        help=(
            "GME price granularity. Use PT60 for hourly datasets used by LEAR. "
            "Default: PT60."
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("GME_REQUEST_SLEEP_SECONDS", "0.2")),
        help="Pause between daily GME requests to avoid overloading the API. Default: 0.2.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=parse_bool(os.getenv("DAM_DATASET_OVERWRITE", "false")),
        help="Overwrite existing CSV files instead of appending/deduplicating.",
    )

    return parser.parse_args()


# ======================================================================
# GME API helpers
# ======================================================================


def get_gme_auth_token(base_url: str) -> str:
    """Authenticate to GME and return a bearer token."""
    username = get_required_env_var("GME_USERNAME")
    password = get_required_env_var("GME_PASSWORD")

    url = f"{base_url}/api/v1/Auth"
    payload = {"Login": username, "Password": password}

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"GME authentication failed: {data.get('Reason')}")

    print("GME authentication successful.")
    return data["token"]



def request_me_zonal_prices(
    token: str,
    base_url: str,
    market_date: dt.date,
    granularity: str,
) -> pd.DataFrame:
    """Request GME ME_ZonalPrices for one date and return the decoded DataFrame."""
    date_str = market_date.strftime("%Y%m%d")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "Platform": "PublicMarketResults",
        "Segment": "MGP",
        "DataName": "ME_ZonalPrices",
        "IntervalStart": date_str,
        "IntervalEnd": date_str,
        "Attributes": {
            "GranularityType": granularity,
        },
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
        raise RuntimeError(f"RequestData failed or malformed response for {date_str}: {data}")

    zip_bytes = base64.b64decode(data["contentResponse"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        json_filename = archive.namelist()[0]
        json_data = json.loads(archive.read(json_filename).decode("utf-8"))

    return pd.DataFrame(json_data)


# ======================================================================
# Dataset assembly helpers
# ======================================================================


def build_local_utc_datetimes(
    base_date: dt.date,
    n_periods: int,
    local_tz: ZoneInfo,
    step_minutes: int = 60,
) -> tuple[list[dt.datetime], list[dt.datetime]]:
    """Build parallel Europe/Rome and UTC timestamps for one market date."""
    start_local = dt.datetime.combine(base_date, dt.time(0, 0), tzinfo=local_tz)

    local_datetimes = [
        start_local + dt.timedelta(minutes=step_minutes * i)
        for i in range(n_periods)
    ]
    utc_datetimes = [timestamp.astimezone(dt.timezone.utc) for timestamp in local_datetimes]

    return local_datetimes, utc_datetimes



def build_daily_zone_dataframe(
    base_date: dt.date,
    prices: pd.Series,
    local_tz: ZoneInfo,
    step_minutes: int = 60,
) -> pd.DataFrame:
    """Create the standard CSV-format DataFrame for one price zone and date."""
    prices = pd.to_numeric(prices, errors="coerce").reset_index(drop=True)
    local_datetimes, utc_datetimes = build_local_utc_datetimes(
        base_date=base_date,
        n_periods=len(prices),
        local_tz=local_tz,
        step_minutes=step_minutes,
    )

    return pd.DataFrame(
        {
            "italian datetime": local_datetimes,
            "datetime": utc_datetimes,
            "price": prices.values.astype(float),
        }
    )



def extract_zone_prices(df_gme: pd.DataFrame, zone: str) -> pd.Series:
    """Extract one zone price series from the raw GME response."""
    required_columns = {"Zone", "Price"}
    missing = required_columns - set(df_gme.columns)
    if missing:
        raise ValueError(f"GME response is missing required columns: {sorted(missing)}")

    zone_df = df_gme[df_gme["Zone"] == zone].copy()
    if zone_df.empty:
        raise ValueError(f"No rows found for zone '{zone}' in the GME response.")

    return pd.to_numeric(zone_df["Price"], errors="coerce").reset_index(drop=True)



def append_or_overwrite_csv(csv_path: Path, new_data: pd.DataFrame, overwrite: bool) -> None:
    """Write a CSV file, optionally appending and deduplicating existing data."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not csv_path.exists():
        output = new_data.copy()
    else:
        existing = pd.read_csv(csv_path)
        output = pd.concat([existing, new_data], ignore_index=True)

    output["datetime"] = pd.to_datetime(output["datetime"], utc=True, errors="coerce")
    output = output.dropna(subset=["datetime"])
    output = output.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

    # Keep ISO-like timestamp strings in the CSV.
    output.to_csv(csv_path, index=False)
    print(f"Saved {len(output)} rows -> {csv_path}")



def date_range_inclusive(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    """Return all dates in the inclusive interval [start_date, end_date]."""
    n_days = (end_date - start_date).days
    return [start_date + dt.timedelta(days=offset) for offset in range(n_days + 1)]


# ======================================================================
# Main pipeline
# ======================================================================


def main() -> None:
    args = parse_arguments()

    if args.end_date < args.start_date:
        raise ValueError(f"End date {args.end_date} must be after start date {args.start_date}.")

    local_tz = ZoneInfo(args.local_timezone)
    pun_csv_path = Path(args.pun_csv_path)
    nord_csv_path = Path(args.nord_csv_path)

    step_minutes = 60 if args.granularity == "PT60" else 15

    print("\n--- DAM price dataset creation configuration ---")
    print(f"Start date:      {args.start_date}")
    print(f"End date:        {args.end_date}")
    print(f"Granularity:     {args.granularity}")
    print(f"PUN CSV path:    {pun_csv_path}")
    print(f"NORD CSV path:   {nord_csv_path}")
    print(f"Local timezone:  {args.local_timezone}")
    print(f"Overwrite:       {args.overwrite}")

    token = get_gme_auth_token(args.gme_base_url)

    pun_parts = []
    nord_parts = []
    failed_dates = []

    all_dates = date_range_inclusive(args.start_date, args.end_date)
    for index, market_date in enumerate(all_dates, start=1):
        print(f"[{index}/{len(all_dates)}] Downloading GME prices for {market_date}...")

        try:
            df_gme = request_me_zonal_prices(
                token=token,
                base_url=args.gme_base_url,
                market_date=market_date,
                granularity=args.granularity,
            )

            pun_prices = extract_zone_prices(df_gme, "PUN")
            nord_prices = extract_zone_prices(df_gme, "NORD")

            if len(pun_prices) != len(nord_prices):
                print(
                    f"  Warning: {market_date} has {len(pun_prices)} PUN points and "
                    f"{len(nord_prices)} NORD points."
                )

            pun_parts.append(
                build_daily_zone_dataframe(
                    base_date=market_date,
                    prices=pun_prices,
                    local_tz=local_tz,
                    step_minutes=step_minutes,
                )
            )
            nord_parts.append(
                build_daily_zone_dataframe(
                    base_date=market_date,
                    prices=nord_prices,
                    local_tz=local_tz,
                    step_minutes=step_minutes,
                )
            )

        except Exception as exc:
            print(f"  Warning: failed to download/process {market_date}: {exc}")
            failed_dates.append(market_date)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not pun_parts or not nord_parts:
        raise RuntimeError("No GME price data were downloaded. Check credentials, dates, and API access.")

    pun_dataset = pd.concat(pun_parts, ignore_index=True)
    nord_dataset = pd.concat(nord_parts, ignore_index=True)

    append_or_overwrite_csv(pun_csv_path, pun_dataset, overwrite=args.overwrite)
    append_or_overwrite_csv(nord_csv_path, nord_dataset, overwrite=args.overwrite)

    print("\nDAM price datasets created successfully.")
    print(f"PUN dataset:  {pun_csv_path}")
    print(f"NORD dataset: {nord_csv_path}")

    if failed_dates:
        failed_as_text = ", ".join(date.isoformat() for date in failed_dates)
        print(f"\nWarning: failed dates were skipped: {failed_as_text}")


if __name__ == "__main__":
    main()
