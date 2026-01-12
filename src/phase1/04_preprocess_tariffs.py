"""
Phase 1, Step 4: Preprocess electricity tariff data.

This script:
1. Loads Primeo Energie tariff data from source JSON
2. Queries ElCom SPARQL endpoint for official tariff data
3. Creates a unified tariff dataset with purchase and feed-in rates
4. Generates time-indexed tariff series for cost modeling

Data sources:
- data/tariffs/primeo_tariffs_source.json (manually collected)
- ElCom LINDAS SPARQL endpoint (official Swiss tariff database)
"""

import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO
import warnings

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARIFF_DIR = DATA_DIR / "tariffs"
OUTPUT_DIR = PROJECT_ROOT / "output" / "phase1"

# ElCom SPARQL endpoint
ELCOM_ENDPOINT = "https://ld.admin.ch/query"
ELCOM_GRAPH = "https://lindas.admin.ch/elcom/electricityprice"

# Primeo Energie operator ID (approximate - may need verification)
PRIMEO_OPERATOR_PATTERN = "Primeo"


def load_source_tariffs() -> dict:
    """Load manually collected tariff data from JSON."""
    source_file = TARIFF_DIR / "primeo_tariffs_source.json"
    with open(source_file, "r", encoding="utf-8") as f:
        return json.load(f)


def query_elcom_sparql(year: int, category: str = "H4") -> pd.DataFrame:
    """
    Query ElCom SPARQL endpoint for tariff data.

    Args:
        year: Tariff year (2023, 2024, 2025)
        category: Consumer category (H1-H8, default H4 = 5-room house, 4500 kWh)

    Returns:
        DataFrame with tariff components
    """
    # SPARQL query for electricity prices by operator
    query = f"""
    PREFIX cube: <https://cube.link/>
    PREFIX schema: <http://schema.org/>
    PREFIX elcom: <https://energy.ld.admin.ch/elcom/electricityprice/dimension/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?operator ?operatorLabel ?municipality ?municipalityLabel
           ?product ?period ?gridusage ?energy ?charge ?total
    WHERE {{
        GRAPH <{ELCOM_GRAPH}> {{
            ?obs a cube:Observation ;
                 elcom:operator ?operator ;
                 elcom:municipality ?municipality ;
                 elcom:category <https://energy.ld.admin.ch/elcom/electricityprice/category/{category}> ;
                 elcom:period ?period ;
                 elcom:product ?product ;
                 elcom:gridusage ?gridusage ;
                 elcom:energy ?energy ;
                 elcom:charge ?charge ;
                 elcom:total ?total .

            ?operator schema:name ?operatorLabel .
            ?municipality schema:name ?municipalityLabel .

            FILTER(YEAR(?period) = {year})
            FILTER(CONTAINS(LCASE(STR(?operatorLabel)), "primeo"))
        }}
    }}
    ORDER BY ?period ?municipality
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Accept": "text/csv"
    }

    try:
        response = requests.post(
            ELCOM_ENDPOINT,
            data={"query": query},
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        if response.text.strip():
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            print(f"  No ElCom data found for year {year}, category {category}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"  ElCom SPARQL query failed: {e}")
        return pd.DataFrame()


def fetch_elcom_tariffs(years: list = None) -> pd.DataFrame:
    """Fetch tariff data from ElCom for multiple years."""
    if years is None:
        years = [2023, 2024, 2025]

    categories = ["H3", "H4", "H5"]  # Different household types
    all_data = []

    print("Fetching ElCom tariff data...")
    for year in years:
        for cat in categories:
            print(f"  Querying year={year}, category={cat}...")
            df = query_elcom_sparql(year, cat)
            if not df.empty:
                df["query_category"] = cat
                df["query_year"] = year
                all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def create_tariff_schedule(source_data: dict) -> pd.DataFrame:
    """
    Create a tariff schedule DataFrame with time-based rates.

    Returns DataFrame with columns:
    - valid_from, valid_to: period validity
    - tariff_type: 'purchase' or 'feedin'
    - rate_type: 'high', 'low', 'single', 'base'
    - rate_rp_kwh: rate in Rappen per kWh
    - notes: additional information
    """
    records = []

    # Purchase tariffs
    for year_key, data in source_data.get("purchase_tariffs", {}).items():
        year = int(year_key.split("_")[0])
        valid_from = data.get("valid_from", f"{year}-01-01")
        valid_to = data.get("valid_to", f"{year}-12-31")

        # NOTE: Single tariff (energy-only component) is not used in this analysis.
        # We use the all-in average estimate which includes network charges, federal
        # levies, and other fees that comprise the actual cost to consumers.
        # The single_tariff_rp_kwh field in the source JSON is preserved for reference
        # but not processed here.

        # High/Low tariffs
        if data.get("high_tariff_rp_kwh"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "purchase",
                "rate_type": "high",
                "rate_rp_kwh": data["high_tariff_rp_kwh"],
                "notes": data.get("notes", "")
            })

        if data.get("low_tariff_rp_kwh"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "purchase",
                "rate_type": "low",
                "rate_rp_kwh": data["low_tariff_rp_kwh"],
                "notes": data.get("notes", "")
            })

        # Estimated average (fallback)
        if data.get("avg_price_rp_kwh_estimate"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "purchase",
                "rate_type": "average_estimate",
                "rate_rp_kwh": data["avg_price_rp_kwh_estimate"],
                "notes": data.get("source", "")
            })

    # Feed-in tariffs
    for period_key, data in source_data.get("feedin_tariffs", {}).items():
        valid_from = data.get("valid_from")
        valid_to = data.get("valid_to")

        # Base rate
        if data.get("base_rate_rp_kwh"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "feedin",
                "rate_type": "base",
                "rate_rp_kwh": data["base_rate_rp_kwh"],
                "notes": data.get("notes", "")
            })

        # Total with HKN (standard)
        if data.get("total_standard_rp_kwh"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "feedin",
                "rate_type": "total_standard",
                "rate_rp_kwh": data["total_standard_rp_kwh"],
                "notes": f"Base + HKN standard. {data.get('notes', '')}"
            })

        # Minimum guarantee
        if data.get("minimum_guarantee_rp_kwh"):
            records.append({
                "valid_from": valid_from,
                "valid_to": valid_to,
                "tariff_type": "feedin",
                "rate_type": "minimum_guarantee",
                "rate_rp_kwh": data["minimum_guarantee_rp_kwh"],
                "notes": f"Guaranteed until {data.get('minimum_guarantee_until', '2028-12-31')}"
            })

    df = pd.DataFrame(records)
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"])
    return df


def create_hourly_tariff_flags(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create hourly flags for high/low tariff periods.

    Based on Primeo Energie tariff times:
    - High tariff: Mon-Fri 06:00-21:00, Sat 06:00-12:00
    - Low tariff: All other times + holidays
    """
    # Swiss holidays (federal holidays when low tariff applies)
    holidays_by_year = {
        2023: [
            "2023-01-01",  # New Year
            "2023-04-07",  # Good Friday
            "2023-04-08",  # Easter Saturday
            "2023-04-10",  # Easter Monday
            "2023-05-18",  # Ascension
            "2023-05-29",  # Whit Monday
            "2023-08-01",  # Swiss National Day
            "2023-12-25",  # Christmas
        ],
        2024: [
            "2024-01-01",
            "2024-03-29",  # Good Friday
            "2024-03-30",  # Easter Saturday
            "2024-04-01",  # Easter Monday
            "2024-05-09",  # Ascension
            "2024-05-20",  # Whit Monday
            "2024-08-01",
            "2024-12-25",
        ],
        2025: [
            "2025-01-01",
            "2025-04-18",  # Good Friday
            "2025-04-19",  # Easter Saturday
            "2025-04-21",  # Easter Monday
            "2025-05-29",  # Ascension
            "2025-06-09",  # Whit Monday
            "2025-08-01",
            "2025-12-25",
        ],
        2026: [
            "2026-01-01",
            "2026-04-03",  # Good Friday
            "2026-04-04",  # Easter Saturday
            "2026-04-06",  # Easter Monday
            "2026-05-14",  # Ascension
            "2026-05-25",  # Whit Monday
            "2026-08-01",
            "2026-12-25",
        ],
    }

    # Flatten holidays
    all_holidays = set()
    for year_holidays in holidays_by_year.values():
        all_holidays.update(pd.to_datetime(year_holidays))

    # Create hourly index
    idx = pd.date_range(start=start_date, end=end_date, freq="h")
    df = pd.DataFrame(index=idx)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df["date"] = df.index.date
    df["is_holiday"] = df.index.normalize().isin(all_holidays)

    # Determine tariff type
    # High tariff: Mon-Fri 06:00-20:59, Sat 06:00-11:59 (and not holiday)
    is_weekday = df["dayofweek"] < 5  # Mon-Fri
    is_saturday = df["dayofweek"] == 5
    is_ht_hours_weekday = (df["hour"] >= 6) & (df["hour"] < 21)
    is_ht_hours_saturday = (df["hour"] >= 6) & (df["hour"] < 12)

    df["is_high_tariff"] = (
        ~df["is_holiday"] &
        (
            (is_weekday & is_ht_hours_weekday) |
            (is_saturday & is_ht_hours_saturday)
        )
    )
    df["is_low_tariff"] = ~df["is_high_tariff"]

    # Add season for SolarAktiv tariff
    df["month"] = df.index.month
    df["is_summer"] = (df["month"] >= 4) & (df["month"] <= 9)  # Apr-Sep
    df["is_winter"] = ~df["is_summer"]

    # SolarAktiv low tariff hours (12:00-15:00)
    df["is_solar_aktiv_low"] = (df["hour"] >= 12) & (df["hour"] < 15)

    return df[["is_high_tariff", "is_low_tariff", "is_holiday",
               "is_summer", "is_winter", "is_solar_aktiv_low"]]


def get_tariff_at_time(timestamp: pd.Timestamp, tariff_schedule: pd.DataFrame,
                       tariff_flags: pd.DataFrame, tariff_type: str = "purchase") -> float:
    """
    Get the applicable tariff rate for a specific timestamp.

    Args:
        timestamp: The timestamp to look up
        tariff_schedule: DataFrame from create_tariff_schedule()
        tariff_flags: DataFrame from create_hourly_tariff_flags()
        tariff_type: 'purchase' or 'feedin'

    Returns:
        Rate in Rp/kWh
    """
    # Find applicable tariff period
    mask = (
        (tariff_schedule["tariff_type"] == tariff_type) &
        (tariff_schedule["valid_from"] <= timestamp) &
        (tariff_schedule["valid_to"] >= timestamp)
    )
    applicable = tariff_schedule[mask]

    if applicable.empty:
        return np.nan

    # Get tariff flag for this hour
    hour_key = timestamp.floor("h")
    if hour_key in tariff_flags.index:
        is_high = tariff_flags.loc[hour_key, "is_high_tariff"]
    else:
        is_high = True  # Default to high tariff

    # Find the right rate
    if tariff_type == "purchase":
        # Prefer high/low, fall back to single or average
        if is_high:
            rate_row = applicable[applicable["rate_type"] == "high"]
        else:
            rate_row = applicable[applicable["rate_type"] == "low"]

        if rate_row.empty:
            rate_row = applicable[applicable["rate_type"] == "single"]
        if rate_row.empty:
            rate_row = applicable[applicable["rate_type"] == "average_estimate"]
    else:
        # Feed-in: use base rate
        rate_row = applicable[applicable["rate_type"] == "base"]
        if rate_row.empty:
            rate_row = applicable[applicable["rate_type"] == "total_standard"]

    if not rate_row.empty:
        return rate_row.iloc[0]["rate_rp_kwh"]

    return np.nan


def create_indexed_tariff_series(start_date: str, end_date: str,
                                  tariff_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Create a time-indexed tariff series for the analysis period.

    Returns DataFrame with columns:
    - purchase_rate_rp_kwh: applicable purchase rate (high or low based on time)
    - purchase_rate_high_rp_kwh: high tariff rate for reference
    - purchase_rate_low_rp_kwh: low tariff rate for reference
    - feedin_rate_rp_kwh: applicable feed-in rate
    - is_high_tariff: boolean flag
    """
    # Create hourly flags
    tariff_flags = create_hourly_tariff_flags(start_date, end_date)

    # Initialize result DataFrame
    result = pd.DataFrame(index=tariff_flags.index)
    result["is_high_tariff"] = tariff_flags["is_high_tariff"]
    result["is_low_tariff"] = tariff_flags["is_low_tariff"]
    result["is_summer"] = tariff_flags["is_summer"]

    # Look up rates for each period
    purchase_rates = []
    purchase_rates_high = []
    purchase_rates_low = []
    feedin_rates = []

    for ts in result.index:
        purchase_rates.append(
            get_tariff_at_time(ts, tariff_schedule, tariff_flags, "purchase")
        )
        feedin_rates.append(
            get_tariff_at_time(ts, tariff_schedule, tariff_flags, "feedin")
        )

        # Also look up explicit high/low rates for reference
        mask = (
            (tariff_schedule["tariff_type"] == "purchase") &
            (tariff_schedule["valid_from"] <= ts) &
            (tariff_schedule["valid_to"] >= ts)
        )
        applicable = tariff_schedule[mask]

        high_rate = applicable[applicable["rate_type"] == "high"]
        low_rate = applicable[applicable["rate_type"] == "low"]

        if not high_rate.empty:
            purchase_rates_high.append(high_rate.iloc[0]["rate_rp_kwh"])
        else:
            # Fall back to average estimate
            avg = applicable[applicable["rate_type"] == "average_estimate"]
            purchase_rates_high.append(avg.iloc[0]["rate_rp_kwh"] * 1.10 if not avg.empty else np.nan)

        if not low_rate.empty:
            purchase_rates_low.append(low_rate.iloc[0]["rate_rp_kwh"])
        else:
            # Fall back to average estimate
            avg = applicable[applicable["rate_type"] == "average_estimate"]
            purchase_rates_low.append(avg.iloc[0]["rate_rp_kwh"] * 0.85 if not avg.empty else np.nan)

    result["purchase_rate_rp_kwh"] = purchase_rates
    result["purchase_rate_high_rp_kwh"] = purchase_rates_high
    result["purchase_rate_low_rp_kwh"] = purchase_rates_low
    result["feedin_rate_rp_kwh"] = feedin_rates

    # Convert to CHF/kWh for convenience
    result["purchase_rate_chf_kwh"] = result["purchase_rate_rp_kwh"] / 100
    result["purchase_rate_high_chf_kwh"] = result["purchase_rate_high_rp_kwh"] / 100
    result["purchase_rate_low_chf_kwh"] = result["purchase_rate_low_rp_kwh"] / 100
    result["feedin_rate_chf_kwh"] = result["feedin_rate_rp_kwh"] / 100

    return result


def generate_report_section(tariff_schedule: pd.DataFrame,
                            elcom_data: pd.DataFrame) -> str:
    """Generate HTML report section for tariffs."""
    html = """
    <h2>Electricity Tariff Data</h2>

    <h3>Data Sources</h3>
    <ul>
        <li>Primeo Energie official announcements and price sheets</li>
        <li>ElCom LINDAS SPARQL endpoint (official Swiss tariff database)</li>
    </ul>

    <h3>Tariff Time Windows</h3>
    <table border="1" cellpadding="5">
        <tr><th>Tariff</th><th>Time Windows</th></tr>
        <tr>
            <td>High Tariff (Hochtarif)</td>
            <td>Mon-Fri 06:00-21:00, Sat 06:00-12:00</td>
        </tr>
        <tr>
            <td>Low Tariff (Niedertarif)</td>
            <td>Mon-Fri 21:00-06:00, Sat 12:00 - Mon 06:00, Federal holidays</td>
        </tr>
    </table>

    <h3>Purchase Tariffs</h3>
    """

    purchase = tariff_schedule[tariff_schedule["tariff_type"] == "purchase"]
    if not purchase.empty:
        html += purchase.to_html(index=False, classes="tariff-table")

    html += """
    <h3>Feed-in Tariffs (Rückliefervergütung)</h3>
    """

    feedin = tariff_schedule[tariff_schedule["tariff_type"] == "feedin"]
    if not feedin.empty:
        html += feedin.to_html(index=False, classes="tariff-table")

    if not elcom_data.empty:
        html += """
        <h3>ElCom Official Data</h3>
        <p>Data retrieved from ElCom LINDAS SPARQL endpoint:</p>
        """
        html += elcom_data.head(20).to_html(index=False, classes="elcom-table")

    html += """
    <h3>Key Observations</h3>
    <ul>
        <li>Feed-in tariffs dropped significantly from 20 Rp/kWh (Jan 2023) to 10.5 Rp/kWh (Apr 2025)</li>
        <li>Minimum guarantee of 9 Rp/kWh applies through 2028 for existing installations</li>
        <li>Purchase tariffs peaked in 2023-2024, now stabilizing with ~1% decrease in 2025</li>
        <li>High/low tariff differential incentivizes off-peak consumption</li>
    </ul>
    """

    return html


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Phase 1, Step 4: Preprocess Electricity Tariffs")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load source data
    print("\n1. Loading source tariff data...")
    source_data = load_source_tariffs()
    print(f"   Loaded tariff data from: {TARIFF_DIR / 'primeo_tariffs_source.json'}")

    # Try to fetch ElCom data
    print("\n2. Fetching ElCom SPARQL data...")
    elcom_data = fetch_elcom_tariffs([2023, 2024, 2025])
    if not elcom_data.empty:
        print(f"   Retrieved {len(elcom_data)} records from ElCom")
        elcom_data.to_csv(OUTPUT_DIR / "elcom_tariffs_raw.csv", index=False)
    else:
        print("   No ElCom data retrieved (network issue or no matching data)")

    # Create tariff schedule
    print("\n3. Creating tariff schedule...")
    tariff_schedule = create_tariff_schedule(source_data)
    print(f"   Created {len(tariff_schedule)} tariff entries")
    tariff_schedule.to_csv(OUTPUT_DIR / "tariff_schedule.csv", index=False)

    # Create hourly tariff flags for full analysis period
    print("\n4. Creating hourly tariff flags...")
    tariff_flags = create_hourly_tariff_flags("2023-01-01", "2026-01-31")
    print(f"   Created {len(tariff_flags)} hourly records")
    tariff_flags.to_parquet(OUTPUT_DIR / "tariff_flags_hourly.parquet")

    # Create indexed tariff series
    print("\n5. Creating indexed tariff series...")
    tariff_series = create_indexed_tariff_series(
        "2023-01-01", "2026-01-31", tariff_schedule
    )
    print(f"   Created tariff series with {len(tariff_series)} hours")
    tariff_series.to_parquet(OUTPUT_DIR / "tariff_series_hourly.parquet")

    # Summary statistics
    print("\n6. Summary Statistics:")
    print(f"   Purchase tariffs: {len(tariff_schedule[tariff_schedule['tariff_type'] == 'purchase'])} entries")
    print(f"   Feed-in tariffs: {len(tariff_schedule[tariff_schedule['tariff_type'] == 'feedin'])} entries")
    print(f"   High tariff hours: {tariff_flags['is_high_tariff'].sum():,}")
    print(f"   Low tariff hours: {tariff_flags['is_low_tariff'].sum():,}")

    # Save report section
    print("\n7. Generating report section...")
    report_html = generate_report_section(tariff_schedule, elcom_data)
    with open(OUTPUT_DIR / "tariff_report_section.html", "w") as f:
        f.write(report_html)

    print("\n" + "=" * 60)
    print("Tariff preprocessing complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - tariff_schedule.csv")
    print("  - tariff_flags_hourly.parquet")
    print("  - tariff_series_hourly.parquet")
    print("  - tariff_report_section.html")
    if not elcom_data.empty:
        print("  - elcom_tariffs_raw.csv")

    return tariff_schedule, tariff_series


if __name__ == "__main__":
    main()
