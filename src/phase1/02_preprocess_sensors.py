"""
Phase 1, Step 2: Preprocess mainic sensor data (InfluxDB export from Home Assistant).

This script:
1. Parses the InfluxDB annotated CSV format
2. Splits by category into separate files (heating, weather, rooms, energy)
3. Standardizes units (e.g., Fahrenheit to Celsius)
4. Resamples to consistent 15-minute intervals
5. Handles missing values and sensor dropouts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import warnings
from typing import Optional, Generator
import sys

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "phase1"

# Sensor categorization
HEATING_SENSORS = [
    "stiebel_eltron_isg_",  # Heat pump sensors
    "wp_",                   # Legacy heat pump
    "danfoss_",              # TRV thermostats
]

WEATHER_SENSORS = [
    "davis_",                # Davis weather station
]

ROOM_SENSORS = [
    # Primary room sensors (temperature + humidity pairs)
    "atelier_temperature",
    "atelier_humidity",
    "bric_temperature",
    "bric_humidity",
    "dorme_temperature",
    "dorme_humidity",
    "halle_temperature",
    "halle_humidity",
    "office1_temperature",
    "office1_humidity",
    "office2_temperature",
    "office2_humidity",
    "simlab_temperature",
    "simlab_humidity",
    "studio_temperature",
    "studio_humidity",
    "guest_temperature",
    "guest_humidity",
    # Additional room/area sensors
    "temp_cave",
    "temphum_plant_temperature",
    "temphum_woz_temperature",
    "temphum_bano_temperature",
    "motion_gang_2_temperature",
    "motion_halle_n_temperature",
    "outfeeler_temperature",
    # Davis inside (indoor reference)
    "davis_inside_temperature",
    "davis_inside_humidity",
]

ENERGY_SENSORS = [
    "swiss_domotique_plug_",  # Smart plugs
]

# Unit conversions
# Note: Davis weather station sensors are already in Celsius (verified from raw CSV unit field "C°")
# No Fahrenheit conversion needed for davis sensors
FAHRENHEIT_SENSORS = [
    # Empty - all sensors already in Celsius
]

# Cumulative counter sensors that may have spurious spikes
CUMULATIVE_COUNTER_SENSORS = [
    "stiebel_eltron_isg_consumed_heating",
    "stiebel_eltron_isg_produced_heating",
    "stiebel_eltron_isg_consumed_water_heating",
    "stiebel_eltron_isg_produced_water_heating",
    "stiebel_eltron_isg_consumed_heating_total",
    "stiebel_eltron_isg_produced_heating_total",
    "stiebel_eltron_isg_consumed_water_heating_total",
    "stiebel_eltron_isg_produced_water_heating_total",
]

# Maximum plausible increment per reading for cumulative counters (kWh)
# Heat pump max ~15kW, readings every ~1h, so 15 kWh is generous upper bound
MAX_COUNTER_INCREMENT = 50.0


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def categorize_sensor(entity_id: str) -> str:
    """Categorize a sensor into heating, weather, rooms, energy, or other.

    Order matters: more specific matches (exact names) are checked before
    prefix matches to ensure correct categorization.
    """
    # Check rooms first (exact matches take precedence)
    for name in ROOM_SENSORS:
        if entity_id == name or entity_id.startswith(name):
            return "rooms"
    # Then check prefix-based categories
    for prefix in HEATING_SENSORS:
        if entity_id.startswith(prefix):
            return "heating"
    for prefix in WEATHER_SENSORS:
        if entity_id.startswith(prefix):
            return "weather"
    for prefix in ENERGY_SENSORS:
        if entity_id.startswith(prefix):
            return "energy"
    return "other"


def parse_influxdb_csv_chunked(filepath: Path, chunksize: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """Parse InfluxDB annotated CSV in chunks."""
    print(f"Parsing {filepath.name} in chunks of {chunksize} lines...")

    current_chunk = []
    line_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1

            if line.startswith("#") or line.startswith(";result"):
                continue

            if line.startswith(";;"):
                parts = line.strip().split(";")
                if len(parts) >= 10:
                    try:
                        record = {
                            "table": parts[2],
                            "_start": parts[3],
                            "_stop": parts[4],
                            "_time": parts[5],
                            "_value": parts[6],
                            "_field": parts[7],
                            "_measurement": parts[8],
                            "domain": parts[9],
                            "entity_id": parts[10] if len(parts) > 10 else "",
                        }
                        current_chunk.append(record)
                    except Exception:
                        pass

            if len(current_chunk) >= chunksize:
                df = pd.DataFrame(current_chunk)
                current_chunk = []
                yield df

            if line_count % 1000000 == 0:
                print(f"  Processed {line_count:,} lines...")

    if current_chunk:
        yield pd.DataFrame(current_chunk)

    print(f"  Finished parsing {line_count:,} lines")


def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk of sensor data."""
    if df.empty:
        return df

    df = df[df["_field"] == "value"].copy()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["_time"], format="ISO8601", utc=True)
    df["value"] = pd.to_numeric(df["_value"], errors="coerce")
    df["category"] = df["entity_id"].apply(categorize_sensor)
    df = df[["datetime", "entity_id", "value", "category", "domain"]].copy()

    return df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert units to standard (Celsius for temperatures)."""
    df = df.copy()

    for sensor in FAHRENHEIT_SENSORS:
        mask = df["entity_id"] == sensor
        if mask.any():
            df.loc[mask, "value"] = df.loc[mask, "value"].apply(fahrenheit_to_celsius)

    return df


def clean_cumulative_counter_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove spurious spikes from cumulative counter sensors.

    Some sensors (e.g., heat pump energy counters) occasionally report
    spurious values that spike up and then correct back down. This function
    detects and removes these spikes by:
    1. Computing increments between consecutive readings
    2. Identifying readings where the increment exceeds MAX_COUNTER_INCREMENT
    3. Removing those spurious readings

    This preserves the cumulative nature of the counter while removing glitches.
    """
    df = df.copy()
    rows_removed = 0

    for sensor in CUMULATIVE_COUNTER_SENSORS:
        mask = df["entity_id"] == sensor
        if not mask.any():
            continue

        sensor_df = df.loc[mask].sort_values("datetime")
        if len(sensor_df) < 2:
            continue

        # Calculate increments
        values = sensor_df["value"].values
        increments = np.diff(values)

        # Find spurious readings: large positive jumps or negative corrections
        # A spike pattern is: normal -> spike (large +) -> correction (large -)
        spurious_mask = np.zeros(len(sensor_df), dtype=bool)

        for i in range(len(increments)):
            inc = increments[i]
            # Large positive jump (spike up)
            if inc > MAX_COUNTER_INCREMENT:
                spurious_mask[i + 1] = True  # Mark the spike value
            # Large negative jump (correction back down)
            elif inc < -MAX_COUNTER_INCREMENT:
                # The value before this correction was the spike
                if i > 0 and not spurious_mask[i]:
                    spurious_mask[i] = True

        # Remove spurious rows
        if spurious_mask.any():
            indices_to_remove = sensor_df.index[spurious_mask]
            n_removed = len(indices_to_remove)
            df = df.drop(indices_to_remove)
            rows_removed += n_removed
            print(f"    Removed {n_removed} spurious readings from {sensor}")

    if rows_removed > 0:
        print(f"  Total spurious readings removed: {rows_removed}")

    return df


def resample_to_15min(df: pd.DataFrame, entity_id: str) -> pd.DataFrame:
    """Resample a single sensor's data to 15-minute intervals."""
    sensor_df = df[df["entity_id"] == entity_id].copy()

    if sensor_df.empty:
        return pd.DataFrame()

    sensor_df = sensor_df.set_index("datetime")
    sensor_df = sensor_df.sort_index()

    resampled = sensor_df["value"].resample("15min").mean()
    resampled = resampled.ffill(limit=4)

    result = pd.DataFrame({
        "datetime": resampled.index,
        "entity_id": entity_id,
        "value": resampled.values,
    })

    return result


def extract_sensor_list(filepath: Path, sample_size: int = 1000000) -> set:
    """Extract list of unique sensors from the file."""
    sensors = set()
    line_count = 0

    print(f"Scanning for sensor list (first {sample_size:,} lines)...")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count > sample_size:
                break

            if line.startswith(";;"):
                parts = line.strip().split(";")
                if len(parts) > 10:
                    entity_id = parts[10]
                    field = parts[7]
                    if field == "value" and entity_id:
                        sensors.add(entity_id)

    print(f"  Found {len(sensors)} unique sensors in first {line_count:,} lines")
    return sensors


def process_mainic_file(filepath: Path, output_dir: Path,
                        categories: Optional[list] = None) -> dict:
    """Process the mainic file and save by category."""
    if categories is None:
        categories = ["heating", "weather", "rooms", "energy"]

    output_dir.mkdir(parents=True, exist_ok=True)

    category_data = {cat: [] for cat in categories}

    chunk_num = 0
    total_records = 0

    for chunk in parse_influxdb_csv_chunked(filepath):
        chunk_num += 1
        processed = process_chunk(chunk)

        if processed.empty:
            continue

        total_records += len(processed)

        for cat in categories:
            cat_df = processed[processed["category"] == cat]
            if not cat_df.empty:
                category_data[cat].append(cat_df)

        if chunk_num % 10 == 0:
            print(f"  Processed chunk {chunk_num}, total records: {total_records:,}")

    print(f"\nTotal records processed: {total_records:,}")

    results = {}

    for cat, dfs in category_data.items():
        if not dfs:
            print(f"  {cat}: no data")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined = convert_units(combined)
        combined = combined.drop_duplicates(subset=["datetime", "entity_id"])
        combined = combined.sort_values(["entity_id", "datetime"])

        # Clean spurious spikes from cumulative counters (heating category only)
        if cat == "heating":
            combined = clean_cumulative_counter_spikes(combined)

        print(f"  {cat}: {len(combined):,} records, {combined['entity_id'].nunique()} sensors")

        output_path = output_dir / f"sensors_{cat}.parquet"
        combined.to_parquet(output_path, index=False)

        results[cat] = combined

    return results


def create_sensor_summary(output_dir: Path):
    """Create summary of all sensors."""
    summaries = []

    for parquet_file in output_dir.glob("sensors_*.parquet"):
        category = parquet_file.stem.replace("sensors_", "")
        df = pd.read_parquet(parquet_file)

        for entity_id in df["entity_id"].unique():
            sensor_df = df[df["entity_id"] == entity_id]
            summaries.append({
                "category": category,
                "entity_id": entity_id,
                "count": len(sensor_df),
                "min_value": sensor_df["value"].min(),
                "max_value": sensor_df["value"].max(),
                "mean_value": sensor_df["value"].mean(),
                "start_time": sensor_df["datetime"].min(),
                "end_time": sensor_df["datetime"].max(),
            })

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(["category", "entity_id"])
    summary_df.to_csv(output_dir / "sensor_summary.csv", index=False)

    print(f"\nSensor summary saved to {output_dir / 'sensor_summary.csv'}")
    return summary_df


def main():
    """Main preprocessing pipeline for sensor data."""
    mainic_files = list(DATA_DIR.glob("mainic*.csv"))

    if not mainic_files:
        print("ERROR: No mainic*.csv file found in", DATA_DIR)
        sys.exit(1)

    mainic_file = mainic_files[0]
    print(f"Processing: {mainic_file.name}")
    print(f"File size: {mainic_file.stat().st_size / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("PHASE 1, STEP 2: SENSOR DATA PREPROCESSING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = process_mainic_file(
        mainic_file,
        OUTPUT_DIR,
        categories=["heating", "weather", "rooms", "energy"]
    )

    create_sensor_summary(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to {OUTPUT_DIR}:")
    for f in OUTPUT_DIR.glob("sensors_*.parquet"):
        print(f"  - {f.name}")
    print(f"  - sensor_summary.csv")

    return results


if __name__ == "__main__":
    main()
