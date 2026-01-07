"""
Integrate energy balance data with sensor data.

This script:
1. Loads preprocessed energy balance and sensor data
2. Merges on timestamp (15-minute intervals)
3. Creates unified dataset for analysis
4. Documents the overlap period
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def load_energy_balance() -> pd.DataFrame:
    """Load preprocessed energy balance data (15-min intervals)."""
    filepath = PROCESSED_DIR / "energy_balance_15min.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"Energy balance data not found: {filepath}")

    df = pd.read_parquet(filepath)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_sensors(category: str) -> pd.DataFrame:
    """Load preprocessed sensor data for a category."""
    filepath = PROCESSED_DIR / f"sensors_{category}.parquet"

    if not filepath.exists():
        return pd.DataFrame()

    df = pd.read_parquet(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def pivot_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sensor data from long to wide format."""
    if df.empty:
        return pd.DataFrame()

    # Resample to 15-min and pivot
    df = df.set_index("datetime")

    # Group by entity_id and resample
    pivoted = df.pivot_table(
        values="value",
        index=df.index,
        columns="entity_id",
        aggfunc="mean"
    )

    # Resample to 15-minute intervals
    pivoted = pivoted.resample("15min").mean()

    return pivoted


def merge_datasets(energy_df: pd.DataFrame,
                   sensor_dfs: dict) -> pd.DataFrame:
    """Merge energy balance with sensor data."""
    # Start with energy data
    merged = energy_df.copy()

    # Add sensor data for each category
    for category, sensor_df in sensor_dfs.items():
        if sensor_df.empty:
            continue

        pivoted = pivot_sensors(sensor_df)

        if pivoted.empty:
            continue

        # Prefix columns with category
        pivoted.columns = [f"{category}_{col}" for col in pivoted.columns]

        # Merge on index
        merged = merged.join(pivoted, how="outer")

    return merged


def analyze_overlap(energy_df: pd.DataFrame,
                    sensor_dfs: dict) -> dict:
    """Analyze the overlap between energy and sensor data."""
    energy_start = energy_df.index.min()
    energy_end = energy_df.index.max()

    overlap_info = {
        "energy_balance": {
            "start": energy_start,
            "end": energy_end,
            "records": len(energy_df),
        }
    }

    for category, sensor_df in sensor_dfs.items():
        if sensor_df.empty:
            continue

        sensor_start = sensor_df["datetime"].min()
        sensor_end = sensor_df["datetime"].max()

        # Calculate overlap
        overlap_start = max(energy_start, sensor_start)
        overlap_end = min(energy_end, sensor_end)

        if overlap_start < overlap_end:
            overlap_days = (overlap_end - overlap_start).days
        else:
            overlap_days = 0

        overlap_info[category] = {
            "start": sensor_start,
            "end": sensor_end,
            "records": len(sensor_df),
            "overlap_start": overlap_start if overlap_days > 0 else None,
            "overlap_end": overlap_end if overlap_days > 0 else None,
            "overlap_days": overlap_days,
        }

    return overlap_info


def main():
    """Main integration pipeline."""
    print("=" * 60)
    print("DATA INTEGRATION")
    print("=" * 60)

    # Load energy balance data
    print("\nLoading energy balance data...")
    try:
        energy_df = load_energy_balance()
        print(f"  Energy balance: {len(energy_df)} records")
        print(f"  Date range: {energy_df.index.min()} to {energy_df.index.max()}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run preprocess_energy_balance.py first")
        return

    # Load sensor data
    print("\nLoading sensor data...")
    sensor_dfs = {}
    for category in ["heating", "weather", "rooms", "energy"]:
        df = load_sensors(category)
        if not df.empty:
            sensor_dfs[category] = df
            print(f"  {category}: {len(df)} records, {df['entity_id'].nunique()} sensors")
        else:
            print(f"  {category}: no data found")

    if not sensor_dfs:
        print("\n  WARNING: No sensor data found. Run preprocess_sensors.py first")

    # Analyze overlap
    print("\nAnalyzing data overlap...")
    overlap = analyze_overlap(energy_df, sensor_dfs)

    for name, info in overlap.items():
        print(f"\n  {name}:")
        print(f"    Range: {info['start']} to {info['end']}")
        print(f"    Records: {info['records']:,}")
        if "overlap_days" in info and info["overlap_days"] > 0:
            print(f"    Overlap with energy: {info['overlap_days']} days")
            print(f"    Overlap period: {info['overlap_start']} to {info['overlap_end']}")

    # Merge datasets
    print("\nMerging datasets...")
    merged = merge_datasets(energy_df, sensor_dfs)
    print(f"  Merged dataset: {len(merged)} records, {len(merged.columns)} columns")

    # Save merged data
    output_path = PROCESSED_DIR / "integrated_dataset.parquet"
    merged.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Save overlap info
    overlap_df = pd.DataFrame(overlap).T
    overlap_path = PROCESSED_DIR / "data_overlap_summary.csv"
    overlap_df.to_csv(overlap_path)
    print(f"Saved overlap summary to: {overlap_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)

    # Identify the overlap period for analysis
    if sensor_dfs:
        all_sensor_starts = [df["datetime"].min() for df in sensor_dfs.values()]
        all_sensor_ends = [df["datetime"].max() for df in sensor_dfs.values()]
        sensor_start = min(all_sensor_starts)
        sensor_end = max(all_sensor_ends)

        overlap_start = max(energy_df.index.min(), sensor_start)
        overlap_end = min(energy_df.index.max(), sensor_end)

        if overlap_start < overlap_end:
            print(f"\nUSABLE OVERLAP PERIOD:")
            print(f"  {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')}")
            print(f"  Duration: {(overlap_end - overlap_start).days} days")

            # Filter to overlap only
            overlap_df = merged.loc[overlap_start:overlap_end]
            overlap_path = PROCESSED_DIR / "integrated_overlap_only.parquet"
            overlap_df.to_parquet(overlap_path)
            print(f"\nSaved overlap-only dataset to: {overlap_path}")
            print(f"  Records: {len(overlap_df)}")
        else:
            print("\nWARNING: No overlap between energy balance and sensor data")

    return merged


if __name__ == "__main__":
    main()
