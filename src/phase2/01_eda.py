#!/usr/bin/env python3
"""
Phase 2, Step 1: Exploratory Data Analysis for Heating Strategy Optimization

Sections:
2.1 Energy Patterns - time-series, seasonal decomposition, heatmaps
2.2 Heating System Analysis - COP, temperature differentials, buffer tank
2.3 Solar-Heating Correlation - overlap analysis, battery utilization
2.4 Summary Statistics - monthly breakdown, heating degree days, peaks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'eda_output'
OUTPUT_DIR.mkdir(exist_ok=True)

# Comfort temperature bounds
COMFORT_MIN = 18.0
COMFORT_MAX = 23.0
BASE_TEMP_HDD = 15.0  # Base temperature for heating degree days (Celsius)


def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sensor data from long format to wide format with datetime index."""
    if df.empty:
        return pd.DataFrame()

    # Ensure datetime column
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')

    # Pivot to wide format
    pivoted = df.pivot_table(
        values='value',
        index=df.index,
        columns='entity_id',
        aggfunc='mean'
    )

    return pivoted


def load_data():
    """Load all processed datasets."""
    print("Loading processed data...")

    data = {}

    # Energy balance - 15-min intervals
    data['energy_15min'] = pd.read_parquet(PROCESSED_DIR / 'energy_balance_15min.parquet')
    data['energy_15min'].index = pd.to_datetime(data['energy_15min'].index)

    # Daily aggregates
    data['energy_daily'] = pd.read_parquet(PROCESSED_DIR / 'energy_balance_daily.parquet')
    data['energy_daily'].index = pd.to_datetime(data['energy_daily'].index)

    # Monthly aggregates
    data['energy_monthly'] = pd.read_parquet(PROCESSED_DIR / 'energy_balance_monthly.parquet')
    data['energy_monthly'].index = pd.to_datetime(data['energy_monthly'].index)

    # Sensor data - load and pivot to wide format
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    data['heating'] = pivot_sensor_data(heating_raw)

    weather_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_weather.parquet')
    data['weather'] = pivot_sensor_data(weather_raw)

    # Room data - use prepared data from wrapper if available, otherwise raw
    rooms_prepared_path = OUTPUT_DIR / 'rooms_prepared.parquet'
    if rooms_prepared_path.exists():
        print("  Using prepared room data (filtered/mapped by wrapper)")
        rooms_raw = pd.read_parquet(rooms_prepared_path)
    else:
        rooms_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_rooms.parquet')
        # Apply office2 → atelier mapping if loading raw data
        rooms_raw = rooms_raw.copy()
        rooms_raw.loc[rooms_raw['entity_id'] == 'office2_temperature', 'entity_id'] = 'atelier_temperature'
        rooms_raw.loc[rooms_raw['entity_id'] == 'office2_humidity', 'entity_id'] = 'atelier_humidity'
    data['rooms'] = pivot_sensor_data(rooms_raw)

    # Integrated dataset (overlap period only)
    data['integrated'] = pd.read_parquet(PROCESSED_DIR / 'integrated_overlap_only.parquet')
    data['integrated'].index = pd.to_datetime(data['integrated'].index)

    for key, df in data.items():
        print(f"  {key}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")

    return data


# =============================================================================
# 2.1 Energy Patterns
# =============================================================================

def analyze_energy_patterns(data):
    """Analyze energy generation and consumption patterns."""
    print("\n" + "="*60)
    print("2.1 ENERGY PATTERNS ANALYSIS")
    print("="*60)

    energy = data['energy_daily'].copy()

    # Add time features
    energy['year'] = energy.index.year
    energy['month'] = energy.index.month
    energy['day_of_week'] = energy.index.dayofweek
    energy['hour'] = 12  # Daily data, set to noon
    energy['season'] = energy['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    # Calculate self-sufficiency ratio
    energy['self_sufficiency'] = (
        energy['direct_consumption_kwh'] /
        energy['total_consumption_kwh'].replace(0, np.nan)
    )

    # Calculate solar utilization (how much PV is self-consumed vs exported)
    energy['pv_self_consumption'] = energy['direct_consumption_kwh'] + energy['battery_charging_kwh']
    energy['solar_utilization'] = (
        energy['pv_self_consumption'] /
        energy['pv_generation_kwh'].replace(0, np.nan)
    )

    # --- Figure 1: Daily Energy Overview Time Series ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # PV Generation
    axes[0].fill_between(energy.index, energy['pv_generation_kwh'], alpha=0.7, color='gold', label='PV Generation')
    axes[0].set_ylabel('kWh/day')
    axes[0].set_title('Daily PV Generation')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Consumption breakdown
    axes[1].fill_between(energy.index, energy['total_consumption_kwh'], alpha=0.7, color='coral', label='Total Consumption')
    axes[1].fill_between(energy.index, energy['direct_consumption_kwh'], alpha=0.7, color='green', label='Direct from PV')
    axes[1].set_ylabel('kWh/day')
    axes[1].set_title('Daily Consumption')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Grid interaction
    axes[2].fill_between(energy.index, energy['grid_feedin_kwh'], alpha=0.7, color='blue', label='Grid Feed-in')
    axes[2].fill_between(energy.index, -energy['external_supply_kwh'], alpha=0.7, color='red', label='Grid Import')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('kWh/day')
    axes[2].set_title('Grid Interaction (positive=export, negative=import)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Battery activity
    axes[3].fill_between(energy.index, energy['battery_charging_kwh'], alpha=0.7, color='purple', label='Battery Charging')
    axes[3].fill_between(energy.index, -energy['battery_discharging_kwh'], alpha=0.7, color='orange', label='Battery Discharging')
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[3].set_ylabel('kWh/day')
    axes[3].set_title('Battery Activity')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig01_daily_energy_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig01_daily_energy_timeseries.png")

    # --- Figure 2: Monthly Aggregated Patterns ---
    monthly = energy.groupby(['year', 'month']).agg({
        'pv_generation_kwh': 'sum',
        'total_consumption_kwh': 'sum',
        'direct_consumption_kwh': 'sum',
        'external_supply_kwh': 'sum',
        'grid_feedin_kwh': 'sum',
        'battery_charging_kwh': 'sum',
        'battery_discharging_kwh': 'sum',
        'self_sufficiency': 'mean'
    }).reset_index()
    monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PV vs Consumption
    ax = axes[0, 0]
    width = 20
    ax.bar(monthly['date'], monthly['pv_generation_kwh'], width=width, alpha=0.7, color='gold', label='PV Generation')
    ax.bar(monthly['date'], monthly['total_consumption_kwh'], width=width, alpha=0.7, color='coral', label='Consumption', bottom=0)
    ax.set_ylabel('kWh/month')
    ax.set_title('Monthly PV Generation vs Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Grid balance
    ax = axes[0, 1]
    net_grid = monthly['grid_feedin_kwh'] - monthly['external_supply_kwh']
    colors = ['green' if x > 0 else 'red' for x in net_grid]
    ax.bar(monthly['date'], net_grid, width=width, alpha=0.7, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('kWh/month')
    ax.set_title('Net Grid Balance (positive=net exporter)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Self-sufficiency by month
    ax = axes[1, 0]
    ax.bar(monthly['date'], monthly['self_sufficiency'] * 100, width=width, alpha=0.7, color='green')
    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Monthly Self-Sufficiency Ratio')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Consumption sources breakdown
    ax = axes[1, 1]
    ax.bar(monthly['date'], monthly['direct_consumption_kwh'], width=width, alpha=0.7,
           color='green', label='Direct PV')
    ax.bar(monthly['date'], monthly['battery_discharging_kwh'], width=width, alpha=0.7,
           color='orange', label='Battery', bottom=monthly['direct_consumption_kwh'])
    ax.bar(monthly['date'], monthly['external_supply_kwh'], width=width, alpha=0.7,
           color='red', label='Grid',
           bottom=monthly['direct_consumption_kwh'] + monthly['battery_discharging_kwh'])
    ax.set_ylabel('kWh/month')
    ax.set_title('Consumption Sources Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig02_monthly_energy_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig02_monthly_energy_patterns.png")

    # --- Figure 3: Hourly Heatmap (using 15-min data) ---
    energy_15min = data['energy_15min'].copy()
    energy_15min['hour'] = energy_15min.index.hour
    energy_15min['day_of_week'] = energy_15min.index.dayofweek
    energy_15min['month'] = energy_15min.index.month

    # Aggregate by hour and day of week
    hourly_dow = energy_15min.groupby(['day_of_week', 'hour']).agg({
        'total_consumption_kwh': 'mean',
        'pv_generation_kwh': 'mean',
        'external_supply_kwh': 'mean'
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Consumption heatmap
    consumption_pivot = hourly_dow['total_consumption_kwh'].unstack(level=0)
    im = axes[0].imshow(consumption_pivot.values, aspect='auto', cmap='YlOrRd', origin='lower')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('Hour of Day')
    axes[0].set_title('Avg Consumption (kWh/15min)')
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0].set_yticks(range(0, 24, 4))
    plt.colorbar(im, ax=axes[0])

    # PV generation heatmap
    pv_pivot = hourly_dow['pv_generation_kwh'].unstack(level=0)
    im = axes[1].imshow(pv_pivot.values, aspect='auto', cmap='YlGn', origin='lower')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Hour of Day')
    axes[1].set_title('Avg PV Generation (kWh/15min)')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1].set_yticks(range(0, 24, 4))
    plt.colorbar(im, ax=axes[1])

    # Grid import heatmap
    grid_pivot = hourly_dow['external_supply_kwh'].unstack(level=0)
    im = axes[2].imshow(grid_pivot.values, aspect='auto', cmap='YlOrBr', origin='lower')
    axes[2].set_xlabel('Day of Week')
    axes[2].set_ylabel('Hour of Day')
    axes[2].set_title('Avg Grid Import (kWh/15min)')
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[2].set_yticks(range(0, 24, 4))
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig03_hourly_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig03_hourly_heatmaps.png")

    # --- Figure 4: Seasonal Patterns ---
    seasonal = energy.groupby('season').agg({
        'pv_generation_kwh': ['mean', 'std'],
        'total_consumption_kwh': ['mean', 'std'],
        'self_sufficiency': ['mean', 'std'],
        'external_supply_kwh': ['mean', 'std']
    })
    seasonal.columns = ['_'.join(col) for col in seasonal.columns]
    seasonal = seasonal.reindex(['Winter', 'Spring', 'Summer', 'Autumn'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = range(4)

    # PV vs Consumption by season
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], seasonal['pv_generation_kwh_mean'], 0.35,
           yerr=seasonal['pv_generation_kwh_std'], label='PV Generation', color='gold', capsize=3)
    ax.bar([i + 0.2 for i in x], seasonal['total_consumption_kwh_mean'], 0.35,
           yerr=seasonal['total_consumption_kwh_std'], label='Consumption', color='coral', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal.index)
    ax.set_ylabel('kWh/day')
    ax.set_title('Seasonal: PV vs Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Self-sufficiency by season
    ax = axes[1]
    ax.bar(x, seasonal['self_sufficiency_mean'] * 100, yerr=seasonal['self_sufficiency_std'] * 100,
           color='green', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal.index)
    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Seasonal Self-Sufficiency')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Grid import by season
    ax = axes[2]
    ax.bar(x, seasonal['external_supply_kwh_mean'], yerr=seasonal['external_supply_kwh_std'],
           color='red', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal.index)
    ax.set_ylabel('kWh/day')
    ax.set_title('Seasonal Grid Import')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig04_seasonal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig04_seasonal_patterns.png")

    # Print summary statistics
    print("\n  Summary Statistics:")
    print(f"    Data period: {energy.index.min().date()} to {energy.index.max().date()}")
    print(f"    Total days: {len(energy)}")
    print(f"\n    Daily averages:")
    print(f"      PV Generation:     {energy['pv_generation_kwh'].mean():.1f} kWh/day")
    print(f"      Total Consumption: {energy['total_consumption_kwh'].mean():.1f} kWh/day")
    print(f"      Grid Import:       {energy['external_supply_kwh'].mean():.1f} kWh/day")
    print(f"      Grid Export:       {energy['grid_feedin_kwh'].mean():.1f} kWh/day")
    print(f"      Self-Sufficiency:  {energy['self_sufficiency'].mean()*100:.1f}%")

    return energy


# =============================================================================
# 2.2 Heating System Analysis
# =============================================================================

def analyze_heating_system(data):
    """Analyze heat pump performance and heating patterns."""
    print("\n" + "="*60)
    print("2.2 HEATING SYSTEM ANALYSIS")
    print("="*60)

    heating = data['heating'].copy()

    # Key heating columns
    hp_cols = {
        'outdoor_temp': 'stiebel_eltron_isg_outdoor_temperature',
        'flow_temp': 'stiebel_eltron_isg_flow_temperature_wp1',
        'return_temp': 'stiebel_eltron_isg_return_temperature_wp1',
        'buffer_actual': 'stiebel_eltron_isg_actual_temperature_buffer',
        'buffer_target': 'stiebel_eltron_isg_target_temperature_buffer',
        'hk1_actual': 'stiebel_eltron_isg_actual_temperature_hk_1',
        'hk1_target': 'stiebel_eltron_isg_target_temperature_hk_1',
        'hk2_actual': 'stiebel_eltron_isg_actual_temperature_hk_2',
        'hk2_target': 'stiebel_eltron_isg_target_temperature_hk_2',
        'water_actual': 'stiebel_eltron_isg_actual_temperature_water',
        'water_target': 'stiebel_eltron_isg_target_temperature_water',
        'consumed_heating': 'stiebel_eltron_isg_consumed_heating',
        'produced_heating': 'stiebel_eltron_isg_produced_heating',
        'consumed_water': 'stiebel_eltron_isg_consumed_water_heating',
        'produced_water': 'stiebel_eltron_isg_produced_water_heating',
        'compressor': 'stiebel_eltron_isg_compressor',
        'is_heating': 'stiebel_eltron_isg_is_heating',
        'hot_gas': 'stiebel_eltron_isg_hot_gas_temperature_wp1',
        'high_pressure': 'stiebel_eltron_isg_high_pressure_wp1',
        'low_pressure': 'stiebel_eltron_isg_low_pressure_wp1',
    }

    # Extract relevant columns
    available_cols = [col for col in hp_cols.values() if col in heating.columns]
    hp_data = heating[available_cols].copy()

    # Rename for easier access
    col_map = {v: k for k, v in hp_cols.items() if v in heating.columns}
    hp_data = hp_data.rename(columns=col_map)

    # --- COP Analysis ---
    # Get daily totals for consumed and produced heating
    # Note: These are cumulative counters, so we need to take the difference
    # We filter out spurious spikes (>100 kWh jumps) that occur due to sensor glitches

    # Initialize daily_heating with default empty DataFrame
    daily_heating = pd.DataFrame()

    if 'consumed_heating' in hp_data.columns and 'produced_heating' in hp_data.columns:
        # Extract non-null values and sort by time (pivoted data is sparse)
        consumed = hp_data['consumed_heating'].dropna().sort_index()
        produced = hp_data['produced_heating'].dropna().sort_index()

        # Calculate daily deltas using OVERLAPPING time periods only
        # This prevents mismatched calculations when one sensor has data gaps
        # (e.g., after spike removal creates gaps in produced but not consumed)

        daily_data = []
        for date in pd.date_range(consumed.index.min().date(), consumed.index.max().date(), freq='D'):
            day_start = pd.Timestamp(date, tz='UTC')
            day_end = day_start + pd.Timedelta(days=1)

            # Get readings for this day
            consumed_day = consumed[(consumed.index >= day_start) & (consumed.index < day_end)]
            produced_day = produced[(produced.index >= day_start) & (produced.index < day_end)]

            if len(consumed_day) < 2 or len(produced_day) < 2:
                continue

            # Find overlapping time window
            overlap_start = max(consumed_day.index.min(), produced_day.index.min())
            overlap_end = min(consumed_day.index.max(), produced_day.index.max())

            # Skip days with less than 20 hours of overlap (data quality filter)
            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            if overlap_hours < 20:
                continue

            # Get first/last values within the overlapping window
            consumed_overlap = consumed_day[(consumed_day.index >= overlap_start) & (consumed_day.index <= overlap_end)]
            produced_overlap = produced_day[(produced_day.index >= overlap_start) & (produced_day.index <= overlap_end)]

            if len(consumed_overlap) >= 2 and len(produced_overlap) >= 2:
                consumed_delta = consumed_overlap.iloc[-1] - consumed_overlap.iloc[0]
                produced_delta = produced_overlap.iloc[-1] - produced_overlap.iloc[0]
                daily_data.append({
                    'date': date,
                    'consumed_delta': consumed_delta,
                    'produced_delta': produced_delta,
                    'overlap_hours': overlap_hours
                })

        daily_heating = pd.DataFrame(daily_data)
        if not daily_heating.empty:
            daily_heating = daily_heating.set_index('date')
            daily_heating.index = pd.to_datetime(daily_heating.index, utc=True)

        # Filter any remaining anomalies (negative values or unrealistic daily totals)
        daily_heating.loc[daily_heating['consumed_delta'] < 0, 'consumed_delta'] = np.nan
        daily_heating.loc[daily_heating['produced_delta'] < 0, 'produced_delta'] = np.nan
        daily_heating.loc[daily_heating['consumed_delta'] > 100, 'consumed_delta'] = np.nan
        daily_heating.loc[daily_heating['produced_delta'] > 500, 'produced_delta'] = np.nan

        # Calculate COP (only where consumed > 0)
        mask = daily_heating['consumed_delta'] > 0
        daily_heating['cop'] = np.nan
        daily_heating.loc[mask, 'cop'] = (
            daily_heating.loc[mask, 'produced_delta'] /
            daily_heating.loc[mask, 'consumed_delta']
        )

        # Filter unrealistic COP values
        daily_heating.loc[daily_heating['cop'] > 10, 'cop'] = np.nan
        daily_heating.loc[daily_heating['cop'] < 1, 'cop'] = np.nan

        # Get outdoor temperature daily average
        if 'outdoor_temp' in hp_data.columns:
            daily_outdoor = hp_data['outdoor_temp'].resample('D').mean()
            daily_heating = daily_heating.join(daily_outdoor.rename('outdoor_temp'))

    # --- Figure 5: Heat Pump COP Analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if not daily_heating.empty and 'cop' in daily_heating.columns:
        # COP time series
        ax = axes[0, 0]
        valid_cop = daily_heating['cop'].dropna()
        ax.plot(valid_cop.index, valid_cop.values, 'o-', markersize=3, alpha=0.7)
        ax.axhline(y=valid_cop.mean(), color='red', linestyle='--', label=f'Mean: {valid_cop.mean():.2f}')
        ax.set_ylabel('COP')
        ax.set_title('Daily Heat Pump COP (Heating)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1, 6)

        # COP vs outdoor temperature
        ax = axes[0, 1]
        if 'outdoor_temp' in daily_heating.columns:
            mask = daily_heating['cop'].notna() & daily_heating['outdoor_temp'].notna()
            ax.scatter(daily_heating.loc[mask, 'outdoor_temp'],
                      daily_heating.loc[mask, 'cop'], alpha=0.5)
            ax.set_xlabel('Outdoor Temperature (C)')
            ax.set_ylabel('COP')
            ax.set_title('COP vs Outdoor Temperature')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(1, 6)

            # Add trendline
            x = daily_heating.loc[mask, 'outdoor_temp'].values
            y = daily_heating.loc[mask, 'cop'].values
            if len(x) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r--', label=f'Trend: {z[0]:.3f}x + {z[1]:.2f}')
                ax.legend()

    # Heating energy consumption by outdoor temp
    ax = axes[1, 0]
    if not daily_heating.empty and 'consumed_delta' in daily_heating.columns and 'outdoor_temp' in daily_heating.columns:
        mask = daily_heating['consumed_delta'].notna() & daily_heating['outdoor_temp'].notna()
        mask &= daily_heating['consumed_delta'] > 0
        ax.scatter(daily_heating.loc[mask, 'outdoor_temp'],
                  daily_heating.loc[mask, 'consumed_delta'], alpha=0.5)
        ax.set_xlabel('Outdoor Temperature (C)')
        ax.set_ylabel('Electricity Consumed (kWh/day)')
        ax.set_title('Heating Electricity vs Outdoor Temperature')
        ax.grid(True, alpha=0.3)

    # Daily heating production
    ax = axes[1, 1]
    if not daily_heating.empty and 'produced_delta' in daily_heating.columns:
        valid = daily_heating['produced_delta'].dropna()
        valid = valid[valid > 0]
        ax.hist(valid, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(x=valid.mean(), color='red', linestyle='--', label=f'Mean: {valid.mean():.1f} kWh')
        ax.set_xlabel('Heat Produced (kWh/day)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Daily Heat Production')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig05_heat_pump_cop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig05_heat_pump_cop.png")

    # --- Figure 6: Temperature Differentials ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Resample to hourly for cleaner visualization
    hourly = hp_data.resample('h').mean()

    # Flow vs Return temperature
    ax = axes[0, 0]
    if 'flow_temp' in hourly.columns and 'return_temp' in hourly.columns:
        ax.plot(hourly.index, hourly['flow_temp'], label='Flow', alpha=0.7)
        ax.plot(hourly.index, hourly['return_temp'], label='Return', alpha=0.7)
        ax.fill_between(hourly.index, hourly['return_temp'], hourly['flow_temp'], alpha=0.3)
        ax.set_ylabel('Temperature (C)')
        ax.set_title('Flow vs Return Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Buffer tank: actual vs target
    ax = axes[0, 1]
    if 'buffer_actual' in hourly.columns and 'buffer_target' in hourly.columns:
        ax.plot(hourly.index, hourly['buffer_actual'], label='Actual', alpha=0.7)
        ax.plot(hourly.index, hourly['buffer_target'], label='Target', alpha=0.7, linestyle='--')
        ax.set_ylabel('Temperature (C)')
        ax.set_title('Buffer Tank Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Heating circuit 1
    ax = axes[1, 0]
    if 'hk1_actual' in hourly.columns and 'hk1_target' in hourly.columns:
        ax.plot(hourly.index, hourly['hk1_actual'], label='Actual', alpha=0.7)
        ax.plot(hourly.index, hourly['hk1_target'], label='Target', alpha=0.7, linestyle='--')
        ax.set_ylabel('Temperature (C)')
        ax.set_title('Heating Circuit 1 (HK1)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Heating circuit 2
    ax = axes[1, 1]
    if 'hk2_actual' in hourly.columns and 'hk2_target' in hourly.columns:
        ax.plot(hourly.index, hourly['hk2_actual'], label='Actual', alpha=0.7)
        ax.plot(hourly.index, hourly['hk2_target'], label='Target', alpha=0.7, linestyle='--')
        ax.set_ylabel('Temperature (C)')
        ax.set_title('Heating Circuit 2 (HK2)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig06_temperature_differentials.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig06_temperature_differentials.png")

    # --- Figure 7: Outdoor vs Indoor Temperature ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Get room temperatures
    rooms = data['rooms'].copy()

    # Find all temperature sensors (exclude humidity)
    temp_sensors = [col for col in rooms.columns if 'temperature' in col.lower()]

    # Color palette for multiple rooms
    colors = plt.cm.tab10(np.linspace(0, 1, len(temp_sensors) + 1))

    ax = axes[0]
    if 'outdoor_temp' in hourly.columns:
        ax.plot(hourly.index, hourly['outdoor_temp'], label='Outdoor', color='blue', alpha=0.8, linewidth=1.5)

    # Plot all room temperature sensors
    for i, sensor in enumerate(sorted(temp_sensors)):
        room_hourly = rooms[sensor].resample('h').mean()
        # Create readable label from sensor name
        label = sensor.replace('_temperature', '').replace('_', ' ').title()
        ax.plot(room_hourly.index, room_hourly.values, label=label, color=colors[i], alpha=0.7, linewidth=1)

    ax.axhline(y=COMFORT_MIN, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=COMFORT_MAX, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Outdoor vs Indoor Temperatures ({len(temp_sensors)} rooms)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(True, alpha=0.3)

    # Temperature difference (indoor - outdoor) for all rooms
    ax = axes[1]
    if 'outdoor_temp' in hourly.columns and len(temp_sensors) > 0:
        outdoor_hourly = hourly['outdoor_temp'].resample('h').mean()

        for i, sensor in enumerate(sorted(temp_sensors)):
            room_hourly = rooms[sensor].resample('h').mean()
            combined = pd.DataFrame({
                'outdoor': outdoor_hourly,
                'indoor': room_hourly
            }).dropna()
            combined['diff'] = combined['indoor'] - combined['outdoor']

            label = sensor.replace('_temperature', '').replace('_', ' ').title()
            ax.plot(combined.index, combined['diff'], label=label, color=colors[i], alpha=0.7, linewidth=1)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Temperature Difference (°C)')
        ax.set_xlabel('Date')
        ax.set_title('Indoor - Outdoor Temperature Difference (all rooms)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig07_indoor_outdoor_temp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig07_indoor_outdoor_temp.png")

    # Print summary statistics
    if not daily_heating.empty and 'cop' in daily_heating.columns:
        valid_cop = daily_heating['cop'].dropna()
        print(f"\n  Heat Pump COP Statistics:")
        print(f"    Mean COP:   {valid_cop.mean():.2f}")
        print(f"    Median COP: {valid_cop.median():.2f}")
        print(f"    Min COP:    {valid_cop.min():.2f}")
        print(f"    Max COP:    {valid_cop.max():.2f}")

    if not daily_heating.empty and 'consumed_delta' in daily_heating.columns:
        consumed = daily_heating['consumed_delta'].dropna()
        consumed = consumed[consumed > 0]
        if len(consumed) > 0:
            print(f"\n  Daily Heating Electricity:")
            print(f"    Mean:   {consumed.mean():.1f} kWh/day")
            print(f"    Median: {consumed.median():.1f} kWh/day")
            print(f"    Max:    {consumed.max():.1f} kWh/day")

    return daily_heating


# =============================================================================
# 2.3 Solar-Heating Correlation
# =============================================================================

def analyze_solar_heating_correlation(data):
    """Analyze the relationship between solar availability and heating demand."""
    print("\n" + "="*60)
    print("2.3 SOLAR-HEATING CORRELATION")
    print("="*60)

    # Use integrated dataset (overlap period)
    integrated = data['integrated'].copy()

    # Key columns
    energy_cols = ['pv_generation_kwh', 'total_consumption_kwh', 'external_supply_kwh',
                   'direct_consumption_kwh', 'battery_discharging_kwh', 'battery_charging_kwh']

    # Check which columns exist
    available_energy = [c for c in energy_cols if c in integrated.columns]

    # Add time features
    integrated['hour'] = integrated.index.hour
    integrated['is_solar_hours'] = (integrated['hour'] >= 8) & (integrated['hour'] <= 17)

    # Check for heating activity indicator
    heating_col = None
    for col in ['stiebel_eltron_isg_is_heating', 'stiebel_eltron_isg_compressor']:
        if col in integrated.columns:
            heating_col = col
            break

    # --- Figure 8: Hourly Patterns - Solar vs Heating ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hourly PV generation profile
    ax = axes[0, 0]
    if 'pv_generation_kwh' in integrated.columns:
        hourly_pv = integrated.groupby('hour')['pv_generation_kwh'].mean()
        ax.bar(hourly_pv.index, hourly_pv.values * 4, color='gold', alpha=0.7)  # *4 to convert to kW approx
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg PV Generation (kWh/h)')
        ax.set_title('Hourly PV Generation Profile')
        ax.grid(True, alpha=0.3)

    # Hourly heating activity
    ax = axes[0, 1]
    if heating_col:
        hourly_heating = integrated.groupby('hour')[heating_col].mean()
        ax.bar(hourly_heating.index, hourly_heating.values * 100, color='coral', alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Heating Active (%)')
        ax.set_title('Hourly Heating Activity Profile')
        ax.grid(True, alpha=0.3)

    # Grid import by hour
    ax = axes[1, 0]
    if 'external_supply_kwh' in integrated.columns:
        hourly_grid = integrated.groupby('hour')['external_supply_kwh'].mean()
        ax.bar(hourly_grid.index, hourly_grid.values * 4, color='red', alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Grid Import (kWh/h)')
        ax.set_title('Hourly Grid Import Profile')
        ax.grid(True, alpha=0.3)

    # Overlap analysis: heating during solar hours vs non-solar hours
    ax = axes[1, 1]
    if heating_col and 'external_supply_kwh' in integrated.columns:
        solar_heating = integrated[integrated['is_solar_hours']].groupby(
            integrated[integrated['is_solar_hours']].index.date
        ).agg({
            heating_col: 'mean',
            'external_supply_kwh': 'sum'
        })

        nonsolar_heating = integrated[~integrated['is_solar_hours']].groupby(
            integrated[~integrated['is_solar_hours']].index.date
        ).agg({
            heating_col: 'mean',
            'external_supply_kwh': 'sum'
        })

        labels = ['Solar Hours\n(8AM-5PM)', 'Non-Solar Hours\n(5PM-8AM)']
        heating_pct = [solar_heating[heating_col].mean() * 100,
                       nonsolar_heating[heating_col].mean() * 100]
        grid_import = [solar_heating['external_supply_kwh'].mean(),
                       nonsolar_heating['external_supply_kwh'].mean()]

        x = np.arange(2)
        width = 0.35

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, heating_pct, width, label='Heating Active %', color='coral', alpha=0.7)
        bars2 = ax2.bar(x + width/2, grid_import, width, label='Grid Import (kWh)', color='red', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Heating Active (%)', color='coral')
        ax2.set_ylabel('Avg Grid Import (kWh)', color='red')
        ax.set_title('Heating vs Grid Import: Solar vs Non-Solar Hours')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig08_solar_heating_hourly.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig08_solar_heating_hourly.png")

    # --- Figure 9: Battery Utilization for Evening/Night Heating ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Hourly battery usage pattern
    ax = axes[0]
    if 'battery_discharging_kwh' in integrated.columns and 'battery_charging_kwh' in integrated.columns:
        hourly_battery = integrated.groupby('hour').agg({
            'battery_charging_kwh': 'mean',
            'battery_discharging_kwh': 'mean'
        })

        ax.bar(hourly_battery.index, hourly_battery['battery_charging_kwh'] * 4,
               color='purple', alpha=0.7, label='Charging')
        ax.bar(hourly_battery.index, -hourly_battery['battery_discharging_kwh'] * 4,
               color='orange', alpha=0.7, label='Discharging')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery Activity (kWh/h)')
        ax.set_title('Hourly Battery Charging/Discharging Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Evening heating from battery vs grid
    ax = axes[1]
    evening_hours = (integrated['hour'] >= 17) | (integrated['hour'] <= 6)
    if 'battery_discharging_kwh' in integrated.columns and 'external_supply_kwh' in integrated.columns:
        evening_data = integrated[evening_hours].resample('D').agg({
            'battery_discharging_kwh': 'sum',
            'external_supply_kwh': 'sum',
            'total_consumption_kwh': 'sum'
        }).dropna()

        # Stack plot
        ax.fill_between(evening_data.index, 0, evening_data['battery_discharging_kwh'],
                        alpha=0.7, color='orange', label='From Battery')
        ax.fill_between(evening_data.index, evening_data['battery_discharging_kwh'],
                        evening_data['battery_discharging_kwh'] + evening_data['external_supply_kwh'],
                        alpha=0.7, color='red', label='From Grid')
        ax.set_ylabel('Energy (kWh)')
        ax.set_xlabel('Date')
        ax.set_title('Evening/Night Energy Sources (5PM-6AM)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig09_battery_evening_heating.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig09_battery_evening_heating.png")

    # --- Figure 10: Forced Grid Consumption Analysis ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Identify periods of forced grid consumption for heating
    # (grid import while heating is active and no/low PV)
    if (heating_col and 'external_supply_kwh' in integrated.columns and
        'pv_generation_kwh' in integrated.columns):

        integrated['forced_grid'] = (
            (integrated[heating_col] > 0.5) &
            (integrated['external_supply_kwh'] > 0.05) &
            (integrated['pv_generation_kwh'] < 0.02)
        )

        # Hourly pattern of forced grid consumption
        ax = axes[0]
        forced_by_hour = integrated.groupby('hour')['forced_grid'].mean() * 100
        colors = ['red' if h < 8 or h >= 17 else 'orange' for h in forced_by_hour.index]
        ax.bar(forced_by_hour.index, forced_by_hour.values, color=colors, alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('% of Time')
        ax.set_title('Forced Grid Consumption While Heating (no PV available)')
        ax.grid(True, alpha=0.3)

        # Daily forced grid consumption
        ax = axes[1]
        daily_forced = integrated.resample('D').agg({
            'forced_grid': 'sum',
            'external_supply_kwh': 'sum'
        })
        # forced_grid is count of 15-min intervals, so divide by 4 for hours
        daily_forced['forced_hours'] = daily_forced['forced_grid'] / 4

        ax.bar(daily_forced.index, daily_forced['forced_hours'], alpha=0.7, color='red')
        ax.set_ylabel('Hours of Forced Grid Heating')
        ax.set_xlabel('Date')
        ax.set_title('Daily Hours of Grid-Dependent Heating')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_forced_grid_heating.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig10_forced_grid_heating.png")

    # Print summary statistics
    if 'pv_generation_kwh' in integrated.columns and heating_col:
        solar_hours = integrated[integrated['is_solar_hours']]
        nonsolar_hours = integrated[~integrated['is_solar_hours']]

        print(f"\n  Solar-Heating Overlap Analysis:")
        print(f"    Solar hours (8AM-5PM):")
        print(f"      Heating active: {solar_hours[heating_col].mean()*100:.1f}% of time")
        print(f"      Avg grid import: {solar_hours['external_supply_kwh'].mean()*4:.2f} kWh/h")
        print(f"    Non-solar hours (5PM-8AM):")
        print(f"      Heating active: {nonsolar_hours[heating_col].mean()*100:.1f}% of time")
        print(f"      Avg grid import: {nonsolar_hours['external_supply_kwh'].mean()*4:.2f} kWh/h")

    if 'forced_grid' in integrated.columns:
        print(f"\n  Forced Grid Consumption:")
        print(f"    % of heating time with forced grid: {integrated['forced_grid'].mean()*100:.1f}%")


# =============================================================================
# 2.4 Summary Statistics
# =============================================================================

def generate_summary_statistics(data, energy_patterns, daily_heating):
    """Generate comprehensive summary statistics."""
    print("\n" + "="*60)
    print("2.4 SUMMARY STATISTICS")
    print("="*60)

    energy = energy_patterns.copy()

    # --- Heating Degree Days Analysis ---
    # Get outdoor temperature from heating sensors
    heating = data['heating']
    outdoor_temp_col = 'stiebel_eltron_isg_outdoor_temperature'

    # Initialize hdd with empty DataFrame
    hdd = pd.DataFrame()

    if outdoor_temp_col in heating.columns:
        daily_temp = heating[outdoor_temp_col].resample('D').mean()

        # Calculate Heating Degree Days (base 15C)
        hdd = daily_temp.apply(lambda x: max(0, BASE_TEMP_HDD - x) if pd.notna(x) else np.nan)
        hdd = hdd.to_frame('hdd')

        # Join with heating consumption
        if not daily_heating.empty and 'consumed_delta' in daily_heating.columns:
            hdd = hdd.join(daily_heating['consumed_delta'].rename('heating_kwh'))
            hdd = hdd.dropna()

    # --- Figure 11: Monthly Breakdown ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Monthly energy breakdown
    ax = axes[0, 0]
    monthly_summary = energy.groupby([energy.index.year, energy.index.month]).agg({
        'pv_generation_kwh': 'sum',
        'total_consumption_kwh': 'sum',
        'external_supply_kwh': 'sum',
        'grid_feedin_kwh': 'sum',
        'self_sufficiency': 'mean'
    })
    monthly_summary.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in monthly_summary.index])

    width = 20
    ax.bar(monthly_summary.index, monthly_summary['pv_generation_kwh'], width=width,
           alpha=0.7, color='gold', label='PV')
    ax.bar(monthly_summary.index, -monthly_summary['total_consumption_kwh'], width=width,
           alpha=0.7, color='coral', label='Consumption')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('kWh/month')
    ax.set_title('Monthly PV Generation vs Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Heating Degree Days vs Consumption
    ax = axes[0, 1]
    if not hdd.empty and 'heating_kwh' in hdd.columns:
        ax.scatter(hdd['hdd'], hdd['heating_kwh'], alpha=0.5)
        ax.set_xlabel(f'Heating Degree Days (base {BASE_TEMP_HDD}C)')
        ax.set_ylabel('Heating Electricity (kWh/day)')
        ax.set_title('Heating Consumption vs Heating Degree Days')
        ax.grid(True, alpha=0.3)

        # Trendline
        mask = hdd['hdd'].notna() & hdd['heating_kwh'].notna()
        if mask.sum() > 5:
            x = hdd.loc[mask, 'hdd'].values
            y = hdd.loc[mask, 'heating_kwh'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', label=f'{z[0]:.2f} kWh/HDD')
            ax.legend()

    # Peak demand analysis
    ax = axes[1, 0]
    daily_peaks = energy['total_consumption_kwh'].copy()
    ax.hist(daily_peaks.dropna(), bins=30, alpha=0.7, color='coral', edgecolor='black')
    p95 = daily_peaks.quantile(0.95)
    ax.axvline(x=p95, color='red', linestyle='--', label=f'95th percentile: {p95:.1f} kWh')
    ax.axvline(x=daily_peaks.mean(), color='blue', linestyle='--', label=f'Mean: {daily_peaks.mean():.1f} kWh')
    ax.set_xlabel('Daily Consumption (kWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Consumption Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Yearly totals
    ax = axes[1, 1]
    yearly = energy.groupby(energy.index.year).agg({
        'pv_generation_kwh': 'sum',
        'total_consumption_kwh': 'sum',
        'external_supply_kwh': 'sum',
        'grid_feedin_kwh': 'sum'
    })

    x = np.arange(len(yearly))
    width = 0.2
    ax.bar(x - 1.5*width, yearly['pv_generation_kwh'], width, label='PV', color='gold')
    ax.bar(x - 0.5*width, yearly['total_consumption_kwh'], width, label='Consumption', color='coral')
    ax.bar(x + 0.5*width, yearly['external_supply_kwh'], width, label='Grid Import', color='red')
    ax.bar(x + 1.5*width, yearly['grid_feedin_kwh'], width, label='Grid Export', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index)
    ax.set_ylabel('kWh/year')
    ax.set_title('Yearly Energy Totals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig11_summary_statistics.png")

    # Seasonal findings for summary
    seasonal = energy.groupby('season').agg({
        'self_sufficiency': 'mean',
        'external_supply_kwh': 'mean'
    })
    best_season = seasonal['self_sufficiency'].idxmax()
    worst_season = seasonal['self_sufficiency'].idxmin()

    # Print summary to console (used by wrapper for stats extraction)
    print(f"""
  Summary:
    Energy period: {energy.index.min().date()} to {energy.index.max().date()} ({len(energy)} days)
    Self-sufficiency: {energy['self_sufficiency'].mean() * 100:.0f}%
    Best season: {best_season} ({seasonal.loc[best_season, 'self_sufficiency']*100:.0f}%)
    Worst season: {worst_season} ({seasonal.loc[worst_season, 'self_sufficiency']*100:.0f}%)""")

    return hdd


def main():
    """Run all Phase 2 EDA analyses."""
    print("="*60)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Load data
    data = load_data()

    # 2.1 Energy Patterns
    energy_patterns = analyze_energy_patterns(data)

    # 2.2 Heating System Analysis
    daily_heating = analyze_heating_system(data)

    # 2.3 Solar-Heating Correlation
    analyze_solar_heating_correlation(data)

    # 2.4 Summary Statistics
    generate_summary_statistics(data, energy_patterns, daily_heating)

    print("\n" + "="*60)
    print(f"Phase 2 EDA complete. Output saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
