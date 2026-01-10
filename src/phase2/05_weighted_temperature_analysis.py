#!/usr/bin/env python3
"""
Phase 2, Step 5: Weighted Temperature Analysis

Analyzes how the weighted indoor temperature (comfort objective) responds to
controllable heating parameters:
- comfort_temperature_target_hk1 (day setpoint)
- eco_temperature_target_hk1 (night setpoint)
- heating_curve_rise_hk1 (slope)
- schedule (comfort start/end times)

Key features:
- Computes weighted indoor temperature from 5 sensors (T_weighted)
- Detects parameter regime changes
- Applies 2-day washout exclusion after each regime change
- Builds sensitivity model for parameter effects
- Generates visualizations and documentation

Output:
- output/phase2/fig13_weighted_temp_parameters.png (4-panel visualization)
- output/phase2/weighted_temp_regimes.csv (regime summary)
- output/phase2/weighted_temp_sensitivity.csv (parameter effects)
- output/phase2/weighted_temp_report_section.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(exist_ok=True)

# Sensor weights for weighted indoor temperature (from Phase 3 thermal model)
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 0.40,
    'office1_temperature': 0.30,
    'atelier_temperature': 0.10,
    'studio_temperature': 0.10,
    'simlab_temperature': 0.10,
}

# Washout period after parameter changes (hours)
WASHOUT_HOURS = 48


def load_heating_data():
    """Load heating sensor data in raw (long) format."""
    print("Loading heating sensor data...")
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)

    n_sensors = heating_raw['entity_id'].nunique()
    print(f"  Loaded {len(heating_raw):,} rows, {n_sensors} sensors")
    print(f"  Date range: {heating_raw['datetime'].min()} to {heating_raw['datetime'].max()}")

    return heating_raw


def load_room_data():
    """Load room temperature sensor data."""
    print("Loading room sensor data...")
    rooms_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_rooms.parquet')
    rooms_raw['datetime'] = pd.to_datetime(rooms_raw['datetime'], utc=True)

    # Apply office2 → atelier mapping
    rooms_raw.loc[rooms_raw['entity_id'] == 'office2_temperature', 'entity_id'] = 'atelier_temperature'

    n_sensors = rooms_raw['entity_id'].nunique()
    print(f"  Loaded {len(rooms_raw):,} rows, {n_sensors} sensors")

    return rooms_raw


def load_weather_data():
    """Load weather sensor data (for davis_inside_temperature)."""
    print("Loading weather sensor data...")
    weather_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_weather.parquet')
    weather_raw['datetime'] = pd.to_datetime(weather_raw['datetime'], utc=True)

    n_sensors = weather_raw['entity_id'].nunique()
    print(f"  Loaded {len(weather_raw):,} rows, {n_sensors} sensors")

    return weather_raw


def compute_weighted_temperature(rooms_df: pd.DataFrame) -> pd.Series:
    """
    Compute weighted indoor temperature from sensor data.

    Returns a Series indexed by datetime with the weighted temperature.
    """
    print("\nComputing weighted indoor temperature...")

    # Pivot room sensors to wide format (davis_inside_temperature is in rooms data)
    rooms_pivot = rooms_df.pivot_table(
        values='value',
        index='datetime',
        columns='entity_id',
        aggfunc='mean'
    )

    # Resample all to 15-minute intervals
    df = rooms_pivot.resample('15min').mean()

    # Compute weighted average
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for sensor, weight in SENSOR_WEIGHTS.items():
        if sensor in df.columns:
            valid_mask = df[sensor].notna()
            weighted_sum[valid_mask] += df.loc[valid_mask, sensor] * weight
            weight_sum[valid_mask] += weight
            n_valid = valid_mask.sum()
            print(f"  {sensor}: {n_valid:,} valid readings (weight={weight:.0%})")

    # Normalize by actual weight sum (handles missing sensors)
    T_weighted = weighted_sum / weight_sum
    T_weighted = T_weighted.replace([np.inf, -np.inf], np.nan)

    print(f"  Weighted temperature: {T_weighted.notna().sum():,} valid readings")
    print(f"  Range: {T_weighted.min():.1f}°C to {T_weighted.max():.1f}°C")
    print(f"  Mean: {T_weighted.mean():.1f}°C")

    return T_weighted


def detect_parameter_regimes(heating_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Detect parameter regime changes from setpoint and curve_rise sensors.

    Returns DataFrame with columns:
    - start_datetime: regime start (UTC)
    - comfort_setpoint: comfort temperature setpoint
    - eco_setpoint: eco temperature setpoint
    - curve_rise: heating curve slope
    """
    print("\nDetecting parameter regimes...")

    # Extract setpoint and curve_rise series
    comfort_entity = 'stiebel_eltron_isg_comfort_temperature_target_hk1'
    eco_entity = 'stiebel_eltron_isg_eco_temperature_target_hk1'
    curve_entity = 'stiebel_eltron_isg_heating_curve_rise_hk1'

    comfort_data = heating_raw[heating_raw['entity_id'] == comfort_entity].copy()
    eco_data = heating_raw[heating_raw['entity_id'] == eco_entity].copy()
    curve_data = heating_raw[heating_raw['entity_id'] == curve_entity].copy()

    comfort_data = comfort_data.set_index('datetime')['value'].sort_index()
    eco_data = eco_data.set_index('datetime')['value'].sort_index()
    curve_data = curve_data.set_index('datetime')['value'].sort_index()

    # Combine all parameters
    all_params = pd.concat([
        comfort_data.rename('comfort'),
        eco_data.rename('eco'),
        curve_data.rename('curve_rise')
    ], axis=1).sort_index()

    # Forward fill to get current value at each point
    all_params = all_params.ffill()

    # Detect actual changes (where any value changes)
    all_params['comfort_changed'] = all_params['comfort'] != all_params['comfort'].shift()
    all_params['eco_changed'] = all_params['eco'] != all_params['eco'].shift()
    all_params['curve_changed'] = all_params['curve_rise'] != all_params['curve_rise'].shift()
    all_params['regime_change'] = (
        all_params['comfort_changed'] |
        all_params['eco_changed'] |
        all_params['curve_changed']
    )

    # Mark first row as regime start
    all_params.iloc[0, all_params.columns.get_loc('regime_change')] = True

    # Extract regime change points
    regime_points = all_params[all_params['regime_change']].copy()
    regime_points = regime_points[['comfort', 'eco', 'curve_rise']].reset_index()
    regime_points.columns = ['start_datetime', 'comfort_setpoint', 'eco_setpoint', 'curve_rise']

    # Add end datetime for each regime
    regime_points['end_datetime'] = regime_points['start_datetime'].shift(-1)
    regime_points.loc[regime_points.index[-1], 'end_datetime'] = all_params.index.max()

    print(f"  Found {len(regime_points)} parameter regimes:")
    for i, row in regime_points.head(10).iterrows():
        dt_local = row['start_datetime'].tz_convert('Europe/Zurich')
        print(f"    {dt_local.strftime('%Y-%m-%d %H:%M')}: "
              f"comfort={row['comfort_setpoint']:.1f}°C, "
              f"eco={row['eco_setpoint']:.1f}°C, "
              f"curve={row['curve_rise']:.2f}")
    if len(regime_points) > 10:
        print(f"    ... and {len(regime_points) - 10} more")

    return regime_points


def create_washout_mask(T_weighted: pd.Series, regimes: pd.DataFrame) -> pd.Series:
    """
    Create boolean mask marking washout periods after parameter changes.

    Returns Series indexed like T_weighted with True for washout periods.
    """
    print(f"\nCreating washout mask ({WASHOUT_HOURS}h after each regime change)...")

    washout_mask = pd.Series(False, index=T_weighted.index)

    for i, row in regimes.iterrows():
        if i == 0:
            continue  # Skip first regime (no prior change)

        start = row['start_datetime']
        end = start + timedelta(hours=WASHOUT_HOURS)

        # Mark washout period
        period_mask = (T_weighted.index >= start) & (T_weighted.index < end)
        washout_mask[period_mask] = True

    n_washout = washout_mask.sum()
    n_total = len(T_weighted)
    print(f"  Washout periods: {n_washout:,} readings ({n_washout/n_total*100:.1f}%)")
    print(f"  Analysis periods: {n_total - n_washout:,} readings ({(n_total-n_washout)/n_total*100:.1f}%)")

    return washout_mask


def build_analysis_dataset(T_weighted: pd.Series, heating_raw: pd.DataFrame,
                           regimes: pd.DataFrame, washout_mask: pd.Series) -> pd.DataFrame:
    """
    Build combined dataset for analysis with weighted temperature and parameters.
    """
    print("\nBuilding analysis dataset...")

    # Pivot heating data to get parameters
    sensors = [
        'stiebel_eltron_isg_comfort_temperature_target_hk1',
        'stiebel_eltron_isg_eco_temperature_target_hk1',
        'stiebel_eltron_isg_heating_curve_rise_hk1',
        'stiebel_eltron_isg_outdoor_temperature',
        'stiebel_eltron_isg_target_temperature_hk_1',
    ]

    series_dict = {}
    for entity_id in sensors:
        name = entity_id.replace('stiebel_eltron_isg_', '')
        sensor_data = heating_raw[heating_raw['entity_id'] == entity_id].copy()
        if len(sensor_data) == 0:
            continue
        sensor_data = sensor_data.set_index('datetime')['value'].sort_index()
        series_dict[name] = sensor_data.resample('15min').last().ffill()

    params = pd.DataFrame(series_dict)

    # Combine with weighted temperature
    df = pd.DataFrame({'T_weighted': T_weighted})
    df = df.join(params, how='left')

    # Forward fill parameters
    for col in ['comfort_temperature_target_hk1', 'eco_temperature_target_hk1',
                'heating_curve_rise_hk1']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Add washout flag
    df['is_washout'] = washout_mask.reindex(df.index, fill_value=False)

    # Add time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Determine comfort/eco mode (simplified: 06:30-20:00 is comfort)
    hour_decimal = df['hour'] + df.index.minute / 60
    df['is_comfort'] = (hour_decimal >= 6.5) & (hour_decimal < 20.0)

    # Current setpoint based on mode
    df['setpoint'] = np.where(
        df['is_comfort'],
        df['comfort_temperature_target_hk1'],
        df['eco_temperature_target_hk1']
    )

    # Drop rows with missing weighted temperature
    df = df.dropna(subset=['T_weighted'])

    print(f"  Dataset: {len(df):,} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Washout periods: {df['is_washout'].sum():,} rows")
    print(f"  Analysis data: {(~df['is_washout']).sum():,} rows")

    return df


def compute_regime_statistics(df: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each parameter regime (excluding washout).
    """
    print("\nComputing regime statistics...")

    stats_list = []
    df_utc = df.copy()
    if df_utc.index.tz is None:
        df_utc.index = df_utc.index.tz_localize('UTC')
    else:
        df_utc.index = df_utc.index.tz_convert('UTC')

    for i, row in regimes.iterrows():
        start = row['start_datetime']
        end = row['end_datetime']

        # Filter data for this regime, excluding washout
        regime_mask = (df_utc.index >= start) & (df_utc.index < end) & (~df_utc['is_washout'])
        regime_data = df_utc[regime_mask]

        if len(regime_data) < 10:
            continue

        stats = {
            'regime_idx': i,
            'start_datetime': start,
            'end_datetime': end,
            'comfort_setpoint': row['comfort_setpoint'],
            'eco_setpoint': row['eco_setpoint'],
            'curve_rise': row['curve_rise'],
            'n_readings': len(regime_data),
            'T_weighted_mean': regime_data['T_weighted'].mean(),
            'T_weighted_std': regime_data['T_weighted'].std(),
            'T_weighted_min': regime_data['T_weighted'].min(),
            'T_weighted_max': regime_data['T_weighted'].max(),
        }

        # Comfort vs eco statistics
        comfort_data = regime_data[regime_data['is_comfort']]
        eco_data = regime_data[~regime_data['is_comfort']]

        if len(comfort_data) > 0:
            stats['T_comfort_mean'] = comfort_data['T_weighted'].mean()
            stats['T_comfort_std'] = comfort_data['T_weighted'].std()
        if len(eco_data) > 0:
            stats['T_eco_mean'] = eco_data['T_weighted'].mean()
            stats['T_eco_std'] = eco_data['T_weighted'].std()

        # Outdoor temperature during regime
        if 'outdoor_temperature' in regime_data.columns:
            stats['outdoor_mean'] = regime_data['outdoor_temperature'].mean()

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    print(f"  Computed statistics for {len(stats_df)} regimes")

    return stats_df


def compute_sensitivity(df: pd.DataFrame) -> dict:
    """
    Compute parameter sensitivity (effect of each parameter on T_weighted).

    Excludes washout periods from analysis.
    """
    print("\nComputing parameter sensitivity...")

    # Filter to non-washout data
    analysis_df = df[~df['is_washout']].copy()

    if len(analysis_df) < 100:
        print("  WARNING: Not enough data for sensitivity analysis")
        return {}

    results = {}

    # Effect of setpoint on T_weighted
    if 'setpoint' in analysis_df.columns and analysis_df['setpoint'].notna().sum() > 50:
        # Simple regression: T_weighted = a + b * setpoint
        x = analysis_df['setpoint'].dropna()
        y = analysis_df.loc[x.index, 'T_weighted']

        if len(x) > 50:
            corr = x.corr(y)
            # Linear regression
            cov_xy = ((x - x.mean()) * (y - y.mean())).mean()
            var_x = x.var()
            slope = cov_xy / var_x if var_x > 0 else 0

            results['setpoint'] = {
                'correlation': corr,
                'slope': slope,
                'description': f'+1°C setpoint → {slope:+.2f}°C T_weighted',
                'n_obs': len(x)
            }
            print(f"  Setpoint effect: {results['setpoint']['description']} (r={corr:.2f}, n={len(x):,})")

    # Effect of curve_rise on T_weighted (indirect, via flow temperature)
    if 'heating_curve_rise_hk1' in analysis_df.columns:
        x = analysis_df['heating_curve_rise_hk1'].dropna()
        y = analysis_df.loc[x.index, 'T_weighted']

        if len(x) > 50:
            corr = x.corr(y)
            cov_xy = ((x - x.mean()) * (y - y.mean())).mean()
            var_x = x.var()
            slope = cov_xy / var_x if var_x > 0 else 0

            results['curve_rise'] = {
                'correlation': corr,
                'slope': slope,
                'description': f'+0.1 curve_rise → {0.1*slope:+.2f}°C T_weighted',
                'n_obs': len(x)
            }
            print(f"  Curve rise effect: {results['curve_rise']['description']} (r={corr:.2f}, n={len(x):,})")

    # Effect of outdoor temperature
    if 'outdoor_temperature' in analysis_df.columns:
        x = analysis_df['outdoor_temperature'].dropna()
        y = analysis_df.loc[x.index, 'T_weighted']

        if len(x) > 50:
            corr = x.corr(y)
            cov_xy = ((x - x.mean()) * (y - y.mean())).mean()
            var_x = x.var()
            slope = cov_xy / var_x if var_x > 0 else 0

            results['outdoor'] = {
                'correlation': corr,
                'slope': slope,
                'description': f'+1°C outdoor → {slope:+.2f}°C T_weighted',
                'n_obs': len(x)
            }
            print(f"  Outdoor effect: {results['outdoor']['description']} (r={corr:.2f}, n={len(x):,})")

    return results


def create_visualization(df: pd.DataFrame, regimes: pd.DataFrame,
                         regime_stats: pd.DataFrame, sensitivity: dict):
    """Create 6-panel visualization of weighted temperature analysis."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Convert index to local time for plotting
    df_local = df.copy()
    if df_local.index.tz is not None:
        df_local.index = df_local.index.tz_convert('Europe/Zurich')
    else:
        df_local.index = df_local.index.tz_localize('UTC').tz_convert('Europe/Zurich')

    # Panel 1: Time series with regime shading and washout markers
    ax = axes[0, 0]

    # Plot weighted temperature
    ax.plot(df_local.index, df_local['T_weighted'],
            color='steelblue', alpha=0.6, linewidth=0.5, label='T_weighted')

    # Add daily mean line
    daily_mean = df_local['T_weighted'].resample('D').mean()
    ax.plot(daily_mean.index, daily_mean.values,
            color='darkblue', linewidth=2, label='Daily mean')

    # Shade washout periods
    washout_starts = df_local[df_local['is_washout'] & ~df_local['is_washout'].shift(1).fillna(False)].index
    washout_ends = df_local[df_local['is_washout'] & ~df_local['is_washout'].shift(-1).fillna(False)].index

    for start, end in zip(washout_starts, washout_ends):
        ax.axvspan(start, end, alpha=0.3, color='red', label='_nolegend_')

    # Add regime change markers
    for i, row in regimes.iterrows():
        if i == 0:
            continue
        dt = row['start_datetime'].tz_convert('Europe/Zurich')
        ax.axvline(x=dt, color='orange', linestyle='--', alpha=0.7, linewidth=1)

    # Add comfort bounds
    ax.axhline(y=18.5, color='gray', linestyle=':', alpha=0.7, label='Comfort bounds')
    ax.axhline(y=22.0, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Weighted Temperature (°C)')
    ax.set_title('Weighted Indoor Temperature (T_weighted)\n'
                 'Orange lines = parameter changes, Red shading = 48h washout')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: T_weighted vs setpoint (excluding washout)
    ax = axes[0, 1]

    analysis_df = df_local[~df_local['is_washout']].copy()

    if 'setpoint' in analysis_df.columns and analysis_df['setpoint'].notna().sum() > 0:
        # Color by comfort/eco mode
        comfort_mask = analysis_df['is_comfort']

        ax.scatter(analysis_df.loc[comfort_mask, 'setpoint'],
                   analysis_df.loc[comfort_mask, 'T_weighted'],
                   c='orange', s=5, alpha=0.3, label='Comfort mode')
        ax.scatter(analysis_df.loc[~comfort_mask, 'setpoint'],
                   analysis_df.loc[~comfort_mask, 'T_weighted'],
                   c='blue', s=5, alpha=0.3, label='Eco mode')

        # Add regression line if we have sensitivity data
        if 'setpoint' in sensitivity:
            x_range = np.array([analysis_df['setpoint'].min(), analysis_df['setpoint'].max()])
            y_mean = analysis_df['T_weighted'].mean()
            x_mean = analysis_df['setpoint'].mean()
            slope = sensitivity['setpoint']['slope']
            y_range = y_mean + slope * (x_range - x_mean)
            ax.plot(x_range, y_range, 'r-', linewidth=2,
                    label=f'Slope: {slope:.2f}°C/°C setpoint')

        ax.set_xlabel('Setpoint Temperature (°C)')
        ax.set_ylabel('Weighted Temperature (°C)')
        ax.set_title('T_weighted vs Setpoint\n(Excluding 48h washout periods)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: T_weighted vs curve_rise (excluding washout)
    ax = axes[1, 0]

    if 'heating_curve_rise_hk1' in analysis_df.columns:
        # Sample for plotting if too many points
        if len(analysis_df) > 5000:
            plot_df = analysis_df.sample(5000)
        else:
            plot_df = analysis_df

        scatter = ax.scatter(plot_df['heating_curve_rise_hk1'],
                            plot_df['T_weighted'],
                            c=plot_df['outdoor_temperature'] if 'outdoor_temperature' in plot_df.columns else 'steelblue',
                            cmap='coolwarm', s=5, alpha=0.5)

        if 'outdoor_temperature' in plot_df.columns:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Outdoor Temp (°C)')

        # Add regression line
        if 'curve_rise' in sensitivity:
            x_range = np.array([analysis_df['heating_curve_rise_hk1'].min(),
                               analysis_df['heating_curve_rise_hk1'].max()])
            y_mean = analysis_df['T_weighted'].mean()
            x_mean = analysis_df['heating_curve_rise_hk1'].mean()
            slope = sensitivity['curve_rise']['slope']
            y_range = y_mean + slope * (x_range - x_mean)
            ax.plot(x_range, y_range, 'k-', linewidth=2,
                    label=f'Slope: {slope:.2f}°C per 1.0 rise')

        ax.set_xlabel('Heating Curve Rise (Steilheit)')
        ax.set_ylabel('Weighted Temperature (°C)')
        ax.set_title('T_weighted vs Curve Rise\n(Colored by outdoor temperature)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 4: Hourly profile by regime
    ax = axes[1, 1]

    # Group by regime and hour
    if len(regime_stats) > 0:
        # Use up to 4 distinct regimes for clarity
        unique_regimes = regime_stats.drop_duplicates(
            subset=['comfort_setpoint', 'eco_setpoint', 'curve_rise']
        ).head(4)

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))

        for idx, (_, regime_row) in enumerate(unique_regimes.iterrows()):
            # Find data for this regime configuration
            regime_mask = (
                (analysis_df['comfort_temperature_target_hk1'].round(1) == round(regime_row['comfort_setpoint'], 1)) &
                (analysis_df['eco_temperature_target_hk1'].round(1) == round(regime_row['eco_setpoint'], 1))
            )

            regime_data = analysis_df[regime_mask]

            if len(regime_data) < 50:
                continue

            hourly_mean = regime_data.groupby('hour')['T_weighted'].mean()
            hourly_std = regime_data.groupby('hour')['T_weighted'].std()

            label = f"Comfort={regime_row['comfort_setpoint']:.1f}°C, Eco={regime_row['eco_setpoint']:.1f}°C"
            ax.plot(hourly_mean.index, hourly_mean.values,
                    color=colors[idx], linewidth=2, label=label)
            ax.fill_between(hourly_mean.index,
                            hourly_mean - hourly_std,
                            hourly_mean + hourly_std,
                            color=colors[idx], alpha=0.2)

        # Mark comfort period (06:30-20:00)
        ax.axvspan(6.5, 20, alpha=0.1, color='orange', label='Comfort hours')

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Weighted Temperature (°C)')
        ax.set_title('Hourly Temperature Profile by Setpoint Regime')
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Panel 5: T_weighted during OCCUPIED hours only (08:00-22:00)
    ax = axes[2, 0]

    # Filter to occupied hours (08:00-22:00) - this is when comfort compliance is evaluated
    occupied_mask = (df_local['hour'] >= 8) & (df_local['hour'] < 22)
    df_occupied = df_local[occupied_mask].copy()

    if len(df_occupied) > 0:
        # Plot occupied hours temperature
        ax.plot(df_occupied.index, df_occupied['T_weighted'],
                color='steelblue', alpha=0.6, linewidth=0.5, label='T_weighted')

        # Add daily mean for occupied hours
        daily_mean = df_occupied['T_weighted'].resample('D').mean()
        ax.plot(daily_mean.index, daily_mean.values,
                color='darkblue', linewidth=2, label='Daily mean')

        # Shade washout periods
        washout_occupied = df_occupied[df_occupied['is_washout']]
        if len(washout_occupied) > 0:
            washout_starts = washout_occupied[washout_occupied['is_washout'] & ~washout_occupied['is_washout'].shift(1).fillna(False)].index
            washout_ends = washout_occupied[washout_occupied['is_washout'] & ~washout_occupied['is_washout'].shift(-1).fillna(False)].index
            for start, end in zip(washout_starts, washout_ends):
                ax.axvspan(start, end, alpha=0.3, color='red', label='_nolegend_')

        # Add comfort bounds
        ax.axhline(y=18.5, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Min comfort (18.5°C)')
        ax.axhline(y=22.0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Max comfort (22°C)')

        # Calculate and display comfort compliance
        analysis_occupied = df_occupied[~df_occupied['is_washout']]
        if len(analysis_occupied) > 0:
            in_bounds = (analysis_occupied['T_weighted'] >= 18.5) & (analysis_occupied['T_weighted'] <= 22.0)
            compliance = in_bounds.mean() * 100
            below_min = (analysis_occupied['T_weighted'] < 18.5).mean() * 100
            t_mean_occ = analysis_occupied['T_weighted'].mean()
            ax.set_title(f'T_weighted During OCCUPIED Hours (08:00-22:00)\n'
                        f'Mean: {t_mean_occ:.1f}°C | Compliance: {compliance:.1f}% | Below 18.5°C: {below_min:.1f}%')
        else:
            ax.set_title('T_weighted During OCCUPIED Hours (08:00-22:00)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Weighted Temperature (°C)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 6: T_weighted during NIGHT hours only (22:00-08:00)
    ax = axes[2, 1]

    # Filter to night hours (22:00-08:00) - outside comfort evaluation window
    night_mask = (df_local['hour'] >= 22) | (df_local['hour'] < 8)
    df_night = df_local[night_mask].copy()

    if len(df_night) > 0:
        # Plot night hours temperature
        ax.plot(df_night.index, df_night['T_weighted'],
                color='darkblue', alpha=0.6, linewidth=0.5, label='T_weighted')

        # Add daily mean for night hours
        daily_mean = df_night['T_weighted'].resample('D').mean()
        ax.plot(daily_mean.index, daily_mean.values,
                color='navy', linewidth=2, label='Daily mean')

        # Shade washout periods
        washout_night = df_night[df_night['is_washout']]
        if len(washout_night) > 0:
            washout_starts = washout_night[washout_night['is_washout'] & ~washout_night['is_washout'].shift(1).fillna(False)].index
            washout_ends = washout_night[washout_night['is_washout'] & ~washout_night['is_washout'].shift(-1).fillna(False)].index
            for start, end in zip(washout_starts, washout_ends):
                ax.axvspan(start, end, alpha=0.3, color='red', label='_nolegend_')

        # Add reference lines (comfort bounds shown for reference, but not enforced at night)
        ax.axhline(y=18.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Comfort ref (18.5°C)')
        ax.axhline(y=22.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Calculate night statistics
        analysis_night = df_night[~df_night['is_washout']]
        if len(analysis_night) > 0:
            t_mean_night = analysis_night['T_weighted'].mean()
            t_min_night = analysis_night['T_weighted'].min()
            below_18 = (analysis_night['T_weighted'] < 18.0).mean() * 100
            ax.set_title(f'T_weighted During NIGHT Hours (22:00-08:00)\n'
                        f'Mean: {t_mean_night:.1f}°C | Min: {t_min_night:.1f}°C | Below 18°C: {below_18:.1f}%')
        else:
            ax.set_title('T_weighted During NIGHT Hours (22:00-08:00)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Weighted Temperature (°C)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig13_weighted_temp_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig13_weighted_temp_parameters.png")


def generate_report_section(regime_stats: pd.DataFrame, sensitivity: dict, df: pd.DataFrame) -> str:
    """Generate HTML section for EDA report."""

    # Calculate summary statistics
    analysis_df = df[~df['is_washout']]
    n_total = len(df)
    n_washout = df['is_washout'].sum()
    n_analysis = len(analysis_df)

    t_mean = analysis_df['T_weighted'].mean()
    t_std = analysis_df['T_weighted'].std()
    t_min = analysis_df['T_weighted'].min()
    t_max = analysis_df['T_weighted'].max()

    # Comfort compliance during occupied hours (08:00-22:00)
    occupied_mask = (analysis_df['hour'] >= 8) & (analysis_df['hour'] < 22)
    occupied_df = analysis_df[occupied_mask]
    in_bounds = (occupied_df['T_weighted'] >= 18.5) & (occupied_df['T_weighted'] <= 22.0)
    comfort_compliance = in_bounds.mean() * 100 if len(occupied_df) > 0 else 0

    html = f"""
        <h2 id="weighted-temp">9. Weighted Indoor Temperature Analysis</h2>

        <div class="card">
            <h4>Overview</h4>
            <p>The <strong>weighted indoor temperature (T_weighted)</strong> is the primary comfort objective
            for the Phase 5 intervention study. It combines readings from five indoor temperature sensors
            to create a representative whole-building temperature metric.</p>

            <h4>Weighted Temperature Formula</h4>
            <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem;">T_weighted = 0.40×davis_inside + 0.30×office1 + 0.10×atelier + 0.10×studio + 0.10×simlab</pre>

            <table>
                <tr><th>Sensor</th><th>Weight</th><th>Rationale</th></tr>
                <tr><td>davis_inside_temperature</td><td>40%</td><td>Primary living area, central location</td></tr>
                <tr><td>office1_temperature</td><td>30%</td><td>Secondary occupied workspace</td></tr>
                <tr><td>atelier_temperature</td><td>10%</td><td>Zone coverage - workshop area</td></tr>
                <tr><td>studio_temperature</td><td>10%</td><td>Zone coverage - studio space</td></tr>
                <tr><td>simlab_temperature</td><td>10%</td><td>Zone coverage - laboratory area</td></tr>
            </table>
        </div>

        <div class="card">
            <h4>Analysis Period Summary</h4>
            <table>
                <tr><td>Total readings</td><td>{n_total:,}</td></tr>
                <tr><td>Washout periods (48h after parameter changes)</td><td>{n_washout:,} ({n_washout/n_total*100:.1f}%)</td></tr>
                <tr><td>Analysis readings (excluding washout)</td><td>{n_analysis:,} ({n_analysis/n_total*100:.1f}%)</td></tr>
            </table>

            <h4>Temperature Statistics (Excluding Washout)</h4>
            <table>
                <tr><td>Mean T_weighted</td><td><strong>{t_mean:.1f}°C</strong></td></tr>
                <tr><td>Std deviation</td><td>{t_std:.1f}°C</td></tr>
                <tr><td>Range</td><td>{t_min:.1f}°C to {t_max:.1f}°C</td></tr>
                <tr><td>Comfort compliance (08:00-22:00, 18.5-22°C)</td><td><strong>{comfort_compliance:.1f}%</strong></td></tr>
            </table>
        </div>

        <div class="card">
            <h4>Parameter Sensitivity Analysis</h4>
            <p>How T_weighted responds to changes in controllable parameters (after 48h washout):</p>
            <table>
                <tr><th>Parameter</th><th>Effect</th><th>Correlation</th><th>Observations</th></tr>
"""

    # Add sensitivity rows
    if 'setpoint' in sensitivity:
        s = sensitivity['setpoint']
        html += f"""                <tr>
                    <td>Setpoint</td>
                    <td>{s['description']}</td>
                    <td>r = {s['correlation']:.2f}</td>
                    <td>{s['n_obs']:,}</td>
                </tr>
"""

    if 'curve_rise' in sensitivity:
        s = sensitivity['curve_rise']
        html += f"""                <tr>
                    <td>Curve rise (Steilheit)</td>
                    <td>{s['description']}</td>
                    <td>r = {s['correlation']:.2f}</td>
                    <td>{s['n_obs']:,}</td>
                </tr>
"""

    if 'outdoor' in sensitivity:
        s = sensitivity['outdoor']
        html += f"""                <tr>
                    <td>Outdoor temperature</td>
                    <td>{s['description']}</td>
                    <td>r = {s['correlation']:.2f}</td>
                    <td>{s['n_obs']:,}</td>
                </tr>
"""

    html += """            </table>
        </div>

        <div class="card">
            <h4>Washout Period Rationale</h4>
            <p>A <strong>48-hour (2-day) washout period</strong> is applied after each parameter change to allow
            the thermal system to reach equilibrium. This is based on:</p>
            <ul>
                <li>Building thermal time constant: ~19 hours (weighted average from Phase 3 thermal model)</li>
                <li>Washout = 2.5× time constant ≈ 48 hours (~87% equilibrium)</li>
                <li>Conservative approach to ensure clean measurements in Phase 5 intervention study</li>
            </ul>
            <p><em>Note: Phase 5 study design uses 3-day (72h) washout for additional safety margin.</em></p>
        </div>

        <div class="card">
            <h4>Parameter Regimes Detected</h4>
            <table>
                <tr><th>Start Date</th><th>Comfort (°C)</th><th>Eco (°C)</th><th>Curve Rise</th><th>Mean T_weighted (°C)</th><th>N Readings</th></tr>
"""

    # Add regime rows (show top regimes)
    for _, row in regime_stats.head(10).iterrows():
        start_local = row['start_datetime'].tz_convert('Europe/Zurich') if hasattr(row['start_datetime'], 'tz_convert') else row['start_datetime']
        html += f"""                <tr>
                    <td>{start_local.strftime('%Y-%m-%d')}</td>
                    <td>{row['comfort_setpoint']:.1f}</td>
                    <td>{row['eco_setpoint']:.1f}</td>
                    <td>{row['curve_rise']:.2f}</td>
                    <td>{row['T_weighted_mean']:.1f}</td>
                    <td>{row['n_readings']:,}</td>
                </tr>
"""

    if len(regime_stats) > 10:
        html += f"""                <tr><td colspan="6"><em>... and {len(regime_stats) - 10} more regimes</em></td></tr>
"""

    html += """            </table>
        </div>
"""

    # Add occupied vs night hours statistics
    occupied_mask = (analysis_df['hour'] >= 8) & (analysis_df['hour'] < 22)
    night_mask = (analysis_df['hour'] >= 22) | (analysis_df['hour'] < 8)

    occupied_df = analysis_df[occupied_mask]
    night_df = analysis_df[night_mask]

    if len(occupied_df) > 0 and len(night_df) > 0:
        occ_mean = occupied_df['T_weighted'].mean()
        occ_min = occupied_df['T_weighted'].min()
        occ_compliance = ((occupied_df['T_weighted'] >= 18.5) & (occupied_df['T_weighted'] <= 22.0)).mean() * 100
        occ_below = (occupied_df['T_weighted'] < 18.5).mean() * 100

        night_mean = night_df['T_weighted'].mean()
        night_min = night_df['T_weighted'].min()
        night_below_18 = (night_df['T_weighted'] < 18.0).mean() * 100

        html += f"""
        <div class="card">
            <h4>Occupied vs Night Hours Comparison</h4>
            <p><strong>Important:</strong> Comfort compliance is evaluated only during <strong>occupied hours (08:00-22:00)</strong>.
            Night temperatures (22:00-08:00) are excluded from the comfort objective, allowing energy-saving setback without penalty.</p>
            <table>
                <tr><th>Metric</th><th>Occupied (08:00-22:00)</th><th>Night (22:00-08:00)</th></tr>
                <tr><td>Mean T_weighted</td><td><strong>{occ_mean:.1f}°C</strong></td><td>{night_mean:.1f}°C</td></tr>
                <tr><td>Minimum T_weighted</td><td>{occ_min:.1f}°C</td><td>{night_min:.1f}°C</td></tr>
                <tr><td>Below 18.5°C</td><td>{occ_below:.1f}%</td><td>—</td></tr>
                <tr><td>Below 18.0°C</td><td>—</td><td>{night_below_18:.1f}%</td></tr>
                <tr><td>Comfort compliance (18.5-22°C)</td><td><strong>{occ_compliance:.1f}%</strong></td><td>Not evaluated</td></tr>
                <tr><td>N readings</td><td>{len(occupied_df):,}</td><td>{len(night_df):,}</td></tr>
            </table>
            <p><em>This explains why the overall mean (18.7°C) appears low - night temperatures bring down the average,
            but comfort compliance during occupied hours ({occ_compliance:.1f}%) is what matters for the intervention study.</em></p>
        </div>
"""

    html += """
        <div class="figure">
            <img src="fig13_weighted_temp_parameters.png" alt="Weighted temperature parameter analysis">
            <div class="figure-caption">Fig 13: Weighted indoor temperature analysis (6 panels).
            <strong>Top row:</strong> Full time series with washout periods (left), T_weighted vs setpoint (right).
            <strong>Middle row:</strong> T_weighted vs curve rise colored by outdoor temp (left), hourly profiles by regime (right).
            <strong>Bottom row:</strong> T_weighted during OCCUPIED hours only with comfort bounds (left),
            T_weighted during NIGHT hours only (right). Red shading indicates 48h washout periods.</div>
        </div>
"""

    return html


def main():
    """Run weighted temperature analysis."""
    print("=" * 60)
    print("WEIGHTED TEMPERATURE ANALYSIS")
    print("=" * 60)

    # Load data
    heating_raw = load_heating_data()
    rooms_raw = load_room_data()

    # Compute weighted temperature (davis_inside_temperature is in rooms data)
    T_weighted = compute_weighted_temperature(rooms_raw)

    # Detect parameter regimes
    regimes = detect_parameter_regimes(heating_raw)

    # Create washout mask
    washout_mask = create_washout_mask(T_weighted, regimes)

    # Build analysis dataset
    df = build_analysis_dataset(T_weighted, heating_raw, regimes, washout_mask)

    # Compute regime statistics
    regime_stats = compute_regime_statistics(df, regimes)

    # Compute sensitivity
    sensitivity = compute_sensitivity(df)

    # Save outputs
    print("\nSaving outputs...")

    # Save regime statistics
    regime_stats_save = regime_stats.copy()
    if 'start_datetime' in regime_stats_save.columns:
        regime_stats_save['start_datetime'] = regime_stats_save['start_datetime'].astype(str)
    if 'end_datetime' in regime_stats_save.columns:
        regime_stats_save['end_datetime'] = regime_stats_save['end_datetime'].astype(str)
    regime_stats_save.to_csv(OUTPUT_DIR / 'weighted_temp_regimes.csv', index=False)
    print(f"  Saved: weighted_temp_regimes.csv")

    # Save sensitivity results
    sensitivity_df = pd.DataFrame([
        {'parameter': k, **v} for k, v in sensitivity.items()
    ])
    sensitivity_df.to_csv(OUTPUT_DIR / 'weighted_temp_sensitivity.csv', index=False)
    print(f"  Saved: weighted_temp_sensitivity.csv")

    # Create visualization
    create_visualization(df, regimes, regime_stats, sensitivity)

    # Generate report section
    report_html = generate_report_section(regime_stats, sensitivity, df)
    with open(OUTPUT_DIR / 'weighted_temp_report_section.html', 'w') as f:
        f.write(report_html)
    print(f"  Saved: weighted_temp_report_section.html")

    print("\n" + "=" * 60)
    print("Weighted temperature analysis complete.")
    print("=" * 60)

    return df, regimes, regime_stats, sensitivity


if __name__ == "__main__":
    main()
