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
    'davis_inside_temperature': 1.0,
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


def build_multivariate_model(df: pd.DataFrame) -> dict:
    """
    Build multivariate regression model predicting T_weighted from controllable parameters.

    Model: T_weighted ~ comfort_setpoint + eco_setpoint + curve_rise + comfort_hours + outdoor_temp

    Uses daily aggregates to capture all parameters in a single observation.
    Excludes washout periods from analysis.

    Returns dict with model coefficients, standard errors, R², and predictions.
    """
    print("\nBuilding multivariate regression model...")

    # Filter to non-washout data
    analysis_df = df[~df['is_washout']].copy()

    if len(analysis_df) < 100:
        print("  WARNING: Not enough data for model")
        return {}

    # Aggregate to daily level
    analysis_df['date'] = analysis_df.index.date

    # Calculate daily metrics
    daily_data = []
    for date, day_df in analysis_df.groupby('date'):
        if len(day_df) < 4:  # Need at least 4 readings
            continue

        # Calculate comfort hours (when is_comfort == True)
        comfort_hours = day_df['is_comfort'].sum() * 0.25  # 15-min intervals to hours

        # Get parameter values (should be constant within day after washout)
        comfort_setpoint = day_df['comfort_temperature_target_hk1'].median()
        eco_setpoint = day_df['eco_temperature_target_hk1'].median()
        curve_rise = day_df['heating_curve_rise_hk1'].median()

        # Outdoor and indoor temperatures
        outdoor_mean = day_df['outdoor_temperature'].mean() if 'outdoor_temperature' in day_df.columns else np.nan
        t_weighted_mean = day_df['T_weighted'].mean()

        # Occupied hours only (08:00-22:00)
        occupied_mask = (day_df['hour'] >= 8) & (day_df['hour'] < 22)
        t_weighted_occupied = day_df.loc[occupied_mask, 'T_weighted'].mean() if occupied_mask.any() else np.nan

        daily_data.append({
            'date': date,
            'comfort_setpoint': comfort_setpoint,
            'eco_setpoint': eco_setpoint,
            'curve_rise': curve_rise,
            'comfort_hours': comfort_hours,
            'outdoor_mean': outdoor_mean,
            'T_weighted_mean': t_weighted_mean,
            'T_weighted_occupied': t_weighted_occupied,
            'n_readings': len(day_df)
        })

    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.dropna()

    print(f"  Daily aggregates: {len(daily_df)} days")

    if len(daily_df) < 10:
        print("  WARNING: Not enough daily data for robust model")
        return {'daily_df': daily_df}

    # Build design matrix for multiple regression
    # T_weighted = b0 + b1*comfort + b2*eco + b3*curve_rise + b4*comfort_hours + b5*outdoor
    X_cols = ['comfort_setpoint', 'eco_setpoint', 'curve_rise', 'comfort_hours', 'outdoor_mean']
    y_col = 'T_weighted_occupied'  # Model occupied hours temperature

    # Check which columns have variance
    valid_cols = []
    for col in X_cols:
        if col in daily_df.columns and daily_df[col].std() > 0.001:
            valid_cols.append(col)
        else:
            print(f"  Skipping {col}: no variance")

    if len(valid_cols) < 2:
        print("  WARNING: Not enough variable parameters for multivariate model")
        return {'daily_df': daily_df}

    X = daily_df[valid_cols].values
    y = daily_df[y_col].values

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # OLS regression: beta = (X'X)^-1 X'y
    try:
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ X_with_intercept.T @ y

        # Predictions and residuals
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred
        n = len(y)
        p = len(beta)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Adjusted R-squared
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else r_squared

        # Standard errors
        mse = ss_res / (n - p) if n > p else ss_res / n
        se = np.sqrt(np.diag(XtX_inv) * mse)

        # T-statistics and p-values (approximate)
        t_stats = beta / se
        # Using normal approximation for p-values
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p))

        # Build results dict
        coef_names = ['intercept'] + valid_cols
        coefficients = {}
        for i, name in enumerate(coef_names):
            coefficients[name] = {
                'estimate': beta[i],
                'std_error': se[i],
                't_stat': t_stats[i],
                'p_value': p_values[i]
            }

        results = {
            'coefficients': coefficients,
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'rmse': np.sqrt(mse),
            'n_obs': n,
            'n_params': p,
            'daily_df': daily_df,
            'y_pred': y_pred,
            'y_actual': y,
            'residuals': residuals,
            'feature_names': valid_cols
        }

        print(f"\n  Model Results (R² = {r_squared:.3f}, Adj R² = {r_squared_adj:.3f}, RMSE = {np.sqrt(mse):.2f}°C):")
        print(f"  {'Parameter':<20} {'Coef':>8} {'SE':>8} {'t':>8} {'p':>8}")
        print(f"  {'-'*52}")
        for name in coef_names:
            c = coefficients[name]
            sig = '***' if c['p_value'] < 0.001 else '**' if c['p_value'] < 0.01 else '*' if c['p_value'] < 0.05 else ''
            print(f"  {name:<20} {c['estimate']:>8.3f} {c['std_error']:>8.3f} {c['t_stat']:>8.2f} {c['p_value']:>7.4f} {sig}")

        # Interpretation
        print(f"\n  Interpretation (controlling for other variables):")
        for name in valid_cols:
            c = coefficients[name]
            if name == 'comfort_setpoint':
                print(f"    +1°C comfort setpoint → {c['estimate']:+.2f}°C T_weighted")
            elif name == 'eco_setpoint':
                print(f"    +1°C eco setpoint → {c['estimate']:+.2f}°C T_weighted")
            elif name == 'curve_rise':
                print(f"    +0.1 curve rise → {0.1*c['estimate']:+.2f}°C T_weighted")
            elif name == 'comfort_hours':
                print(f"    +1h comfort duration → {c['estimate']:+.2f}°C T_weighted")
            elif name == 'outdoor_mean':
                print(f"    +1°C outdoor temp → {c['estimate']:+.2f}°C T_weighted")

        return results

    except np.linalg.LinAlgError as e:
        print(f"  ERROR: Model fitting failed: {e}")
        return {'daily_df': daily_df}


def compute_sensitivity(df: pd.DataFrame) -> dict:
    """
    Compute parameter sensitivity using simple correlations (legacy function).
    Kept for backwards compatibility with HTML report generation.
    """
    print("\nComputing simple correlations (for reference)...")

    analysis_df = df[~df['is_washout']].copy()
    results = {}

    if len(analysis_df) < 100:
        return results

    # Simple correlations
    for param, col in [('setpoint', 'setpoint'),
                       ('curve_rise', 'heating_curve_rise_hk1'),
                       ('outdoor', 'outdoor_temperature')]:
        if col in analysis_df.columns:
            x = analysis_df[col].dropna()
            y = analysis_df.loc[x.index, 'T_weighted']
            if len(x) > 50:
                corr = x.corr(y)
                cov_xy = ((x - x.mean()) * (y - y.mean())).mean()
                var_x = x.var()
                slope = cov_xy / var_x if var_x > 0 else 0
                results[param] = {
                    'correlation': corr,
                    'slope': slope,
                    'n_obs': len(x)
                }

    return results


def create_visualization(df: pd.DataFrame, regimes: pd.DataFrame,
                         regime_stats: pd.DataFrame, model_results: dict):
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

    # Panel 2: Model coefficients with confidence intervals
    ax = axes[0, 1]

    if 'coefficients' in model_results and len(model_results['coefficients']) > 1:
        coefs = model_results['coefficients']
        feature_names = model_results.get('feature_names', [])

        # Exclude intercept, show only feature coefficients
        names = []
        estimates = []
        errors = []
        colors_coef = []

        # Nice labels for parameters
        label_map = {
            'comfort_setpoint': 'Comfort\nSetpoint',
            'eco_setpoint': 'Eco\nSetpoint',
            'curve_rise': 'Curve\nRise',
            'comfort_hours': 'Comfort\nHours',
            'outdoor_mean': 'Outdoor\nTemp'
        }

        for name in feature_names:
            if name in coefs:
                c = coefs[name]
                names.append(label_map.get(name, name))
                estimates.append(c['estimate'])
                errors.append(1.96 * c['std_error'])  # 95% CI
                # Color based on significance
                if c['p_value'] < 0.05:
                    colors_coef.append('darkblue' if c['estimate'] > 0 else 'darkred')
                else:
                    colors_coef.append('gray')

        if names:
            y_pos = np.arange(len(names))
            ax.barh(y_pos, estimates, xerr=errors, color=colors_coef, alpha=0.7,
                    capsize=5, error_kw={'linewidth': 2})
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Coefficient (°C per unit)')

            r2 = model_results.get('r_squared', 0)
            rmse = model_results.get('rmse', 0)
            n = model_results.get('n_obs', 0)
            ax.set_title(f'Model Coefficients (Daily Aggregates)\n'
                        f'R² = {r2:.2f}, RMSE = {rmse:.2f}°C, n = {n} days')
            ax.grid(True, alpha=0.3, axis='x')

            # Add annotation
            ax.text(0.02, 0.98, 'Blue/Red = significant (p<0.05)\nGray = not significant',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Model not available\n(insufficient data)',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Model Coefficients')

    # Panel 3: Predicted vs Actual (model validation)
    ax = axes[1, 0]

    if 'y_pred' in model_results and 'y_actual' in model_results:
        y_pred = model_results['y_pred']
        y_actual = model_results['y_actual']
        daily_df = model_results.get('daily_df', pd.DataFrame())

        ax.scatter(y_actual, y_pred, c='steelblue', s=50, alpha=0.7, edgecolors='white')

        # Add 1:1 line
        lims = [min(y_actual.min(), y_pred.min()) - 0.5,
                max(y_actual.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line')

        # Add regression line through points
        z = np.polyfit(y_actual, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), 'r-', linewidth=2, alpha=0.7, label=f'Fit (slope={z[0]:.2f})')

        ax.set_xlabel('Actual T_weighted (°C)')
        ax.set_ylabel('Predicted T_weighted (°C)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r2 = model_results.get('r_squared', 0)
        rmse = model_results.get('rmse', 0)
        ax.set_title(f'Model Validation: Predicted vs Actual\n'
                    f'R² = {r2:.2f}, RMSE = {rmse:.2f}°C')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'Model validation not available',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Model Validation')

    # Panel 4: Partial effects visualization
    ax = axes[1, 1]

    if 'daily_df' in model_results and 'coefficients' in model_results:
        daily_df = model_results['daily_df']
        coefs = model_results['coefficients']

        # Show how predicted T_weighted changes with comfort_setpoint
        # holding other variables at their means
        if 'comfort_setpoint' in coefs and len(daily_df) > 0:
            # Create prediction grid
            comfort_range = np.linspace(
                daily_df['comfort_setpoint'].min(),
                daily_df['comfort_setpoint'].max(),
                20
            )

            # Get means of other variables
            means = {col: daily_df[col].mean() for col in model_results.get('feature_names', [])}

            # Calculate partial effect
            intercept = coefs.get('intercept', {}).get('estimate', 0)
            base_pred = intercept
            for name, mean_val in means.items():
                if name != 'comfort_setpoint' and name in coefs:
                    base_pred += coefs[name]['estimate'] * mean_val

            comfort_coef = coefs.get('comfort_setpoint', {}).get('estimate', 0)
            comfort_se = coefs.get('comfort_setpoint', {}).get('std_error', 0)

            # Predictions for comfort setpoint
            y_comfort = base_pred + comfort_coef * comfort_range + comfort_coef * (means.get('comfort_setpoint', 20) - comfort_range) * 0  # Simplified

            # Actually compute properly
            y_comfort = base_pred + comfort_coef * (comfort_range - means.get('comfort_setpoint', 20)) + \
                       coefs.get('comfort_setpoint', {}).get('estimate', 0) * means.get('comfort_setpoint', 20)

            # Simpler: just show the slope effect
            y_base = daily_df['T_weighted_occupied'].mean()
            comfort_mean = daily_df['comfort_setpoint'].mean()
            y_comfort = y_base + comfort_coef * (comfort_range - comfort_mean)
            y_upper = y_comfort + 1.96 * comfort_se * np.abs(comfort_range - comfort_mean)
            y_lower = y_comfort - 1.96 * comfort_se * np.abs(comfort_range - comfort_mean)

            ax.plot(comfort_range, y_comfort, 'b-', linewidth=2, label='Comfort setpoint effect')
            ax.fill_between(comfort_range, y_lower, y_upper, alpha=0.2, color='blue')

            # Also show eco setpoint effect
            if 'eco_setpoint' in coefs:
                eco_range = np.linspace(
                    daily_df['eco_setpoint'].min(),
                    daily_df['eco_setpoint'].max(),
                    20
                )
                eco_coef = coefs['eco_setpoint']['estimate']
                eco_se = coefs['eco_setpoint']['std_error']
                eco_mean = daily_df['eco_setpoint'].mean()

                y_eco = y_base + eco_coef * (eco_range - eco_mean)
                y_eco_upper = y_eco + 1.96 * eco_se * np.abs(eco_range - eco_mean)
                y_eco_lower = y_eco - 1.96 * eco_se * np.abs(eco_range - eco_mean)

                ax.plot(eco_range, y_eco, 'r-', linewidth=2, label='Eco setpoint effect')
                ax.fill_between(eco_range, y_eco_lower, y_eco_upper, alpha=0.2, color='red')

            ax.axhline(y=y_base, color='gray', linestyle=':', label=f'Mean T_weighted ({y_base:.1f}°C)')
            ax.axhline(y=18.5, color='green', linestyle='--', alpha=0.5, label='Min comfort')

            ax.set_xlabel('Setpoint Temperature (°C)')
            ax.set_ylabel('Predicted T_weighted (°C)')
            ax.set_title('Partial Effects: Setpoint → T_weighted\n'
                        '(holding other variables at means)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Partial effects not available',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Partial Effects')

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


def generate_report_section(regime_stats: pd.DataFrame, sensitivity: dict, df: pd.DataFrame,
                            model_results: dict = None) -> str:
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
            <p>The <strong>indoor temperature (T_weighted)</strong> is the primary comfort objective
            for the Phase 5 intervention study. It uses the davis_inside sensor as the single
            reference temperature (least noise, central location).</p>

            <h4>Temperature Sensor</h4>
            <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem;">T_weighted = davis_inside_temperature</pre>

            <table>
                <tr><th>Sensor</th><th>Weight</th><th>Rationale</th></tr>
                <tr><td>davis_inside_temperature</td><td>100%</td><td>Primary living area, central location, least noise</td></tr>
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
            <h4>Multivariate Regression Model</h4>
            <p>A multivariate model predicts T_weighted (during occupied hours) from all controllable parameters simultaneously,
            using daily aggregates to capture parameter effects while controlling for confounding variables.</p>
"""

    # Add model results if available
    if model_results and 'coefficients' in model_results:
        r2 = model_results.get('r_squared', 0)
        r2_adj = model_results.get('r_squared_adj', 0)
        rmse = model_results.get('rmse', 0)
        n_obs = model_results.get('n_obs', 0)

        html += f"""
            <p><strong>Model Performance:</strong> R² = {r2:.3f}, Adjusted R² = {r2_adj:.3f}, RMSE = {rmse:.2f}°C, n = {n_obs} days</p>
            <table>
                <tr><th>Parameter</th><th>Coefficient</th><th>Std Error</th><th>t-stat</th><th>p-value</th><th>Interpretation</th></tr>
"""
        # Nice labels and interpretations for parameters
        label_map = {
            'intercept': 'Intercept',
            'comfort_setpoint': 'Comfort Setpoint',
            'eco_setpoint': 'Eco Setpoint',
            'curve_rise': 'Curve Rise',
            'comfort_hours': 'Comfort Hours',
            'outdoor_mean': 'Outdoor Temp'
        }
        interp_map = {
            'comfort_setpoint': '+1°C setpoint → {coef:+.2f}°C T_weighted',
            'eco_setpoint': '+1°C setpoint → {coef:+.2f}°C T_weighted',
            'curve_rise': '+0.1 curve → {coef:+.2f}°C T_weighted',
            'comfort_hours': '+1h comfort → {coef:+.2f}°C T_weighted',
            'outdoor_mean': '+1°C outdoor → {coef:+.2f}°C T_weighted'
        }

        for name, coef_data in model_results['coefficients'].items():
            label = label_map.get(name, name)
            est = coef_data['estimate']
            se = coef_data['std_error']
            t_stat = coef_data['t_stat']
            p_val = coef_data['p_value']

            # Significance stars
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

            # Interpretation
            if name in interp_map:
                if name == 'curve_rise':
                    interp = interp_map[name].format(coef=0.1*est)
                else:
                    interp = interp_map[name].format(coef=est)
            elif name == 'intercept':
                interp = f'Baseline: {est:.1f}°C'
            else:
                interp = '—'

            html += f"""                <tr>
                    <td>{label}</td>
                    <td><strong>{est:.3f}</strong></td>
                    <td>{se:.3f}</td>
                    <td>{t_stat:.2f}</td>
                    <td>{p_val:.4f} {sig}</td>
                    <td>{interp}</td>
                </tr>
"""

        html += """            </table>
            <p><em>Significance: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05</em></p>
"""
    else:
        html += """
            <p><em>Model not available (insufficient data or parameter variation).</em></p>
"""

    html += """        </div>

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
            <strong>Top row:</strong> Full time series with washout periods and regime changes (left),
            multivariate model coefficients with 95% CI (right).
            <strong>Middle row:</strong> Model validation - predicted vs actual T_weighted (left),
            partial effects showing setpoint impact on T_weighted (right).
            <strong>Bottom row:</strong> T_weighted during OCCUPIED hours (08:00-22:00) with comfort bounds (left),
            T_weighted during NIGHT hours (22:00-08:00) for reference (right).
            Orange dashed lines indicate parameter changes, red shading indicates 48h washout periods.</div>
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

    # Build multivariate model
    model_results = build_multivariate_model(df)

    # Compute simple correlations (for reference/backwards compatibility)
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

    # Save sensitivity results (simple correlations)
    sensitivity_df = pd.DataFrame([
        {'parameter': k, **v} for k, v in sensitivity.items()
    ])
    sensitivity_df.to_csv(OUTPUT_DIR / 'weighted_temp_sensitivity.csv', index=False)
    print(f"  Saved: weighted_temp_sensitivity.csv")

    # Save model coefficients if available
    if 'coefficients' in model_results:
        model_df = pd.DataFrame([
            {'parameter': k, **v} for k, v in model_results['coefficients'].items()
        ])
        model_df.to_csv(OUTPUT_DIR / 'weighted_temp_model.csv', index=False)
        print(f"  Saved: weighted_temp_model.csv")

    # Create visualization with model results
    create_visualization(df, regimes, regime_stats, model_results)

    # Generate report section
    report_html = generate_report_section(regime_stats, sensitivity, df, model_results)
    with open(OUTPUT_DIR / 'weighted_temp_report_section.html', 'w') as f:
        f.write(report_html)
    print(f"  Saved: weighted_temp_report_section.html")

    print("\n" + "=" * 60)
    print("Weighted temperature analysis complete.")
    print("=" * 60)

    return df, regimes, regime_stats, sensitivity


if __name__ == "__main__":
    main()
