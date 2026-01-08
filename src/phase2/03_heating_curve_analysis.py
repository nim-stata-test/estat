#!/usr/bin/env python3
"""
Phase 2, Step 3: Heating Curve Analysis

Analyzes how target_temperature_hk_1 depends on controllable parameters:
- comfort_temperature_target_hk1 (day setpoint)
- eco_temperature_target_hk1 (night setpoint)
- heating_curve_rise_hk1 (slope)

Key features:
- Detects comfort/eco schedule transitions from step changes in target temperature
- Estimates time-varying comfort start/end times
- Builds regression model with schedule parameters
- Generates visualizations and documentation

Output:
- eda_output/fig12_heating_curve_schedule.png (4-panel visualization)
- eda_output/heating_curve_analysis.csv (schedule regimes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'eda_output'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_heating_data():
    """Load heating sensor data in raw (long) format."""
    print("Loading heating sensor data...")
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)

    n_sensors = heating_raw['entity_id'].nunique()
    print(f"  Loaded {len(heating_raw):,} rows, {n_sensors} sensors")
    print(f"  Date range: {heating_raw['datetime'].min()} to {heating_raw['datetime'].max()}")

    return heating_raw


def load_heating_data_pivoted():
    """Load and pivot heating sensor data (for transition detection)."""
    heating_raw = load_heating_data()

    # Pivot to wide format
    heating = heating_raw.pivot_table(
        values='value',
        index='datetime',
        columns='entity_id',
        aggfunc='mean'
    )

    return heating


def detect_schedule_transitions(heating: pd.DataFrame) -> pd.DataFrame:
    """
    Detect comfort/eco transitions from step changes in target temperature.

    Returns DataFrame with columns:
    - datetime: transition time (local timezone)
    - transition: 'morning' (ECO→COMFORT) or 'evening' (COMFORT→ECO)
    - time_decimal: hour as decimal (e.g., 6.5 = 06:30)
    - magnitude: temperature change magnitude
    """
    print("\nDetecting schedule transitions...")

    target_col = 'stiebel_eltron_isg_target_temperature_hk_1'
    if target_col not in heating.columns:
        raise ValueError(f"Column {target_col} not found")

    # Get target temperature series
    target = heating[target_col].dropna().sort_index()

    # Convert to local time
    target_local = target.copy()
    target_local.index = target_local.index.tz_convert('Europe/Zurich')

    # Calculate differences
    diff = target_local.diff()
    time_diff = target_local.index.to_series().diff().dt.total_seconds() / 60

    # Identify step changes (>3°C in <30 min)
    step_mask = (abs(diff) > 3) & (time_diff < 30)
    steps = pd.DataFrame({
        'value': target_local[step_mask],
        'diff': diff[step_mask],
        'time_diff': time_diff[step_mask]
    })

    # Classify transitions
    steps['transition'] = steps['diff'].apply(lambda x: 'morning' if x > 0 else 'evening')
    steps['time_decimal'] = steps.index.hour + steps.index.minute / 60
    steps['magnitude'] = abs(steps['diff'])
    steps['date'] = steps.index.date

    print(f"  Found {len(steps)} transitions")
    print(f"    Morning (ECO→COMFORT): {(steps['transition']=='morning').sum()}")
    print(f"    Evening (COMFORT→ECO): {(steps['transition']=='evening').sum()}")

    return steps


def estimate_schedule_regimes(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate schedule regimes from transitions.

    Returns DataFrame with columns:
    - start_date: regime start date
    - end_date: regime end date
    - comfort_start: comfort period start time (decimal hours)
    - comfort_end: comfort period end time (decimal hours)
    - days: number of days in regime
    """
    print("\nEstimating schedule regimes...")

    # Build daily schedule
    daily_schedule = []
    for date in sorted(set(transitions['date'])):
        day_trans = transitions[transitions['date'] == date]
        morning = day_trans[day_trans['transition'] == 'morning']
        evening = day_trans[day_trans['transition'] == 'evening']

        if len(morning) > 0 and len(evening) > 0:
            daily_schedule.append({
                'date': date,
                'comfort_start': morning['time_decimal'].iloc[0],
                'comfort_end': evening['time_decimal'].iloc[0]
            })

    daily_df = pd.DataFrame(daily_schedule)

    if daily_df.empty:
        print("  No complete daily schedules found")
        return pd.DataFrame()

    # Round to nearest 30 minutes for regime detection
    daily_df['start_rounded'] = (daily_df['comfort_start'] * 2).round() / 2
    daily_df['end_rounded'] = (daily_df['comfort_end'] * 2).round() / 2

    # Detect regime changes
    daily_df['regime_change'] = (
        (daily_df['start_rounded'] != daily_df['start_rounded'].shift()) |
        (daily_df['end_rounded'] != daily_df['end_rounded'].shift())
    )
    daily_df.loc[daily_df.index[0], 'regime_change'] = True

    # Build regime table
    regimes = []
    regime_starts = daily_df[daily_df['regime_change']].index.tolist()

    for i, start_idx in enumerate(regime_starts):
        end_idx = regime_starts[i + 1] - 1 if i + 1 < len(regime_starts) else daily_df.index[-1]

        regime_data = daily_df.loc[start_idx:end_idx]
        regimes.append({
            'start_date': regime_data['date'].iloc[0],
            'end_date': regime_data['date'].iloc[-1],
            'comfort_start': regime_data['start_rounded'].iloc[0],
            'comfort_end': regime_data['end_rounded'].iloc[0],
            'comfort_start_mean': regime_data['comfort_start'].mean(),
            'comfort_end_mean': regime_data['comfort_end'].mean(),
            'comfort_start_std': regime_data['comfort_start'].std(),
            'comfort_end_std': regime_data['comfort_end'].std(),
            'days': len(regime_data)
        })

    regimes_df = pd.DataFrame(regimes)

    print(f"  Found {len(regimes_df)} schedule regimes:")
    for _, row in regimes_df.iterrows():
        start_h, start_m = int(row['comfort_start']), int((row['comfort_start'] % 1) * 60)
        end_h, end_m = int(row['comfort_end']), int((row['comfort_end'] % 1) * 60)
        print(f"    {row['start_date']} to {row['end_date']}: "
              f"Comfort {start_h:02d}:{start_m:02d} - {end_h:02d}:{end_m:02d} ({row['days']} days)")

    return regimes_df


def get_schedule_for_datetime(dt, regimes_df):
    """Get the comfort start/end times for a given datetime."""
    date = dt.date() if hasattr(dt, 'date') else dt

    for _, row in regimes_df.iterrows():
        if row['start_date'] <= date <= row['end_date']:
            return row['comfort_start'], row['comfort_end']

    # Default if no regime found (use most common)
    return 6.5, 20.0


def is_comfort_mode(dt, regimes_df):
    """Check if datetime is in comfort mode based on schedule regimes."""
    # Convert to local time if needed
    if hasattr(dt, 'tz') and dt.tz is not None:
        dt_local = dt.tz_convert('Europe/Zurich')
    else:
        dt_local = dt

    hour_decimal = dt_local.hour + dt_local.minute / 60
    comfort_start, comfort_end = get_schedule_for_datetime(dt_local, regimes_df)

    return comfort_start <= hour_decimal < comfort_end


def build_heating_curve_model(heating_raw: pd.DataFrame, regimes_df: pd.DataFrame) -> dict:
    """
    Build heating curve model with time-varying schedules.

    Model: T_target = T_setpoint + curve_rise * (T_ref - T_outdoor)

    Where T_setpoint is comfort_temp during comfort hours, eco_temp otherwise.
    T_ref is estimated from the data.

    Note: Takes raw (long format) heating data, not pivoted.
    """
    print("\nBuilding heating curve model...")

    # Required sensors
    sensors = {
        'target': 'stiebel_eltron_isg_target_temperature_hk_1',
        'outdoor': 'stiebel_eltron_isg_outdoor_temperature',
        'comfort': 'stiebel_eltron_isg_comfort_temperature_target_hk1',
        'eco': 'stiebel_eltron_isg_eco_temperature_target_hk1',
        'curve_rise': 'stiebel_eltron_isg_heating_curve_rise_hk1'
    }

    # Extract each sensor and resample to 1-minute
    series_dict = {}
    for name, entity_id in sensors.items():
        sensor_data = heating_raw[heating_raw['entity_id'] == entity_id].copy()
        if len(sensor_data) == 0:
            raise ValueError(f"No data for sensor {entity_id}")
        sensor_data = sensor_data.set_index('datetime')['value'].sort_index()
        # Resample to 1-minute, forward fill parameters
        if name in ['comfort', 'eco', 'curve_rise']:
            series_dict[name] = sensor_data.resample('1min').last().ffill()
        else:
            series_dict[name] = sensor_data.resample('1min').mean()

    # Combine into DataFrame
    df = pd.DataFrame(series_dict)

    # Forward/backward fill parameters to cover all timestamps
    for col in ['comfort', 'eco', 'curve_rise']:
        df[col] = df[col].ffill().bfill()

    # Drop rows where target or outdoor is missing
    df = df.dropna(subset=['target', 'outdoor'])

    if len(df) == 0:
        raise ValueError("No rows with target and outdoor temperature")

    # Convert index to local time
    df.index = df.index.tz_convert('Europe/Zurich')

    # Determine comfort/eco mode for each row using vectorized approach
    hour_decimal = df.index.hour + df.index.minute / 60
    dates = df.index.date

    # Build is_comfort array
    is_comfort = np.zeros(len(df), dtype=bool)
    for _, row in regimes_df.iterrows():
        mask = (dates >= row['start_date']) & (dates <= row['end_date'])
        is_comfort[mask] = (hour_decimal[mask] >= row['comfort_start']) & (hour_decimal[mask] < row['comfort_end'])

    # For dates outside regime ranges, use default schedule
    dates_in_regime = np.zeros(len(df), dtype=bool)
    for _, row in regimes_df.iterrows():
        dates_in_regime |= (dates >= row['start_date']) & (dates <= row['end_date'])

    # Default schedule (6:30 - 20:00) for dates not in any regime
    default_mask = ~dates_in_regime
    is_comfort[default_mask] = (hour_decimal[default_mask] >= 6.5) & (hour_decimal[default_mask] < 20.0)

    df['is_comfort'] = is_comfort
    df['setpoint'] = np.where(df['is_comfort'], df['comfort'], df['eco'])

    print(f"  Analysis dataset: {len(df):,} rows")
    print(f"  Comfort mode: {df['is_comfort'].sum():,} ({df['is_comfort'].mean()*100:.1f}%)")
    print(f"  Eco mode: {(~df['is_comfort']).sum():,} ({(~df['is_comfort']).mean()*100:.1f}%)")

    # Estimate reference temperature for each mode
    # T_target = T_setpoint + curve_rise * (T_ref - T_outdoor)
    # Rearrange: T_ref = (T_target - T_setpoint) / curve_rise + T_outdoor

    df['implied_ref'] = (df['target'] - df['setpoint']) / df['curve_rise'] + df['outdoor']

    t_ref_comfort = df[df['is_comfort']]['implied_ref'].median()
    t_ref_eco = df[~df['is_comfort']]['implied_ref'].median()

    print(f"  Estimated T_ref (comfort): {t_ref_comfort:.2f}°C")
    print(f"  Estimated T_ref (eco): {t_ref_eco:.2f}°C")

    # Calculate predicted target temperature
    df['t_ref'] = np.where(df['is_comfort'], t_ref_comfort, t_ref_eco)
    df['predicted'] = df['setpoint'] + df['curve_rise'] * (df['t_ref'] - df['outdoor'])
    df['residual'] = df['target'] - df['predicted']

    # Model statistics
    ss_res = (df['residual'] ** 2).sum()
    ss_tot = ((df['target'] - df['target'].mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt((df['residual'] ** 2).mean())

    # By mode
    comfort_rmse = np.sqrt((df[df['is_comfort']]['residual'] ** 2).mean())
    eco_rmse = np.sqrt((df[~df['is_comfort']]['residual'] ** 2).mean())

    print(f"\n  Model Performance:")
    print(f"    R-squared: {r_squared:.4f}")
    print(f"    RMSE: {rmse:.2f}°C")
    print(f"    RMSE (comfort): {comfort_rmse:.2f}°C")
    print(f"    RMSE (eco): {eco_rmse:.2f}°C")

    results = {
        'data': df,
        't_ref_comfort': t_ref_comfort,
        't_ref_eco': t_ref_eco,
        'r_squared': r_squared,
        'rmse': rmse,
        'rmse_comfort': comfort_rmse,
        'rmse_eco': eco_rmse
    }

    return results


def create_visualization(transitions: pd.DataFrame,
                         regimes_df: pd.DataFrame, model_results: dict):
    """Create 4-panel visualization of heating curve analysis."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    df = model_results['data']

    # Panel 1: Schedule transitions over time
    ax = axes[0, 0]

    morning = transitions[transitions['transition'] == 'morning']
    evening = transitions[transitions['transition'] == 'evening']

    ax.scatter(morning.index, morning['time_decimal'],
               c='orange', s=30, alpha=0.7, label='Morning (ECO→COMFORT)')
    ax.scatter(evening.index, evening['time_decimal'],
               c='blue', s=30, alpha=0.7, label='Evening (COMFORT→ECO)')

    # Add regime boundaries
    for _, row in regimes_df.iterrows():
        start_date = pd.Timestamp(row['start_date'], tz='Europe/Zurich')
        ax.axvline(x=start_date, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_ylabel('Time of Day (hours)')
    ax.set_xlabel('Date')
    ax.set_title('Detected Schedule Transitions')
    ax.set_ylim(0, 24)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_yticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Target temperature vs outdoor temperature
    ax = axes[0, 1]

    comfort_data = df[df['is_comfort']]
    eco_data = df[~df['is_comfort']]

    # Sample for plotting (avoid overplotting)
    sample_n = min(2000, len(comfort_data), len(eco_data))
    if len(comfort_data) > sample_n:
        comfort_sample = comfort_data.sample(sample_n)
    else:
        comfort_sample = comfort_data
    if len(eco_data) > sample_n:
        eco_sample = eco_data.sample(sample_n)
    else:
        eco_sample = eco_data

    ax.scatter(comfort_sample['outdoor'], comfort_sample['target'],
               c='orange', s=10, alpha=0.3, label='Comfort mode')
    ax.scatter(eco_sample['outdoor'], eco_sample['target'],
               c='blue', s=10, alpha=0.3, label='Eco mode')

    # Add model lines
    outdoor_range = np.linspace(df['outdoor'].min(), df['outdoor'].max(), 100)

    # Use median setpoints for visualization
    comfort_setpoint = df[df['is_comfort']]['comfort'].median()
    eco_setpoint = df[~df['is_comfort']]['eco'].median()
    curve_rise = df['curve_rise'].median()

    comfort_line = comfort_setpoint + curve_rise * (model_results['t_ref_comfort'] - outdoor_range)
    eco_line = eco_setpoint + curve_rise * (model_results['t_ref_eco'] - outdoor_range)

    ax.plot(outdoor_range, comfort_line, 'orange', linewidth=2, linestyle='--',
            label=f'Comfort model (setpoint={comfort_setpoint:.1f}°C)')
    ax.plot(outdoor_range, eco_line, 'blue', linewidth=2, linestyle='--',
            label=f'Eco model (setpoint={eco_setpoint:.1f}°C)')

    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('Target Flow Temperature (°C)')
    ax.set_title('Heating Curve: Target vs Outdoor Temperature')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Model residuals
    ax = axes[1, 0]

    # Hourly residuals
    df['hour'] = df.index.hour
    hourly_residuals = df.groupby('hour')['residual'].agg(['mean', 'std'])

    ax.bar(hourly_residuals.index, hourly_residuals['mean'],
           yerr=hourly_residuals['std'], capsize=2, alpha=0.7, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Mark comfort/eco regions (using first regime as example)
    comfort_start = regimes_df.iloc[0]['comfort_start']
    comfort_end = regimes_df.iloc[0]['comfort_end']
    ax.axvspan(comfort_start, comfort_end, alpha=0.2, color='orange', label='Comfort period')
    ax.axvspan(0, comfort_start, alpha=0.2, color='blue', label='Eco period')
    ax.axvspan(comfort_end, 24, alpha=0.2, color='blue')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Residual (°C)')
    ax.set_title(f'Model Residuals by Hour (RMSE={model_results["rmse"]:.2f}°C, R²={model_results["r_squared"]:.3f})')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 4: Schedule regimes table and parameter effects
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    summary_text = "HEATING CURVE MODEL\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += "Formula:\n"
    summary_text += "  T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)\n\n"
    summary_text += "Parameters:\n"
    summary_text += f"  • T_ref (comfort): {model_results['t_ref_comfort']:.2f}°C\n"
    summary_text += f"  • T_ref (eco): {model_results['t_ref_eco']:.2f}°C\n"
    summary_text += f"  • Curve rise (median): {curve_rise:.2f}\n\n"
    summary_text += "Schedule Regimes:\n"

    for _, row in regimes_df.iterrows():
        start_h, start_m = int(row['comfort_start']), int((row['comfort_start'] % 1) * 60)
        end_h, end_m = int(row['comfort_end']), int((row['comfort_end'] % 1) * 60)
        summary_text += f"  • {row['start_date']} to {row['end_date']}:\n"
        summary_text += f"    Comfort {start_h:02d}:{start_m:02d} - {end_h:02d}:{end_m:02d} ({row['days']} days)\n"

    summary_text += f"\nModel Performance:\n"
    summary_text += f"  • R²: {model_results['r_squared']:.4f}\n"
    summary_text += f"  • RMSE overall: {model_results['rmse']:.2f}°C\n"
    summary_text += f"  • RMSE comfort: {model_results['rmse_comfort']:.2f}°C\n"
    summary_text += f"  • RMSE eco: {model_results['rmse_eco']:.2f}°C\n"

    summary_text += f"\nParameter Effects:\n"
    summary_text += f"  • +1°C setpoint → +1°C target\n"
    summary_text += f"  • +0.1 curve rise → +{0.1 * (model_results['t_ref_comfort'] - 5):.1f}°C target (at 5°C outdoor)\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_heating_curve_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig12_heating_curve_schedule.png")


def generate_report_section(regimes_df: pd.DataFrame, model_results: dict) -> str:
    """Generate HTML section for EDA report."""

    df = model_results['data']
    curve_rise = df['curve_rise'].median()
    comfort_setpoint = df[df['is_comfort']]['comfort'].median()
    eco_setpoint = df[~df['is_comfort']]['eco'].median()

    html = """
        <h2 id="heating-curve">8. Heating Curve Analysis</h2>

        <div class="card">
            <h4>Model Overview</h4>
            <p>The target flow temperature for heating circuit 1 (HK1) follows a linear heating curve formula:</p>
            <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem;">T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)</pre>

            <h4>Parameters</h4>
            <table>
                <tr><th>Parameter</th><th>Description</th><th>Value</th></tr>
                <tr>
                    <td><strong>T_setpoint</strong></td>
                    <td>Room temperature setpoint (comfort or eco depending on schedule)</td>
                    <td>Comfort: {comfort_setpoint:.1f}°C, Eco: {eco_setpoint:.1f}°C</td>
                </tr>
                <tr>
                    <td><strong>curve_rise</strong></td>
                    <td>Heating curve slope (controllable parameter)</td>
                    <td>{curve_rise:.2f} (range: {curve_min:.2f} - {curve_max:.2f})</td>
                </tr>
                <tr>
                    <td><strong>T_ref</strong></td>
                    <td>Internal reference temperature (derived from data)</td>
                    <td>Comfort: {t_ref_comfort:.2f}°C, Eco: {t_ref_eco:.2f}°C</td>
                </tr>
                <tr>
                    <td><strong>T_outdoor</strong></td>
                    <td>Outdoor temperature sensor</td>
                    <td>Range: {outdoor_min:.1f}°C to {outdoor_max:.1f}°C</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h4>Schedule Regimes (Detected from Data)</h4>
            <p>The comfort/eco schedule was detected by analyzing step changes (>3°C) in the target temperature.
            The system automatically switches between comfort (higher setpoint) during the day and eco (lower setpoint) at night.</p>
            <table>
                <tr><th>Period</th><th>Comfort Start</th><th>Comfort End</th><th>Duration</th><th>Days</th></tr>
{schedule_rows}
            </table>
            <p><em>Note: Times are in local timezone (Europe/Zurich). Comfort mode uses the higher setpoint;
            eco mode (outside comfort hours) uses the lower setpoint.</em></p>
        </div>

        <div class="card">
            <h4>Model Performance</h4>
            <table>
                <tr><td>R-squared</td><td><strong>{r_squared:.4f}</strong> (explains {r_pct:.1f}% of variance)</td></tr>
                <tr><td>RMSE (overall)</td><td>{rmse:.2f}°C</td></tr>
                <tr><td>RMSE (comfort mode)</td><td>{rmse_comfort:.2f}°C</td></tr>
                <tr><td>RMSE (eco mode)</td><td>{rmse_eco:.2f}°C</td></tr>
            </table>

            <h4>Parameter Effects</h4>
            <ul>
                <li><strong>Setpoint</strong>: +1°C setpoint → +1°C target flow temperature (direct relationship)</li>
                <li><strong>Curve rise</strong>: +0.1 rise → +{effect_5c:.1f}°C target at 5°C outdoor, +{effect_0c:.1f}°C at 0°C outdoor</li>
                <li><strong>Outdoor temperature</strong>: +1°C outdoor → -{curve_rise:.2f}°C target (inverse relationship via curve)</li>
            </ul>
        </div>

        <div class="card">
            <h4>Example Calculations</h4>
            <p>For outdoor temperature = 5°C, comfort_temp = {comfort_setpoint:.1f}°C, eco_temp = {eco_setpoint:.1f}°C, curve_rise = {curve_rise:.2f}:</p>
            <table>
                <tr><th>Outdoor</th><th>Comfort Target</th><th>Eco Target</th></tr>
                <tr><td>-3°C</td><td>{t_neg3_comfort:.1f}°C</td><td>{t_neg3_eco:.1f}°C</td></tr>
                <tr><td>0°C</td><td>{t_0_comfort:.1f}°C</td><td>{t_0_eco:.1f}°C</td></tr>
                <tr><td>5°C</td><td>{t_5_comfort:.1f}°C</td><td>{t_5_eco:.1f}°C</td></tr>
                <tr><td>10°C</td><td>{t_10_comfort:.1f}°C</td><td>{t_10_eco:.1f}°C</td></tr>
                <tr><td>15°C</td><td>{t_15_comfort:.1f}°C</td><td>{t_15_eco:.1f}°C</td></tr>
            </table>
        </div>

        <div class="figure">
            <img src="fig12_heating_curve_schedule.png" alt="Heating curve analysis with schedule detection">
            <div class="figure-caption">Heating curve analysis: schedule transitions (top-left), target vs outdoor temperature by mode (top-right),
            model residuals by hour (bottom-left), and model summary (bottom-right)</div>
        </div>
"""

    # Build schedule rows
    schedule_rows = ""
    for _, row in regimes_df.iterrows():
        start_h, start_m = int(row['comfort_start']), int((row['comfort_start'] % 1) * 60)
        end_h, end_m = int(row['comfort_end']), int((row['comfort_end'] % 1) * 60)
        duration_h = row['comfort_end'] - row['comfort_start']
        schedule_rows += f"""                <tr>
                    <td>{row['start_date']} to {row['end_date']}</td>
                    <td>{start_h:02d}:{start_m:02d}</td>
                    <td>{end_h:02d}:{end_m:02d}</td>
                    <td>{duration_h:.1f} hours</td>
                    <td>{row['days']}</td>
                </tr>
"""

    # Calculate example temperatures
    t_ref_comfort = model_results['t_ref_comfort']
    t_ref_eco = model_results['t_ref_eco']

    def calc_target(outdoor, setpoint, t_ref):
        return setpoint + curve_rise * (t_ref - outdoor)

    html = html.format(
        comfort_setpoint=comfort_setpoint,
        eco_setpoint=eco_setpoint,
        curve_rise=curve_rise,
        curve_min=df['curve_rise'].min(),
        curve_max=df['curve_rise'].max(),
        t_ref_comfort=t_ref_comfort,
        t_ref_eco=t_ref_eco,
        outdoor_min=df['outdoor'].min(),
        outdoor_max=df['outdoor'].max(),
        schedule_rows=schedule_rows,
        r_squared=model_results['r_squared'],
        r_pct=model_results['r_squared'] * 100,
        rmse=model_results['rmse'],
        rmse_comfort=model_results['rmse_comfort'],
        rmse_eco=model_results['rmse_eco'],
        effect_5c=0.1 * (t_ref_comfort - 5),
        effect_0c=0.1 * (t_ref_comfort - 0),
        t_neg3_comfort=calc_target(-3, comfort_setpoint, t_ref_comfort),
        t_neg3_eco=calc_target(-3, eco_setpoint, t_ref_eco),
        t_0_comfort=calc_target(0, comfort_setpoint, t_ref_comfort),
        t_0_eco=calc_target(0, eco_setpoint, t_ref_eco),
        t_5_comfort=calc_target(5, comfort_setpoint, t_ref_comfort),
        t_5_eco=calc_target(5, eco_setpoint, t_ref_eco),
        t_10_comfort=calc_target(10, comfort_setpoint, t_ref_comfort),
        t_10_eco=calc_target(10, eco_setpoint, t_ref_eco),
        t_15_comfort=calc_target(15, comfort_setpoint, t_ref_comfort),
        t_15_eco=calc_target(15, eco_setpoint, t_ref_eco),
    )

    return html


def main():
    """Run heating curve analysis."""
    print("=" * 60)
    print("HEATING CURVE ANALYSIS")
    print("=" * 60)

    # Load data - raw format for model, pivoted for transitions
    heating_raw = load_heating_data()

    # Pivot for transition detection
    print("\nPivoting data for transition detection...")
    heating_pivoted = heating_raw.pivot_table(
        values='value',
        index='datetime',
        columns='entity_id',
        aggfunc='mean'
    )

    # Detect schedule transitions (needs pivoted data)
    transitions = detect_schedule_transitions(heating_pivoted)

    # Estimate schedule regimes
    regimes_df = estimate_schedule_regimes(transitions)

    # Save regimes to CSV
    regimes_df.to_csv(OUTPUT_DIR / 'heating_curve_schedules.csv', index=False)
    print(f"\n  Saved: heating_curve_schedules.csv")

    # Build model (uses raw data for proper resampling)
    model_results = build_heating_curve_model(heating_raw, regimes_df)

    # Create visualization
    create_visualization(transitions, regimes_df, model_results)

    # Generate report section
    report_html = generate_report_section(regimes_df, model_results)

    # Save report section for integration
    with open(OUTPUT_DIR / 'heating_curve_report_section.html', 'w') as f:
        f.write(report_html)
    print(f"  Saved: heating_curve_report_section.html")

    print("\n" + "=" * 60)
    print("Heating curve analysis complete.")
    print("=" * 60)

    return regimes_df, model_results


if __name__ == "__main__":
    main()
