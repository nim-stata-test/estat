#!/usr/bin/env python3
"""
Phase 3: Extended Model Decomposition

Creates comprehensive decomposition figures showing:
1. Room temperature (actual vs predicted)
2. Outdoor temperature
3. Heating effort contribution
4. Solar/PV contribution
5. Battery state of charge
6. Power consumption
7. Grid feed-in
8. Grid import
9. Heat pump COP

This replaces fig18c and is placed after fig21.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
WEEKLY_DIR = OUTPUT_DIR / 'weekly_decomposition'
OUTPUT_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)

# Figure style
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
})

COLORS = {
    'actual': '#1e4557',
    'predicted': '#2a9749',
    'outdoor': '#079bca',
    'heating': '#E85A4F',
    'solar': '#FFB800',
    'battery': '#2a9749',
    'consumption': '#9156b4',
    'grid_import': '#E85A4F',
    'grid_export': '#2a9749',
    'cop': '#1e4557',
    'baseline': '#bababa',
}


def load_data():
    """Load all required datasets."""
    print("Loading data...")

    # Load integrated dataset
    integrated = pd.read_parquet(DATA_DIR / 'integrated_dataset.parquet')
    integrated.index = pd.to_datetime(integrated.index)
    if integrated.index.tz is not None:
        integrated.index = integrated.index.tz_localize(None)

    # Load energy balance data
    energy = pd.read_parquet(DATA_DIR / 'energy_balance_15min.parquet')
    energy.index = pd.to_datetime(energy.index)
    if energy.index.tz is not None:
        energy.index = energy.index.tz_localize(None)

    # Load heating sensors - need to pivot to wide format
    heating_raw = pd.read_parquet(DATA_DIR / 'sensors_heating.parquet')
    if 'datetime' in heating_raw.columns:
        # Pivot to wide format with datetime as index
        heating = heating_raw.pivot_table(
            index='datetime', columns='entity_id', values='value', aggfunc='first'
        )
        heating.index = pd.to_datetime(heating.index)
        if heating.index.tz is not None:
            heating.index = heating.index.tz_localize(None)
    else:
        heating = heating_raw
        heating.index = pd.to_datetime(heating.index)
        if heating.index.tz is not None:
            heating.index = heating.index.tz_localize(None)

    print(f"  Integrated: {len(integrated)} rows")
    print(f"  Energy: {len(energy)} rows")
    print(f"  Heating: {len(heating)} rows")

    return integrated, energy, heating


def load_thermal_model_params():
    """Load thermal model parameters from CSV."""
    params_file = OUTPUT_DIR / 'thermal_model_results.csv'
    if params_file.exists():
        df = pd.read_csv(params_file)
        # Use davis_inside parameters
        row = df[df['room'] == 'davis_inside'].iloc[0]
        return {
            'c0': row['offset'],
            'tau_out': row['tau_outdoor_h'],
            'tau_eff': row['tau_effort_h'],
            'tau_pv': row['tau_pv_h'],
            'g_out': row['gain_outdoor'],
            'g_eff': row['gain_effort'],
            'g_pv': row['gain_pv'],
        }
    return None


def load_heating_curve_params():
    """Load heating curve parameters."""
    params_file = OUTPUT_DIR.parent / 'phase2' / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            return json.load(f)
    return {'intercept': 39.9, 'slope': -0.869}


def apply_lpf(x, tau_hours, dt_hours=0.25):
    """Apply first-order low-pass filter."""
    tau_samples = tau_hours / dt_hours
    alpha = 1 - np.exp(-1 / tau_samples) if tau_samples > 0 else 1

    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            y[i] = y[i-1]
        else:
            y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def compute_model_terms(df, params, hc_params):
    """Compute thermal model terms for decomposition."""
    # Get outdoor temperature
    t_out = df['stiebel_eltron_isg_outdoor_temperature'].values

    # Get HK2 target temperature (flow temperature setpoint)
    t_hk2 = df['stiebel_eltron_isg_target_temperature_hk_2'].values

    # Compute heating effort (deviation from heating curve)
    t_hk2_expected = hc_params.get('intercept', 39.9) + hc_params.get('slope', -0.869) * t_out
    effort = t_hk2 - t_hk2_expected

    # Get PV generation (column name is pv_generation_kwh)
    pv = df['pv_generation_kwh'].values if 'pv_generation_kwh' in df.columns else np.zeros(len(df))

    # Apply low-pass filters
    lpf_out = apply_lpf(t_out, params['tau_out'])
    lpf_eff = apply_lpf(effort, params['tau_eff'])
    lpf_pv = apply_lpf(pv, params['tau_pv'])

    # Compute contributions
    contrib_out = params['g_out'] * lpf_out
    contrib_eff = params['g_eff'] * lpf_eff
    contrib_pv = params['g_pv'] * lpf_pv

    # Predicted temperature
    t_pred = params['c0'] + contrib_out + contrib_eff + contrib_pv

    return {
        't_out': t_out,
        'effort': effort,
        'pv': pv,
        'lpf_out': lpf_out,
        'lpf_eff': lpf_eff,
        'lpf_pv': lpf_pv,
        'contrib_out': contrib_out,
        'contrib_eff': contrib_eff,
        'contrib_pv': contrib_pv,
        't_pred': t_pred,
    }


def create_extended_decomposition(df, energy_df, heating_df, params, hc_params,
                                   start_date, end_date, output_path, title_suffix=''):
    """Create extended decomposition figure with all panels in single column layout."""

    # Filter data to date range
    mask = (df.index >= start_date) & (df.index < end_date)
    df_week = df[mask].copy()

    if len(df_week) < 10:
        print(f"  Insufficient data for {start_date}")
        return None

    # Get energy data for the same period
    energy_mask = (energy_df.index >= start_date) & (energy_df.index < end_date)
    energy_week = energy_df[energy_mask].copy()

    # Get heating data for the same period
    heating_mask = (heating_df.index >= start_date) & (heating_df.index < end_date)
    heating_week = heating_df[heating_mask].copy()

    # Compute model terms
    terms = compute_model_terms(df_week, params, hc_params)

    # Get actual room temperature
    t_actual = df_week['davis_inside_temperature'].values

    # Create figure with 10 panels in single column
    fig, axes = plt.subplots(10, 1, figsize=(14, 28))
    fig.suptitle(f'Extended Model Decomposition{title_suffix}', fontsize=14, fontweight='bold', y=0.995)

    time_idx = df_week.index

    # === Panel 1: Room Temperature (actual vs predicted) ===
    ax = axes[0]
    ax.plot(time_idx, t_actual, color=COLORS['actual'], label='Observed', linewidth=1.5)
    ax.plot(time_idx, terms['t_pred'], color=COLORS['predicted'], label='Predicted',
            linewidth=1.5, linestyle='--')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('1. Room Temperature: Observed vs Predicted')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Calculate R² for this period
    valid = ~(np.isnan(t_actual) | np.isnan(terms['t_pred']))
    if valid.sum() > 10:
        ss_res = np.sum((t_actual[valid] - terms['t_pred'][valid])**2)
        ss_tot = np.sum((t_actual[valid] - np.mean(t_actual[valid]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((t_actual[valid] - terms['t_pred'][valid])**2))
        ax.text(0.02, 0.95, f'R²={r2:.3f}, RMSE={rmse:.2f}°C',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === Panel 2: Outdoor Temperature Contribution ===
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(time_idx, terms['t_out'], color=COLORS['outdoor'], alpha=0.6,
            linewidth=1, label='T_outdoor (raw)')
    ax2.plot(time_idx, terms['contrib_out'], color=COLORS['outdoor'],
             linewidth=1.5, label=f'Contribution (g={params["g_out"]:.3f})')
    ax.set_ylabel('Outdoor Temp (°C)', color=COLORS['outdoor'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['outdoor'])
    ax.set_title(f'2. Outdoor Temperature → Room Temp (τ={params["tau_out"]:.0f}h, g={params["g_out"]:.3f})')
    ax.grid(True, alpha=0.3)

    # === Panel 3: Heating Effort Contribution ===
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(time_idx, terms['effort'], color=COLORS['heating'], alpha=0.4,
            linewidth=0.8, label='Effort (raw)')
    ax2.plot(time_idx, terms['contrib_eff'], color=COLORS['heating'],
             linewidth=1.5, label=f'Contribution (g={params["g_eff"]:.3f})')
    ax.set_ylabel('Heating Effort (°C)', color=COLORS['heating'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['heating'])
    ax.set_title(f'3. Heating Effort → Room Temp (τ={params["tau_eff"]:.0f}h, g={params["g_eff"]:.3f})')
    ax.grid(True, alpha=0.3)

    # === Panel 4: Solar/PV Contribution ===
    ax = axes[3]
    ax2 = ax.twinx()
    ax.fill_between(time_idx, 0, terms['pv'], color=COLORS['solar'], alpha=0.3, label='PV (raw)')
    ax2.plot(time_idx, terms['contrib_pv'], color=COLORS['solar'],
             linewidth=1.5, label=f'Contribution (g={params["g_pv"]:.3f})')
    ax.set_ylabel('PV Generation (kW)', color=COLORS['solar'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['solar'])
    ax.set_title(f'4. Solar Gain → Room Temp (τ={params["tau_pv"]:.0f}h, g={params["g_pv"]:.3f})')
    ax.grid(True, alpha=0.3)

    # === Panel 5: Battery State of Charge ===
    ax = axes[4]
    if len(energy_week) > 0 and 'battery_charging_kwh' in energy_week.columns:
        # Calculate cumulative battery state (observed)
        charge = energy_week['battery_charging_kwh'].fillna(0).values
        discharge = energy_week['battery_discharging_kwh'].fillna(0).values
        net_flow = charge - discharge  # Positive = charging

        # Cumulative SoC (observed) - normalized around 50%
        cumsum = np.cumsum(net_flow)
        soc_observed = 50 + (cumsum - np.mean(cumsum)) / 10 * 100
        soc_observed = np.clip(soc_observed, 0, 100)

        # "Predicted" baseline: what we'd expect from daily pattern (24h rolling mean)
        soc_baseline = uniform_filter1d(soc_observed, size=96, mode='nearest')

        ax.fill_between(energy_week.index, 0, soc_observed, color=COLORS['battery'],
                        alpha=0.3, label='Observed')
        ax.plot(energy_week.index, soc_baseline, color=COLORS['baseline'],
                linewidth=2, linestyle='--', label='24h baseline')
        ax.set_ylabel('Est. SoC (%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No battery data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('5. Battery State of Charge: Observed vs Baseline')
    ax.grid(True, alpha=0.3)

    # === Panel 6: Power Consumption ===
    ax = axes[5]
    if len(energy_week) > 0 and 'total_consumption_kwh' in energy_week.columns:
        consumption = energy_week['total_consumption_kwh'].fillna(0).values
        # 24-hour moving average as "predicted/baseline"
        baseline = uniform_filter1d(consumption, size=96, mode='nearest')

        ax.fill_between(energy_week.index, 0, consumption, color=COLORS['consumption'],
                        alpha=0.3, label='Observed')
        ax.plot(energy_week.index, baseline, color=COLORS['baseline'],
                linewidth=2, linestyle='--', label='24h baseline')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No consumption data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('6. Power Consumption: Observed vs Baseline')
    ax.grid(True, alpha=0.3)

    # === Panel 7: Grid Feed-in ===
    ax = axes[6]
    if len(energy_week) > 0 and 'grid_feedin_kwh' in energy_week.columns:
        feedin = energy_week['grid_feedin_kwh'].fillna(0).values
        baseline = uniform_filter1d(feedin, size=96, mode='nearest')

        ax.fill_between(energy_week.index, 0, feedin, color=COLORS['grid_export'],
                        alpha=0.3, label='Observed')
        ax.plot(energy_week.index, baseline, color=COLORS['baseline'],
                linewidth=2, linestyle='--', label='24h baseline')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No feed-in data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('7. Grid Feed-in: Observed vs Baseline')
    ax.grid(True, alpha=0.3)

    # === Panel 8: Grid Import ===
    ax = axes[7]
    if len(energy_week) > 0 and 'external_supply_kwh' in energy_week.columns:
        grid_import = energy_week['external_supply_kwh'].fillna(0).values
        baseline = uniform_filter1d(grid_import, size=96, mode='nearest')

        ax.fill_between(energy_week.index, 0, grid_import, color=COLORS['grid_import'],
                        alpha=0.3, label='Observed')
        ax.plot(energy_week.index, baseline, color=COLORS['baseline'],
                linewidth=2, linestyle='--', label='24h baseline')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No grid import data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('8. Grid Import: Observed vs Baseline')
    ax.grid(True, alpha=0.3)

    # === Panel 9: Outdoor Temperature ===
    ax = axes[8]
    ax.plot(time_idx, terms['t_out'], color=COLORS['outdoor'], linewidth=1.2, label='Observed')
    mean_temp = np.nanmean(terms['t_out'])
    ax.axhline(y=mean_temp, color=COLORS['baseline'], linestyle='--',
               label=f'Mean: {mean_temp:.1f}°C')
    ax.fill_between(time_idx, mean_temp, terms['t_out'],
                    where=terms['t_out'] > mean_temp,
                    color=COLORS['heating'], alpha=0.2, label='Above mean')
    ax.fill_between(time_idx, mean_temp, terms['t_out'],
                    where=terms['t_out'] < mean_temp,
                    color=COLORS['outdoor'], alpha=0.2, label='Below mean')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(loc='upper right')
    ax.set_title('9. Outdoor Temperature')
    ax.grid(True, alpha=0.3)

    # === Panel 10: Heat Pump COP ===
    ax = axes[9]
    if len(heating_week) > 0:
        # Calculate instantaneous COP from today's counters
        q_heat = heating_week.get('stiebel_eltron_isg_produced_heating_today', pd.Series())
        e_elec = heating_week.get('stiebel_eltron_isg_consumed_heating_today', pd.Series())

        if len(q_heat) > 0 and len(e_elec) > 0 and q_heat.notna().sum() > 10:
            # Use differences for incremental COP calculation
            q_diff = q_heat.diff().fillna(0)
            e_diff = e_elec.diff().fillna(0)

            # Calculate COP where both are positive
            valid_cop = (e_diff > 0.01) & (q_diff > 0)
            cop = pd.Series(index=heating_week.index, dtype=float)
            cop[valid_cop] = q_diff[valid_cop] / e_diff[valid_cop]
            cop = cop.clip(1, 8)  # Reasonable COP range

            if valid_cop.sum() > 0:
                # Smooth COP
                cop_smooth = cop.rolling(window=16, min_periods=1).mean()

                ax.scatter(heating_week.index[valid_cop], cop[valid_cop],
                          color=COLORS['cop'], alpha=0.2, s=8, label='Instantaneous')
                ax.plot(heating_week.index, cop_smooth, color=COLORS['cop'],
                       linewidth=2, label='Smoothed (4h)')
                mean_cop = cop[valid_cop].mean()
                ax.axhline(y=mean_cop, color=COLORS['baseline'], linestyle='--',
                          label=f'Mean: {mean_cop:.2f}')
                ax.set_ylabel('COP')
                ax.set_ylim(1, 7)
                ax.legend(loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No valid COP values', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Insufficient COP data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No heating data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
    ax.set_title('10. Heat Pump COP')
    ax.grid(True, alpha=0.3)

    # Format x-axes for all panels
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def main():
    """Generate extended decomposition figures."""
    print("=" * 60)
    print("Phase 3: Extended Model Decomposition")
    print("=" * 60)

    # Load data
    integrated, energy, heating = load_data()

    # Load model parameters
    params = load_thermal_model_params()
    if params is None:
        print("ERROR: Could not load thermal model parameters")
        print("Run 01_thermal_model.py first")
        return

    print(f"\nModel parameters loaded:")
    print(f"  τ_out={params['tau_out']}h, τ_eff={params['tau_eff']}h, τ_pv={params['tau_pv']}h")
    print(f"  g_out={params['g_out']:.3f}, g_eff={params['g_eff']:.3f}, g_pv={params['g_pv']:.3f}")

    # Load heating curve parameters
    hc_params = load_heating_curve_params()

    # Find overlapping data period
    overlap_start = max(integrated.index.min(), energy.index.min(), heating.index.min())
    overlap_end = min(integrated.index.max(), energy.index.max(), heating.index.max())
    print(f"\nData overlap: {overlap_start.date()} to {overlap_end.date()}")

    # Generate main extended decomposition figure (representative week)
    print("\nGenerating main extended decomposition figure...")

    # Find a representative week with good data
    # Use the most recent complete week
    end_date = overlap_end
    start_date = end_date - pd.Timedelta(days=7)

    output_path = OUTPUT_DIR / 'fig22_extended_decomposition.png'
    success = create_extended_decomposition(
        integrated, energy, heating, params, hc_params,
        start_date, end_date, output_path,
        title_suffix=f'\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
    )

    if success:
        print(f"  Saved: {output_path.name}")

    # Generate weekly decomposition figures
    print("\nGenerating weekly extended decomposition figures...")

    # Find all weeks with sufficient data
    weeks = []
    current = overlap_start
    while current < overlap_end:
        week_end = current + pd.Timedelta(days=7)
        if week_end <= overlap_end:
            weeks.append((current, week_end))
        current = week_end

    print(f"  Found {len(weeks)} weeks")

    for i, (start, end) in enumerate(weeks):
        week_num = i + 1
        output_path = WEEKLY_DIR / f'week_{week_num:02d}_extended.png'

        success = create_extended_decomposition(
            integrated, energy, heating, params, hc_params,
            start, end, output_path,
            title_suffix=f' - Week {week_num}\n{start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}'
        )

        if success:
            print(f"  Week {week_num}: {start.strftime('%Y-%m-%d')} - Saved")
        else:
            print(f"  Week {week_num}: {start.strftime('%Y-%m-%d')} - Skipped (insufficient data)")

    print("\n" + "=" * 60)
    print("EXTENDED DECOMPOSITION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'fig22_extended_decomposition.png'}")
    print(f"  - {WEEKLY_DIR / 'week_XX_extended.png'} (multiple)")


if __name__ == '__main__':
    main()
