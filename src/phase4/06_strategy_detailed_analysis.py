#!/usr/bin/env python3
"""
Phase 4, Step 6: Detailed Strategy Analysis for Phase 5

Generates comprehensive summary statistics and visualizations for the strategies
selected for Phase 5 intervention study.

NOW INCLUDES PROPER ENERGY SIMULATION:
- Uses the same energy model as Pareto optimization
- Simulates grid consumption for each strategy
- Shows meaningful energy differences between strategies

Outputs:
- fig28_strategy_detailed_timeseries.png
- fig29_strategy_hourly_patterns.png
- fig30_strategy_energy_patterns.png
- strategy_detailed_stats.csv
- strategy_detailed_report.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE4_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR = PHASE4_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL CONSTANTS (must match 04_pareto_optimization.py)
# =============================================================================

# COP model from Phase 3
COP_PARAMS = {
    'intercept': 6.52,
    'outdoor_coef': 0.1319,
    'flow_coef': -0.1007,
}

# Heating curve reference temperatures from Phase 2
HEATING_CURVE_PARAMS = {
    't_ref_comfort': 21.32,
    't_ref_eco': 19.18,
}

# Energy model constants (calibrated from historical data)
BASE_LOAD_KWH = 11.0   # Non-heating daily consumption
THERMAL_COEF = 10.0    # Thermal kWh per HDD (before COP division)

# T_weighted regression coefficients from Phase 2
TEMP_REGRESSION = {
    'intercept': -15.31,
    'comfort_setpoint': 1.218,
    'eco_setpoint': -0.090,
    'curve_rise': 9.73,
    'comfort_hours': -0.020,
}

# Sensor weights for weighted temperature
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 0.40,
    'office1_temperature': 0.30,
    'atelier_temperature': 0.10,
    'studio_temperature': 0.10,
    'simlab_temperature': 0.10,
}

# Baseline parameters
BASELINE = {
    'setpoint_comfort': 20.2,
    'setpoint_eco': 18.5,
    'comfort_start': 6.5,
    'comfort_end': 20.0,
    'curve_rise': 1.08,
}

# Comfort constraint
COMFORT_THRESHOLD = 18.5
OCCUPIED_START = 8
OCCUPIED_END = 22

# Phase 5 strategies to include (plus baseline)
PHASE5_STRATEGY_LABELS = ['Grid-Minimal', 'Balanced', 'Cost-Minimal']


def load_data():
    """Load integrated dataset and selected strategies."""
    print("Loading data...")

    # Load integrated dataset
    df = pd.read_parquet(PHASE1_DIR / 'integrated_overlap_only.parquet')
    df.index = pd.to_datetime(df.index)

    # Calculate weighted temperature
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for sensor, weight in SENSOR_WEIGHTS.items():
        if sensor in df.columns:
            valid = df[sensor].notna()
            weighted_sum[valid] += df.loc[valid, sensor] * weight
            weight_sum[valid] += weight
    df['T_weighted'] = weighted_sum / weight_sum
    df.loc[weight_sum == 0, 'T_weighted'] = np.nan

    # Add outdoor temperature
    df['T_outdoor'] = df['stiebel_eltron_isg_outdoor_temperature']

    # Add time features
    df['hour'] = df.index.hour + df.index.minute / 60
    df['date'] = df.index.date
    df['day_of_week'] = df.index.dayofweek

    # Load tariff data
    tariff = pd.read_parquet(PHASE1_DIR / 'tariff_series_hourly.parquet')
    tariff.index = pd.to_datetime(tariff.index)
    if tariff.index.tz is not None:
        tariff.index = tariff.index.tz_localize(None)

    # Map tariff to df index
    hourly_idx = df.index.floor('h')
    if hourly_idx.tz is not None:
        hourly_idx = hourly_idx.tz_localize(None)

    purchase_map = tariff['purchase_rate_rp_kwh'].to_dict()
    feedin_map = tariff['feedin_rate_rp_kwh'].to_dict()
    df['purchase_rate'] = [purchase_map.get(h, 32.0) for h in hourly_idx]
    df['feedin_rate'] = [feedin_map.get(h, 14.0) for h in hourly_idx]

    # Load selected strategies
    with open(PHASE4_DIR / 'selected_strategies.json', 'r') as f:
        all_strategies = json.load(f)

    # Filter to Phase 5 strategies
    strategies = [s for s in all_strategies if s.get('label') in PHASE5_STRATEGY_LABELS]

    # Add baseline as reference
    baseline_strategy = {
        'id': 'baseline',
        'label': 'Baseline',
        'variables': BASELINE,
        'objectives': {'mean_temp': 0, 'grid_kwh': 0, 'cost_chf': 0}
    }
    strategies.insert(0, baseline_strategy)

    # Filter to valid rows (need T_outdoor, T_weighted, pv_generation)
    valid_mask = df['T_outdoor'].notna() & df['T_weighted'].notna() & df['pv_generation_kwh'].notna()
    df = df[valid_mask].copy()

    print(f"  Loaded {len(df):,} valid timesteps ({len(df)//96:.0f} days)")
    print(f"  Loaded {len(strategies)} strategies for analysis")

    return df, strategies


def simulate_strategy_energy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Simulate energy consumption for a strategy using the corrected model.

    Returns DataFrame with per-timestep:
    - heating_kwh: Electrical heating consumption
    - grid_import: Grid import (demand - PV)
    - grid_export: Grid export (PV surplus)
    - cop: Instantaneous COP
    """
    setpoint_comfort = params.get('setpoint_comfort', BASELINE['setpoint_comfort'])
    setpoint_eco = params.get('setpoint_eco', BASELINE['setpoint_eco'])
    comfort_start = params.get('comfort_start', BASELINE['comfort_start'])
    comfort_end = params.get('comfort_end', BASELINE['comfort_end'])
    curve_rise = params.get('curve_rise', BASELINE['curve_rise'])

    hours = df['hour'].values
    T_outdoor = df['T_outdoor'].values
    pv_gen = df['pv_generation_kwh'].fillna(0).values
    dates = df['date'].values

    # Determine comfort mode
    is_comfort = (hours >= comfort_start) & (hours < comfort_end)

    # Flow temperature from heating curve
    setpoint = np.where(is_comfort, setpoint_comfort, setpoint_eco)
    T_ref = np.where(is_comfort, HEATING_CURVE_PARAMS['t_ref_comfort'], HEATING_CURVE_PARAMS['t_ref_eco'])
    T_flow = setpoint + curve_rise * (T_ref - T_outdoor)
    T_flow = np.clip(T_flow, 20, 55)

    # COP calculation
    cop = COP_PARAMS['intercept'] + COP_PARAMS['outdoor_coef'] * T_outdoor + COP_PARAMS['flow_coef'] * T_flow
    cop = np.maximum(cop, 1.5)

    # Mode factor for eco periods
    setback_range = setpoint_comfort - 12.0
    actual_setback = setpoint_comfort - setpoint_eco
    eco_mode_factor = max(0.1, 1.0 - 0.9 * (actual_setback / setback_range)) if setback_range > 0 else 1.0
    mode_factor = np.where(is_comfort, 1.0, eco_mode_factor)

    # Thermal demand weight
    thermal_demand_weight = np.maximum(0, T_flow - T_outdoor) * mode_factor

    # Calculate daily heating energy with COP
    unique_dates = np.unique(dates)
    heating_kwh = np.zeros(len(df))

    for date in unique_dates:
        date_mask = dates == date
        T_outdoor_mean = np.mean(T_outdoor[date_mask])
        hdd = max(0, 18 - T_outdoor_mean)
        daily_thermal_kwh = THERMAL_COEF * hdd

        if daily_thermal_kwh < 0.1:
            continue

        day_thermal_weights = thermal_demand_weight[date_mask]
        day_cops = cop[date_mask]

        total_thermal_weight = np.sum(day_thermal_weights)
        if total_thermal_weight > 0.01:
            avg_cop = np.sum(day_cops * day_thermal_weights) / total_thermal_weight
            normalized_weights = day_thermal_weights / total_thermal_weight
        else:
            avg_cop = np.mean(day_cops)
            normalized_weights = np.ones(np.sum(date_mask)) / np.sum(date_mask)

        # Electrical heating = Thermal / COP
        daily_electrical_kwh = daily_thermal_kwh / avg_cop
        heating_kwh[date_mask] = daily_electrical_kwh * normalized_weights

    # Total demand
    BASE_LOAD_TIMESTEP = BASE_LOAD_KWH / 96
    total_demand = BASE_LOAD_TIMESTEP + heating_kwh

    # Grid calculations
    grid_import = np.maximum(0, total_demand - pv_gen)
    grid_export = np.maximum(0, pv_gen - total_demand)

    return pd.DataFrame({
        'heating_kwh': heating_kwh,
        'total_demand': total_demand,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'cop': cop,
        'T_flow': T_flow,
        'is_comfort': is_comfort,
    }, index=df.index)


def calculate_delta_T(params: dict) -> float:
    """Calculate temperature adjustment from strategy parameters."""
    baseline_hours = BASELINE['comfort_end'] - BASELINE['comfort_start']
    comfort_hours = params.get('comfort_end', 20) - params.get('comfort_start', 6.5)

    delta_T = (
        TEMP_REGRESSION['comfort_setpoint'] * (params.get('setpoint_comfort', 20.2) - BASELINE['setpoint_comfort']) +
        TEMP_REGRESSION['eco_setpoint'] * (params.get('setpoint_eco', 18.5) - BASELINE['setpoint_eco']) +
        TEMP_REGRESSION['curve_rise'] * (params.get('curve_rise', 1.08) - BASELINE['curve_rise']) +
        TEMP_REGRESSION['comfort_hours'] * (comfort_hours - baseline_hours)
    )

    return delta_T


def compute_summary_statistics(df: pd.DataFrame, strategies: list, energy_results: dict) -> pd.DataFrame:
    """Compute detailed summary statistics for each strategy."""
    print("\nComputing summary statistics...")

    stats = []

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        delta_T = calculate_delta_T(params)

        # Temperature data
        T_adj = df['T_weighted'] + delta_T
        occupied_mask = (df['hour'] >= OCCUPIED_START) & (df['hour'] < OCCUPIED_END)
        T_occupied = T_adj[occupied_mask].dropna()

        # Energy data
        energy = energy_results[label]
        daily_energy = energy.resample('D').sum()

        # Violations
        n_violations = (T_occupied < COMFORT_THRESHOLD).sum()
        violation_pct = n_violations / len(T_occupied) * 100 if len(T_occupied) > 0 else 0

        stat = {
            'strategy': label,
            'setpoint_comfort': params.get('setpoint_comfort', BASELINE['setpoint_comfort']),
            'setpoint_eco': params.get('setpoint_eco', BASELINE['setpoint_eco']),
            'comfort_start': params.get('comfort_start', BASELINE['comfort_start']),
            'comfort_end': params.get('comfort_end', BASELINE['comfort_end']),
            'curve_rise': params.get('curve_rise', BASELINE['curve_rise']),
            'delta_T': delta_T,
            # Temperature stats (occupied hours)
            'T_mean_occupied': T_occupied.mean(),
            'T_std_occupied': T_occupied.std(),
            'T_min_occupied': T_occupied.min(),
            'T_max_occupied': T_occupied.max(),
            'T_p05_occupied': T_occupied.quantile(0.05),
            'T_p50_occupied': T_occupied.quantile(0.50),
            'T_p95_occupied': T_occupied.quantile(0.95),
            'violation_pct': violation_pct,
            'violation_hours': n_violations * 0.25,
            # Energy stats
            'total_heating_kwh': energy['heating_kwh'].sum(),
            'total_grid_import': energy['grid_import'].sum(),
            'total_grid_export': energy['grid_export'].sum(),
            'daily_heating_kwh': daily_energy['heating_kwh'].mean(),
            'daily_grid_import': daily_energy['grid_import'].mean(),
            'avg_cop': energy['cop'].mean(),
        }

        stats.append(stat)
        print(f"  {label}: T={stat['T_mean_occupied']:.1f}°C, Grid={stat['total_grid_import']:.0f}kWh, COP={stat['avg_cop']:.2f}")

    return pd.DataFrame(stats)


def plot_detailed_timeseries(df: pd.DataFrame, strategies: list, energy_results: dict):
    """Create detailed time series visualization with simulated energy."""
    print("\nCreating time series visualization...")

    fig = plt.figure(figsize=(16, 18))

    colors = {'Baseline': '#2E86AB', 'Grid-Minimal': '#A23B72',
              'Balanced': '#00A896', 'Cost-Minimal': '#F18F01'}

    # Panel 1: Weighted temperature by strategy
    ax1 = fig.add_subplot(4, 1, 1)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        delta_T = calculate_delta_T(params)
        T_adj = df['T_weighted'] + delta_T

        ax1.plot(df.index, T_adj, label=label,
                color=colors.get(label, 'gray'), alpha=0.7, linewidth=0.8)

    ax1.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                label=f'Threshold ({COMFORT_THRESHOLD}°C)')
    ax1.axhspan(COMFORT_THRESHOLD, 26, alpha=0.1, color='green')
    ax1.axhspan(14, COMFORT_THRESHOLD, alpha=0.1, color='red')

    ax1.set_ylabel('Weighted Temperature (°C)')
    ax1.set_title('Panel A: Simulated Temperature by Strategy')
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(14, 26)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Panel 2: Outdoor temperature and PV generation
    ax2 = fig.add_subplot(4, 1, 2)

    ax2_twin = ax2.twinx()

    ax2.plot(df.index, df['T_outdoor'], color='#5D5D5D', linewidth=0.8, label='Outdoor Temp')
    ax2.set_ylabel('Outdoor Temperature (°C)', color='#5D5D5D')
    ax2.tick_params(axis='y', labelcolor='#5D5D5D')

    ax2_twin.fill_between(df.index, df['pv_generation_kwh'] * 4, alpha=0.5,
                          color='gold', label='PV (kW)')
    ax2_twin.set_ylabel('PV Generation (kW)', color='goldenrod')
    ax2_twin.tick_params(axis='y', labelcolor='goldenrod')
    ax2_twin.set_ylim(0, 10)

    ax2.set_title('Panel B: Outdoor Temperature and Solar Generation')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Panel 3: Simulated grid import by strategy
    ax3 = fig.add_subplot(4, 1, 3)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        # Resample to hourly for cleaner visualization
        hourly_grid = energy['grid_import'].resample('h').sum()
        ax3.plot(hourly_grid.index, hourly_grid.values,
                label=label, color=colors.get(label, 'gray'), alpha=0.8, linewidth=0.8)

    ax3.set_ylabel('Grid Import (kWh/hour)')
    ax3.set_title('Panel C: Simulated Grid Import by Strategy')
    ax3.legend(loc='upper right', ncol=4, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Panel 4: Daily grid comparison
    ax4 = fig.add_subplot(4, 1, 4)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        daily_grid = energy['grid_import'].resample('D').sum()
        ax4.plot(daily_grid.index, daily_grid.values,
                label=label, color=colors.get(label, 'gray'), linewidth=2, marker='o', markersize=3)

    ax4.set_ylabel('Daily Grid Import (kWh)')
    ax4.set_xlabel('Date')
    ax4.set_title('Panel D: Daily Grid Import Comparison')
    ax4.legend(loc='upper right', ncol=4, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig28_strategy_detailed_timeseries.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig28_strategy_detailed_timeseries.png")


def plot_hourly_patterns(df: pd.DataFrame, strategies: list, energy_results: dict):
    """Create hourly pattern visualization."""
    print("\nCreating hourly pattern visualization...")

    fig = plt.figure(figsize=(16, 16))

    colors = {'Baseline': '#2E86AB', 'Grid-Minimal': '#A23B72',
              'Balanced': '#00A896', 'Cost-Minimal': '#F18F01'}

    # Panel 1: Hourly temperature profiles by strategy
    ax1 = fig.add_subplot(3, 2, 1)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        delta_T = calculate_delta_T(params)
        T_adj = df['T_weighted'] + delta_T

        hourly = T_adj.groupby(df['hour'].astype(int)).agg(['mean', 'std'])
        ax1.plot(hourly.index, hourly['mean'], label=label,
                color=colors.get(label, 'gray'), linewidth=2)
        ax1.fill_between(hourly.index,
                        hourly['mean'] - hourly['std'],
                        hourly['mean'] + hourly['std'],
                        color=colors.get(label, 'gray'), alpha=0.2)

    ax1.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=1.5)
    ax1.axvspan(OCCUPIED_START, OCCUPIED_END, alpha=0.1, color='yellow', label='Occupied')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Panel A: Hourly Temperature Profile (Mean ± Std)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)

    # Panel 2: Hourly grid import by strategy
    ax2 = fig.add_subplot(3, 2, 2)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        hourly_grid = energy['grid_import'].groupby(energy.index.hour).mean() * 4  # kW

        ax2.plot(range(24), hourly_grid.values, label=label,
                color=colors.get(label, 'gray'), linewidth=2)

    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Grid Import (kW average)')
    ax2.set_title('Panel B: Hourly Grid Import Profile')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)

    # Panel 3: Hourly COP by strategy
    ax3 = fig.add_subplot(3, 2, 3)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        hourly_cop = energy['cop'].groupby(energy.index.hour).mean()

        ax3.plot(range(24), hourly_cop.values, label=label,
                color=colors.get(label, 'gray'), linewidth=2)

    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Average COP')
    ax3.set_title('Panel C: Hourly COP Profile')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)

    # Panel 4: Hourly PV vs demand
    ax4 = fig.add_subplot(3, 2, 4)

    hourly_pv = df['pv_generation_kwh'].groupby(df.index.hour).mean() * 4

    ax4.fill_between(range(24), hourly_pv.values, alpha=0.5, color='gold', label='PV Available')

    for strategy in strategies[:2]:  # Just baseline and grid-minimal for clarity
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        hourly_demand = energy['total_demand'].groupby(energy.index.hour).mean() * 4

        ax4.plot(range(24), hourly_demand.values, label=f'{label} Demand',
                color=colors.get(label, 'gray'), linewidth=2)

    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Power (kW)')
    ax4.set_title('Panel D: PV Generation vs Demand')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)

    # Panel 5: Grid heatmap - Baseline
    ax5 = fig.add_subplot(3, 2, 5)

    baseline_energy = energy_results['Baseline']
    grid_pivot = baseline_energy['grid_import'].to_frame()
    grid_pivot['date'] = grid_pivot.index.date
    grid_pivot['hour'] = grid_pivot.index.hour
    grid_heatmap = grid_pivot.pivot_table(values='grid_import', index='date',
                                          columns='hour', aggfunc='sum')

    sns.heatmap(grid_heatmap * 4, ax=ax5, cmap='YlOrRd',
                cbar_kws={'label': 'kW'}, vmin=0, vmax=2)
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Date')
    ax5.set_title('Panel E: Baseline Grid Import Heatmap')

    n_dates = len(grid_heatmap.index)
    step = max(1, n_dates // 8)
    ax5.set_yticks(range(0, n_dates, step))
    ax5.set_yticklabels([str(grid_heatmap.index[i])[:10] for i in range(0, n_dates, step)], fontsize=8)

    # Panel 6: Grid heatmap - Grid-Minimal
    ax6 = fig.add_subplot(3, 2, 6)

    if 'Grid-Minimal' in energy_results:
        gm_energy = energy_results['Grid-Minimal']
        grid_pivot_gm = gm_energy['grid_import'].to_frame()
        grid_pivot_gm['date'] = grid_pivot_gm.index.date
        grid_pivot_gm['hour'] = grid_pivot_gm.index.hour
        grid_heatmap_gm = grid_pivot_gm.pivot_table(values='grid_import', index='date',
                                                    columns='hour', aggfunc='sum')

        sns.heatmap(grid_heatmap_gm * 4, ax=ax6, cmap='YlOrRd',
                    cbar_kws={'label': 'kW'}, vmin=0, vmax=2)
        ax6.set_xlabel('Hour of Day')
        ax6.set_ylabel('Date')
        ax6.set_title('Panel F: Grid-Minimal Grid Import Heatmap')

        ax6.set_yticks(range(0, n_dates, step))
        ax6.set_yticklabels([str(grid_heatmap_gm.index[i])[:10] for i in range(0, n_dates, step)], fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig29_strategy_hourly_patterns.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig29_strategy_hourly_patterns.png")


def plot_energy_patterns(df: pd.DataFrame, strategies: list, energy_results: dict, stats_df: pd.DataFrame):
    """Create energy-focused visualization."""
    print("\nCreating energy pattern visualization...")

    fig = plt.figure(figsize=(16, 14))

    colors = {'Baseline': '#2E86AB', 'Grid-Minimal': '#A23B72',
              'Balanced': '#00A896', 'Cost-Minimal': '#F18F01'}

    # Panel 1: Total energy comparison bar chart
    ax1 = fig.add_subplot(3, 2, 1)

    labels = [s['strategy'] for _, s in stats_df.iterrows()]
    grid_values = stats_df['total_grid_import'].values
    heating_values = stats_df['total_heating_kwh'].values

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, grid_values, width, label='Grid Import', color='#E74C3C', alpha=0.8)
    bars2 = ax1.bar(x + width/2, heating_values, width, label='Heating', color='#3498DB', alpha=0.8)

    ax1.set_ylabel('Total Energy (kWh)')
    ax1.set_title('Panel A: Total Energy by Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, grid_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # Panel 2: Grid savings vs baseline
    ax2 = fig.add_subplot(3, 2, 2)

    baseline_grid = stats_df[stats_df['strategy'] == 'Baseline']['total_grid_import'].values[0]
    savings = [(baseline_grid - g) / baseline_grid * 100 for g in grid_values]

    bar_colors = ['green' if s > 0 else 'red' for s in savings]
    bars = ax2.bar(labels, savings, color=bar_colors, alpha=0.7, edgecolor='black')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Grid Savings vs Baseline (%)')
    ax2.set_title('Panel B: Grid Import Reduction')
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, savings):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

    # Panel 3: COP comparison
    ax3 = fig.add_subplot(3, 2, 3)

    cop_values = stats_df['avg_cop'].values
    bars = ax3.bar(labels, cop_values, color=[colors.get(l, 'gray') for l in labels], alpha=0.8)

    ax3.set_ylabel('Average COP')
    ax3.set_title('Panel C: Average Heat Pump COP')
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, cop_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # Panel 4: Daily grid distribution
    ax4 = fig.add_subplot(3, 2, 4)

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        energy = energy_results[label]
        daily_grid = energy['grid_import'].resample('D').sum()

        ax4.hist(daily_grid.values, bins=15, alpha=0.5, label=label,
                color=colors.get(label, 'gray'), edgecolor='black')

    ax4.set_xlabel('Daily Grid Import (kWh)')
    ax4.set_ylabel('Frequency (days)')
    ax4.set_title('Panel D: Distribution of Daily Grid Import')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Temperature box plots
    ax5 = fig.add_subplot(3, 2, 5)

    occupied_mask = (df['hour'] >= OCCUPIED_START) & (df['hour'] < OCCUPIED_END)
    occupied_data = df[occupied_mask]

    box_data = []
    labels_list = []
    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        delta_T = calculate_delta_T(params)
        T_adj = occupied_data['T_weighted'] + delta_T
        box_data.append(T_adj.dropna().values)
        labels_list.append(label)

    bp = ax5.boxplot(box_data, tick_labels=labels_list, patch_artist=True)
    for i, (box, label) in enumerate(zip(bp['boxes'], labels_list)):
        box.set_facecolor(colors.get(label, 'gray'))
        box.set_alpha(0.7)

    ax5.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=1.5)
    ax5.set_ylabel('Temperature (°C)')
    ax5.set_title('Panel E: Occupied Hours Temperature (08:00-22:00)')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Summary table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    table_data = []
    headers = ['Strategy', 'Grid\n(kWh)', 'Savings', 'COP', 'T_mean\n(°C)', 'Violation']

    for _, row in stats_df.iterrows():
        savings_pct = (baseline_grid - row['total_grid_import']) / baseline_grid * 100
        table_data.append([
            row['strategy'],
            f"{row['total_grid_import']:.0f}",
            f"{savings_pct:+.1f}%",
            f"{row['avg_cop']:.2f}",
            f"{row['T_mean_occupied']:.1f}",
            f"{row['violation_pct']:.1f}%"
        ])

    table = ax6.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center',
                     colColours=['lightblue'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    ax6.set_title('Panel F: Strategy Summary',
                 fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig30_strategy_energy_patterns.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig30_strategy_energy_patterns.png")


def generate_report(df: pd.DataFrame, strategies: list, stats_df: pd.DataFrame) -> str:
    """Generate HTML report section."""
    print("\nGenerating HTML report...")

    n_days = len(df['date'].unique())
    date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"

    baseline_stats = stats_df[stats_df['strategy'] == 'Baseline'].iloc[0]

    html = f"""
    <section id="strategy-detailed-analysis">
    <h2>4.6 Detailed Strategy Analysis for Phase 5</h2>

    <h3>Analysis Overview</h3>
    <p>This analysis simulates energy consumption for each strategy using the corrected energy model
    (BASE_LOAD={BASE_LOAD_KWH} kWh/day, THERMAL_COEF={THERMAL_COEF} kWh/HDD).</p>

    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Date Range</td><td>{date_range}</td></tr>
        <tr><td>Total Days</td><td>{n_days}</td></tr>
        <tr><td>Timesteps</td><td>{len(df):,}</td></tr>
        <tr><td>Comfort Threshold</td><td>{COMFORT_THRESHOLD}°C</td></tr>
        <tr><td>Occupied Hours</td><td>{OCCUPIED_START:02d}:00 - {OCCUPIED_END:02d}:00</td></tr>
    </table>

    <h3>Strategy Parameters</h3>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Comfort</th>
            <th>Eco</th>
            <th>Schedule</th>
            <th>Curve Rise</th>
        </tr>
    """

    for _, row in stats_df.iterrows():
        html += f"""
        <tr>
            <td><strong>{row['strategy']}</strong></td>
            <td>{row['setpoint_comfort']:.1f}°C</td>
            <td>{row['setpoint_eco']:.1f}°C</td>
            <td>{int(row['comfort_start']):02d}:00-{int(row['comfort_end']):02d}:00</td>
            <td>{row['curve_rise']:.2f}</td>
        </tr>
        """
    html += "</table>"

    # Energy comparison table
    html += """
    <h3>Energy Performance Comparison</h3>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Grid Import<br>(kWh)</th>
            <th>vs Baseline</th>
            <th>Heating<br>(kWh)</th>
            <th>Avg COP</th>
        </tr>
    """

    for _, row in stats_df.iterrows():
        savings = (baseline_stats['total_grid_import'] - row['total_grid_import']) / baseline_stats['total_grid_import'] * 100
        savings_style = 'color: green' if savings > 0 else 'color: red' if savings < 0 else ''

        html += f"""
        <tr>
            <td><strong>{row['strategy']}</strong></td>
            <td>{row['total_grid_import']:.0f}</td>
            <td style="{savings_style}">{savings:+.1f}%</td>
            <td>{row['total_heating_kwh']:.0f}</td>
            <td>{row['avg_cop']:.2f}</td>
        </tr>
        """
    html += "</table>"

    # Temperature comparison
    html += """
    <h3>Temperature Performance (Occupied Hours)</h3>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Mean (°C)</th>
            <th>Min (°C)</th>
            <th>5th %ile</th>
            <th>Violation %</th>
        </tr>
    """

    for _, row in stats_df.iterrows():
        viol_style = 'background-color: #ffcccc' if row['violation_pct'] > 5 else 'background-color: #ccffcc'

        html += f"""
        <tr>
            <td><strong>{row['strategy']}</strong></td>
            <td>{row['T_mean_occupied']:.1f}</td>
            <td>{row['T_min_occupied']:.1f}</td>
            <td>{row['T_p05_occupied']:.1f}</td>
            <td style="{viol_style}">{row['violation_pct']:.1f}%</td>
        </tr>
        """
    html += "</table>"

    # Figures
    html += """
    <h3>Visualizations</h3>

    <figure>
        <img src="fig28_strategy_detailed_timeseries.png" alt="Strategy Time Series">
        <figcaption><strong>Figure 28:</strong> Time series showing (A) simulated temperature by strategy,
        (B) outdoor temperature and PV generation, (C) simulated hourly grid import by strategy,
        (D) daily grid import comparison.</figcaption>
    </figure>

    <figure>
        <img src="fig29_strategy_hourly_patterns.png" alt="Hourly Patterns">
        <figcaption><strong>Figure 29:</strong> Hourly patterns showing (A) temperature profiles,
        (B) grid import profiles, (C) COP profiles, (D) PV vs demand,
        (E-F) grid import heatmaps for Baseline and Grid-Minimal strategies.</figcaption>
    </figure>

    <figure>
        <img src="fig30_strategy_energy_patterns.png" alt="Energy Patterns">
        <figcaption><strong>Figure 30:</strong> Energy analysis showing (A) total energy comparison,
        (B) grid savings vs baseline, (C) average COP, (D) daily grid distribution,
        (E) temperature box plots, (F) summary table.</figcaption>
    </figure>

    <h3>Key Findings</h3>
    <ul>
    """

    # Generate insights
    best_grid = stats_df.loc[stats_df['total_grid_import'].idxmin()]
    best_cop = stats_df.loc[stats_df['avg_cop'].idxmax()]

    html += f"""
        <li><strong>Best grid efficiency:</strong> {best_grid['strategy']} saves
            {(baseline_stats['total_grid_import'] - best_grid['total_grid_import'])/baseline_stats['total_grid_import']*100:.1f}%
            grid import vs Baseline</li>
        <li><strong>Best COP:</strong> {best_cop['strategy']} achieves COP {best_cop['avg_cop']:.2f}
            vs Baseline COP {baseline_stats['avg_cop']:.2f}</li>
        <li><strong>Energy model calibration:</strong> Uses BASE_LOAD={BASE_LOAD_KWH} kWh/day,
            THERMAL_COEF={THERMAL_COEF} kWh/HDD (electrical = thermal/COP)</li>
    </ul>
    </section>
    """

    return html


def main():
    """Main function."""
    print("=" * 60)
    print("Phase 4, Step 6: Detailed Strategy Analysis")
    print("=" * 60)

    # Load data
    df, strategies = load_data()

    # Simulate energy for each strategy
    print("\nSimulating energy for each strategy...")
    energy_results = {}
    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        energy_results[label] = simulate_strategy_energy(df, params)
        total_grid = energy_results[label]['grid_import'].sum()
        avg_cop = energy_results[label]['cop'].mean()
        print(f"  {label}: Grid={total_grid:.0f} kWh, COP={avg_cop:.2f}")

    # Compute statistics
    stats_df = compute_summary_statistics(df, strategies, energy_results)

    # Save statistics
    stats_df.to_csv(OUTPUT_DIR / 'strategy_detailed_stats.csv', index=False)
    print(f"\nSaved: strategy_detailed_stats.csv")

    # Create visualizations
    plot_detailed_timeseries(df, strategies, energy_results)
    plot_hourly_patterns(df, strategies, energy_results)
    plot_energy_patterns(df, strategies, energy_results, stats_df)

    # Generate report
    report_html = generate_report(df, strategies, stats_df)
    with open(OUTPUT_DIR / 'strategy_detailed_report.html', 'w') as f:
        f.write(report_html)
    print("Saved: strategy_detailed_report.html")

    # Summary
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS SUMMARY")
    print("=" * 60)

    baseline = stats_df[stats_df['strategy'] == 'Baseline'].iloc[0]

    print(f"\n{'Strategy':<15} {'Grid (kWh)':<12} {'Savings':<10} {'COP':<8} {'Violation':<10}")
    print("-" * 60)
    for _, row in stats_df.iterrows():
        savings = (baseline['total_grid_import'] - row['total_grid_import']) / baseline['total_grid_import'] * 100
        print(f"{row['strategy']:<15} {row['total_grid_import']:<12.0f} {savings:+7.1f}%   {row['avg_cop']:<8.2f} {row['violation_pct']:.1f}%")

    print("\n" + "=" * 60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
