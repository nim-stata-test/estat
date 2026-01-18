#!/usr/bin/env python3
"""
Phase 4, Step 4b: Grid Search Optimization for Heating Strategies

Exhaustive grid search alternative to NSGA-II for finding Pareto-optimal
heating parameter configurations. Guarantees complete coverage of the
parameter space within the specified resolution.

Decision Variables (5):
- setpoint_comfort: [19.0, 22.0] °C, step 0.5°C
- setpoint_eco: [12.0, 22.0] °C, step 1.0°C
- comfort_start: [6.0, 12.0] hours, step 0.5h
- comfort_end: [16.0, 22.0] hours, step 0.5h
- curve_rise: [0.80, 1.20], step 0.05

Objectives (3, all minimized internally):
1. Negative mean T_weighted during occupied hours (minimizing negative = maximizing temp)
2. Grid import (kWh)
3. Net electricity cost (CHF)

Constraints:
- setpoint_eco <= setpoint_comfort
- comfort_end > comfort_start
- violation_pct <= 5% (T_weighted < 18.5°C for no more than 5% of daytime hours)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse
import sys
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import time

# Module-level globals for parallel workers (initialized in main)
_WORKER_DATA = {}

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from shared.energy_system import (
    simulate_battery_soc,
    predict_cop,
    predict_t_hk2_variable_setpoint,
    is_high_tariff,
    calculate_electricity_cost,
    BATTERY_PARAMS,
)
from shared.report_style import CSS, COLORS

# Project paths
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Grid Definition
# ============================================================================

GRID_CONFIG = {
    'setpoint_comfort': {'min': 19.0, 'max': 22.0, 'step': 0.5},  # 7 values
    'setpoint_eco': {'min': 12.0, 'max': 22.0, 'step': 1.0},      # 11 values
    'comfort_start': {'min': 6.0, 'max': 12.0, 'step': 0.5},      # 13 values
    'comfort_end': {'min': 16.0, 'max': 22.0, 'step': 0.5},       # 13 values
    'curve_rise': {'min': 0.80, 'max': 1.20, 'step': 0.05},       # 9 values
}

# Occupied hours for comfort evaluation
OCCUPIED_START = 8
OCCUPIED_END = 22

# Baseline parameters (reference point)
BASELINE = {
    'setpoint_comfort': 20.2,
    'setpoint_eco': 18.5,
    'comfort_start': 6.5,
    'comfort_end': 20.0,
    'curve_rise': 1.08,
}

# ============================================================================
# Model Parameters
# ============================================================================

# Empirical hourly heating intensity profile (derived from observed consumption)
# Values represent relative heating intensity at each hour (0-23)
# Derived by subtracting base load from hourly consumption and normalizing
HOURLY_HEATING_PROFILE = np.array([
    0.0349, 0.0213, 0.0256, 0.0341, 0.0252, 0.0254,  # 00-05: night
    0.0364, 0.0750, 0.0573, 0.0380, 0.0633, 0.0474,  # 06-11: morning
    0.0601, 0.0415, 0.0489, 0.0354, 0.0477, 0.0519,  # 12-17: afternoon
    0.0512, 0.0532, 0.0493, 0.0285, 0.0173, 0.0310,  # 18-23: evening
])


def load_heating_curve_params():
    """Load parametric heating curve from Phase 2 JSON."""
    params_file = PHASE2_DIR / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        return {
            't_ref_comfort': params['t_ref_comfort'],
            't_ref_eco': params['t_ref_eco'],
        }
    else:
        return {'t_ref_comfort': 21.32, 't_ref_eco': 19.18}


def load_causal_coefficients():
    """Load causal coefficients from Phase 3 transfer function analysis."""
    causal_file = PHASE3_DIR / 'causal_coefficients.json'
    if causal_file.exists():
        with open(causal_file) as f:
            data = json.load(f)
        return {
            'intercept': 0.0,
            'comfort_setpoint': data['coefficients']['comfort_setpoint'],
            'eco_setpoint': data['coefficients']['eco_setpoint'],
            'curve_rise': data['coefficients']['curve_rise'],
            'comfort_hours': data['coefficients']['comfort_hours'],
        }
    else:
        # Fallback to Phase 2 regression
        return {
            'intercept': -15.31,
            'comfort_setpoint': 1.218,
            'eco_setpoint': -0.090,
            'curve_rise': 9.73,
            'comfort_hours': -0.020,
        }


HEATING_CURVE_PARAMS = load_heating_curve_params()
TEMP_REGRESSION = load_causal_coefficients()

# ============================================================================
# Grid Generation
# ============================================================================

def generate_grid():
    """Generate all parameter combinations on the grid."""
    grids = {}
    for name, cfg in GRID_CONFIG.items():
        grids[name] = np.arange(cfg['min'], cfg['max'] + cfg['step']/2, cfg['step'])

    # Generate all combinations
    all_combos = list(product(
        grids['setpoint_comfort'],
        grids['setpoint_eco'],
        grids['comfort_start'],
        grids['comfort_end'],
        grids['curve_rise'],
    ))

    # Filter valid combinations
    valid_combos = []
    for combo in all_combos:
        setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise = combo

        # Constraint 1: eco <= comfort
        if setpoint_eco > setpoint_comfort:
            continue

        # Constraint 2: end > start (at least 2 hours)
        if comfort_end <= comfort_start + 2:
            continue

        valid_combos.append({
            'setpoint_comfort': setpoint_comfort,
            'setpoint_eco': setpoint_eco,
            'comfort_start': comfort_start,
            'comfort_end': comfort_end,
            'curve_rise': curve_rise,
        })

    return valid_combos


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_parameters(params, sim_data, T_weighted, hours, T_outdoor, pv_gen, dates, timestamps):
    """
    Evaluate a single parameter configuration.

    Returns dict with objectives and metrics.
    """
    setpoint_comfort = params['setpoint_comfort']
    setpoint_eco = params['setpoint_eco']
    comfort_start = params['comfort_start']
    comfort_end = params['comfort_end']
    curve_rise = params['curve_rise']

    comfort_hours = comfort_end - comfort_start

    # --- Temperature Prediction using Transfer Function Model ---
    delta_T = (
        TEMP_REGRESSION['comfort_setpoint'] * (setpoint_comfort - BASELINE['setpoint_comfort']) +
        TEMP_REGRESSION['eco_setpoint'] * (setpoint_eco - BASELINE['setpoint_eco']) +
        TEMP_REGRESSION['curve_rise'] * (curve_rise - BASELINE['curve_rise']) +
        TEMP_REGRESSION['comfort_hours'] * (comfort_hours - (BASELINE['comfort_end'] - BASELINE['comfort_start']))
    )
    T_weighted_adj = T_weighted + delta_T

    occupied_mask = (hours >= OCCUPIED_START) & (hours < OCCUPIED_END)
    T_weighted_occupied = T_weighted_adj[occupied_mask]

    # Objective 1: Mean temperature (we want to maximize, so store as positive)
    mean_temp = np.mean(T_weighted_occupied)
    min_temp = np.min(T_weighted_occupied)

    # Violation percentage
    threshold_temp = 18.5
    violation_count = np.sum(T_weighted_occupied < threshold_temp)
    violation_pct = violation_count / len(T_weighted_occupied) if len(T_weighted_occupied) > 0 else 0.0

    # --- Energy/Cost Simulation ---
    # Calibrated from daily consumption = BASE + THERMAL * HDD / COP (R² = 0.81)
    BASE_LOAD_KWH = 10.5
    THERMAL_COEF = 8.4
    BASE_LOAD_TIMESTEP = BASE_LOAD_KWH / 96
    DT_HOURS = 0.25

    is_comfort = (hours >= comfort_start) & (hours < comfort_end)

    T_HK2 = predict_t_hk2_variable_setpoint(
        T_outdoor, setpoint_comfort, setpoint_eco, curve_rise, is_comfort,
        params=HEATING_CURVE_PARAMS
    )

    cop = predict_cop(T_outdoor, T_HK2)

    # Eco mode reduction factor based on setback
    setback_range = setpoint_comfort - 12.0
    actual_setback = setpoint_comfort - setpoint_eco
    eco_mode_factor = max(0.1, 1.0 - 0.7 * (actual_setback / setback_range)) if setback_range > 0 else 1.0

    # Build heating weights using empirical hourly profile + comfort/eco mode
    # Get base profile weights for each timestep (4 slots per hour)
    hour_indices = hours.astype(int)
    base_weights = HOURLY_HEATING_PROFILE[hour_indices]

    # Apply comfort/eco mode factors:
    # - During comfort: use full profile weight
    # - During eco: reduce weight to reflect lower heating activity
    mode_factor = np.where(is_comfort, 1.0, eco_mode_factor)
    thermal_demand_weight = base_weights * mode_factor

    unique_dates = np.unique(dates)
    heating_kwh = np.zeros(len(sim_data))

    for date in unique_dates:
        date_mask = dates == date
        T_outdoor_mean = np.mean(T_outdoor[date_mask])
        hdd = max(0, 18 - T_outdoor_mean)

        daily_thermal_kwh = THERMAL_COEF * hdd
        if daily_thermal_kwh < 0.1:
            continue

        day_weights = thermal_demand_weight[date_mask]
        day_cops = cop[date_mask]

        total_weight = np.sum(day_weights)
        if total_weight > 0.01:
            # Weight COP by heating intensity to get effective COP
            avg_cop = np.sum(day_cops * day_weights) / total_weight
            normalized_weights = day_weights / total_weight
        else:
            avg_cop = np.mean(day_cops)
            normalized_weights = np.ones(np.sum(date_mask)) / np.sum(date_mask)

        daily_electrical_kwh = daily_thermal_kwh / avg_cop
        heating_kwh[date_mask] = daily_electrical_kwh * normalized_weights

    total_consumption = BASE_LOAD_TIMESTEP + heating_kwh

    battery_soc, grid_import, grid_export, battery_flow = simulate_battery_soc(
        pv_generation=pv_gen,
        consumption=total_consumption,
        dt_hours=DT_HOURS,
        battery_params=BATTERY_PARAMS,
        timestamps=timestamps,
    )

    is_high = is_high_tariff(timestamps)
    grid_cost, feedin_revenue, _ = calculate_electricity_cost(
        grid_import, grid_export, is_high
    )

    grid_import_total = np.sum(grid_import)
    net_cost_total = grid_cost - feedin_revenue

    return {
        'setpoint_comfort': setpoint_comfort,
        'setpoint_eco': setpoint_eco,
        'comfort_start': comfort_start,
        'comfort_end': comfort_end,
        'curve_rise': curve_rise,
        'mean_temp': mean_temp,
        'min_temp': min_temp,
        'violation_pct': violation_pct,
        'grid_import_kwh': grid_import_total,
        'net_cost_chf': net_cost_total,
        'comfort_hours': comfort_hours,
        'feasible': violation_pct <= 0.05,  # 5% violation limit
    }


def _init_worker(sim_data, T_weighted, hours, T_outdoor, pv_gen, dates, timestamps):
    """Initialize worker process with shared data."""
    global _WORKER_DATA
    _WORKER_DATA['sim_data'] = sim_data
    _WORKER_DATA['T_weighted'] = T_weighted
    _WORKER_DATA['hours'] = hours
    _WORKER_DATA['T_outdoor'] = T_outdoor
    _WORKER_DATA['pv_gen'] = pv_gen
    _WORKER_DATA['dates'] = dates
    _WORKER_DATA['timestamps'] = timestamps


def _evaluate_params_worker(params):
    """Worker function that uses global data."""
    return evaluate_parameters(
        params,
        _WORKER_DATA['sim_data'],
        _WORKER_DATA['T_weighted'],
        _WORKER_DATA['hours'],
        _WORKER_DATA['T_outdoor'],
        _WORKER_DATA['pv_gen'],
        _WORKER_DATA['dates'],
        _WORKER_DATA['timestamps'],
    )


# ============================================================================
# Pareto Front Extraction
# ============================================================================

def is_dominated(a, b):
    """Check if solution a is dominated by solution b (for minimization)."""
    # For our objectives: minimize -temp (maximize temp), minimize grid, minimize cost
    obj_a = np.array([-a['mean_temp'], a['grid_import_kwh'], a['net_cost_chf']])
    obj_b = np.array([-b['mean_temp'], b['grid_import_kwh'], b['net_cost_chf']])

    return np.all(obj_b <= obj_a) and np.any(obj_b < obj_a)


def extract_pareto_front(results, feasible_only=True):
    """Extract Pareto-optimal solutions from results."""
    if feasible_only:
        candidates = [r for r in results if r['feasible']]
    else:
        candidates = results

    if not candidates:
        return []

    pareto = []
    for i, sol in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i != j and is_dominated(sol, other):
                dominated = True
                break
        if not dominated:
            pareto.append(sol)

    return pareto


def apply_epsilon_dominance(pareto_front, eps_temp=0.1, eps_grid=50.0, eps_cost=10.0):
    """Apply epsilon-dominance to reduce Pareto front size."""
    if not pareto_front:
        return []

    # Snap to epsilon grid
    snapped = []
    for sol in pareto_front:
        key = (
            round(sol['mean_temp'] / eps_temp) * eps_temp,
            round(sol['grid_import_kwh'] / eps_grid) * eps_grid,
            round(sol['net_cost_chf'] / eps_cost) * eps_cost,
        )
        snapped.append((key, sol))

    # Keep best solution per epsilon cell
    cells = {}
    for key, sol in snapped:
        if key not in cells:
            cells[key] = sol
        else:
            # Keep solution with higher mean temp (primary objective)
            if sol['mean_temp'] > cells[key]['mean_temp']:
                cells[key] = sol

    return list(cells.values())


# ============================================================================
# Visualization
# ============================================================================

def create_grid_search_visualization(results_df, pareto_df, output_path):
    """Create visualization of grid search results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Filter feasible solutions
    feasible = results_df[results_df['feasible']]
    infeasible = results_df[~results_df['feasible']]

    # Panel 1: Temperature vs Grid Import
    ax = axes[0, 0]
    if len(infeasible) > 0:
        ax.scatter(infeasible['grid_import_kwh'], infeasible['mean_temp'],
                   c='lightgray', alpha=0.3, s=5, label='Infeasible')
    ax.scatter(feasible['grid_import_kwh'], feasible['mean_temp'],
               c=feasible['net_cost_chf'], cmap='viridis', alpha=0.5, s=10)
    ax.scatter(pareto_df['grid_import_kwh'], pareto_df['mean_temp'],
               c='red', s=100, marker='*', edgecolors='black', linewidths=0.5,
               label='Pareto Front', zorder=10)
    ax.set_xlabel('Grid Import (kWh)')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('Temperature vs Grid Import')
    ax.legend()

    # Panel 2: Temperature vs Cost
    ax = axes[0, 1]
    if len(infeasible) > 0:
        ax.scatter(infeasible['net_cost_chf'], infeasible['mean_temp'],
                   c='lightgray', alpha=0.3, s=5)
    ax.scatter(feasible['net_cost_chf'], feasible['mean_temp'],
               c=feasible['grid_import_kwh'], cmap='viridis', alpha=0.5, s=10)
    ax.scatter(pareto_df['net_cost_chf'], pareto_df['mean_temp'],
               c='red', s=100, marker='*', edgecolors='black', linewidths=0.5,
               zorder=10)
    ax.set_xlabel('Net Cost (CHF)')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('Temperature vs Cost')

    # Panel 3: Grid Import vs Cost
    ax = axes[0, 2]
    if len(infeasible) > 0:
        ax.scatter(infeasible['grid_import_kwh'], infeasible['net_cost_chf'],
                   c='lightgray', alpha=0.3, s=5)
    ax.scatter(feasible['grid_import_kwh'], feasible['net_cost_chf'],
               c=feasible['mean_temp'], cmap='coolwarm', alpha=0.5, s=10)
    ax.scatter(pareto_df['grid_import_kwh'], pareto_df['net_cost_chf'],
               c='red', s=100, marker='*', edgecolors='black', linewidths=0.5,
               zorder=10)
    ax.set_xlabel('Grid Import (kWh)')
    ax.set_ylabel('Net Cost (CHF)')
    ax.set_title('Grid Import vs Cost')

    # Panel 4: Parameter distribution - Curve Rise vs Comfort Hours
    ax = axes[1, 0]
    ax.scatter(feasible['curve_rise'], feasible['comfort_hours'],
               c=feasible['mean_temp'], cmap='coolwarm', alpha=0.5, s=10)
    ax.scatter(pareto_df['curve_rise'], pareto_df['comfort_hours'],
               c='red', s=100, marker='*', edgecolors='black', linewidths=0.5,
               zorder=10)
    ax.set_xlabel('Curve Rise')
    ax.set_ylabel('Comfort Hours')
    ax.set_title('Parameter Space: Curve Rise vs Comfort Hours')

    # Panel 5: Setpoint distribution
    ax = axes[1, 1]
    ax.scatter(feasible['setpoint_comfort'], feasible['setpoint_eco'],
               c=feasible['mean_temp'], cmap='coolwarm', alpha=0.5, s=10)
    ax.scatter(pareto_df['setpoint_comfort'], pareto_df['setpoint_eco'],
               c='red', s=100, marker='*', edgecolors='black', linewidths=0.5,
               zorder=10)
    ax.plot([12, 22], [12, 22], 'k--', alpha=0.3, label='eco=comfort')
    ax.set_xlabel('Setpoint Comfort (°C)')
    ax.set_ylabel('Setpoint Eco (°C)')
    ax.set_title('Parameter Space: Setpoints')
    ax.legend()

    # Panel 6: Pareto front table
    ax = axes[1, 2]
    ax.axis('off')

    # Sort Pareto solutions by temperature
    pareto_sorted = pareto_df.sort_values('mean_temp', ascending=False)

    table_data = []
    for i, (_, row) in enumerate(pareto_sorted.head(10).iterrows()):
        table_data.append([
            f"{row['setpoint_comfort']:.1f}",
            f"{row['setpoint_eco']:.0f}",
            f"{row['comfort_start']:.1f}-{row['comfort_end']:.1f}",
            f"{row['curve_rise']:.2f}",
            f"{row['mean_temp']:.1f}",
            f"{row['grid_import_kwh']:.0f}",
            f"{row['net_cost_chf']:.0f}",
        ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Comf°C', 'Eco°C', 'Schedule', 'Rise', 'Temp°C', 'Grid', 'Cost'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    ax.set_title(f'Top 10 Pareto Solutions (of {len(pareto_df)})')

    plt.suptitle(f'Grid Search Results: {len(results_df):,} evaluations, {len(pareto_df)} Pareto solutions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_objective_landscape(results_df, output_path):
    """Create heatmap visualization of objective landscape."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    feasible = results_df[results_df['feasible']]

    # Panel 1: Mean temp vs curve_rise and comfort_hours
    ax = axes[0, 0]
    pivot = feasible.groupby(['curve_rise', 'comfort_hours'])['mean_temp'].mean().unstack()
    if not pivot.empty:
        im = ax.imshow(pivot.values, aspect='auto', cmap='coolwarm', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
        ax.set_xlabel('Comfort Hours')
        ax.set_ylabel('Curve Rise')
        ax.set_title('Mean Temperature (°C)')
        plt.colorbar(im, ax=ax)

    # Panel 2: Grid import vs curve_rise and comfort_hours
    ax = axes[0, 1]
    pivot = feasible.groupby(['curve_rise', 'comfort_hours'])['grid_import_kwh'].mean().unstack()
    if not pivot.empty:
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
        ax.set_xlabel('Comfort Hours')
        ax.set_ylabel('Curve Rise')
        ax.set_title('Grid Import (kWh)')
        plt.colorbar(im, ax=ax)

    # Panel 3: Mean temp vs setpoints
    ax = axes[1, 0]
    pivot = feasible.groupby(['setpoint_comfort', 'setpoint_eco'])['mean_temp'].mean().unstack()
    if not pivot.empty:
        im = ax.imshow(pivot.values, aspect='auto', cmap='coolwarm', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
        ax.set_xlabel('Setpoint Eco (°C)')
        ax.set_ylabel('Setpoint Comfort (°C)')
        ax.set_title('Mean Temperature (°C)')
        plt.colorbar(im, ax=ax)

    # Panel 4: Cost vs setpoints
    ax = axes[1, 1]
    pivot = feasible.groupby(['setpoint_comfort', 'setpoint_eco'])['net_cost_chf'].mean().unstack()
    if not pivot.empty:
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
        ax.set_xlabel('Setpoint Eco (°C)')
        ax.set_ylabel('Setpoint Comfort (°C)')
        ax.set_title('Net Cost (CHF)')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Objective Landscape (averaged over other parameters)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ============================================================================
# Main Execution
# ============================================================================

def load_simulation_data():
    """Load and prepare simulation data."""
    print("Loading simulation data...")

    # Load integrated dataset
    integrated = pd.read_parquet(PHASE1_DIR / 'integrated_dataset.parquet')

    # Load energy data for PV
    energy = pd.read_parquet(PHASE1_DIR / 'energy_balance_15min.parquet')

    # Ensure both have consistent timezone handling
    if integrated.index.tz is not None:
        integrated.index = integrated.index.tz_localize(None)
    if energy.index.tz is not None:
        energy.index = energy.index.tz_localize(None)

    # Get overlap period
    start_date = max(integrated.index.min(), energy.index.min())
    end_date = min(integrated.index.max(), energy.index.max())

    # Filter to overlap
    integrated = integrated[start_date:end_date].copy()
    energy = energy[start_date:end_date].copy()

    # Prepare simulation data
    sim_data = pd.DataFrame(index=integrated.index)
    sim_data['T_outdoor'] = integrated['stiebel_eltron_isg_outdoor_temperature']
    sim_data['T_weighted'] = integrated['davis_inside_temperature']  # Primary comfort sensor
    sim_data['pv_generation'] = energy['pv_generation_kwh']
    sim_data['hour'] = sim_data.index.hour + sim_data.index.minute / 60
    sim_data['date'] = sim_data.index.date

    # Drop rows with missing values
    sim_data = sim_data.dropna()

    print(f"  Loaded {len(sim_data):,} timesteps")
    print(f"  Period: {sim_data.index.min().date()} to {sim_data.index.max().date()}")

    return sim_data


def main():
    parser = argparse.ArgumentParser(description='Grid search optimization for heating strategies')
    parser.add_argument('--coarse', action='store_true', help='Use coarser grid (faster)')
    parser.add_argument('--parallel', '-j', type=int, default=1,
                        help='Number of parallel workers (0=auto, 1=sequential, N=N workers)')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4: Grid Search Optimization")
    print("=" * 60)

    # Adjust grid for coarse mode
    if args.coarse:
        print("\nUsing COARSE grid (faster, lower resolution)")
        GRID_CONFIG['setpoint_comfort']['step'] = 1.0
        GRID_CONFIG['setpoint_eco']['step'] = 2.0
        GRID_CONFIG['comfort_start']['step'] = 1.0
        GRID_CONFIG['comfort_end']['step'] = 1.0
        GRID_CONFIG['curve_rise']['step'] = 0.1

    # Load data
    sim_data = load_simulation_data()

    # Pre-extract arrays for evaluation
    T_weighted = sim_data['T_weighted'].values
    hours = sim_data['hour'].values
    T_outdoor = sim_data['T_outdoor'].values
    pv_gen = sim_data['pv_generation'].fillna(0).values
    dates = sim_data['date'].values
    timestamps = pd.DatetimeIndex(sim_data.index)

    # Generate grid
    print("\nGenerating parameter grid...")
    grid = generate_grid()
    print(f"  Total valid combinations: {len(grid):,}")

    # Determine number of workers
    n_workers = args.parallel
    if n_workers <= 0:
        n_workers = cpu_count()
    if n_workers > 1:
        print(f"\nUsing {n_workers} parallel workers")

    # Evaluate all combinations
    print(f"\nEvaluating {len(grid):,} parameter combinations...", flush=True)

    start_time = time.time()
    report_interval = max(1, len(grid) // 20)  # Report 20 times

    if n_workers > 1:
        # Parallel evaluation using multiprocessing Pool
        # Initialize worker data in this process first (for fork inheritance)
        _init_worker(sim_data, T_weighted, hours, T_outdoor, pv_gen, dates, timestamps)

        # Use imap_unordered for better performance with progress tracking
        results = []
        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(sim_data, T_weighted, hours, T_outdoor, pv_gen, dates, timestamps)) as pool:
            for i, result in enumerate(pool.imap_unordered(_evaluate_params_worker, grid, chunksize=100)):
                results.append(result)

                if (i + 1) % report_interval == 0 or i == len(grid) - 1:
                    elapsed = time.time() - start_time
                    pct = 100 * (i + 1) / len(grid)
                    rate = (i + 1) / elapsed
                    eta = (len(grid) - i - 1) / rate if rate > 0 else 0
                    print(f"  {i+1:,}/{len(grid):,} ({pct:.0f}%) - {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining", flush=True)
    else:
        # Sequential evaluation (original code path)
        results = []
        for i, params in enumerate(grid):
            result = evaluate_parameters(
                params, sim_data, T_weighted, hours, T_outdoor, pv_gen, dates, timestamps
            )
            results.append(result)

            if (i + 1) % report_interval == 0 or i == len(grid) - 1:
                elapsed = time.time() - start_time
                pct = 100 * (i + 1) / len(grid)
                rate = (i + 1) / elapsed
                eta = (len(grid) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:,}/{len(grid):,} ({pct:.0f}%) - {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Statistics
    n_feasible = results_df['feasible'].sum()
    print(f"\n  Feasible solutions: {n_feasible:,} ({100*n_feasible/len(results_df):.1f}%)")

    # Extract Pareto front
    print("\nExtracting Pareto front...")
    pareto_front = extract_pareto_front(results, feasible_only=True)
    print(f"  Raw Pareto solutions: {len(pareto_front)}")

    # Apply epsilon dominance
    eps_pareto = apply_epsilon_dominance(pareto_front, eps_temp=0.1, eps_grid=50.0, eps_cost=10.0)
    print(f"  After ε-dominance: {len(eps_pareto)}")

    pareto_df = pd.DataFrame(eps_pareto)

    # Sort by temperature (descending)
    pareto_df = pareto_df.sort_values('mean_temp', ascending=False)

    # Label strategies
    labels = []
    for i, _ in enumerate(pareto_df.iterrows()):
        if i == 0:
            labels.append('Comfort-First')
        elif i == len(pareto_df) - 1:
            labels.append('Grid-Minimal')
        else:
            labels.append(f'Balanced-{i}')
    pareto_df['label'] = labels

    # Save results
    print("\nSaving results...")

    results_df.to_csv(OUTPUT_DIR / 'grid_search_all_results.csv', index=False)
    print(f"  Saved: grid_search_all_results.csv ({len(results_df):,} rows)")

    pareto_df.to_csv(OUTPUT_DIR / 'grid_search_pareto.csv', index=False)
    print(f"  Saved: grid_search_pareto.csv ({len(pareto_df)} solutions)")

    # Save as JSON for compatibility
    pareto_json = pareto_df.to_dict('records')
    with open(OUTPUT_DIR / 'grid_search_pareto.json', 'w') as f:
        json.dump(pareto_json, f, indent=2, default=str)
    print(f"  Saved: grid_search_pareto.json")

    # Create visualizations
    print("\nCreating visualizations...")
    create_grid_search_visualization(
        results_df, pareto_df,
        OUTPUT_DIR / 'fig4.11_grid_search_results.png'
    )
    create_objective_landscape(
        results_df,
        OUTPUT_DIR / 'fig4.12_objective_landscape.png'
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)

    print(f"\nPareto-optimal strategies ({len(pareto_df)} solutions):")
    print("-" * 90)
    print(f"{'Label':<15} {'Comf°C':<8} {'Eco°C':<8} {'Schedule':<12} {'Rise':<6} {'Temp°C':<8} {'Grid':<8} {'Cost':<8}")
    print("-" * 90)
    for _, row in pareto_df.iterrows():
        print(f"{row['label']:<15} {row['setpoint_comfort']:<8.1f} {row['setpoint_eco']:<8.0f} "
              f"{row['comfort_start']:.1f}-{row['comfort_end']:.1f}  {row['curve_rise']:<6.2f} "
              f"{row['mean_temp']:<8.1f} {row['grid_import_kwh']:<8.0f} {row['net_cost_chf']:<8.0f}")


if __name__ == '__main__':
    main()
