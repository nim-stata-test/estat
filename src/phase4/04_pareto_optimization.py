#!/usr/bin/env python3
"""
Phase 4, Step 4: Multi-Objective Pareto Optimization for Heating Strategies

Uses NSGA-II algorithm to find Pareto-optimal heating parameter configurations.

Decision Variables (5):
- setpoint_comfort: [19.0, 22.0] °C
- setpoint_eco: [12.0, 19.0] °C (12°C = frost protection)
- comfort_start: [6.0, 12.0] hours
- comfort_end: [16.0, 22.0] hours
- curve_rise: [0.80, 1.20]

Objectives (3, all minimized):
1. Negative mean T_weighted during occupied hours (minimizing negative = maximizing temp)
2. Grid import (kWh)
3. Net electricity cost (CHF)

Constraints (soft penalty):
- setpoint_eco <= setpoint_comfort (eco must not exceed comfort)
- violation_pct <= 20% (T_weighted < 18.5°C for no more than 20% of daytime hours)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse
import sys

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Grey-box thermal simulator disabled - forward simulation diverges (R² = -6.7)
# Using TEMP_REGRESSION instead (transfer function model)
# from shared.thermal_simulator import ThermalSimulator, compute_T_HK2_from_schedule_vectorized

# Import shared energy system module with battery model
from shared.energy_system import (
    simulate_battery_soc,
    predict_cop,
    predict_t_hk2_variable_setpoint,
    is_high_tariff,
    calculate_electricity_cost,
    BATTERY_PARAMS,
)

try:
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.core.repair import Repair
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.core.population import Population
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    print("Warning: pymoo not installed. Run: pip install pymoo>=0.6.0")

# Project paths (PROJECT_ROOT already defined above for shared imports)
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)

# Buffer tank sensor column (for grey-box simulation initial conditions)
BUFFER_COL = 'stiebel_eltron_isg_actual_temperature_buffer'

# Model parameters from Phase 3
# Uses T_HK2 (target flow from heating curve) not actual measured flow
COP_PARAMS = {
    'intercept': 6.52,
    'outdoor_coef': 0.1319,
    't_hk2_coef': -0.1007,
}

# Load heating curve parameters from Phase 2 (or use defaults)
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
        print(f"WARNING: {params_file} not found, using defaults")
        return {'t_ref_comfort': 21.32, 't_ref_eco': 19.18}

HEATING_CURVE_PARAMS = load_heating_curve_params()

# Load causal coefficients from Phase 3 transfer function analysis
# These replace the Phase 2 regression coefficients which overestimate effects by 3-6x
def load_causal_coefficients():
    """Load causal coefficients from Phase 3 transfer function analysis."""
    causal_file = PHASE3_DIR / 'causal_coefficients.json'
    if causal_file.exists():
        with open(causal_file) as f:
            data = json.load(f)
        print(f"  Loaded causal coefficients from Phase 3 (g_eff={data['g_eff']:.3f})")
        return {
            'intercept': 0.0,  # No intercept needed for delta-based adjustment
            'comfort_setpoint': data['coefficients']['comfort_setpoint'],
            'eco_setpoint': data['coefficients']['eco_setpoint'],
            'curve_rise': data['coefficients']['curve_rise'],
            'comfort_hours': data['coefficients']['comfort_hours'],
            'outdoor_mean': 0.0,  # Outdoor is exogenous, not controllable
        }
    else:
        # Fallback to Phase 2 regression (with warning)
        print(f"  WARNING: {causal_file} not found, using Phase 2 regression (may overestimate effects)")
        return {
            'intercept': -15.31,
            'comfort_setpoint': 1.218,
            'eco_setpoint': -0.090,
            'curve_rise': 9.73,
            'comfort_hours': -0.020,
            'outdoor_mean': 0.090,
        }

# Temperature adjustment coefficients
# Source: Phase 3 transfer function (causal) or Phase 2 regression (fallback)
TEMP_REGRESSION = load_causal_coefficients()

# Grey-box thermal model disabled for optimization
# Forward simulation diverges (R² = -6.7 on validation)
# Using TEMP_REGRESSION (transfer function) instead
GREYBOX_PARAMS = None

# Baseline parameters (reference point)
BASELINE = {
    'setpoint_comfort': 20.2,
    'setpoint_eco': 18.5,
    'comfort_start': 6.5,
    'comfort_end': 20.0,
    'curve_rise': 1.08,
}

# Target sensor (single sensor for simplicity)
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 1.0,
}

# Occupied hours for comfort evaluation
OCCUPIED_START = 8   # 08:00
OCCUPIED_END = 22    # 22:00

# Epsilon values for ε-dominance filtering
# Solutions must differ by at least these amounts to be considered meaningfully different
EPSILON = {
    'mean_temp': 0.1,   # °C - below this, comfort difference is imperceptible
    'grid_kwh': 100.0,  # kWh - ~5% of typical total range
    'cost_chf': 10.0,   # CHF - fine-grained cost differences
}


def snap_to_epsilon_grid(F: np.ndarray, epsilon: dict = None) -> np.ndarray:
    """
    Snap objective values to epsilon grid for ε-dominance comparison.

    Args:
        F: Array of shape (n_solutions, 3) with objectives [neg_mean_temp, grid_kwh, cost_chf]
        epsilon: Dict with epsilon values for each objective (default: EPSILON)

    Returns:
        F_snapped: Array with objectives rounded to epsilon precision
    """
    if epsilon is None:
        epsilon = EPSILON

    F_snapped = F.copy()
    # Objective 0: neg_mean_temp (negative, so use epsilon for mean_temp)
    F_snapped[:, 0] = np.round(F[:, 0] / epsilon['mean_temp']) * epsilon['mean_temp']
    # Objective 1: grid_kwh
    F_snapped[:, 1] = np.round(F[:, 1] / epsilon['grid_kwh']) * epsilon['grid_kwh']
    # Objective 2: cost_chf
    F_snapped[:, 2] = np.round(F[:, 2] / epsilon['cost_chf']) * epsilon['cost_chf']

    return F_snapped


def epsilon_nondominated_sort(F: np.ndarray, epsilon: dict = None) -> list:
    """
    Perform non-dominated sorting with ε-dominance.

    Uses ε-box dominance: solutions are mapped to grid cells of size epsilon,
    and dominance is determined by grid cell positions. Solutions in the same
    grid cell are considered equivalent.

    Args:
        F: Array of shape (n_solutions, 3) with objectives (all minimized)
        epsilon: Dict with epsilon values for each objective

    Returns:
        List of indices of ε-non-dominated solutions
    """
    if epsilon is None:
        epsilon = EPSILON

    n = len(F)
    if n == 0:
        return []

    # Snap to epsilon grid
    F_snapped = snap_to_epsilon_grid(F, epsilon)

    # Find unique grid cells and their best representatives
    # For each unique grid cell, keep the solution with best average rank
    grid_to_solutions = {}
    for i in range(n):
        key = tuple(F_snapped[i])
        if key not in grid_to_solutions:
            grid_to_solutions[key] = []
        grid_to_solutions[key].append(i)

    # For each grid cell, pick representative (first one, or best actual values)
    representatives = []
    for key, indices in grid_to_solutions.items():
        # Pick the solution with best sum of actual objective values
        best_idx = min(indices, key=lambda i: sum(F[i]))
        representatives.append(best_idx)

    # Now do standard non-dominated sorting on representatives' snapped values
    rep_F = F_snapped[representatives]

    # Standard non-dominated sorting
    is_dominated = np.zeros(len(representatives), dtype=bool)
    for i in range(len(representatives)):
        if is_dominated[i]:
            continue
        for j in range(len(representatives)):
            if i == j or is_dominated[j]:
                continue
            # Check if i dominates j (all <= and at least one <)
            if np.all(rep_F[i] <= rep_F[j]) and np.any(rep_F[i] < rep_F[j]):
                is_dominated[j] = True
            # Check if j dominates i
            elif np.all(rep_F[j] <= rep_F[i]) and np.any(rep_F[j] < rep_F[i]):
                is_dominated[i] = True
                break

    # Return original indices of non-dominated representatives
    pareto_indices = [representatives[i] for i in range(len(representatives)) if not is_dominated[i]]

    return pareto_indices


class SimulationData:
    """Container for preloaded simulation data and thermal simulator."""

    def __init__(self):
        self.sim_data = None
        self.tariff_data = None
        self.baseline_metrics = None
        self.thermal_simulator = None  # Grey-box thermal model for forward simulation
        # Pre-cached arrays for simulation performance
        self._cached_T_outdoor = None
        self._cached_hours = None
        self._cached_PV = None
        self._cached_x0 = None

    def load(self):
        """Load all required data and initialize thermal simulator."""
        print("Loading simulation data...")

        # Load integrated dataset
        df = pd.read_parquet(PHASE1_DIR / 'integrated_overlap_only.parquet')
        df.index = pd.to_datetime(df.index)

        # Load tariff series
        tariff = pd.read_parquet(PHASE1_DIR / 'tariff_series_hourly.parquet')
        tariff.index = pd.to_datetime(tariff.index)
        if tariff.index.tz is not None:
            tariff.index = tariff.index.tz_localize(None)

        # Prepare simulation data
        self.sim_data = self._prepare_sim_data(df, tariff)
        self.tariff_data = tariff

        # Calculate baseline metrics for reference
        self.baseline_metrics = self._compute_baseline_metrics()

        # Grey-box thermal simulator disabled (forward simulation diverges)
        # Using TEMP_REGRESSION (transfer function model) instead
        self.thermal_simulator = None

        # Pre-cache arrays for simulation performance
        self._cached_T_outdoor = self.sim_data['T_outdoor'].values.copy()
        self._cached_hours = self.sim_data['hour'].values.copy()
        self._cached_PV = self.sim_data['pv_generation'].fillna(0).values.copy()
        # Initial state: [T_buffer, T_room] from first valid observation
        T_buffer_init = self.sim_data['T_buffer'].iloc[0] if 'T_buffer' in self.sim_data else 30.0
        T_room_init = self.sim_data['T_weighted'].iloc[0]
        self._cached_x0 = np.array([T_buffer_init, T_room_init])

        print(f"  Loaded {len(self.sim_data):,} timesteps ({len(self.sim_data)//96:.0f} days)")

    def _prepare_sim_data(self, df: pd.DataFrame, tariff: pd.DataFrame) -> pd.DataFrame:
        """Prepare simulation data with all required columns."""
        sim = pd.DataFrame(index=df.index)

        # Outdoor temperature: Use the heat pump's built-in sensor (mounted near house).
        # This is intentionally NOT true ambient temperature - it's what the heat pump
        # uses for heating curve calculations, so the model matches actual HP behavior.
        sim['T_outdoor'] = df['stiebel_eltron_isg_outdoor_temperature']

        # Weighted indoor temperature
        weighted_sum = pd.Series(0.0, index=df.index)
        weight_sum = pd.Series(0.0, index=df.index)
        for sensor, weight in SENSOR_WEIGHTS.items():
            if sensor in df.columns:
                valid = df[sensor].notna()
                weighted_sum[valid] += df.loc[valid, sensor] * weight
                weight_sum[valid] += weight
        sim['T_weighted'] = weighted_sum / weight_sum
        sim.loc[weight_sum == 0, 'T_weighted'] = np.nan

        # Flow temperature
        sim['T_HK2'] = df.get('stiebel_eltron_isg_actual_temperature_hk_2',
                               df.get('stiebel_eltron_isg_flow_temperature_wp1'))

        # Buffer tank temperature (for grey-box simulation initial conditions)
        if BUFFER_COL in df.columns:
            sim['T_buffer'] = df[BUFFER_COL]
        else:
            # Fallback: estimate from T_HK2 (buffer is typically near flow temp)
            sim['T_buffer'] = sim['T_HK2'] * 0.9 if sim['T_HK2'] is not None else 30.0

        # Energy columns
        sim['pv_generation'] = df.get('pv_generation_kwh', 0)
        sim['grid_import'] = df.get('external_supply_kwh', 0)
        sim['total_consumption'] = df.get('total_consumption_kwh', 0)
        sim['direct_solar'] = df.get('direct_consumption_kwh', 0)

        # Hour of day
        sim['hour'] = sim.index.hour + sim.index.minute / 60
        sim['date'] = sim.index.date

        # Tariff data
        hourly_idx = sim.index.floor('h')
        if hourly_idx.tz is not None:
            hourly_idx = hourly_idx.tz_localize(None)
        for col in ['is_high_tariff', 'purchase_rate_rp_kwh', 'feedin_rate_rp_kwh']:
            if col in tariff.columns:
                sim[col] = hourly_idx.map(tariff[col].to_dict())
        sim['is_high_tariff'] = sim['is_high_tariff'].fillna(True)
        sim['purchase_rate_rp_kwh'] = sim['purchase_rate_rp_kwh'].ffill().bfill()
        sim['feedin_rate_rp_kwh'] = sim['feedin_rate_rp_kwh'].ffill().bfill()

        # Drop rows with missing essential data
        sim = sim.dropna(subset=['T_outdoor', 'T_weighted', 'T_HK2'])

        return sim

    def _compute_baseline_metrics(self) -> dict:
        """Compute baseline metrics for normalization."""
        occupied = self.sim_data[
            (self.sim_data['hour'] >= OCCUPIED_START) &
            (self.sim_data['hour'] < OCCUPIED_END)
        ]

        return {
            'mean_T_weighted': occupied['T_weighted'].mean(),
            'min_T_weighted': occupied['T_weighted'].min(),
            'grid_import': self.sim_data['grid_import'].sum(),
            'net_cost_chf': (
                self.sim_data['grid_import'] * self.sim_data['purchase_rate_rp_kwh'] / 100 -
                (self.sim_data['pv_generation'] - self.sim_data['direct_solar']).clip(lower=0) *
                self.sim_data['feedin_rate_rp_kwh'] / 100
            ).sum(),
        }


def estimate_flow_temp(curve_rise: float, T_outdoor: float, setpoint: float,
                       is_comfort: bool = True) -> float:
    """Estimate target flow temperature from heating curve."""
    T_ref = HEATING_CURVE_PARAMS['t_ref_comfort'] if is_comfort else HEATING_CURVE_PARAMS['t_ref_eco']
    return setpoint + curve_rise * (T_ref - T_outdoor)


def calculate_cop(T_outdoor: float, T_HK2: float) -> float:
    """Calculate COP from temperatures."""
    return (COP_PARAMS['intercept'] +
            COP_PARAMS['outdoor_coef'] * T_outdoor +
            COP_PARAMS['t_hk2_coef'] * T_HK2)


def simulate_parameters(params: dict, sim_data_obj: 'SimulationData') -> dict:
    """
    Simulate heating strategy using grey-box forward simulation.

    Uses physics-based thermal model to predict room temperature trajectory,
    then calculates comfort objectives and energy/cost metrics.

    Models energy time-shifting: when comfort schedule changes, heating energy
    shifts to different hours, affecting:
    1. Grid import (heating during solar hours uses PV instead of grid)
    2. Tariff costs (heating during low-tariff hours is cheaper)
    3. Feed-in revenue (more self-consumption means less feed-in)

    Args:
        params: Dict with setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise
        sim_data_obj: SimulationData object with thermal simulator and cached arrays

    Returns:
        Dict with objective values and constraint violations
    """
    setpoint_comfort = params['setpoint_comfort']
    setpoint_eco = params['setpoint_eco']
    comfort_start = params['comfort_start']
    comfort_end = params['comfort_end']
    curve_rise = params['curve_rise']

    # Extract DataFrame for energy simulation
    sim_data = sim_data_obj.sim_data

    # Calculate comfort hours
    comfort_hours = comfort_end - comfort_start

    # --- Temperature Prediction using Transfer Function Model ---
    # Uses regression-based delta_T from Phase 2 multivariate analysis
    # (Grey-box forward simulation disabled due to divergence, R² = -6.7)
    delta_T = (
        TEMP_REGRESSION['comfort_setpoint'] * (setpoint_comfort - BASELINE['setpoint_comfort']) +
        TEMP_REGRESSION['eco_setpoint'] * (setpoint_eco - BASELINE['setpoint_eco']) +
        TEMP_REGRESSION['curve_rise'] * (curve_rise - BASELINE['curve_rise']) +
        TEMP_REGRESSION['comfort_hours'] * (comfort_hours - (BASELINE['comfort_end'] - BASELINE['comfort_start']))
    )
    T_weighted_adj = sim_data['T_weighted'].values + delta_T
    occupied_mask = (sim_data['hour'].values >= OCCUPIED_START) & (sim_data['hour'].values < OCCUPIED_END)
    T_weighted_occupied = T_weighted_adj[occupied_mask]

    # Objective 1: Negative mean temperature (minimize to maximize avg temp)
    mean_temp = np.mean(T_weighted_occupied)
    neg_mean_temp = -mean_temp

    # Calculate violation percentage for constraint
    threshold_temp = 18.5
    violation_count = np.sum(T_weighted_occupied < threshold_temp)
    violation_pct = violation_count / len(T_weighted_occupied) if len(T_weighted_occupied) > 0 else 0.0

    # --- Battery-Aware Energy/Cost Simulation ---
    # Uses shared energy_system module with capacity-constrained battery model

    # Constants (calibrated from historical data analysis)
    BASE_LOAD_KWH = 11.0  # Non-heating consumption per day
    THERMAL_COEF = 10.0   # Thermal kWh needed per HDD (before COP division)
    BASE_LOAD_TIMESTEP = BASE_LOAD_KWH / 96  # Per 15-min interval
    DT_HOURS = 0.25  # Time step in hours

    # Extract arrays for vectorized ops
    hours = sim_data['hour'].values
    T_outdoor = sim_data['T_outdoor'].values
    pv_gen = sim_data['pv_generation'].fillna(0).values
    dates = sim_data['date'].values

    # Determine comfort mode for each timestep
    is_comfort = (hours >= comfort_start) & (hours < comfort_end)

    # Calculate T_HK2 using shared module
    T_HK2 = predict_t_hk2_variable_setpoint(
        T_outdoor, setpoint_comfort, setpoint_eco, curve_rise, is_comfort,
        params=HEATING_CURVE_PARAMS
    )

    # Calculate COP using shared module
    cop = predict_cop(T_outdoor, T_HK2)

    # Mode factor: comfort=1.0, eco reduces heating effort based on setback
    setback_range = setpoint_comfort - 12.0
    actual_setback = setpoint_comfort - setpoint_eco
    eco_mode_factor = max(0.1, 1.0 - 0.9 * (actual_setback / setback_range)) if setback_range > 0 else 1.0
    mode_factor = np.where(is_comfort, 1.0, eco_mode_factor)

    # Thermal demand weight at each timestep
    thermal_demand_weight = np.maximum(0, T_HK2 - T_outdoor) * mode_factor

    # Calculate daily heating energy with COP
    unique_dates = np.unique(dates)
    heating_kwh = np.zeros(len(sim_data))

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

        daily_electrical_kwh = daily_thermal_kwh / avg_cop
        heating_kwh[date_mask] = daily_electrical_kwh * normalized_weights

    # Total consumption at each timestep (kWh per 15-min interval)
    total_consumption = BASE_LOAD_TIMESTEP + heating_kwh

    # --- Battery-aware grid simulation ---
    # Simulate battery SoC with capacity constraints (11 kWh, 84% efficiency)
    battery_soc, grid_import, grid_export, battery_flow = simulate_battery_soc(
        pv_generation=pv_gen,
        consumption=total_consumption,
        dt_hours=DT_HOURS,
        battery_params=BATTERY_PARAMS,
    )

    # Calculate tariff periods
    timestamps = pd.DatetimeIndex(sim_data.index)
    is_high = is_high_tariff(timestamps)

    # Calculate costs with tariff awareness
    grid_cost, feedin_revenue, _ = calculate_electricity_cost(
        grid_import, grid_export, is_high
    )

    # Objective 2: Total grid import (kWh)
    grid_import_total = np.sum(grid_import)

    # Objective 3: Net cost (CHF) = grid cost - feed-in revenue
    net_cost_total = grid_cost - feedin_revenue

    # Constraints (g <= 0 means satisfied, g > 0 means violation - soft penalty)
    g1 = setpoint_eco - setpoint_comfort  # Eco must be <= comfort
    g2 = violation_pct - 0.05  # T_weighted < 18.5°C for no more than 5% of daytime hours

    return {
        'objectives': np.array([neg_mean_temp, grid_import_total, net_cost_total]),
        'constraints': np.array([g1, g2]),
        'metrics': {
            'mean_T_weighted': mean_temp,
            'min_T_weighted': np.min(T_weighted_occupied),
            'violation_pct': violation_pct,
            'grid_import_kwh': grid_import_total,
            'net_cost_chf': net_cost_total,
            'comfort_hours': comfort_hours,
        }
    }


class HeatingOptimizationProblem(Problem):
    """Multi-objective optimization problem for heating parameters."""

    # Parameter grid definitions
    GRID_SETPOINT = 0.1       # °C grid for setpoint_comfort and setpoint_eco
    GRID_TIME = 0.25          # 15-minute intervals (0.25 hours)
    GRID_CURVE_RISE = 0.01    # Curve rise grid

    def __init__(self, sim_data: SimulationData):
        """
        Initialize problem with simulation data.

        Variables: [setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise]
        Objectives (3): neg_mean_temp, grid_import, net_cost
        Constraints (2): eco_leq_comfort, violation_pct

        Parameters are constrained to discrete grids:
        - Setpoints: 0.1°C grid (19.0, 19.1, 19.2, ...)
        - Times: 15-minute intervals (6.0, 6.25, 6.5, ...)
        - Curve rise: 0.01 grid (0.80, 0.81, 0.82, ...)
        """
        super().__init__(
            n_var=5,
            n_obj=3,
            n_constr=2,
            xl=np.array([19.0, 12.0, 6.0, 16.0, 0.80]),   # lower bounds (eco 12°C = frost protection)
            xu=np.array([22.0, 19.0, 12.0, 22.0, 1.20]),  # upper bounds
        )
        self.sim_data = sim_data

    @classmethod
    def snap_to_grid(cls, X: np.ndarray) -> np.ndarray:
        """Snap parameter values to their respective grids."""
        X_snapped = X.copy()
        # Setpoint comfort (index 0): 0.1°C grid
        X_snapped[:, 0] = np.round(X[:, 0] / cls.GRID_SETPOINT) * cls.GRID_SETPOINT
        # Setpoint eco (index 1): 0.1°C grid
        X_snapped[:, 1] = np.round(X[:, 1] / cls.GRID_SETPOINT) * cls.GRID_SETPOINT
        # Comfort start (index 2): 15-min grid
        X_snapped[:, 2] = np.round(X[:, 2] / cls.GRID_TIME) * cls.GRID_TIME
        # Comfort end (index 3): 15-min grid
        X_snapped[:, 3] = np.round(X[:, 3] / cls.GRID_TIME) * cls.GRID_TIME
        # Curve rise (index 4): 0.01 grid
        X_snapped[:, 4] = np.round(X[:, 4] / cls.GRID_CURVE_RISE) * cls.GRID_CURVE_RISE
        return X_snapped

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population of solutions."""
        # Snap parameters to grid before evaluation
        X = self.snap_to_grid(X)

        n_pop = X.shape[0]
        F = np.zeros((n_pop, 3))  # objectives (3)
        G = np.zeros((n_pop, 2))  # constraints (2)

        for i in range(n_pop):
            params = {
                'setpoint_comfort': X[i, 0],
                'setpoint_eco': X[i, 1],
                'comfort_start': X[i, 2],
                'comfort_end': X[i, 3],
                'curve_rise': X[i, 4],
            }
            result = simulate_parameters(params, self.sim_data)
            F[i] = result['objectives']
            G[i] = result['constraints']

        out["F"] = F
        out["G"] = G


class GridRepair(Repair):
    """Repair operator to snap parameters to discrete grid values."""

    def _do(self, problem, X, **kwargs):
        """Snap all solutions to their respective parameter grids."""
        return HeatingOptimizationProblem.snap_to_grid(X)


class OptimizationHistoryCallback(Callback):
    """
    Callback to track all evaluated solutions across generations.

    Records:
    - All unique parameter sets evaluated
    - Generation when each was first seen
    - Pareto front membership at each generation
    """

    def __init__(self):
        super().__init__()
        self.history = []  # List of generation snapshots
        self.all_solutions = {}  # Dict mapping solution hash -> solution data
        self.generation_pareto_fronts = []  # Pareto front indices per generation

    def _solution_hash(self, x: np.ndarray) -> str:
        """Create a hash for a solution vector (rounded for floating point stability)."""
        rounded = tuple(np.round(x, 4))
        return str(rounded)

    def notify(self, algorithm):
        """Called after each generation."""
        gen = algorithm.n_gen
        pop = algorithm.pop
        timestamp = datetime.now().isoformat()

        # Get current population
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G") if pop.get("G") is not None else np.zeros((len(X), 2))

        # Determine which solutions are on the Pareto front
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        pareto_indices = set(fronts[0]) if len(fronts) > 0 else set()

        # Record generation snapshot
        gen_snapshot = {
            'generation': gen,
            'timestamp': timestamp,
            'population_size': len(X),
            'n_pareto': len(pareto_indices),
            'solutions': []
        }

        # Process each solution
        for i in range(len(X)):
            sol_hash = self._solution_hash(X[i])
            is_pareto = i in pareto_indices

            # Convert objective 0 back to mean_temp (it's stored as negative)
            mean_temp = -float(F[i, 0])

            sol_data = {
                'hash': sol_hash,
                'variables': {
                    'setpoint_comfort': float(X[i, 0]),
                    'setpoint_eco': float(X[i, 1]),
                    'comfort_start': float(X[i, 2]),
                    'comfort_end': float(X[i, 3]),
                    'curve_rise': float(X[i, 4]),
                },
                'objectives': {
                    'mean_temp': mean_temp,
                    'grid_kwh': float(F[i, 1]),
                    'cost_chf': float(F[i, 2]),
                },
                'constraints': {
                    'eco_leq_comfort': float(G[i, 0]),
                    'violation_pct': float(G[i, 1]),
                },
                'is_pareto': is_pareto,
            }

            # Track in all_solutions (first appearance)
            if sol_hash not in self.all_solutions:
                self.all_solutions[sol_hash] = {
                    **sol_data,
                    'first_gen': gen,
                    'pareto_generations': [],
                }

            # Record Pareto membership for this generation
            if is_pareto:
                self.all_solutions[sol_hash]['pareto_generations'].append(gen)

            gen_snapshot['solutions'].append({
                'hash': sol_hash,
                'is_pareto': is_pareto,
                'rank': 0 if is_pareto else self._get_rank(i, fronts),
            })

        self.history.append(gen_snapshot)

    def _get_rank(self, idx: int, fronts: list) -> int:
        """Get the Pareto rank (front number) for a solution."""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

    def get_full_history(self) -> dict:
        """Return the complete optimization history."""
        return {
            'generations': self.history,
            'all_solutions': list(self.all_solutions.values()),
            'summary': {
                'total_generations': len(self.history),
                'unique_solutions': len(self.all_solutions),
                'final_pareto_size': self.history[-1]['n_pareto'] if self.history else 0,
            }
        }


def run_optimization(sim_data: SimulationData,
                     n_gen: int = 200,
                     pop_size: int = 100,
                     seed: int = 42,
                     warm_start: str = None) -> dict:
    """
    Run NSGA-II optimization.

    Args:
        sim_data: Prepared simulation data
        n_gen: Number of generations
        pop_size: Population size
        seed: Random seed
        warm_start: Path to previous archive for warm start

    Returns:
        Dict with optimization results
    """
    if not HAS_PYMOO:
        raise ImportError("pymoo not installed. Run: pip install pymoo>=0.6.0")

    print(f"\nRunning NSGA-II optimization...")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Seed: {seed}")

    # Create problem
    problem = HeatingOptimizationProblem(sim_data)

    # Create algorithm with grid repair operator
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=GridRepair(),
        eliminate_duplicates=True
    )

    # Warm start from previous archive
    if warm_start and Path(warm_start).exists():
        print(f"  Warm starting from: {warm_start}")
        archive = load_archive(warm_start)
        if archive and archive['solutions']:
            # Extract Pareto solutions as initial population
            initial_X = np.array([
                [s['variables']['setpoint_comfort'],
                 s['variables']['setpoint_eco'],
                 s['variables']['comfort_start'],
                 s['variables']['comfort_end'],
                 s['variables']['curve_rise']]
                for s in archive['solutions'] if s.get('is_pareto', False)
            ])
            if len(initial_X) > 0:
                # Pad with random solutions if needed
                if len(initial_X) < pop_size:
                    n_random = pop_size - len(initial_X)
                    random_X = np.random.uniform(
                        problem.xl, problem.xu, (n_random, problem.n_var)
                    )
                    initial_X = np.vstack([initial_X, random_X])
                elif len(initial_X) > pop_size:
                    initial_X = initial_X[:pop_size]

                # Create initial population
                sampling = initial_X
                algorithm = NSGA2(
                    pop_size=pop_size,
                    sampling=sampling,
                    crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(eta=20),
                    repair=GridRepair(),
                    eliminate_duplicates=True
                )
                print(f"  Initialized with {len(initial_X)} solutions from archive")

    # Create history callback
    history_callback = OptimizationHistoryCallback()

    # Run optimization with callback
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True,
        callback=history_callback
    )

    # Get optimization history
    opt_history = history_callback.get_full_history()

    print(f"\nOptimization complete!")
    print(f"  Total evaluations: {result.algorithm.n_gen * pop_size:,}")
    print(f"  Pareto solutions: {len(result.F)}")
    print(f"  Unique solutions tracked: {opt_history['summary']['unique_solutions']}")

    return {
        'X': result.X,  # Decision variables
        'F': result.F,  # Objective values
        'G': result.G if result.G is not None else np.zeros((len(result.X), 2)),  # Constraints
        'n_gen': n_gen,
        'pop_size': pop_size,
        'seed': seed,
        'history': opt_history,  # Full optimization history
    }


def extract_pareto_front(result: dict, use_epsilon: bool = True, epsilon: dict = None) -> list:
    """
    Extract non-dominated solutions from optimization result.

    Args:
        result: Optimization result dict with 'X' and 'F' arrays
        use_epsilon: If True, use ε-dominance to filter meaningfully different solutions
        epsilon: Custom epsilon values (default: EPSILON)

    Returns:
        List of Pareto-optimal solution dicts
    """
    X = result['X']
    F = result['F']

    if use_epsilon:
        # ε-dominance: keep only meaningfully different solutions
        pareto_idx = epsilon_nondominated_sort(F, epsilon)
        print(f"  ε-dominance filtering: {len(F)} → {len(pareto_idx)} solutions")
    else:
        # Standard non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        pareto_idx = fronts[0]  # First front is Pareto-optimal

    solutions = []
    for i, idx in enumerate(pareto_idx):
        # F[idx, 0] is neg_mean_temp, convert back to mean_temp
        mean_temp = -float(F[idx, 0])
        sol = {
            'id': f'sol_{i+1:03d}',
            'variables': {
                'setpoint_comfort': float(X[idx, 0]),
                'setpoint_eco': float(X[idx, 1]),
                'comfort_start': float(X[idx, 2]),
                'comfort_end': float(X[idx, 3]),
                'curve_rise': float(X[idx, 4]),
            },
            'objectives': {
                'mean_temp': mean_temp,  # Average daytime temperature (higher is better)
                'grid_kwh': float(F[idx, 1]),
                'cost_chf': float(F[idx, 2]),
            },
            'is_pareto': True,
        }
        solutions.append(sol)

    # Sort by grid_kwh for consistent ordering
    solutions.sort(key=lambda x: x['objectives']['grid_kwh'])

    return solutions


def select_diverse_strategies(solutions: list, n_select: int = 10) -> list:
    """
    Select diverse strategies from Pareto front using crowding distance.

    Returns n_select strategies spanning the Pareto front with descriptive labels.
    """
    if len(solutions) <= n_select:
        selected = solutions.copy()
    else:
        # Extract objective values (3 objectives: mean_temp, grid_kwh, cost_chf)
        # Note: mean_temp is stored as positive (higher is better), so we negate for crowding
        F = np.array([
            [-s['objectives']['mean_temp'],  # Negate so lower is better for crowding calc
             s['objectives']['grid_kwh'],
             s['objectives']['cost_chf']]
            for s in solutions
        ])

        # Normalize objectives to [0, 1]
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_norm = (F - F_min) / (F_max - F_min + 1e-10)

        # Calculate crowding distance
        n_sol = len(solutions)
        crowding = np.zeros(n_sol)

        for obj_idx in range(3):  # 3 objectives
            sorted_idx = np.argsort(F_norm[:, obj_idx])
            crowding[sorted_idx[0]] = np.inf
            crowding[sorted_idx[-1]] = np.inf

            for i in range(1, n_sol - 1):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                crowding[idx] += F_norm[next_idx, obj_idx] - F_norm[prev_idx, obj_idx]

        # Select solutions with highest crowding distance
        selected_idx = np.argsort(crowding)[-n_select:]
        selected = [solutions[i] for i in sorted(selected_idx)]

    # Assign descriptive labels based on objectives
    labels = []
    for sol in selected:
        obj = sol['objectives']
        # Determine dominant characteristic
        if obj['mean_temp'] >= max(s['objectives']['mean_temp'] for s in selected) - 0.1:
            label = "Comfort-First"
        elif obj['grid_kwh'] == min(s['objectives']['grid_kwh'] for s in selected):
            label = "Grid-Minimal"
        elif obj['cost_chf'] == min(s['objectives']['cost_chf'] for s in selected):
            label = "Cost-Minimal"
        elif obj['mean_temp'] >= 20.5:  # Good average temp
            label = "Warm-Balanced"
        else:
            label = "Balanced"

        # Ensure unique labels
        base_label = label
        counter = 1
        while label in labels:
            counter += 1
            label = f"{base_label}-{counter}"
        labels.append(label)
        sol['label'] = label

    # Re-number IDs
    for i, sol in enumerate(selected):
        sol['id'] = f'strategy_{i+1:02d}'

    return selected


def save_archive(solutions: list, metadata: dict, path: Path, history: dict = None):
    """Save optimization archive to JSON, optionally including optimization history."""
    archive = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_solutions': len(solutions),
            **metadata
        },
        'solutions': solutions,
    }

    # Include optimization history if provided
    if history:
        archive['optimization_history'] = history

    with open(path, 'w') as f:
        json.dump(archive, f, indent=2)

    print(f"Saved archive: {path}")
    if history:
        print(f"  Includes history: {history['summary']['total_generations']} generations, "
              f"{history['summary']['unique_solutions']} unique solutions")


def load_archive(path: str) -> dict:
    """Load optimization archive from JSON."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load archive: {e}")
        return None


def plot_pareto_front(solutions: list, selected: list = None):
    """Create Pareto front visualization."""
    print("\nCreating Pareto front visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract objective values (3 objectives: mean_temp, grid_kwh, cost_chf)
    F = np.array([
        [s['objectives']['mean_temp'],
         s['objectives']['grid_kwh'],
         s['objectives']['cost_chf']]
        for s in solutions
    ])

    # If selected, extract their objectives too
    if selected:
        F_sel = np.array([
            [s['objectives']['mean_temp'],
             s['objectives']['grid_kwh'],
             s['objectives']['cost_chf']]
            for s in selected
        ])

    obj_labels = [
        'Avg Temp (°C)',
        'Grid Import (kWh)',
        'Net Cost (CHF)'
    ]

    # Plot pairs of objectives (3 unique pairs + one repeated for 4 panels)
    pairs = [(0, 1), (0, 2), (1, 2), (0, 1)]  # Last panel repeats temp vs grid
    panel_titles = ['Temp vs Grid', 'Temp vs Cost', 'Grid vs Cost', 'Temp vs Grid (zoom)']

    for ax_idx, (ax, (i, j)) in enumerate(zip(axes.flat, pairs)):
        ax.scatter(F[:, i], F[:, j], c='lightgray', alpha=0.5, s=30, label='All Pareto')
        if selected:
            ax.scatter(F_sel[:, i], F_sel[:, j], c='blue', s=80, edgecolors='black',
                      label='Selected', zorder=5)
            for k, sol in enumerate(selected):
                ax.annotate(sol.get('label', f'{k+1}')[:8],
                           (F_sel[k, i], F_sel[k, j]),
                           fontsize=7, ha='left', va='bottom')
        ax.set_xlabel(obj_labels[i])
        ax.set_ylabel(obj_labels[j])
        ax.set_title(panel_titles[ax_idx], fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0, 0]:
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Pareto Front: Multi-Objective Heating Optimization\n'
                 '(Maximize Avg Temp, Minimize Grid Import, Minimize Net Cost)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig25_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig25_pareto_front.png")


def plot_strategy_comparison(selected: list):
    """Create radar chart comparing selected strategies."""
    print("Creating strategy comparison visualization...")

    # Normalize metrics for radar chart
    metrics = ['setpoint_comfort', 'setpoint_eco', 'comfort_start', 'comfort_end', 'curve_rise']
    metric_labels = ['Comfort\nSetpoint', 'Eco\nSetpoint', 'Comfort\nStart', 'Comfort\nEnd', 'Curve\nRise']

    # Extract values
    values = np.array([
        [s['variables'][m] for m in metrics]
        for s in selected
    ])

    # Normalize to [0, 1]
    v_min = values.min(axis=0)
    v_max = values.max(axis=0)
    values_norm = (values - v_min) / (v_max - v_min + 1e-10)

    # Number of variables
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    for i, sol in enumerate(selected):
        vals = values_norm[i].tolist()
        vals += vals[:1]  # Close the polygon
        ax.plot(angles, vals, 'o-', linewidth=2, label=sol.get('label', f'Strategy {i+1}'),
               color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.set_title('Strategy Parameter Comparison (Normalized)', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig26_pareto_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig26_pareto_strategy_comparison.png")


def generate_report(solutions: list, selected: list, metadata: dict) -> str:
    """Generate HTML report for Pareto optimization results."""

    def time_str(decimal_hour: float) -> str:
        h = int(decimal_hour)
        m = int((decimal_hour - h) * 60)
        return f"{h:02d}:{m:02d}"

    # Get grey-box model info if available
    greybox_info = ""
    if GREYBOX_PARAMS is not None:
        r2 = GREYBOX_PARAMS['fit_stats']['r2_room']
        greybox_info = f"""
    <h3>Thermal Model</h3>
    <p>Temperature predictions use <strong>grey-box forward simulation</strong> with physics-based
    state-space model (R² = {r2:.3f}):</p>
    <pre>
T_buf[k+1] = T_buf[k] + (dt/τ_buf) × [(T_HK2[k] - T_buf[k]) - r_emit×(T_buf[k] - T_room[k])]
T_room[k+1] = T_room[k] + (dt/τ_room) × [r_heat×(T_buf[k] - T_room[k]) - (T_room[k] - T_out[k])] + k_solar×PV[k]
    </pre>
    <p>This replaces the previous regression-based delta_T approach with explicit thermal dynamics modeling.</p>
"""

    html = f"""
    <section id="pareto-optimization">
    <h2>4.4 Multi-Objective Pareto Optimization</h2>

    <h3>Methodology</h3>
    <p>Used NSGA-II algorithm to find Pareto-optimal heating configurations:</p>
    <ul>
        <li><strong>Algorithm</strong>: NSGA-II (Non-dominated Sorting Genetic Algorithm II)</li>
        <li><strong>Population</strong>: {metadata.get('pop_size', 100)} individuals</li>
        <li><strong>Generations</strong>: {metadata.get('n_gen', 200)}</li>
        <li><strong>Evaluations</strong>: {metadata.get('pop_size', 100) * metadata.get('n_gen', 200):,}</li>
        <li><strong>Pareto solutions</strong>: {len(solutions)}</li>
    </ul>
    {greybox_info}

    <h3>Decision Variables</h3>
    <table>
        <tr><th>Variable</th><th>Range</th><th>Description</th></tr>
        <tr><td>setpoint_comfort</td><td>[19.0, 22.0] °C</td><td>Comfort mode target</td></tr>
        <tr><td>setpoint_eco</td><td>[12.0, 19.0] °C</td><td>Eco mode target (12°C = frost protection)</td></tr>
        <tr><td>comfort_start</td><td>[06:00, 12:00]</td><td>Comfort period start</td></tr>
        <tr><td>comfort_end</td><td>[16:00, 22:00]</td><td>Comfort period end</td></tr>
        <tr><td>curve_rise</td><td>[0.80, 1.20]</td><td>Heating curve slope</td></tr>
    </table>

    <h3>Objectives</h3>
    <ol>
        <li><strong>Average Temperature</strong>: Maximize mean T_weighted during 08:00-22:00 (higher is better)</li>
        <li><strong>Grid import</strong>: Minimize total kWh purchased from grid</li>
        <li><strong>Net cost</strong>: Minimize grid cost - feed-in revenue (CHF)</li>
    </ol>

    <h3>Constraint (Soft Penalty)</h3>
    <p><strong>Low-temperature violation</strong>: T_weighted &lt; 18.5°C for no more than 20% of daytime hours (08:00-22:00).
    Solutions exceeding this threshold are penalized but not excluded.</p>

    <h3>Selected Strategies (10 Diverse)</h3>
    <table>
        <tr>
            <th>Label</th>
            <th>Comfort<br>Setpoint</th>
            <th>Eco<br>Setpoint</th>
            <th>Schedule</th>
            <th>Curve<br>Rise</th>
            <th>Avg Temp<br>(°C)</th>
            <th>Grid<br>(kWh)</th>
            <th>Cost<br>(CHF)</th>
        </tr>
    """

    for sol in selected:
        v = sol['variables']
        o = sol['objectives']
        html += f"""
        <tr>
            <td><strong>{sol.get('label', sol['id'])}</strong></td>
            <td>{v['setpoint_comfort']:.1f}°C</td>
            <td>{v['setpoint_eco']:.1f}°C</td>
            <td>{time_str(v['comfort_start'])}-{time_str(v['comfort_end'])}</td>
            <td>{v['curve_rise']:.2f}</td>
            <td>{o['mean_temp']:.1f}</td>
            <td>{o['grid_kwh']:.0f}</td>
            <td>{o['cost_chf']:.1f}</td>
        </tr>
        """

    html += """
    </table>

    <p><em>Select 3 strategies from this table for Phase 5 intervention study.
    Consider including a baseline-like strategy, an energy-focused strategy,
    and a cost-focused strategy for comparison.</em></p>

    <h3>Pareto Front Visualization</h3>
    <figure>
        <img src="fig25_pareto_front.png" alt="Pareto Front">
        <figcaption><strong>Figure 26:</strong> Pareto front showing trade-offs between objectives.
        Blue points are the 10 selected strategies.</figcaption>
    </figure>

    <figure>
        <img src="fig26_pareto_strategy_comparison.png" alt="Strategy Comparison">
        <figcaption><strong>Figure 26:</strong> Radar chart comparing parameter values across selected strategies.</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for Pareto optimization."""
    # Default archive path
    DEFAULT_ARCHIVE = OUTPUT_DIR / 'pareto_archive.json'

    parser = argparse.ArgumentParser(description='Multi-objective Pareto optimization for heating')
    parser.add_argument('--generations', '-g', type=int, default=200, help='Number of generations (default: 200)')
    parser.add_argument('--population', '-p', type=int, default=100, help='Population size')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--warm-start', '-w', type=str, help='Path to previous archive (default: auto-detect)')
    parser.add_argument('--fresh', '-f', action='store_true', help='Start fresh, ignore existing archive')
    parser.add_argument('--n-select', '-n', type=int, default=10, help='Number of strategies to select')
    parser.add_argument('--no-epsilon', action='store_true', help='Disable ε-dominance filtering (keep all Pareto solutions)')
    parser.add_argument('--eps-temp', type=float, default=EPSILON['mean_temp'],
                        help=f'Epsilon for temperature (default: {EPSILON["mean_temp"]}°C)')
    parser.add_argument('--eps-grid', type=float, default=EPSILON['grid_kwh'],
                        help=f'Epsilon for grid import (default: {EPSILON["grid_kwh"]} kWh)')
    parser.add_argument('--eps-cost', type=float, default=EPSILON['cost_chf'],
                        help=f'Epsilon for cost (default: {EPSILON["cost_chf"]} CHF)')
    args = parser.parse_args()

    # Build custom epsilon dict if provided
    epsilon = {
        'mean_temp': args.eps_temp,
        'grid_kwh': args.eps_grid,
        'cost_chf': args.eps_cost,
    }
    use_epsilon = not args.no_epsilon

    print("="*60)
    print("Phase 4, Step 4: Multi-Objective Pareto Optimization")
    print("="*60)

    if use_epsilon:
        print(f"\nε-dominance enabled:")
        print(f"  Temperature: {epsilon['mean_temp']}°C")
        print(f"  Grid import: {epsilon['grid_kwh']} kWh")
        print(f"  Cost: {epsilon['cost_chf']} CHF")
    else:
        print("\nε-dominance disabled (keeping all Pareto solutions)")

    if not HAS_PYMOO:
        print("\nERROR: pymoo not installed. Run: pip install pymoo>=0.6.0")
        return 1

    # Determine warm start path
    warm_start_path = None
    if args.fresh:
        print("\nStarting fresh (--fresh flag set)")
    elif args.warm_start:
        warm_start_path = args.warm_start
    elif DEFAULT_ARCHIVE.exists():
        warm_start_path = str(DEFAULT_ARCHIVE)
        print(f"\nAuto-detected existing archive: {DEFAULT_ARCHIVE.name}")

    # Load data
    sim_data = SimulationData()
    sim_data.load()

    # Run optimization
    result = run_optimization(
        sim_data,
        n_gen=args.generations,
        pop_size=args.population,
        seed=args.seed,
        warm_start=warm_start_path
    )

    # Extract Pareto front from new optimization
    new_solutions = extract_pareto_front(result, use_epsilon=use_epsilon, epsilon=epsilon)
    print(f"\nExtracted {len(new_solutions)} ε-Pareto solutions from this run")

    # Merge with existing archive to keep ε-Pareto-optimal solutions
    if warm_start_path and Path(warm_start_path).exists():
        existing_archive = load_archive(warm_start_path)
        if existing_archive and existing_archive.get('solutions'):
            existing_solutions = existing_archive['solutions']
            print(f"Merging with {len(existing_solutions)} existing solutions...")

            # Combine all solutions
            all_solutions = existing_solutions + new_solutions

            # Re-compute Pareto front on combined set using ε-dominance
            # Use negative mean_temp so lower values are better for Pareto sorting
            all_F = np.array([
                [-s['objectives']['mean_temp'],
                 s['objectives']['grid_kwh'],
                 s['objectives']['cost_chf']]
                for s in all_solutions
            ])

            if use_epsilon:
                # ε-dominance sorting on combined set
                pareto_idx = epsilon_nondominated_sort(all_F, epsilon)
                print(f"  ε-dominance merge: {len(all_solutions)} → {len(pareto_idx)} solutions")
            else:
                # Standard non-dominated sorting on combined set
                nds = NonDominatedSorting()
                fronts = nds.do(all_F)
                pareto_idx = fronts[0]

            # Keep only Pareto-optimal solutions
            solutions = []
            for i, idx in enumerate(pareto_idx):
                sol = all_solutions[idx].copy()
                sol['id'] = f'sol_{i+1:03d}'
                sol['is_pareto'] = True
                solutions.append(sol)

            # Sort by grid_kwh for consistent ordering
            solutions.sort(key=lambda x: x['objectives']['grid_kwh'])
            print(f"Combined ε-Pareto front: {len(solutions)} solutions")
        else:
            solutions = new_solutions
    else:
        solutions = new_solutions

    print(f"Total ε-Pareto solutions: {len(solutions)}")

    # Select diverse strategies
    selected = select_diverse_strategies(solutions, n_select=args.n_select)
    print(f"Selected {len(selected)} diverse strategies")

    # Save results
    metadata = {
        'n_gen': args.generations,
        'pop_size': args.population,
        'seed': args.seed,
        'use_epsilon': use_epsilon,
        'epsilon': epsilon if use_epsilon else None,
    }

    # Get optimization history from this run
    opt_history = result.get('history')

    # Save full archive with optimization history
    save_archive(solutions, metadata, OUTPUT_DIR / 'pareto_archive.json', history=opt_history)

    # Save Pareto front as CSV
    pareto_df = pd.DataFrame([
        {**sol['variables'], **sol['objectives'], 'id': sol['id']}
        for sol in solutions
    ])
    pareto_df.to_csv(OUTPUT_DIR / 'pareto_front.csv', index=False)
    print(f"Saved: pareto_front.csv ({len(solutions)} solutions)")

    # Save selected strategies
    selected_df = pd.DataFrame([
        {**sol['variables'], **sol['objectives'], 'id': sol['id'], 'label': sol.get('label', '')}
        for sol in selected
    ])
    selected_df.to_csv(OUTPUT_DIR / 'selected_strategies.csv', index=False)
    print(f"Saved: selected_strategies.csv ({len(selected)} strategies)")

    with open(OUTPUT_DIR / 'selected_strategies.json', 'w') as f:
        json.dump(selected, f, indent=2)
    print("Saved: selected_strategies.json")

    # Create visualizations
    plot_pareto_front(solutions, selected)
    plot_strategy_comparison(selected)

    # Generate report
    report_html = generate_report(solutions, selected, metadata)
    with open(OUTPUT_DIR / 'pareto_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: pareto_report_section.html")

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)

    print("\nSelected strategies for Phase 5:")
    print("-" * 90)
    print(f"{'Label':<15} {'Comfort':<8} {'Eco':<6} {'Schedule':<13} {'Rise':<6} {'AvgTemp':<8} {'Grid':<8} {'Cost':<8}")
    print("-" * 90)

    for sol in selected:
        v = sol['variables']
        o = sol['objectives']
        schedule = f"{int(v['comfort_start']):02d}:00-{int(v['comfort_end']):02d}:00"
        print(f"{sol.get('label', sol['id']):<15} {v['setpoint_comfort']:.1f}°C   {v['setpoint_eco']:.1f}°C "
              f"{schedule:<13} {v['curve_rise']:.2f}  {o['mean_temp']:<8.1f} {o['grid_kwh']:<8.0f} {o['cost_chf']:<8.1f}")

    print("\n" + "="*60)
    print("STEP COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
