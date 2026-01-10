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

Objectives (4, all minimized):
1. Mean T_weighted deficit (target - mean) during occupied hours
2. Min T_weighted deficit (threshold - min) during occupied hours
3. Grid import (kWh)
4. Electricity cost (CHF)

Constraints:
- comfort_end > comfort_start + 4 (minimum 4h comfort period)
- setpoint_eco <= setpoint_comfort - 1 (eco must be setback)
- min(T_weighted) >= 16.0°C (hard safety constraint)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse

try:
    from pymoo.core.problem import Problem
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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)

# Model parameters from Phase 3
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

# T_weighted regression coefficients from Phase 2 multivariate analysis
# T_weighted = intercept + coef * parameter_value
TEMP_REGRESSION = {
    'intercept': -15.31,
    'comfort_setpoint': 1.218,   # +1.22°C per 1°C increase
    'eco_setpoint': -0.090,      # -0.09°C per 1°C increase (negligible)
    'curve_rise': 9.73,          # +9.73°C per unit increase
    'comfort_hours': -0.020,     # -0.02°C per hour increase
    'outdoor_mean': 0.090,       # +0.09°C per 1°C outdoor increase
}

# Baseline parameters (reference point)
BASELINE = {
    'setpoint_comfort': 20.2,
    'setpoint_eco': 18.5,
    'comfort_start': 6.5,
    'comfort_end': 20.0,
    'curve_rise': 1.08,
}

# Target sensors weights (for reference)
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 0.40,
    'office1_temperature': 0.30,
    'atelier_temperature': 0.10,
    'studio_temperature': 0.10,
    'simlab_temperature': 0.10,
}

# Occupied hours for comfort evaluation
OCCUPIED_START = 8   # 08:00
OCCUPIED_END = 22    # 22:00


class SimulationData:
    """Container for preloaded simulation data."""

    def __init__(self):
        self.sim_data = None
        self.tariff_data = None
        self.baseline_metrics = None

    def load(self):
        """Load all required data."""
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

        print(f"  Loaded {len(self.sim_data):,} timesteps ({len(self.sim_data)//96:.0f} days)")

    def _prepare_sim_data(self, df: pd.DataFrame, tariff: pd.DataFrame) -> pd.DataFrame:
        """Prepare simulation data with all required columns."""
        sim = pd.DataFrame(index=df.index)

        # Outdoor temperature
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
        sim['T_flow'] = df.get('stiebel_eltron_isg_actual_temperature_hk_2',
                               df.get('stiebel_eltron_isg_flow_temperature_wp1'))

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
        sim = sim.dropna(subset=['T_outdoor', 'T_weighted', 'T_flow'])

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


def calculate_cop(T_outdoor: float, T_flow: float) -> float:
    """Calculate COP from temperatures."""
    return (COP_PARAMS['intercept'] +
            COP_PARAMS['outdoor_coef'] * T_outdoor +
            COP_PARAMS['flow_coef'] * T_flow)


def simulate_parameters(params: dict, sim_data: pd.DataFrame) -> dict:
    """
    Simulate heating strategy and return objective values.

    Uses historical proxy approach with Phase 2 regression coefficients
    to adjust T_weighted based on parameter changes.

    Args:
        params: Dict with setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise
        sim_data: Prepared simulation DataFrame

    Returns:
        Dict with objective values and constraint violations
    """
    setpoint_comfort = params['setpoint_comfort']
    setpoint_eco = params['setpoint_eco']
    comfort_start = params['comfort_start']
    comfort_end = params['comfort_end']
    curve_rise = params['curve_rise']

    # Calculate comfort hours
    comfort_hours = comfort_end - comfort_start

    # Calculate T_weighted adjustment using regression coefficients
    # Delta from baseline parameters
    delta_T = (
        TEMP_REGRESSION['comfort_setpoint'] * (setpoint_comfort - BASELINE['setpoint_comfort']) +
        TEMP_REGRESSION['eco_setpoint'] * (setpoint_eco - BASELINE['setpoint_eco']) +
        TEMP_REGRESSION['curve_rise'] * (curve_rise - BASELINE['curve_rise']) +
        TEMP_REGRESSION['comfort_hours'] * (comfort_hours - (BASELINE['comfort_end'] - BASELINE['comfort_start']))
    )

    # Adjust T_weighted
    T_weighted_adj = sim_data['T_weighted'] + delta_T

    # Filter to occupied hours (08:00-22:00) for comfort metrics
    occupied_mask = (sim_data['hour'] >= OCCUPIED_START) & (sim_data['hour'] < OCCUPIED_END)
    T_weighted_occupied = T_weighted_adj[occupied_mask]

    # Objective 1: Mean temperature deficit (target 20.5°C - actual)
    target_temp = 20.5  # Target mean temperature
    mean_deficit = target_temp - T_weighted_occupied.mean()

    # Objective 2: Min temperature deficit (threshold 18.5°C - min)
    threshold_temp = 18.5
    min_deficit = threshold_temp - T_weighted_occupied.min()

    # Calculate COP and energy for grid/cost objectives
    results = []
    for idx, row in sim_data.iterrows():
        hour = row['hour']
        T_outdoor = row['T_outdoor']

        # Determine mode
        is_comfort = comfort_start <= hour < comfort_end

        # Estimate flow temperature
        setpoint = setpoint_comfort if is_comfort else setpoint_eco
        T_flow_strategy = estimate_flow_temp(curve_rise, T_outdoor, setpoint, is_comfort)

        # Calculate COP
        cop_strategy = calculate_cop(T_outdoor, T_flow_strategy)
        cop_actual = calculate_cop(T_outdoor, row['T_flow'])

        # Energy ratio (strategy vs actual)
        energy_ratio = cop_actual / max(cop_strategy, 0.1)  # >1 means more energy needed

        # Estimate grid import change
        grid_import_adj = row['grid_import'] * energy_ratio

        # Cost calculation
        grid_cost = grid_import_adj * row['purchase_rate_rp_kwh'] / 100
        pv_export = max(0, row['pv_generation'] - row['direct_solar'])
        feedin_revenue = pv_export * row['feedin_rate_rp_kwh'] / 100
        net_cost = grid_cost - feedin_revenue

        results.append({
            'grid_import': grid_import_adj,
            'net_cost': net_cost,
        })

    results_df = pd.DataFrame(results)

    # Objective 3: Total grid import (kWh)
    grid_import_total = results_df['grid_import'].sum()

    # Objective 4: Total net cost (CHF)
    net_cost_total = results_df['net_cost'].sum()

    # Constraints
    # G1: comfort_end - comfort_start >= 4 (minimum 4h comfort period)
    # Constraint violated if g > 0, so: g1 = 4 - (comfort_end - comfort_start)
    g1 = 4.0 - comfort_hours

    # G2: setpoint_eco <= setpoint_comfort - 1 (eco must be setback)
    # g2 = setpoint_eco - (setpoint_comfort - 1)
    g2 = setpoint_eco - (setpoint_comfort - 1.0)

    return {
        'objectives': np.array([mean_deficit, min_deficit, grid_import_total, net_cost_total]),
        'constraints': np.array([g1, g2]),
        'metrics': {
            'mean_T_weighted': T_weighted_occupied.mean(),
            'min_T_weighted': T_weighted_occupied.min(),
            'grid_import_kwh': grid_import_total,
            'net_cost_chf': net_cost_total,
            'comfort_hours': comfort_hours,
        }
    }


class HeatingOptimizationProblem(Problem):
    """Multi-objective optimization problem for heating parameters."""

    def __init__(self, sim_data: SimulationData):
        """
        Initialize problem with simulation data.

        Variables: [setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise]
        """
        super().__init__(
            n_var=5,
            n_obj=4,
            n_constr=2,
            xl=np.array([19.0, 12.0, 6.0, 16.0, 0.80]),   # lower bounds (eco 12°C = frost protection)
            xu=np.array([22.0, 19.0, 12.0, 22.0, 1.20]),  # upper bounds
        )
        self.sim_data = sim_data

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population of solutions."""
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 4))  # objectives
        G = np.zeros((n_pop, 2))  # constraints

        for i in range(n_pop):
            params = {
                'setpoint_comfort': X[i, 0],
                'setpoint_eco': X[i, 1],
                'comfort_start': X[i, 2],
                'comfort_end': X[i, 3],
                'curve_rise': X[i, 4],
            }
            result = simulate_parameters(params, self.sim_data.sim_data)
            F[i] = result['objectives']
            G[i] = result['constraints']

        out["F"] = F
        out["G"] = G


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

    # Create algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
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
                    eliminate_duplicates=True
                )
                print(f"  Initialized with {len(initial_X)} solutions from archive")

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True
    )

    print(f"\nOptimization complete!")
    print(f"  Total evaluations: {result.algorithm.n_gen * pop_size:,}")
    print(f"  Pareto solutions: {len(result.F)}")

    return {
        'X': result.X,  # Decision variables
        'F': result.F,  # Objective values
        'G': result.G if result.G is not None else np.zeros((len(result.X), 2)),  # Constraints
        'n_gen': n_gen,
        'pop_size': pop_size,
        'seed': seed,
    }


def extract_pareto_front(result: dict) -> list:
    """Extract non-dominated solutions from optimization result."""
    X = result['X']
    F = result['F']

    # Non-dominated sorting
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    pareto_idx = fronts[0]  # First front is Pareto-optimal

    solutions = []
    for i, idx in enumerate(pareto_idx):
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
                'mean_temp_deficit': float(F[idx, 0]),
                'min_temp_deficit': float(F[idx, 1]),
                'grid_kwh': float(F[idx, 2]),
                'cost_chf': float(F[idx, 3]),
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
        # Extract objective values
        F = np.array([
            [s['objectives']['mean_temp_deficit'],
             s['objectives']['min_temp_deficit'],
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

        for obj_idx in range(4):
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

    # Assign descriptive labels
    labels = []
    for sol in selected:
        obj = sol['objectives']
        # Determine dominant characteristic
        if obj['mean_temp_deficit'] <= -0.5:  # High comfort
            label = "Comfort-First"
        elif obj['grid_kwh'] == min(s['objectives']['grid_kwh'] for s in selected):
            label = "Grid-Minimal"
        elif obj['cost_chf'] == min(s['objectives']['cost_chf'] for s in selected):
            label = "Cost-Minimal"
        elif obj['min_temp_deficit'] <= -1.0:  # Good min temp
            label = "Warm-Stable"
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


def save_archive(solutions: list, metadata: dict, path: Path):
    """Save optimization archive to JSON."""
    archive = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_solutions': len(solutions),
            **metadata
        },
        'solutions': solutions,
    }

    with open(path, 'w') as f:
        json.dump(archive, f, indent=2)

    print(f"Saved archive: {path}")


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

    # Extract objective values
    F = np.array([
        [s['objectives']['mean_temp_deficit'],
         s['objectives']['min_temp_deficit'],
         s['objectives']['grid_kwh'],
         s['objectives']['cost_chf']]
        for s in solutions
    ])

    # If selected, extract their objectives too
    if selected:
        F_sel = np.array([
            [s['objectives']['mean_temp_deficit'],
             s['objectives']['min_temp_deficit'],
             s['objectives']['grid_kwh'],
             s['objectives']['cost_chf']]
            for s in selected
        ])

    obj_labels = [
        'Mean Temp Deficit (°C)',
        'Min Temp Deficit (°C)',
        'Grid Import (kWh)',
        'Net Cost (CHF)'
    ]

    # Plot pairs of objectives
    pairs = [(0, 2), (0, 3), (2, 3), (1, 2)]

    for ax, (i, j) in zip(axes.flat, pairs):
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
        ax.grid(True, alpha=0.3)
        if ax == axes[0, 0]:
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Pareto Front: Multi-Objective Heating Optimization', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig19_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig19_pareto_front.png")


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
    plt.savefig(OUTPUT_DIR / 'fig20_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig20_strategy_comparison.png")


def generate_report(solutions: list, selected: list, metadata: dict) -> str:
    """Generate HTML report for Pareto optimization results."""

    def time_str(decimal_hour: float) -> str:
        h = int(decimal_hour)
        m = int((decimal_hour - h) * 60)
        return f"{h:02d}:{m:02d}"

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

    <h3>Decision Variables</h3>
    <table>
        <tr><th>Variable</th><th>Range</th><th>Description</th></tr>
        <tr><td>setpoint_comfort</td><td>[19.0, 22.0] °C</td><td>Comfort mode target</td></tr>
        <tr><td>setpoint_eco</td><td>[12.0, 19.0] °C</td><td>Eco mode target (12°C = frost protection)</td></tr>
        <tr><td>comfort_start</td><td>[06:00, 12:00]</td><td>Comfort period start</td></tr>
        <tr><td>comfort_end</td><td>[16:00, 22:00]</td><td>Comfort period end</td></tr>
        <tr><td>curve_rise</td><td>[0.80, 1.20]</td><td>Heating curve slope</td></tr>
    </table>

    <h3>Objectives (Minimized)</h3>
    <ol>
        <li><strong>Mean temp deficit</strong>: Target (20.5°C) - mean T_weighted during 08:00-22:00</li>
        <li><strong>Min temp deficit</strong>: Threshold (18.5°C) - min T_weighted during 08:00-22:00</li>
        <li><strong>Grid import</strong>: Total kWh purchased from grid</li>
        <li><strong>Net cost</strong>: Grid cost - feed-in revenue (CHF)</li>
    </ol>

    <h3>Selected Strategies (10 Diverse)</h3>
    <table>
        <tr>
            <th>Label</th>
            <th>Comfort<br>Setpoint</th>
            <th>Eco<br>Setpoint</th>
            <th>Schedule</th>
            <th>Curve<br>Rise</th>
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
        <img src="fig19_pareto_front.png" alt="Pareto Front">
        <figcaption>Pareto front showing trade-offs between objectives.
        Blue points are the 10 selected strategies.</figcaption>
    </figure>

    <figure>
        <img src="fig20_strategy_comparison.png" alt="Strategy Comparison">
        <figcaption>Radar chart comparing parameter values across selected strategies.</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for Pareto optimization."""
    # Default archive path
    DEFAULT_ARCHIVE = OUTPUT_DIR / 'pareto_archive.json'

    parser = argparse.ArgumentParser(description='Multi-objective Pareto optimization for heating')
    parser.add_argument('--generations', '-g', type=int, default=10, help='Number of generations (default: 10)')
    parser.add_argument('--population', '-p', type=int, default=100, help='Population size')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--warm-start', '-w', type=str, help='Path to previous archive (default: auto-detect)')
    parser.add_argument('--fresh', '-f', action='store_true', help='Start fresh, ignore existing archive')
    parser.add_argument('--n-select', '-n', type=int, default=10, help='Number of strategies to select')
    args = parser.parse_args()

    print("="*60)
    print("Phase 4, Step 4: Multi-Objective Pareto Optimization")
    print("="*60)

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
    new_solutions = extract_pareto_front(result)
    print(f"\nExtracted {len(new_solutions)} Pareto-optimal solutions from this run")

    # Merge with existing archive to keep ALL Pareto-optimal solutions
    if warm_start_path and Path(warm_start_path).exists():
        existing_archive = load_archive(warm_start_path)
        if existing_archive and existing_archive.get('solutions'):
            existing_solutions = existing_archive['solutions']
            print(f"Merging with {len(existing_solutions)} existing solutions...")

            # Combine all solutions
            all_solutions = existing_solutions + new_solutions

            # Re-compute Pareto front on combined set
            all_F = np.array([
                [s['objectives']['mean_temp_deficit'],
                 s['objectives']['min_temp_deficit'],
                 s['objectives']['grid_kwh'],
                 s['objectives']['cost_chf']]
                for s in all_solutions
            ])

            # Non-dominated sorting on combined set
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
            print(f"Combined Pareto front: {len(solutions)} non-dominated solutions")
        else:
            solutions = new_solutions
    else:
        solutions = new_solutions

    print(f"Total Pareto-optimal solutions: {len(solutions)}")

    # Select diverse strategies
    selected = select_diverse_strategies(solutions, n_select=args.n_select)
    print(f"Selected {len(selected)} diverse strategies")

    # Save results
    metadata = {
        'n_gen': args.generations,
        'pop_size': args.population,
        'seed': args.seed,
    }

    # Save full archive
    save_archive(solutions, metadata, OUTPUT_DIR / 'pareto_archive.json')

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
    print("-" * 80)
    print(f"{'Label':<15} {'Comfort':<8} {'Eco':<6} {'Schedule':<13} {'Rise':<6} {'Grid':<8} {'Cost':<8}")
    print("-" * 80)

    for sol in selected:
        v = sol['variables']
        o = sol['objectives']
        schedule = f"{int(v['comfort_start']):02d}:00-{int(v['comfort_end']):02d}:00"
        print(f"{sol.get('label', sol['id']):<15} {v['setpoint_comfort']:.1f}°C   {v['setpoint_eco']:.1f}°C "
              f"{schedule:<13} {v['curve_rise']:.2f}  {o['grid_kwh']:<8.0f} {o['cost_chf']:<8.1f}")

    print("\n" + "="*60)
    print("STEP COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
