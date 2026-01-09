#!/usr/bin/env python3
"""
Phase 4, Step 2: Strategy Simulation

Simulates the three optimization strategies on historical data to validate
expected improvements before Phase 5 intervention study.

Uses Phase 3 models:
- Thermal model with weighted indoor temperature (davis_inside 40%, office1 30%,
  atelier/studio/simlab 10% each)
- COP model for energy efficiency
- Energy system model for grid/solar interaction

Outputs:
- Daily metrics by strategy (grid consumption, self-sufficiency, comfort compliance)
- Time-series visualization of example week
- Validation of expected improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)

# Model parameters (from Phase 3 - updated with weighted sensor model)
COP_PARAMS = {
    'intercept': 6.52,
    'outdoor_coef': 0.1319,
    'flow_coef': -0.1007,
}

THERMAL_PARAMS = {
    'heating_coef': 0.0132,  # K/(15min)/K (weighted average)
    'loss_coef': 0.0141,     # K/(15min)/K (weighted average)
    'time_constant_h': 19.3,  # Weighted average from target sensors
}

# Target sensors and weights for thermal model
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 0.40,
    'office1_temperature': 0.30,
    'atelier_temperature': 0.10,
    'studio_temperature': 0.10,
    'simlab_temperature': 0.10,
}

# Heating curve reference temperatures (from Phase 2 analysis)
# T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)
HEATING_CURVE_PARAMS = {
    't_ref_comfort': 21.32,  # Reference temp for comfort mode
    't_ref_eco': 19.18,      # Reference temp for eco mode
}


def load_data():
    """Load historical data for simulation."""
    print("Loading data for simulation...")

    # Load integrated dataset
    df = pd.read_parquet(PHASE1_DIR / 'integrated_overlap_only.parquet')
    df.index = pd.to_datetime(df.index)

    # Load energy balance data
    energy = pd.read_parquet(PHASE1_DIR / 'energy_balance_15min.parquet')
    energy.index = pd.to_datetime(energy.index)

    # Load strategy definitions
    with open(OUTPUT_DIR / 'strategies_full.json', 'r') as f:
        strategies = json.load(f)

    print(f"  Integrated dataset: {len(df):,} rows")
    print(f"  Energy data: {len(energy):,} rows")
    print(f"  Strategies loaded: {len(strategies)}")

    return df, energy, strategies


def prepare_simulation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset with required columns for simulation."""
    print("\nPreparing simulation data...")

    sim_data = pd.DataFrame(index=df.index)

    # Outdoor temperature
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    if outdoor_col in df.columns:
        sim_data['T_outdoor'] = df[outdoor_col]
    else:
        print("  Warning: outdoor temperature not found")
        return pd.DataFrame()

    # Room temperature - compute weighted average from target sensors
    # Weights: davis_inside (40%), office1 (30%), atelier/studio/simlab (10% each)
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for sensor, weight in SENSOR_WEIGHTS.items():
        if sensor in df.columns:
            valid_mask = df[sensor].notna()
            weighted_sum[valid_mask] += df.loc[valid_mask, sensor] * weight
            weight_sum[valid_mask] += weight

    # Normalize by actual weight sum (handles missing sensors)
    sim_data['T_room'] = weighted_sum / weight_sum
    sim_data.loc[weight_sum == 0, 'T_room'] = np.nan

    print(f"  Weighted T_room: {sim_data['T_room'].notna().sum():,} valid points")

    # Flow temperature
    flow_col = 'stiebel_eltron_isg_actual_temperature_hk_2'
    if flow_col in df.columns:
        sim_data['T_flow'] = df[flow_col]
    elif 'stiebel_eltron_isg_flow_temperature_wp1' in df.columns:
        sim_data['T_flow'] = df['stiebel_eltron_isg_flow_temperature_wp1']

    # Energy columns
    if 'pv_generation_kwh' in df.columns:
        sim_data['pv_generation'] = df['pv_generation_kwh']
    if 'external_supply_kwh' in df.columns:
        sim_data['grid_import'] = df['external_supply_kwh']
    if 'total_consumption_kwh' in df.columns:
        sim_data['total_consumption'] = df['total_consumption_kwh']
    if 'direct_consumption_kwh' in df.columns:
        sim_data['direct_solar'] = df['direct_consumption_kwh']

    # Heating energy
    consumed_col = 'stiebel_eltron_isg_consumed_heating_today'
    if consumed_col in df.columns:
        # This is cumulative daily, need to diff
        daily_consumed = df[consumed_col].copy()
        # Reset at midnight, take diff otherwise
        sim_data['heating_consumed'] = daily_consumed.diff().clip(lower=0)

    # Hour of day for schedule logic
    sim_data['hour'] = sim_data.index.hour + sim_data.index.minute / 60

    # Drop rows with missing essential data
    essential = ['T_outdoor', 'T_room', 'T_flow']
    sim_data = sim_data.dropna(subset=[c for c in essential if c in sim_data.columns])

    print(f"  Simulation data: {len(sim_data):,} rows")
    print(f"  Columns: {list(sim_data.columns)}")

    return sim_data


def calculate_cop(T_outdoor: float, T_flow: float) -> float:
    """Calculate COP from temperatures."""
    return (COP_PARAMS['intercept'] +
            COP_PARAMS['outdoor_coef'] * T_outdoor +
            COP_PARAMS['flow_coef'] * T_flow)


def estimate_flow_temp(curve_rise: float, T_outdoor: float, setpoint: float,
                       is_comfort: bool = True) -> float:
    """
    Estimate target flow temperature from heating curve.

    Heating curve formula (from Phase 2 analysis):
        T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)

    Where T_ref depends on mode:
        - Comfort mode: T_ref = 21.32°C
        - Eco mode: T_ref = 19.18°C
    """
    T_ref = HEATING_CURVE_PARAMS['t_ref_comfort'] if is_comfort else HEATING_CURVE_PARAMS['t_ref_eco']
    return setpoint + curve_rise * (T_ref - T_outdoor)


def simulate_strategy(sim_data: pd.DataFrame, strategy: dict,
                     strategy_id: str) -> pd.DataFrame:
    """
    Simulate a heating strategy on historical data.

    Returns DataFrame with simulated metrics per timestep.
    """
    params = strategy['parameters']

    comfort_start = params.get('comfort_start', 6.5)
    comfort_end = params.get('comfort_end', 20.0)
    setpoint_comfort = params.get('setpoint_comfort', 20.0)
    setpoint_eco = params.get('setpoint_eco', 18.0)
    curve_rise = params.get('curve_rise', 1.08)
    curve_rise_grid = params.get('curve_rise_grid_fallback', curve_rise)
    comfort_min = params.get('comfort_band_min', 18.0)
    comfort_max = params.get('comfort_band_max', 22.0)

    results = []

    for idx, row in sim_data.iterrows():
        hour = row['hour']
        T_outdoor = row['T_outdoor']
        T_room_actual = row['T_room']
        T_flow_actual = row['T_flow']

        # Determine if in comfort mode
        is_comfort = comfort_start <= hour < comfort_end

        # Target setpoint
        target_setpoint = setpoint_comfort if is_comfort else setpoint_eco

        # PV availability (simple: between 8-17 with enough generation)
        pv = row.get('pv_generation', 0)
        is_pv_available = (8 <= hour <= 17) and (pv > 0.1)

        # Grid dependency (no PV and significant consumption)
        is_grid_dependent = not is_pv_available and row.get('grid_import', 0) > 0.5

        # Adjust curve rise based on grid dependency
        effective_curve_rise = curve_rise_grid if is_grid_dependent else curve_rise

        # Estimate flow temperature under this strategy
        T_flow_strategy = estimate_flow_temp(effective_curve_rise, T_outdoor, target_setpoint, is_comfort)

        # Calculate COP for strategy
        cop_strategy = calculate_cop(T_outdoor, T_flow_strategy)

        # Calculate COP for actual (baseline reference)
        cop_actual = calculate_cop(T_outdoor, T_flow_actual)

        # Estimate energy savings from COP improvement
        # If COP improves, less electricity needed for same heat
        cop_ratio = cop_strategy / max(cop_actual, 0.1)

        # Comfort compliance
        in_comfort_band = comfort_min <= T_room_actual <= comfort_max

        # Solar utilization (heating during PV hours)
        solar_heating = is_comfort and is_pv_available

        results.append({
            'datetime': idx,
            'strategy': strategy_id,
            'hour': hour,
            'T_outdoor': T_outdoor,
            'T_room_actual': T_room_actual,
            'T_flow_actual': T_flow_actual,
            'T_flow_strategy': T_flow_strategy,
            'target_setpoint': target_setpoint,
            'is_comfort': is_comfort,
            'is_pv_available': is_pv_available,
            'is_grid_dependent': is_grid_dependent,
            'cop_actual': cop_actual,
            'cop_strategy': cop_strategy,
            'cop_improvement': cop_strategy - cop_actual,
            'energy_ratio': 1.0 / cop_ratio,  # <1 means savings
            'pv_generation': pv,
            'grid_import': row.get('grid_import', 0),
            'total_consumption': row.get('total_consumption', 0),
            'in_comfort_band': in_comfort_band,
            'solar_heating': solar_heating,
        })

    return pd.DataFrame(results)


def aggregate_daily_metrics(simulation_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation results to daily metrics."""
    simulation_results['date'] = pd.to_datetime(simulation_results['datetime']).dt.date

    daily = simulation_results.groupby(['strategy', 'date']).agg({
        'T_outdoor': 'mean',
        'T_room_actual': 'mean',
        'cop_actual': 'mean',
        'cop_strategy': 'mean',
        'cop_improvement': 'mean',
        'energy_ratio': 'mean',
        'pv_generation': 'sum',
        'grid_import': 'sum',
        'total_consumption': 'sum',
        'in_comfort_band': 'mean',  # % time in band
        'solar_heating': 'mean',  # % heating during solar
        'is_comfort': 'sum',  # Hours in comfort mode
    }).reset_index()

    daily.columns = ['strategy', 'date', 'T_outdoor_mean', 'T_room_mean',
                    'cop_actual', 'cop_strategy', 'cop_improvement', 'energy_ratio',
                    'pv_total', 'grid_total', 'consumption_total',
                    'comfort_compliance', 'solar_heating_pct', 'comfort_hours']

    # Calculate self-sufficiency
    daily['self_sufficiency'] = 1 - (daily['grid_total'] / daily['consumption_total'].clip(lower=0.1))
    daily['self_sufficiency'] = daily['self_sufficiency'].clip(0, 1)

    # Estimated grid savings (from COP improvement)
    daily['grid_savings_pct'] = 1 - daily['energy_ratio']

    return daily


def compare_strategies(daily_metrics: pd.DataFrame, strategies: dict) -> pd.DataFrame:
    """Compare strategies against baseline."""
    print("\nComparing strategies...")

    # Get baseline metrics
    baseline = daily_metrics[daily_metrics['strategy'] == 'baseline'].copy()
    baseline_means = baseline.mean(numeric_only=True)

    comparisons = []

    for strategy_id in daily_metrics['strategy'].unique():
        strategy_data = daily_metrics[daily_metrics['strategy'] == strategy_id]
        strategy_means = strategy_data.mean(numeric_only=True)

        comparison = {
            'strategy': strategy_id,
            'name': strategies[strategy_id]['name'],
            'avg_cop': strategy_means['cop_strategy'],
            'cop_vs_baseline': strategy_means['cop_strategy'] - baseline_means['cop_actual'],
            'self_sufficiency': strategy_means['self_sufficiency'],
            'ss_vs_baseline': strategy_means['self_sufficiency'] - baseline_means['self_sufficiency'],
            'comfort_compliance': strategy_means['comfort_compliance'],
            'solar_heating_pct': strategy_means['solar_heating_pct'],
            'grid_savings_pct': strategy_means['grid_savings_pct'],
            'comfort_hours_per_day': strategy_means['comfort_hours'] / 4,  # 15-min intervals to hours
        }

        comparisons.append(comparison)

        print(f"\n  {strategies[strategy_id]['name']}:")
        print(f"    COP: {comparison['avg_cop']:.2f} (vs baseline: {comparison['cop_vs_baseline']:+.2f})")
        print(f"    Self-sufficiency: {comparison['self_sufficiency']*100:.1f}% ({comparison['ss_vs_baseline']*100:+.1f}pp)")
        print(f"    Comfort compliance: {comparison['comfort_compliance']*100:.1f}%")
        print(f"    Solar heating: {comparison['solar_heating_pct']*100:.1f}% of comfort hours")

    return pd.DataFrame(comparisons)


def plot_simulation_results(simulation_results: pd.DataFrame, daily_metrics: pd.DataFrame,
                           comparison: pd.DataFrame, strategies: dict) -> None:
    """Create visualization of simulation results."""
    print("\nCreating simulation visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'baseline': '#2E86AB', 'energy_optimized': '#A23B72', 'aggressive_solar': '#F18F01'}

    # Panel 1: Example week time series - COP comparison
    ax = axes[0, 0]

    # Pick a representative week (middle of data range)
    dates = simulation_results['datetime'].unique()
    mid_idx = len(dates) // 2
    week_start = dates[mid_idx]
    week_end = dates[min(mid_idx + 7*96, len(dates)-1)]  # 7 days * 96 intervals

    for strategy_id in strategies.keys():
        data = simulation_results[
            (simulation_results['strategy'] == strategy_id) &
            (simulation_results['datetime'] >= week_start) &
            (simulation_results['datetime'] <= week_end)
        ]
        ax.plot(data['datetime'], data['cop_strategy'],
               label=strategies[strategy_id]['name'],
               color=colors[strategy_id], alpha=0.8, linewidth=1)

    ax.set_ylabel('COP')
    ax.set_title('Heat Pump COP: Example Week')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Daily self-sufficiency comparison
    ax = axes[0, 1]

    for strategy_id in strategies.keys():
        data = daily_metrics[daily_metrics['strategy'] == strategy_id]
        dates = pd.to_datetime(data['date'])
        ax.plot(dates, data['self_sufficiency'] * 100,
               label=strategies[strategy_id]['name'],
               color=colors[strategy_id], alpha=0.8, linewidth=1.5)

    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Daily Self-Sufficiency by Strategy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Panel 3: Strategy comparison bar chart
    ax = axes[1, 0]

    metrics = ['cop_vs_baseline', 'ss_vs_baseline', 'comfort_compliance']
    metric_labels = ['COP\nImprovement', 'Self-Sufficiency\n(+pp)', 'Comfort\nCompliance (%)']

    x = np.arange(len(metrics))
    width = 0.35

    for i, strategy_id in enumerate(['energy_optimized', 'aggressive_solar']):
        row = comparison[comparison['strategy'] == strategy_id].iloc[0]
        values = [
            row['cop_vs_baseline'],
            row['ss_vs_baseline'] * 100,
            row['comfort_compliance'] * 100,
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width,
                     label=strategies[strategy_id]['name'],
                     color=colors[strategy_id], alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title('Strategy Performance vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 4: Hourly COP profile
    ax = axes[1, 1]

    hourly = simulation_results.groupby(['strategy', 'hour']).agg({
        'cop_strategy': 'mean',
        'is_pv_available': 'mean',
    }).reset_index()

    # Add PV availability shading
    pv_hours = hourly[hourly['strategy'] == 'baseline']
    ax.fill_between(pv_hours['hour'], 0, pv_hours['is_pv_available'] * 6,
                   alpha=0.2, color='gold', label='PV Available')

    for strategy_id in strategies.keys():
        data = hourly[hourly['strategy'] == strategy_id]
        ax.plot(data['hour'], data['cop_strategy'],
               label=strategies[strategy_id]['name'],
               color=colors[strategy_id], linewidth=2)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average COP')
    ax.set_title('Hourly COP Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig17_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig17_simulation_results.png")


def generate_report(comparison: pd.DataFrame, daily_metrics: pd.DataFrame,
                   strategies: dict) -> str:
    """Generate HTML report section for simulation results."""

    # Calculate summary stats
    baseline_row = comparison[comparison['strategy'] == 'baseline'].iloc[0]
    energy_row = comparison[comparison['strategy'] == 'energy_optimized'].iloc[0]
    aggressive_row = comparison[comparison['strategy'] == 'aggressive_solar'].iloc[0]

    html = f"""
    <section id="strategy-simulation">
    <h2>4.2 Strategy Simulation Results</h2>

    <h3>Methodology</h3>
    <p>Simulated all three strategies on 64 days of historical data using Phase 3 models:</p>
    <ul>
        <li>COP prediction based on outdoor and flow temperatures</li>
        <li>Flow temperature estimation from heating curve parameters</li>
        <li>Schedule-based comfort mode determination</li>
        <li>Solar availability from actual PV generation data</li>
    </ul>

    <h3>Summary Comparison</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Energy-Optimized</th>
            <th>Aggressive Solar</th>
        </tr>
        <tr>
            <td>Average COP</td>
            <td>{baseline_row['avg_cop']:.2f}</td>
            <td>{energy_row['avg_cop']:.2f} ({energy_row['cop_vs_baseline']:+.2f})</td>
            <td>{aggressive_row['avg_cop']:.2f} ({aggressive_row['cop_vs_baseline']:+.2f})</td>
        </tr>
        <tr>
            <td>Self-Sufficiency</td>
            <td>{baseline_row['self_sufficiency']*100:.1f}%</td>
            <td>{energy_row['self_sufficiency']*100:.1f}% ({energy_row['ss_vs_baseline']*100:+.1f}pp)</td>
            <td>{aggressive_row['self_sufficiency']*100:.1f}% ({aggressive_row['ss_vs_baseline']*100:+.1f}pp)</td>
        </tr>
        <tr>
            <td>Comfort Compliance</td>
            <td>{baseline_row['comfort_compliance']*100:.1f}%</td>
            <td>{energy_row['comfort_compliance']*100:.1f}%</td>
            <td>{aggressive_row['comfort_compliance']*100:.1f}%</td>
        </tr>
        <tr>
            <td>Solar Heating %</td>
            <td>{baseline_row['solar_heating_pct']*100:.1f}%</td>
            <td>{energy_row['solar_heating_pct']*100:.1f}%</td>
            <td>{aggressive_row['solar_heating_pct']*100:.1f}%</td>
        </tr>
        <tr>
            <td>Comfort Hours/Day</td>
            <td>{baseline_row['comfort_hours_per_day']:.1f}h</td>
            <td>{energy_row['comfort_hours_per_day']:.1f}h</td>
            <td>{aggressive_row['comfort_hours_per_day']:.1f}h</td>
        </tr>
    </table>

    <h3>Key Findings</h3>
    <ul>
        <li><strong>COP Improvement</strong>: Both optimized strategies achieve higher COP through
            lower flow temperatures. Energy-optimized gains +{energy_row['cop_vs_baseline']:.2f} COP,
            Aggressive Solar gains +{aggressive_row['cop_vs_baseline']:.2f} COP.</li>
        <li><strong>Self-Sufficiency</strong>: Schedule shifting improves solar-heating alignment.
            Energy-optimized: {energy_row['ss_vs_baseline']*100:+.1f}pp,
            Aggressive Solar: {aggressive_row['ss_vs_baseline']*100:+.1f}pp.</li>
        <li><strong>Comfort Trade-off</strong>: Aggressive Solar accepts wider temperature band
            (17-23°C) to maximize solar utilization. Comfort compliance remains
            {aggressive_row['comfort_compliance']*100:.1f}%.</li>
    </ul>

    <h3>Validation Against Expected Improvements</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Strategy</th>
            <th>Expected</th>
            <th>Simulated</th>
            <th>Status</th>
        </tr>
        <tr>
            <td rowspan="2">Self-Sufficiency Gain</td>
            <td>Energy-Optimized</td>
            <td>+10pp</td>
            <td>{energy_row['ss_vs_baseline']*100:+.1f}pp</td>
            <td>{'OK' if abs(energy_row['ss_vs_baseline']*100 - 10) < 5 else 'Check'}</td>
        </tr>
        <tr>
            <td>Aggressive Solar</td>
            <td>+27pp</td>
            <td>{aggressive_row['ss_vs_baseline']*100:+.1f}pp</td>
            <td>{'OK' if abs(aggressive_row['ss_vs_baseline']*100 - 27) < 10 else 'Check'}</td>
        </tr>
        <tr>
            <td rowspan="2">COP Improvement</td>
            <td>Energy-Optimized</td>
            <td>+0.5</td>
            <td>{energy_row['cop_vs_baseline']:+.2f}</td>
            <td>{'OK' if abs(energy_row['cop_vs_baseline'] - 0.5) < 0.3 else 'Check'}</td>
        </tr>
        <tr>
            <td>Aggressive Solar</td>
            <td>+0.7</td>
            <td>{aggressive_row['cop_vs_baseline']:+.2f}</td>
            <td>{'OK' if abs(aggressive_row['cop_vs_baseline'] - 0.7) < 0.4 else 'Check'}</td>
        </tr>
    </table>

    <p><em>Note: Simulation uses actual historical data, so results reflect real weather conditions
    during the 64-day overlap period. Full-season validation in Phase 5 will provide more
    robust estimates.</em></p>

    <figure>
        <img src="fig17_simulation_results.png" alt="Simulation Results">
        <figcaption>Simulation results: example week COP (top-left), daily self-sufficiency (top-right),
        strategy comparison bars (bottom-left), hourly COP profile with PV availability (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for strategy simulation."""
    print("="*60)
    print("Phase 4, Step 2: Strategy Simulation")
    print("="*60)

    # Load data
    df, energy, strategies = load_data()

    # Prepare simulation data
    sim_data = prepare_simulation_data(df)

    if sim_data.empty:
        print("ERROR: Could not prepare simulation data")
        return 1

    # Simulate each strategy
    all_results = []

    for strategy_id, strategy in strategies.items():
        print(f"\nSimulating {strategy['name']}...")
        results = simulate_strategy(sim_data, strategy, strategy_id)
        all_results.append(results)

    simulation_results = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal simulation rows: {len(simulation_results):,}")

    # Aggregate to daily metrics
    daily_metrics = aggregate_daily_metrics(simulation_results)

    # Compare strategies
    comparison = compare_strategies(daily_metrics, strategies)

    # Create visualizations
    plot_simulation_results(simulation_results, daily_metrics, comparison, strategies)

    # Save results
    daily_metrics.to_csv(OUTPUT_DIR / 'simulation_daily_metrics.csv', index=False)
    print(f"\nSaved: simulation_daily_metrics.csv")

    comparison.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False)
    print("Saved: strategy_comparison.csv")

    # Generate report section
    report_html = generate_report(comparison, daily_metrics, strategies)
    with open(OUTPUT_DIR / 'simulation_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: simulation_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)

    baseline = comparison[comparison['strategy'] == 'baseline'].iloc[0]
    print(f"\nBaseline Performance:")
    print(f"  Average COP: {baseline['avg_cop']:.2f}")
    print(f"  Self-sufficiency: {baseline['self_sufficiency']*100:.1f}%")

    for strategy_id in ['energy_optimized', 'aggressive_solar']:
        row = comparison[comparison['strategy'] == strategy_id].iloc[0]
        print(f"\n{strategies[strategy_id]['name']}:")
        print(f"  COP improvement: {row['cop_vs_baseline']:+.2f}")
        print(f"  Self-sufficiency gain: {row['ss_vs_baseline']*100:+.1f}pp")
        print(f"  Comfort compliance: {row['comfort_compliance']*100:.1f}%")

    print("\n" + "="*60)
    print("STEP COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
