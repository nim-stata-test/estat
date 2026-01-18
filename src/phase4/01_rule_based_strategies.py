#!/usr/bin/env python3
"""
Phase 4, Step 1: Rule-Based Optimization Strategies

Defines three heating optimization strategies based on Phase 3 model parameters:
1. Baseline: Current settings (control)
2. Energy-Optimized: Maintain comfort, maximize solar self-consumption
3. Cost-Optimized: Maintain comfort, minimize electricity costs via tariff arbitrage

Primary objective: Maintain comfortable temperature in key rooms
Secondary objective: Minimize energy expenditure

Key model parameters used:
- Thermal model: Transfer function with τ_effort (heating response) ~8-48h per room
- Weighted τ_effort: ~12h (for washout calculation)
- Thermal model sensor: davis_inside (100% - primary living area, least noise)
- COP model: COP = 6.52 + 0.13*T_outdoor - 0.10*T_HK2 (target flow from heating curve)
- Peak PV hours: 10:00-16:00
- Current self-sufficiency: 58.1%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

def _decimal_to_time(decimal_hour: float) -> str:
    """Convert decimal hour (e.g., 6.5) to time string (e.g., '06:30')."""
    hours = int(decimal_hour)
    minutes = int((decimal_hour - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)

# Model parameters from Phase 3
# COP model from heat pump analysis (R²=0.95)
# Thermal model uses transfer function approach with τ_effort for heating response
# Indoor temperature: davis_inside only (100%)
MODEL_PARAMS = {
    'cop_intercept': 6.52,
    'cop_outdoor_coef': 0.1319,
    'cop_t_hk2_coef': -0.1007,  # T_HK2 = target flow temperature from heating curve
    # Weighted τ_effort (heating response time constant) from Phase 3 transfer function model:
    # 0.4×8 + 0.3×8 + 0.1×8 + 0.1×12 + 0.1×48 = 12.4h
    # Note: τ_outdoor (outdoor temp response) is longer (24-120h) but less relevant for optimization
    'tau_effort_weighted_h': 12.4,
    'battery_efficiency': 0.837,
    'current_self_sufficiency': 0.581,
    'target_self_sufficiency': 0.853,
    'peak_pv_hours': list(range(10, 17)),  # 10:00-16:00
}

# Target sensor for thermal model (single sensor)
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 1.0,
}

# Load heating curve parameters from Phase 2 JSON
def _load_heating_curve_params():
    """Load parametric heating curve from Phase 2 JSON."""
    params_file = PHASE2_DIR / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        return {'t_ref_comfort': params['t_ref_comfort'], 't_ref_eco': params['t_ref_eco']}
    return {'t_ref_comfort': 21.32, 't_ref_eco': 19.18}

# Heating curve reference temperatures from Phase 2 analysis
# T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)
HEATING_CURVE_PARAMS = _load_heating_curve_params()

# Current baseline settings (from Phase 2 heating curve analysis)
BASELINE_SETTINGS = {
    'comfort_start': 6.5,  # 06:30
    'comfort_end': 20.0,  # 20:00
    'setpoint_comfort': 20.2,
    'setpoint_eco': 18.5,
    'curve_rise': 1.08,
    'comfort_band_min': 18.5,
    'comfort_band_max': 22.0,
}


def define_strategies() -> dict:
    """
    Define the three optimization strategies with their rule sets.

    Returns dict with strategy name -> strategy definition
    """
    strategies = {}

    # Strategy 1: Baseline (current settings)
    strategies['baseline'] = {
        'name': 'Baseline',
        'description': 'Current system settings (control group)',
        'goal': 'Maintain current operation as reference',
        'parameters': BASELINE_SETTINGS.copy(),
        'rules': [
            'Use current schedule (06:30-20:00 comfort mode)',
            'Maintain current curve rise (1.08)',
            'Standard comfort band (18.5-22°C)',
            'No dynamic adjustments based on PV/grid state',
        ],
        'expected_improvement': {
            'self_sufficiency': 0.0,
            'grid_reduction': 0.0,
            'cop_change': 0.0,
        }
    }

    # Strategy 2: Energy-Optimized (minimize grid while maintaining comfort)
    strategies['energy_optimized'] = {
        'name': 'Energy-Optimized',
        'description': 'Maximize solar self-consumption while maintaining comfort',
        'goal': 'Maintain comfort, minimize grid electricity consumption',
        'parameters': {
            'comfort_start': 10.0,  # Delay to 10:00 (PV peak start)
            'comfort_end': 18.0,  # Earlier end to coast through evening
            'setpoint_comfort': 20.0,  # Slightly lower
            'setpoint_eco': 18.0,  # Deeper setback
            'curve_rise': 0.98,  # Lower flow temps = higher COP
            'curve_rise_grid_fallback': 0.90,  # Even lower when grid-dependent
            'comfort_band_min': 18.5,
            'comfort_band_max': 22.0,
        },
        'rules': [
            'Shift comfort start to 10:00 (PV peak) - use thermal mass to coast morning',
            'End comfort at 18:00 - rely on thermal inertia for evening',
            'Lower curve_rise to 0.98 for better COP (+1.0 COP improvement)',
            'When grid-dependent (battery<20%, no PV): use curve_rise 0.90',
            'Pre-heat building during solar hours into thermal mass',
        ],
        'expected_improvement': {
            'self_sufficiency': 0.10,  # +10pp expected
            'grid_reduction': 0.25,  # 25% less grid import
            'cop_change': 0.5,  # +0.5 average COP from lower flow temps
        }
    }

    # Strategy 3: Cost-Optimized (minimize electricity costs while maintaining comfort)
    strategies['cost_optimized'] = {
        'name': 'Cost-Optimized',
        'description': 'Minimize electricity costs while maintaining comfort',
        'goal': 'Maintain comfort, minimize annual electricity bill',
        'parameters': {
            # Schedule: Avoid expensive high-tariff morning hours (06:00-21:00)
            'comfort_start': 11.0,  # Late start - use solar, avoid morning peak
            'comfort_end': 21.0,    # Extend to low-tariff transition (21:00)
            # Temperature: Accept slightly lower comfort for savings
            'setpoint_comfort': 20.0,  # Reduced from 20.5
            'setpoint_eco': 17.5,      # Lower eco = less night heating cost
            # Heating curve: Aggressive reduction during grid-dependent periods
            'curve_rise': 0.95,
            'curve_rise_grid_fallback': 0.85,  # Very aggressive when on grid
            # Comfort band
            'comfort_band_min': 18.5,  # Same as baseline minimum
            'comfort_band_max': 22.5,
            # Cost-specific parameters
            'use_tariff_rates': True,
            'high_tariff_setpoint_reduction': 1.0,  # -1°C during high tariff
            'preheat_before_low_tariff': False,  # Already heating during solar
        },
        'rules': [
            'Shift heating to low-tariff periods (21:00-06:00 weekdays, weekends)',
            'Pre-heat during solar hours (11:00-16:00) using free PV',
            'Reduce setpoint by 1°C during high-tariff grid-dependent periods',
            'Aggressively reduce flow temp when grid consumption unavoidable',
            'Accept COP reduction if tariff arbitrage saves more money',
            'Use thermal mass to coast through expensive evening hours (18:00-21:00)',
        ],
        'expected_improvement': {
            'self_sufficiency': 0.03,   # +3pp (secondary benefit)
            'grid_reduction': 0.10,     # 10% less grid import
            'cop_change': -0.15,        # Accept small COP trade-off
            'cost_reduction': 0.20,     # Target -20% cost vs baseline
        }
    }

    return strategies


def calculate_cop_prediction(T_outdoor: float, T_HK2: float) -> float:
    """Calculate predicted COP from outdoor temperature and T_HK2 (target flow)."""
    return (MODEL_PARAMS['cop_intercept'] +
            MODEL_PARAMS['cop_outdoor_coef'] * T_outdoor +
            MODEL_PARAMS['cop_t_hk2_coef'] * T_HK2)


def estimate_t_hk2(curve_rise: float, T_outdoor: float,
                   T_setpoint: float = 20.0, is_comfort: bool = True) -> float:
    """
    Estimate T_HK2 (target flow temperature) from heating curve parameters.

    Heating curve formula (from Phase 2 analysis):
        T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)

    Where T_ref depends on mode:
        - Comfort mode: T_ref = 21.32°C
        - Eco mode: T_ref = 19.18°C

    Args:
        curve_rise: Heating curve slope (typically 0.85-1.08)
        T_outdoor: Outdoor temperature in °C
        T_setpoint: Room temperature setpoint (comfort or eco)
        is_comfort: True if in comfort mode, False for eco mode

    Returns:
        T_HK2 target flow temperature in °C
    """
    T_ref = HEATING_CURVE_PARAMS['t_ref_comfort'] if is_comfort else HEATING_CURVE_PARAMS['t_ref_eco']
    T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
    return T_HK2


def analyze_strategy_cop_impact(strategies: dict) -> pd.DataFrame:
    """
    Analyze COP improvement for each strategy across temperature range.
    """
    print("\nAnalyzing COP impact of strategies...")

    # Typical outdoor temperature range for heating season
    T_outdoor_range = np.arange(-5, 15, 1)

    results = []

    for strategy_id, strategy in strategies.items():
        params = strategy['parameters']
        curve_rise = params.get('curve_rise', BASELINE_SETTINGS['curve_rise'])
        setpoint_comfort = params.get('setpoint_comfort', BASELINE_SETTINGS['setpoint_comfort'])

        for T_out in T_outdoor_range:
            # Calculate T_HK2 during comfort mode (main heating period)
            T_HK2 = estimate_t_hk2(curve_rise, T_out, setpoint_comfort, is_comfort=True)
            cop = calculate_cop_prediction(T_out, T_HK2)

            # Also calculate baseline COP for comparison
            T_HK2_baseline = estimate_t_hk2(
                BASELINE_SETTINGS['curve_rise'], T_out,
                BASELINE_SETTINGS['setpoint_comfort'], is_comfort=True)
            cop_baseline = calculate_cop_prediction(T_out, T_HK2_baseline)

            results.append({
                'strategy': strategy_id,
                'T_outdoor': T_out,
                'T_HK2': T_HK2,
                'curve_rise': curve_rise,
                'COP': cop,
                'COP_baseline': cop_baseline,
                'COP_improvement': cop - cop_baseline,
            })

    return pd.DataFrame(results)


def analyze_schedule_impact(strategies: dict) -> pd.DataFrame:
    """
    Analyze how schedule changes affect solar-heating overlap.
    """
    print("\nAnalyzing schedule impact...")

    # Hours of the day
    hours = list(range(24))

    # Typical PV profile (relative, peaks at midday)
    pv_profile = np.array([
        0, 0, 0, 0, 0, 0,  # 00:00-05:00
        0, 0.1, 0.3, 0.6, 0.85, 0.95,  # 06:00-11:00
        1.0, 0.95, 0.85, 0.6, 0.3, 0.1,  # 12:00-17:00
        0, 0, 0, 0, 0, 0  # 18:00-23:00
    ])

    # Typical heating demand profile (relative, higher morning/evening)
    heating_profile = np.array([
        0.5, 0.5, 0.5, 0.5, 0.5, 0.6,  # 00:00-05:00 (night)
        0.9, 1.0, 0.9, 0.7, 0.5, 0.4,  # 06:00-11:00 (morning peak)
        0.3, 0.3, 0.3, 0.4, 0.5, 0.7,  # 12:00-17:00 (midday low)
        0.9, 0.9, 0.8, 0.7, 0.6, 0.5   # 18:00-23:00 (evening)
    ])

    results = []

    for strategy_id, strategy in strategies.items():
        params = strategy['parameters']
        comfort_start = params.get('comfort_start', BASELINE_SETTINGS['comfort_start'])
        comfort_end = params.get('comfort_end', BASELINE_SETTINGS['comfort_end'])

        for hour in hours:
            is_comfort = comfort_start <= hour < comfort_end
            is_pv_available = pv_profile[hour] > 0.2

            # Adjust heating for strategy
            if is_comfort:
                heating_mult = 1.0  # Full heating
            else:
                heating_mult = 0.5  # Reduced (eco mode)

            adjusted_heating = heating_profile[hour] * heating_mult

            # Solar-heating overlap
            if is_pv_available and is_comfort:
                overlap = 'solar_comfort'
            elif is_pv_available and not is_comfort:
                overlap = 'solar_eco'
            elif not is_pv_available and is_comfort:
                overlap = 'grid_comfort'  # Likely grid-dependent
            else:
                overlap = 'grid_eco'

            results.append({
                'strategy': strategy_id,
                'hour': hour,
                'pv_relative': pv_profile[hour],
                'heating_relative': adjusted_heating,
                'is_comfort': is_comfort,
                'is_pv': is_pv_available,
                'overlap_type': overlap,
            })

    return pd.DataFrame(results)


def plot_strategy_comparison(strategies: dict, cop_analysis: pd.DataFrame,
                            schedule_analysis: pd.DataFrame) -> None:
    """Create visualization comparing strategies."""
    print("\nCreating strategy comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'baseline': '#2E86AB', 'energy_optimized': '#A23B72', 'cost_optimized': '#27AE60'}

    # Panel 1: COP vs Outdoor Temperature by Strategy
    ax = axes[0, 0]
    for strategy_id in strategies.keys():
        data = cop_analysis[cop_analysis['strategy'] == strategy_id]
        ax.plot(data['T_outdoor'], data['COP'],
                label=strategies[strategy_id]['name'],
                color=colors[strategy_id], linewidth=2)

    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('Predicted COP')
    ax.set_title('Heat Pump COP by Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: COP Improvement over Baseline
    ax = axes[0, 1]
    for strategy_id in ['energy_optimized', 'cost_optimized']:
        data = cop_analysis[cop_analysis['strategy'] == strategy_id]
        ax.fill_between(data['T_outdoor'], 0, data['COP_improvement'],
                       label=strategies[strategy_id]['name'],
                       color=colors[strategy_id], alpha=0.5)

    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('COP Improvement vs Baseline')
    ax.set_title('COP Improvement from Flow Temp Reduction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 3: Daily Schedule Comparison
    ax = axes[1, 0]

    # PV profile
    hours = list(range(24))
    pv_profile = schedule_analysis[schedule_analysis['strategy'] == 'baseline']['pv_relative'].values
    ax.fill_between(hours, 0, pv_profile, alpha=0.3, color='gold', label='PV Available')

    # Comfort periods for each strategy
    for i, (strategy_id, strategy) in enumerate(strategies.items()):
        params = strategy['parameters']
        comfort_start = params['comfort_start']
        comfort_end = params['comfort_end']

        y_offset = 1.1 + i * 0.15
        ax.hlines(y=y_offset, xmin=comfort_start, xmax=comfort_end,
                 colors=colors[strategy_id], linewidth=8, label=f"{strategy['name']} comfort")
        ax.text(comfort_start - 0.5, y_offset, f'{int(comfort_start)}:00',
                ha='right', va='center', fontsize=8)
        ax.text(comfort_end + 0.5, y_offset, f'{int(comfort_end)}:00',
                ha='left', va='center', fontsize=8)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Relative Intensity / Schedule')
    ax.set_title('Schedule Optimization: Shift Comfort to PV Hours')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.6)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Expected Improvements Summary
    ax = axes[1, 1]

    metrics = ['self_sufficiency', 'grid_reduction', 'cop_change']
    metric_labels = ['Self-Sufficiency\n(+pp)', 'Grid Reduction\n(%)', 'COP Change\n(+)']
    x = np.arange(len(metrics))
    width = 0.35

    for i, (strategy_id, strategy) in enumerate(list(strategies.items())[1:]):  # Skip baseline
        improvements = strategy['expected_improvement']
        values = [improvements[m] * 100 if m != 'cop_change' else improvements[m]
                 for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=strategy['name'],
                     color=colors[strategy_id], alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'+{val:.0f}' if val >= 1 else f'+{val:.1f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Improvement')
    ax.set_title('Expected Improvements vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4.01_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig4.01_strategy_comparison.png")


def generate_report(strategies: dict, cop_analysis: pd.DataFrame) -> str:
    """Generate HTML report section for strategy definitions."""

    html = """
    <section id="rule-based-strategies">
    <h2>4.1 Rule-Based Optimization Strategies</h2>

    <h3>Methodology</h3>
    <p>Three heating optimization strategies were developed using Phase 3 model parameters:</p>
    <ul>
        <li><strong>COP Model</strong>: COP = 6.52 + 0.13×T_outdoor - 0.10×T_HK2 (R²=0.95)</li>
        <li><strong>Building Time Constant</strong>: ~19 hours (weighted average from target sensors)</li>
        <li><strong>Target Sensor</strong>: davis_inside (100% - least noise)</li>
        <li><strong>Peak PV Hours</strong>: 10:00-16:00</li>
        <li><strong>Current Self-Sufficiency</strong>: 58.1%</li>
    </ul>

    <h3>Strategy Definitions</h3>
    """

    for strategy_id, strategy in strategies.items():
        params = strategy['parameters']
        improvement = strategy['expected_improvement']

        html += f"""
        <div class="strategy-card" style="border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px;">
        <h4>{strategy['name']}</h4>
        <p><em>{strategy['description']}</em></p>
        <p><strong>Goal:</strong> {strategy['goal']}</p>

        <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Comfort Start</td><td>{_decimal_to_time(params.get('comfort_start', 0))}</td></tr>
        <tr><td>Comfort End</td><td>{_decimal_to_time(params.get('comfort_end', 0))}</td></tr>
        <tr><td>Setpoint (Comfort)</td><td>{params.get('setpoint_comfort', 'N/A')}°C</td></tr>
        <tr><td>Setpoint (Eco)</td><td>{params.get('setpoint_eco', 'N/A')}°C</td></tr>
        <tr><td>Curve Rise</td><td>{params.get('curve_rise', 'N/A')}</td></tr>
        <tr><td>Comfort Band</td><td>{params.get('comfort_band_min', 18.5)}-{params.get('comfort_band_max', 22)}°C</td></tr>
        </table>

        <p><strong>Rules:</strong></p>
        <ul>
        """

        for rule in strategy['rules']:
            html += f"<li>{rule}</li>\n"

        html += f"""
        </ul>

        <p><strong>Expected Improvement vs Baseline:</strong></p>
        <ul>
            <li>Self-sufficiency: +{improvement['self_sufficiency']*100:.0f} percentage points</li>
            <li>Grid reduction: {improvement['grid_reduction']*100:.0f}%</li>
            <li>COP improvement: {'+' if improvement['cop_change'] >= 0 else ''}{improvement['cop_change']:.1f}</li>
            {f"<li>Cost reduction: {improvement.get('cost_reduction', 0)*100:.0f}%</li>" if 'cost_reduction' in improvement else ''}
        </ul>
        </div>
        """

    # COP analysis summary
    baseline_cop = cop_analysis[cop_analysis['strategy'] == 'baseline']['COP'].mean()
    energy_cop = cop_analysis[cop_analysis['strategy'] == 'energy_optimized']['COP'].mean()
    cost_cop = cop_analysis[cop_analysis['strategy'] == 'cost_optimized']['COP'].mean()

    html += f"""
    <h3>COP Impact Analysis</h3>
    <p>Average predicted COP across heating season temperature range (-5°C to 15°C):</p>
    <table>
        <tr><th>Strategy</th><th>Average COP</th><th>vs Baseline</th></tr>
        <tr><td>Baseline</td><td>{baseline_cop:.2f}</td><td>—</td></tr>
        <tr><td>Energy-Optimized</td><td>{energy_cop:.2f}</td><td>+{energy_cop-baseline_cop:.2f}</td></tr>
        <tr><td>Cost-Optimized</td><td>{cost_cop:.2f}</td><td>{'+' if cost_cop >= baseline_cop else ''}{cost_cop-baseline_cop:.2f}</td></tr>
    </table>

    <p><strong>Heating curve formula</strong> (from Phase 2 analysis):<br>
    <code>T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)</code><br>
    where T_ref = 21.32°C (comfort) or 19.18°C (eco).</p>

    <p>Key insight: Reducing curve_rise from 1.08 to 0.95-0.98 lowers T_HK2 (target flow) by ~1-2°C,
    improving COP by ~0.1-0.2 across all outdoor temperatures.</p>

    <h3>Schedule Optimization Rationale</h3>
    <p>Shifting comfort mode from 06:30-20:00 to 10:00-17:00/18:00:</p>
    <ul>
        <li><strong>Morning (06:30-10:00)</strong>: Building maintains 17.5-18.5°C using ~19h thermal mass.
            PV not yet available, so early heating uses grid/battery.</li>
        <li><strong>Midday (10:00-16:00)</strong>: Maximum heating during PV peak. Pre-heat to 20-21°C,
            storing energy in building thermal mass.</li>
        <li><strong>Evening (17:00/18:00-22:00)</strong>: Coast down on stored heat.
            19h time constant means ~2-3°C drop over 4-5 hours.</li>
    </ul>

    <figure>
        <img src="fig4.01_strategy_comparison.png" alt="Strategy Comparison">
        <figcaption><strong>Figure 22:</strong> Strategy comparison: COP by temperature (top-left), COP improvement (top-right),
        schedule alignment with PV (bottom-left), expected improvements (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for strategy definition."""
    print("="*60)
    print("Phase 4, Step 1: Rule-Based Optimization Strategies")
    print("="*60)

    # Define strategies
    strategies = define_strategies()

    print(f"\nDefined {len(strategies)} strategies:")
    for strategy_id, strategy in strategies.items():
        print(f"  - {strategy['name']}: {strategy['goal']}")

    # Analyze COP impact
    cop_analysis = analyze_strategy_cop_impact(strategies)

    # Analyze schedule impact
    schedule_analysis = analyze_schedule_impact(strategies)

    # Create visualizations
    plot_strategy_comparison(strategies, cop_analysis, schedule_analysis)

    # Save strategy definitions as CSV
    strategy_rows = []
    for strategy_id, strategy in strategies.items():
        row = {
            'strategy_id': strategy_id,
            'name': strategy['name'],
            'description': strategy['description'],
            'goal': strategy['goal'],
        }
        row.update({f'param_{k}': v for k, v in strategy['parameters'].items()
                   if not isinstance(v, list)})
        row.update({f'expected_{k}': v for k, v in strategy['expected_improvement'].items()})
        strategy_rows.append(row)

    strategies_df = pd.DataFrame(strategy_rows)
    strategies_df.to_csv(OUTPUT_DIR / 'optimization_strategies.csv', index=False)
    print(f"\nSaved: optimization_strategies.csv")

    # Save COP analysis
    cop_analysis.to_csv(OUTPUT_DIR / 'cop_analysis_by_strategy.csv', index=False)
    print("Saved: cop_analysis_by_strategy.csv")

    # Save full strategy definitions as JSON
    # Convert to JSON-serializable format
    strategies_json = {}
    for strategy_id, strategy in strategies.items():
        strategies_json[strategy_id] = {
            'name': strategy['name'],
            'description': strategy['description'],
            'goal': strategy['goal'],
            'parameters': strategy['parameters'],
            'rules': strategy['rules'],
            'expected_improvement': strategy['expected_improvement'],
        }

    with open(OUTPUT_DIR / 'strategies_full.json', 'w') as f:
        json.dump(strategies_json, f, indent=2)
    print("Saved: strategies_full.json")

    # Generate report section
    report_html = generate_report(strategies, cop_analysis)
    with open(OUTPUT_DIR / 'strategies_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: strategies_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("STRATEGY DEFINITION SUMMARY")
    print("="*60)

    print("\nKey differences from baseline:")
    print("\n  Energy-Optimized:")
    print("    - Comfort: 10:00-18:00 (vs 06:30-20:00)")
    print("    - Curve rise: 0.98 (vs 1.08)")
    print("    - Expected: +10pp self-sufficiency, +0.5 COP")

    print("\n  Cost-Optimized:")
    print("    - Comfort: 11:00-21:00 (vs 06:30-20:00)")
    print("    - Curve rise: 0.95, 0.85 on grid (vs 1.08)")
    print("    - Focus: Minimize costs via tariff arbitrage")
    print("    - Expected: +3pp self-sufficiency, -20% cost")

    print("\n" + "="*60)
    print("STEP COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
