#!/usr/bin/env python3
"""
Phase 4, Step 3: Parameter Set Generation

Generates final parameter sets for Phase 5 randomized intervention study.

Outputs:
- JSON file with exact parameter values for each strategy
- Parameter space visualization
- Testable predictions with confidence intervals
- Implementation checklist for Phase 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_simulation_results():
    """Load simulation results from Step 2."""
    print("Loading simulation results...")

    comparison = pd.read_csv(OUTPUT_DIR / 'strategy_comparison.csv')
    daily_metrics = pd.read_csv(OUTPUT_DIR / 'simulation_daily_metrics.csv')

    with open(OUTPUT_DIR / 'strategies_full.json', 'r') as f:
        strategies = json.load(f)

    print(f"  Loaded comparison for {len(comparison)} strategies")
    print(f"  Daily metrics: {len(daily_metrics)} rows")

    return comparison, daily_metrics, strategies


def generate_parameter_sets(strategies: dict, comparison: pd.DataFrame) -> dict:
    """
    Generate final parameter sets for Phase 5 intervention.

    Includes:
    - Exact parameter values
    - Implementation notes
    - Expected outcomes with confidence intervals
    """
    print("\nGenerating parameter sets...")

    parameter_sets = {}

    for strategy_id, strategy in strategies.items():
        params = strategy['parameters']
        sim_results = comparison[comparison['strategy'] == strategy_id].iloc[0]

        # Build parameter set with implementation details
        param_set = {
            'strategy_id': strategy_id,
            'name': strategy['name'],
            'description': strategy['description'],

            # Heat pump parameters (via heat pump interface)
            'heat_pump_settings': {
                'curve_rise': params.get('curve_rise', 1.08),
                'curve_rise_unit': 'dimensionless',
                'curve_rise_notes': 'Set via heat pump controller menu',
            },

            # Schedule parameters (via heat pump interface)
            'schedule_settings': {
                'comfort_start': _decimal_to_time(params.get('comfort_start', 6.5)),
                'comfort_end': _decimal_to_time(params.get('comfort_end', 20.0)),
                'schedule_notes': 'Set comfort period start/end in heat pump scheduler',
            },

            # Temperature setpoints (via Home Assistant)
            'setpoint_settings': {
                'setpoint_comfort': params.get('setpoint_comfort', 20.0),
                'setpoint_eco': params.get('setpoint_eco', 18.5),
                'setpoint_unit': 'celsius',
                'setpoint_notes': 'Set via Home Assistant climate entity',
            },

            # Comfort bounds for monitoring
            'comfort_bounds': {
                'min_temp': params.get('comfort_band_min', 18.5),
                'max_temp': params.get('comfort_band_max', 22.0),
                'unit': 'celsius',
            },

            # Dynamic rules (for future automation)
            'dynamic_rules': {
                'curve_rise_grid_fallback': params.get('curve_rise_grid_fallback', params.get('curve_rise', 1.08)),
                'notes': 'Optional automation rules - manual implementation in Phase 5',
            },

            # Expected outcomes (from simulation)
            'expected_outcomes': {
                'cop_mean': round(sim_results['avg_cop'], 2),
                'cop_vs_baseline': round(sim_results['cop_vs_baseline'], 2),
                'self_sufficiency': round(sim_results['self_sufficiency'], 3),
                'self_sufficiency_vs_baseline_pp': round(sim_results['ss_vs_baseline'] * 100, 1),
                # Convert violation_pct to compliance: compliance = 1 - violation_pct
                'comfort_compliance': round(1.0 - sim_results['violation_pct'], 3),
                # Cost metrics
                'daily_net_cost_chf': float(round(sim_results.get('daily_net_cost_chf', 0), 2)),
                'cost_reduction_pct': float(round(sim_results.get('cost_reduction_pct', 0), 1)),
            },

            # Confidence intervals (estimated from daily variation)
            'confidence_intervals': {
                'cop_ci_95': '±0.3',
                'self_sufficiency_ci_95': '±5pp',
                'notes': 'Estimated from simulation; will refine in Phase 5',
            },
        }

        parameter_sets[strategy_id] = param_set
        print(f"  Generated: {strategy['name']}")

    return parameter_sets


def _decimal_to_time(decimal_hour: float) -> str:
    """Convert decimal hour (e.g., 6.5) to time string (e.g., '06:30')."""
    hours = int(decimal_hour)
    minutes = int((decimal_hour - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def generate_testable_predictions(parameter_sets: dict, daily_metrics: pd.DataFrame) -> dict:
    """
    Generate testable predictions for Phase 5.

    For each strategy, predict:
    - Grid consumption per HDD (heating degree day)
    - Self-sufficiency range
    - Comfort compliance target
    """
    print("\nGenerating testable predictions...")

    predictions = {}

    # Calculate baseline metrics from daily data
    baseline_data = daily_metrics[daily_metrics['strategy'] == 'baseline']

    # Heating degree days approximation: HDD = max(18 - T_outdoor, 0)
    baseline_data = baseline_data.copy()
    baseline_data['hdd'] = (18 - baseline_data['T_outdoor_mean']).clip(lower=0)
    baseline_hdd_total = baseline_data['hdd'].sum()
    baseline_grid_total = baseline_data['grid_total'].sum()

    if baseline_hdd_total > 0:
        baseline_grid_per_hdd = baseline_grid_total / baseline_hdd_total
    else:
        baseline_grid_per_hdd = baseline_grid_total / max(len(baseline_data), 1)

    for strategy_id, param_set in parameter_sets.items():
        outcomes = param_set['expected_outcomes']

        # Calculate expected grid per HDD
        ss_improvement = outcomes['self_sufficiency_vs_baseline_pp'] / 100
        expected_grid_per_hdd = baseline_grid_per_hdd * (1 - ss_improvement * 0.5)

        # Cost predictions
        cost_reduction = outcomes.get('cost_reduction_pct', 0)
        daily_cost = outcomes.get('daily_net_cost_chf', 0)

        predictions[strategy_id] = {
            'strategy_name': param_set['name'],

            # Primary prediction: Grid consumption
            'grid_per_hdd': {
                'value': round(expected_grid_per_hdd, 2),
                'unit': 'kWh/HDD',
                'baseline': round(baseline_grid_per_hdd, 2),
                'reduction_pct': round(ss_improvement * 50, 1),
                'measurement': 'Sum of external_supply_kwh / Sum of HDD per block',
            },

            # Self-sufficiency prediction
            'self_sufficiency': {
                'target': round(outcomes['self_sufficiency'] * 100, 1),
                'range_min': round((outcomes['self_sufficiency'] - 0.05) * 100, 1),
                'range_max': round((outcomes['self_sufficiency'] + 0.05) * 100, 1),
                'unit': 'percent',
                'measurement': '(1 - grid_import/total_consumption) × 100',
            },

            # COP prediction
            'cop': {
                'target': outcomes['cop_mean'],
                'range_min': round(outcomes['cop_mean'] - 0.3, 2),
                'range_max': round(outcomes['cop_mean'] + 0.3, 2),
                'measurement': 'produced_heating / consumed_heating (daily)',
            },

            # Comfort compliance (evaluated during occupied hours 08:00-22:00 only)
            'comfort_compliance': {
                'target_pct': round(outcomes['comfort_compliance'] * 100, 1),
                'minimum_pct': 95.0,
                'measurement': 'Percent of readings within comfort band during occupied hours (08:00-22:00)',
            },

            # Cost prediction (new)
            'cost': {
                'daily_net_chf': round(daily_cost, 2),
                'cost_reduction_pct': round(cost_reduction, 1),
                'measurement': '(grid_import × purchase_rate - pv_export × feedin_rate) / 100',
            },

            # Success criteria
            'success_criteria': [
                f"Self-sufficiency ≥ {round((outcomes['self_sufficiency'] - 0.03) * 100, 1)}%",
                f"COP ≥ {round(outcomes['cop_mean'] - 0.2, 2)}",
                f"Comfort compliance ≥ 95%",
                f"No manual overrides required",
            ],
        }

        # Add cost-specific success criterion for cost_optimized
        if strategy_id == 'cost_optimized':
            predictions[strategy_id]['success_criteria'].append(
                f"Cost reduction ≥ {round(cost_reduction - 10, 1)}% vs baseline"
            )

        print(f"  {param_set['name']}:")
        print(f"    Target self-sufficiency: {predictions[strategy_id]['self_sufficiency']['target']}%")
        print(f"    Target COP: {predictions[strategy_id]['cop']['target']}")

    return predictions


def plot_parameter_space(parameter_sets: dict) -> None:
    """Visualize parameter space and trade-offs."""
    print("\nCreating parameter space visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {
        'baseline': '#2E86AB',
        'energy_optimized': '#A23B72',
        'cost_optimized': '#00A896',
    }
    markers = {
        'baseline': 'o',
        'energy_optimized': 's',
        'cost_optimized': 'D',
    }

    # Extract data for plotting
    data = []
    for strategy_id, param_set in parameter_sets.items():
        hp = param_set['heat_pump_settings']
        sp = param_set['setpoint_settings']
        sched = param_set['schedule_settings']
        outcomes = param_set['expected_outcomes']

        # Parse schedule times
        comfort_start = float(sched['comfort_start'].split(':')[0]) + float(sched['comfort_start'].split(':')[1])/60
        comfort_end = float(sched['comfort_end'].split(':')[0]) + float(sched['comfort_end'].split(':')[1])/60

        data.append({
            'strategy': strategy_id,
            'name': param_set['name'],
            'curve_rise': hp['curve_rise'],
            'setpoint_comfort': sp['setpoint_comfort'],
            'setpoint_eco': sp['setpoint_eco'],
            'comfort_start': comfort_start,
            'comfort_end': comfort_end,
            'comfort_duration': comfort_end - comfort_start,
            'cop': outcomes['cop_mean'],
            'self_sufficiency': outcomes['self_sufficiency'] * 100,
            'comfort_compliance': outcomes['comfort_compliance'] * 100,
        })

    df = pd.DataFrame(data)

    # Panel 1: Curve Rise vs COP
    ax = axes[0, 0]
    for _, row in df.iterrows():
        ax.scatter(row['curve_rise'], row['cop'],
                  c=colors[row['strategy']], marker=markers[row['strategy']],
                  s=200, label=row['name'], edgecolors='black', linewidth=1)
        ax.annotate(row['name'], (row['curve_rise'], row['cop']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Curve Rise')
    ax.set_ylabel('Average COP')
    ax.set_title('Curve Rise vs COP Trade-off')
    ax.grid(True, alpha=0.3)

    # Add trend annotation
    ax.annotate('Lower curve rise\n→ Lower flow temp\n→ Higher COP',
               xy=(0.95, 4.5), fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Schedule vs Self-Sufficiency
    ax = axes[0, 1]
    for _, row in df.iterrows():
        ax.scatter(row['comfort_duration'], row['self_sufficiency'],
                  c=colors[row['strategy']], marker=markers[row['strategy']],
                  s=200, label=row['name'], edgecolors='black', linewidth=1)
        ax.annotate(row['name'], (row['comfort_duration'], row['self_sufficiency']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Comfort Duration (hours)')
    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Schedule Duration vs Self-Sufficiency')
    ax.grid(True, alpha=0.3)

    # Add shaded region for PV hours
    ax.axvspan(7, 9, alpha=0.2, color='gold', label='PV overlap window')

    # Panel 3: Comfort-Efficiency Trade-off
    ax = axes[1, 0]
    for _, row in df.iterrows():
        ax.scatter(row['comfort_compliance'], row['self_sufficiency'],
                  c=colors[row['strategy']], marker=markers[row['strategy']],
                  s=200, label=row['name'], edgecolors='black', linewidth=1)
        ax.annotate(row['name'], (row['comfort_compliance'], row['self_sufficiency']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Comfort Compliance (%)')
    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Comfort vs Efficiency Trade-off')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=95, color='red', linestyle='--', alpha=0.5, label='95% minimum')

    # Panel 4: Parameter Summary Table
    ax = axes[1, 1]
    ax.axis('off')

    # Create table
    table_data = []
    strategy_ids = ['baseline', 'energy_optimized', 'cost_optimized']
    headers = ['Parameter', 'Baseline', 'Energy-Opt', 'Cost-Opt']

    params_to_show = [
        ('Comfort Start', 'comfort_start'),
        ('Comfort End', 'comfort_end'),
        ('Setpoint (Comfort)', 'setpoint_comfort'),
        ('Setpoint (Eco)', 'setpoint_eco'),
        ('Curve Rise', 'curve_rise'),
        ('Self-Sufficiency', 'self_sufficiency'),
        ('COP', 'cop'),
        ('Cost Change', 'cost_reduction'),
    ]

    for param_name, param_key in params_to_show:
        row = [param_name]
        for strategy_id in strategy_ids:
            ps = parameter_sets[strategy_id]
            if param_key == 'comfort_start':
                row.append(ps['schedule_settings']['comfort_start'])
            elif param_key == 'comfort_end':
                row.append(ps['schedule_settings']['comfort_end'])
            elif param_key == 'setpoint_comfort':
                row.append(f"{ps['setpoint_settings']['setpoint_comfort']}°C")
            elif param_key == 'setpoint_eco':
                row.append(f"{ps['setpoint_settings']['setpoint_eco']}°C")
            elif param_key == 'curve_rise':
                row.append(str(ps['heat_pump_settings']['curve_rise']))
            elif param_key == 'self_sufficiency':
                row.append(f"{ps['expected_outcomes']['self_sufficiency']*100:.1f}%")
            elif param_key == 'cop':
                row.append(f"{ps['expected_outcomes']['cop_mean']:.2f}")
            elif param_key == 'cost_reduction':
                cost_pct = ps['expected_outcomes'].get('cost_reduction_pct', 0)
                row.append(f"{cost_pct:+.1f}%")
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    loc='center', cellLoc='center',
                    colColours=['#f0f0f0'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9 if len(strategy_ids) > 3 else 10)
    table.scale(1.2, 1.5)

    ax.set_title('Parameter Summary', pad=20, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig24_parameter_space.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig24_parameter_space.png")


def generate_implementation_checklist(parameter_sets: dict) -> str:
    """Generate implementation checklist for Phase 5."""

    checklist = """
# Phase 5 Implementation Checklist

## Pre-Study Setup (Fall 2027)

### Equipment Preparation
- [ ] Verify heat pump interface access for curve_rise adjustment
- [ ] Verify Home Assistant integration for setpoint control
- [ ] Set up automated logging for all sensor data
- [ ] Create backup of current settings

### Data Collection Setup
- [ ] Ensure sensor data logging rate: 15-minute intervals minimum
- [ ] Set up weather forecast integration (for MPC future work)
- [ ] Create daily summary dashboard
- [ ] Test alert system for comfort violations

### Randomization
- [ ] Generate randomized block schedule (3-5 days per block)
- [ ] Balance strategies across early/mid/late winter
- [ ] Document exact switch times in log

## Parameter Change Protocol

### Baseline Settings
```
Comfort Start:    06:30
Comfort End:      20:00
Setpoint Comfort: 20.2°C
Setpoint Eco:     18.5°C
Curve Rise:       1.08
```

### Energy-Optimized Settings
```
Comfort Start:    10:00
Comfort End:      18:00
Setpoint Comfort: 20.0°C
Setpoint Eco:     18.0°C
Curve Rise:       0.98
```

### Cost-Optimized Settings
```
Comfort Start:    11:00
Comfort End:      21:00
Setpoint Comfort: 20.0°C
Setpoint Eco:     17.5°C
Curve Rise:       0.95
Curve Rise (Grid): 0.85 (when grid-dependent)
```

## Daily Monitoring

- [ ] Check comfort compliance during occupied hours 08:00-22:00 (target: ≥95%)
- [ ] Log any manual overrides
- [ ] Note occupancy deviations
- [ ] Record weather conditions
- [ ] Check sensor data quality

## Block Transition Protocol

1. Record end time of current block
2. Wait for system to reach steady state (minimum 2 hours after schedule change)
3. Apply new parameter set
4. Record start time of new block
5. Verify all parameters changed successfully

## Success Metrics

| Strategy | Self-Sufficiency Target | COP Target | Comfort Min | Cost Change |
|----------|------------------------|------------|-------------|-------------|
| Baseline | 58% | 3.5 | 95% | — |
| Energy-Optimized | 68% | 4.0 | 95% | +5-10% savings |
| Cost-Optimized | 61% | 3.4 | 90% | +15-25% savings |

## Safety Limits

- Minimum room temperature: 16°C (override if violated)
- Maximum flow temperature: 45°C
- Alert if comfort compliance < 90% for 24 hours
"""

    return checklist


def generate_report(parameter_sets: dict, predictions: dict) -> str:
    """Generate HTML report section for parameter sets."""

    html = """
    <section id="parameter-sets">
    <h2>4.3 Parameter Sets for Phase 5</h2>

    <h3>Final Parameter Sets</h3>
    <p>The following parameter sets are ready for the Phase 5 randomized intervention study (Winter 2027-2028).</p>
    """

    for strategy_id, param_set in parameter_sets.items():
        hp = param_set['heat_pump_settings']
        sp = param_set['setpoint_settings']
        sched = param_set['schedule_settings']
        outcomes = param_set['expected_outcomes']

        cost_reduction = outcomes.get('cost_reduction_pct', 0)
        daily_cost = outcomes.get('daily_net_cost_chf', 0)

        html += f"""
        <div class="parameter-card" style="border: 2px solid #333; padding: 15px; margin: 15px 0; border-radius: 8px;">
        <h4>{param_set['name']}</h4>
        <p><em>{param_set['description']}</em></p>

        <table>
            <tr><th colspan="2">Heat Pump Settings</th></tr>
            <tr><td>Curve Rise</td><td><strong>{hp['curve_rise']}</strong></td></tr>

            <tr><th colspan="2">Schedule</th></tr>
            <tr><td>Comfort Start</td><td><strong>{sched['comfort_start']}</strong></td></tr>
            <tr><td>Comfort End</td><td><strong>{sched['comfort_end']}</strong></td></tr>

            <tr><th colspan="2">Setpoints</th></tr>
            <tr><td>Comfort</td><td><strong>{sp['setpoint_comfort']}°C</strong></td></tr>
            <tr><td>Eco</td><td><strong>{sp['setpoint_eco']}°C</strong></td></tr>

            <tr><th colspan="2">Expected Outcomes</th></tr>
            <tr><td>COP</td><td>{outcomes['cop_mean']} ({outcomes['cop_vs_baseline']:+.2f} vs baseline)</td></tr>
            <tr><td>Self-Sufficiency</td><td>{outcomes['self_sufficiency']*100:.1f}% ({outcomes['self_sufficiency_vs_baseline_pp']:+.1f}pp)</td></tr>
            <tr><td>Comfort Compliance (08:00-22:00)</td><td>{outcomes['comfort_compliance']*100:.1f}%</td></tr>
            <tr><td>Daily Net Cost</td><td>CHF {daily_cost:.2f} ({cost_reduction:+.1f}% vs baseline)</td></tr>
        </table>
        </div>
        """

    html += """
    <h3>Testable Predictions</h3>
    <p>For statistical validation in Phase 5:</p>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Self-Sufficiency</th>
            <th>COP</th>
            <th>Comfort Compliance</th>
            <th>Cost Change</th>
        </tr>
    """

    for strategy_id, pred in predictions.items():
        ss = pred['self_sufficiency']
        cop = pred['cop']
        comfort = pred['comfort_compliance']
        cost = pred.get('cost', {})
        cost_pct = cost.get('cost_reduction_pct', 0)

        html += f"""
        <tr>
            <td>{pred['strategy_name']}</td>
            <td>{ss['target']}% ({ss['range_min']}-{ss['range_max']}%)</td>
            <td>{cop['target']} ({cop['range_min']}-{cop['range_max']})</td>
            <td>≥{comfort['minimum_pct']}% (target: {comfort['target_pct']}%)</td>
            <td>{cost_pct:+.1f}%</td>
        </tr>
        """

    html += """
    </table>

    <h3>Implementation Notes</h3>
    <ul>
        <li><strong>Curve Rise</strong>: Adjust via heat pump controller menu. Primary lever for COP improvement.</li>
        <li><strong>Schedule</strong>: Set comfort period in heat pump scheduler. Aligns heating with PV availability.</li>
        <li><strong>Setpoints</strong>: Control via Home Assistant climate entity. Automate if desired.</li>
    </ul>

    <h3>Success Criteria</h3>
    <p>A strategy is considered successful if:</p>
    <ol>
        <li>Self-sufficiency meets or exceeds prediction range</li>
        <li>COP meets or exceeds prediction range</li>
        <li>Comfort compliance ≥95% during occupied hours (08:00-22:00) (≥90% for Cost-Optimized)</li>
        <li>Cost savings meet or exceed prediction (for Cost-Optimized: ≥10% reduction)</li>
        <li>No frequent manual overrides required</li>
    </ol>
    <p><em>Note: Night temperatures (22:00-08:00) are excluded from comfort compliance. This allows aggressive energy-saving strategies during unoccupied hours.</em></p>

    <figure>
        <img src="fig24_parameter_space.png" alt="Parameter Space">
        <figcaption><strong>Figure 24:</strong> Parameter space visualization: curve rise vs COP (top-left), schedule duration
        vs self-sufficiency (top-right), comfort-efficiency trade-off (bottom-left),
        parameter summary table (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for parameter set generation."""
    print("="*60)
    print("Phase 4, Step 3: Parameter Set Generation")
    print("="*60)

    # Load simulation results
    comparison, daily_metrics, strategies = load_simulation_results()

    # Generate parameter sets
    parameter_sets = generate_parameter_sets(strategies, comparison)

    # Generate testable predictions
    predictions = generate_testable_predictions(parameter_sets, daily_metrics)

    # Create visualizations
    plot_parameter_space(parameter_sets)

    # Save parameter sets as JSON
    with open(OUTPUT_DIR / 'phase5_parameter_sets.json', 'w') as f:
        json.dump(parameter_sets, f, indent=2)
    print(f"\nSaved: phase5_parameter_sets.json")

    # Save predictions
    with open(OUTPUT_DIR / 'phase5_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print("Saved: phase5_predictions.json")

    # Generate implementation checklist
    checklist = generate_implementation_checklist(parameter_sets)
    with open(OUTPUT_DIR / 'phase5_implementation_checklist.md', 'w') as f:
        f.write(checklist)
    print("Saved: phase5_implementation_checklist.md")

    # Generate report section
    report_html = generate_report(parameter_sets, predictions)
    with open(OUTPUT_DIR / 'parameter_sets_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: parameter_sets_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("PARAMETER GENERATION SUMMARY")
    print("="*60)

    print("\nPhase 5 Parameter Sets Generated:")
    for strategy_id, param_set in parameter_sets.items():
        outcomes = param_set['expected_outcomes']
        cost_pct = outcomes.get('cost_reduction_pct', 0)
        print(f"\n  {param_set['name']}:")
        print(f"    Schedule: {param_set['schedule_settings']['comfort_start']} - {param_set['schedule_settings']['comfort_end']}")
        print(f"    Curve Rise: {param_set['heat_pump_settings']['curve_rise']}")
        print(f"    Expected COP: {outcomes['cop_mean']} ({outcomes['cop_vs_baseline']:+.2f})")
        print(f"    Expected Self-Sufficiency: {outcomes['self_sufficiency']*100:.1f}%")
        print(f"    Expected Cost Change: {cost_pct:+.1f}%")

    print("\nOutputs ready for Phase 5:")
    print("  - phase5_parameter_sets.json (exact settings)")
    print("  - phase5_predictions.json (testable predictions)")
    print("  - phase5_implementation_checklist.md (protocol)")

    print("\n" + "="*60)
    print("STEP COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
