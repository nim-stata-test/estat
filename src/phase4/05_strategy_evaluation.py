#!/usr/bin/env python3
"""
Phase 4, Step 5: Strategy Evaluation and Temperature Prediction

Evaluates selected Pareto strategies for comfort violations and predicts
temperature profiles for winter 2026/2027.

Outputs:
- fig27_strategy_temperature_predictions.png
- strategy_violation_analysis.csv
- strategy_evaluation_report.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
from datetime import datetime, timedelta

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
PHASE4_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR = PHASE4_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

# Model parameters from Phase 3
# Uses T_HK2 (target flow from heating curve) not actual measured flow
COP_PARAMS = {
    'intercept': 6.52,
    'outdoor_coef': 0.1319,
    't_hk2_coef': -0.1007,
}

# Load heating curve parameters from Phase 2 JSON
def _load_heating_curve_params():
    params_file = PHASE2_DIR / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        return {'t_ref_comfort': params['t_ref_comfort'], 't_ref_eco': params['t_ref_eco']}
    return {'t_ref_comfort': 21.32, 't_ref_eco': 19.18}

HEATING_CURVE_PARAMS = _load_heating_curve_params()

# T_weighted regression coefficients from Phase 2 multivariate analysis
TEMP_REGRESSION = {
    'intercept': -15.31,
    'comfort_setpoint': 1.218,
    'eco_setpoint': -0.090,
    'curve_rise': 9.73,
    'comfort_hours': -0.020,
    'outdoor_mean': 0.090,
}

# Baseline parameters
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

# Comfort constraint parameters
COMFORT_THRESHOLD = 18.5  # °C - minimum acceptable temperature
OCCUPIED_START = 8  # 08:00
OCCUPIED_END = 22   # 22:00
VIOLATION_LIMIT = 0.05  # 5% max violation allowed


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

    # Add hour column
    df['hour'] = df.index.hour + df.index.minute / 60

    # Load selected strategies
    with open(PHASE4_DIR / 'selected_strategies.json', 'r') as f:
        strategies = json.load(f)

    print(f"  Loaded {len(df):,} timesteps ({len(df)//96:.0f} days)")
    print(f"  Loaded {len(strategies)} selected strategies")

    return df, strategies


def calculate_delta_T(params: dict) -> float:
    """Calculate temperature adjustment delta from parameters."""
    comfort_hours = params['comfort_end'] - params['comfort_start']
    baseline_hours = BASELINE['comfort_end'] - BASELINE['comfort_start']

    delta_T = (
        TEMP_REGRESSION['comfort_setpoint'] * (params['setpoint_comfort'] - BASELINE['setpoint_comfort']) +
        TEMP_REGRESSION['eco_setpoint'] * (params['setpoint_eco'] - BASELINE['setpoint_eco']) +
        TEMP_REGRESSION['curve_rise'] * (params['curve_rise'] - BASELINE['curve_rise']) +
        TEMP_REGRESSION['comfort_hours'] * (comfort_hours - baseline_hours)
    )

    return delta_T


def evaluate_strategy(params: dict, df: pd.DataFrame) -> dict:
    """
    Evaluate a strategy for comfort violations.

    Returns dict with:
    - T_weighted_adj: adjusted temperature series
    - violation_pct: percentage of daytime hours below threshold
    - mean_temp: mean daytime temperature
    - min_temp: minimum daytime temperature
    - hours_below_threshold: total hours below threshold
    """
    delta_T = calculate_delta_T(params)

    # Adjust temperatures
    T_weighted_adj = df['T_weighted'].values + delta_T

    # Filter to occupied hours only
    occupied_mask = (df['hour'].values >= OCCUPIED_START) & (df['hour'].values < OCCUPIED_END)
    T_occupied = T_weighted_adj[occupied_mask]

    # Calculate violations
    n_occupied = len(T_occupied)
    n_violations = np.sum(T_occupied < COMFORT_THRESHOLD)
    violation_pct = n_violations / n_occupied if n_occupied > 0 else 0.0

    # Calculate hours (15-min intervals -> hours)
    hours_below = n_violations * 0.25
    total_occupied_hours = n_occupied * 0.25

    return {
        'T_weighted_adj': T_weighted_adj,
        'delta_T': delta_T,
        'violation_pct': violation_pct,
        'violation_pct_display': f"{violation_pct * 100:.1f}%",
        'mean_temp': np.nanmean(T_occupied),
        'min_temp': np.nanmin(T_occupied),
        'max_temp': np.nanmax(T_occupied),
        'hours_below_threshold': hours_below,
        'total_occupied_hours': total_occupied_hours,
        'constraint_satisfied': violation_pct <= VIOLATION_LIMIT,
    }


def generate_winter_predictions(df: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """
    Generate predicted T_weighted for winter 2026/2027 (Nov 2026 - Feb 2027).

    Uses historical outdoor temperatures from the overlap period, repeated
    to simulate a full winter season.
    """
    print("\nGenerating winter 2026/2027 predictions...")

    # Create winter date range
    winter_start = pd.Timestamp('2026-11-01')
    winter_end = pd.Timestamp('2027-02-28 23:45:00')
    winter_idx = pd.date_range(winter_start, winter_end, freq='15min')

    print(f"  Winter period: {winter_start.date()} to {winter_end.date()}")
    print(f"  Total timesteps: {len(winter_idx):,}")

    # Get historical data for repeating pattern
    hist_T = df['T_weighted'].dropna()
    hist_len = len(hist_T)

    # Create predictions dataframe
    predictions = pd.DataFrame(index=winter_idx)
    predictions['hour'] = predictions.index.hour + predictions.index.minute / 60
    predictions['date'] = predictions.index.date
    predictions['month'] = predictions.index.month

    # Repeat historical pattern
    n_repeats = (len(winter_idx) // hist_len) + 1
    hist_repeated = np.tile(hist_T.values, n_repeats)[:len(winter_idx)]
    predictions['T_weighted_baseline'] = hist_repeated

    # Add strategy predictions
    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        delta_T = calculate_delta_T(params)
        predictions[f'T_weighted_{label}'] = hist_repeated + delta_T

    return predictions


def plot_strategy_predictions(predictions: pd.DataFrame, strategies: list, evaluations: dict):
    """Create comprehensive visualization of strategy temperature predictions."""
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(16, 14))

    # Panel 1: Daily mean temperature over winter
    ax1 = fig.add_subplot(3, 2, 1)

    # Calculate daily means for occupied hours only
    occupied_mask = (predictions['hour'] >= OCCUPIED_START) & (predictions['hour'] < OCCUPIED_END)
    # Select only numeric columns for resampling
    numeric_cols = [c for c in predictions.columns if c.startswith('T_weighted')]
    daily_means = predictions.loc[occupied_mask, numeric_cols].resample('D').mean()

    ax1.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Comfort threshold ({COMFORT_THRESHOLD}°C)', zorder=1)

    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for i, strategy in enumerate(strategies):
        label = strategy.get('label', strategy['id'])
        col = f'T_weighted_{label}'
        if col in daily_means.columns:
            ax1.plot(daily_means.index, daily_means[col],
                    color=colors[i], linewidth=1.5, alpha=0.8,
                    label=f"{label} (mean: {evaluations[label]['mean_temp']:.1f}°C)")

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Mean T_weighted (°C)')
    ax1.set_title('Predicted Daily Mean Temperature (Daytime 08:00-22:00)')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.set_ylim(16, 26)

    # Panel 2: Monthly distribution boxplots
    ax2 = fig.add_subplot(3, 2, 2)

    # Prepare data for boxplot
    occupied_data = predictions[occupied_mask].copy()
    monthly_data = []
    month_labels = []
    strategy_labels = []

    for month in sorted(occupied_data['month'].unique()):
        month_mask = occupied_data['month'] == month
        year = 2027 if month in [1, 2] else 2026
        month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%b')

        for i, strategy in enumerate(strategies[:4]):  # Limit to 4 for clarity
            label = strategy.get('label', strategy['id'])
            col = f'T_weighted_{label}'
            if col in occupied_data.columns:
                monthly_data.append(occupied_data.loc[month_mask, col].values)
                month_labels.append(f"{month_name}\n{label[:8]}")
                strategy_labels.append(label)

    bp = ax2.boxplot(monthly_data, labels=month_labels, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(strategies[:4])])
        box.set_alpha(0.7)

    ax2.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('T_weighted (°C)')
    ax2.set_title('Monthly Temperature Distribution (Top 4 Strategies)')
    ax2.tick_params(axis='x', rotation=45, labelsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Violation percentage by strategy
    ax3 = fig.add_subplot(3, 2, 3)

    labels = [s.get('label', s['id']) for s in strategies]
    violations = [evaluations[l]['violation_pct'] * 100 for l in labels]
    bar_colors = ['red' if v > VIOLATION_LIMIT * 100 else 'green' for v in violations]

    bars = ax3.bar(range(len(labels)), violations, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=VIOLATION_LIMIT * 100, color='orange', linestyle='--', linewidth=2,
                label=f'Constraint limit ({VIOLATION_LIMIT * 100:.0f}%)')

    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Violation Percentage (%)')
    ax3.set_title(f'Hours Below {COMFORT_THRESHOLD}°C Threshold (Daytime)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for bar, pct in zip(bars, violations):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # Panel 4: Hourly profile for typical cold day
    ax4 = fig.add_subplot(3, 2, 4)

    # Find a cold day in the predictions
    daily_min = predictions[numeric_cols].resample('D').min()
    cold_day_idx = daily_min['T_weighted_baseline'].idxmin()
    cold_day = predictions[predictions.index.date == cold_day_idx.date()]

    ax4.axhline(y=COMFORT_THRESHOLD, color='red', linestyle='--', linewidth=2)
    ax4.axvspan(OCCUPIED_START, OCCUPIED_END, color='yellow', alpha=0.2, label='Occupied hours')

    for i, strategy in enumerate(strategies[:5]):  # Limit to 5
        label = strategy.get('label', strategy['id'])
        col = f'T_weighted_{label}'
        if col in cold_day.columns:
            ax4.plot(cold_day['hour'], cold_day[col],
                    color=colors[i], linewidth=2, label=label[:12])

    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('T_weighted (°C)')
    ax4.set_title(f'Hourly Profile on Coldest Day ({cold_day_idx.date()})')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)
    ax4.set_xticks(range(0, 25, 4))

    # Panel 5: Cumulative violation hours
    ax5 = fig.add_subplot(3, 2, 5)

    # Calculate cumulative violations over winter
    for i, strategy in enumerate(strategies[:5]):
        label = strategy.get('label', strategy['id'])
        col = f'T_weighted_{label}'
        if col in predictions.columns:
            occupied = predictions[occupied_mask].copy()
            violations_cum = (occupied[col] < COMFORT_THRESHOLD).cumsum() * 0.25  # hours
            ax5.plot(occupied.index, violations_cum,
                    color=colors[i], linewidth=1.5, label=f"{label[:12]}")

    ax5.set_xlabel('Date')
    ax5.set_ylabel('Cumulative Hours Below Threshold')
    ax5.set_title(f'Cumulative Cold Hours (<{COMFORT_THRESHOLD}°C during 08:00-22:00)')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Panel 6: Summary table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    # Create summary table
    table_data = []
    headers = ['Strategy', 'Mean\n(°C)', 'Min\n(°C)', 'Violation\n(%)', 'Cold\nHours', 'Constraint']

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        e = evaluations[label]
        constraint_status = '✓ PASS' if e['constraint_satisfied'] else '✗ FAIL'
        table_data.append([
            label[:12],
            f"{e['mean_temp']:.1f}",
            f"{e['min_temp']:.1f}",
            f"{e['violation_pct']*100:.1f}%",
            f"{e['hours_below_threshold']:.0f}h",
            constraint_status
        ])

    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['lightblue'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code constraint column
    for i, row in enumerate(table_data):
        if '✗ FAIL' in row[-1]:
            table[(i+1, 5)].set_facecolor('#ffcccc')
        else:
            table[(i+1, 5)].set_facecolor('#ccffcc')

    ax6.set_title('Strategy Evaluation Summary', fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig27_strategy_temperature_predictions.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig27_strategy_temperature_predictions.png")


def generate_report(strategies: list, evaluations: dict) -> str:
    """Generate HTML report section."""

    html = f"""
    <section id="strategy-evaluation">
    <h2>4.5 Strategy Evaluation and Comfort Analysis</h2>

    <h3>Comfort Constraint Parameters</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr>
            <td><code>COMFORT_THRESHOLD</code></td>
            <td><strong>{COMFORT_THRESHOLD}°C</strong></td>
            <td>Minimum acceptable weighted indoor temperature</td>
        </tr>
        <tr>
            <td><code>VIOLATION_LIMIT</code></td>
            <td><strong>{VIOLATION_LIMIT * 100:.0f}%</strong></td>
            <td>Maximum allowed proportion of occupied hours below threshold</td>
        </tr>
        <tr>
            <td><code>OCCUPIED_HOURS</code></td>
            <td><strong>{OCCUPIED_START:02d}:00 - {OCCUPIED_END:02d}:00</strong></td>
            <td>Hours during which comfort is evaluated</td>
        </tr>
    </table>

    <h3>Penalization Mechanism</h3>
    <p>The Pareto optimization uses a <strong>soft constraint</strong> approach:</p>
    <ul>
        <li>Constraint: <code>g = violation_pct - {VIOLATION_LIMIT}</code></li>
        <li>If <code>g ≤ 0</code>: Solution is <em>feasible</em> (violation ≤ {VIOLATION_LIMIT * 100:.0f}%)</li>
        <li>If <code>g > 0</code>: Solution is <em>infeasible</em> but not excluded</li>
        <li>NSGA-II ranks feasible solutions higher than infeasible ones</li>
        <li>Among infeasible solutions, smaller constraint violation is preferred</li>
    </ul>

    <h3>Strategy Evaluation Results</h3>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Setpoint<br>Comfort</th>
            <th>Setpoint<br>Eco</th>
            <th>Schedule</th>
            <th>Mean<br>Temp</th>
            <th>Min<br>Temp</th>
            <th>Violation<br>%</th>
            <th>Cold<br>Hours</th>
            <th>Status</th>
        </tr>
    """

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        v = strategy['variables']
        e = evaluations[label]

        status_class = 'pass' if e['constraint_satisfied'] else 'fail'
        status_text = 'PASS' if e['constraint_satisfied'] else 'FAIL'
        status_color = '#ccffcc' if e['constraint_satisfied'] else '#ffcccc'

        html += f"""
        <tr>
            <td><strong>{label}</strong></td>
            <td>{v['setpoint_comfort']:.1f}°C</td>
            <td>{v['setpoint_eco']:.1f}°C</td>
            <td>{int(v['comfort_start']):02d}:00-{int(v['comfort_end']):02d}:00</td>
            <td>{e['mean_temp']:.1f}°C</td>
            <td>{e['min_temp']:.1f}°C</td>
            <td style="background-color: {status_color}">{e['violation_pct']*100:.1f}%</td>
            <td>{e['hours_below_threshold']:.0f}h</td>
            <td style="background-color: {status_color}"><strong>{status_text}</strong></td>
        </tr>
        """

    html += """
    </table>

    <h3>Key Findings</h3>
    """

    # Calculate summary statistics
    n_pass = sum(1 for e in evaluations.values() if e['constraint_satisfied'])
    n_fail = len(evaluations) - n_pass
    max_violation = max(e['violation_pct'] for e in evaluations.values())
    min_violation = min(e['violation_pct'] for e in evaluations.values())

    html += f"""
    <ul>
        <li><strong>Constraint satisfaction:</strong> {n_pass}/{len(evaluations)} strategies pass the {VIOLATION_LIMIT*100:.0f}% limit</li>
        <li><strong>Violation range:</strong> {min_violation*100:.1f}% to {max_violation*100:.1f}%</li>
    """

    if n_fail > 0:
        html += f"""
        <li><strong>Warning:</strong> {n_fail} strategies exceed the comfort constraint and may cause discomfort</li>
        """

    html += """
    </ul>

    <h3>Winter 2026/2027 Temperature Predictions</h3>
    <figure>
        <img src="fig27_strategy_temperature_predictions.png" alt="Strategy Temperature Predictions">
        <figcaption><strong>Figure 27:</strong> Predicted weighted indoor temperatures for each strategy
        over winter 2026/2027. Red dashed line indicates the comfort threshold. Strategies with
        violation percentages exceeding the constraint limit are highlighted.</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function."""
    print("=" * 60)
    print("Phase 4, Step 5: Strategy Evaluation")
    print("=" * 60)

    # Load data
    df, strategies = load_data()

    # Evaluate each strategy
    print("\nEvaluating strategies...")
    evaluations = {}

    for strategy in strategies:
        label = strategy.get('label', strategy['id'])
        params = strategy['variables']
        evaluation = evaluate_strategy(params, df)
        evaluations[label] = evaluation

        status = "✓" if evaluation['constraint_satisfied'] else "✗"
        print(f"  {label:20s}: violation={evaluation['violation_pct']*100:5.1f}%, "
              f"mean={evaluation['mean_temp']:.1f}°C, min={evaluation['min_temp']:.1f}°C {status}")

    # Generate winter predictions
    predictions = generate_winter_predictions(df, strategies)

    # Create visualization
    plot_strategy_predictions(predictions, strategies, evaluations)

    # Save evaluation results
    eval_df = pd.DataFrame([
        {
            'strategy': label,
            'setpoint_comfort': strategies[i]['variables']['setpoint_comfort'],
            'setpoint_eco': strategies[i]['variables']['setpoint_eco'],
            'comfort_start': strategies[i]['variables']['comfort_start'],
            'comfort_end': strategies[i]['variables']['comfort_end'],
            'curve_rise': strategies[i]['variables']['curve_rise'],
            'delta_T': e['delta_T'],
            'mean_temp': e['mean_temp'],
            'min_temp': e['min_temp'],
            'max_temp': e['max_temp'],
            'violation_pct': e['violation_pct'],
            'hours_below_threshold': e['hours_below_threshold'],
            'constraint_satisfied': e['constraint_satisfied'],
        }
        for i, (label, e) in enumerate(evaluations.items())
    ])
    eval_df.to_csv(OUTPUT_DIR / 'strategy_violation_analysis.csv', index=False)
    print(f"\nSaved: strategy_violation_analysis.csv")

    # Generate report
    report_html = generate_report(strategies, evaluations)
    with open(OUTPUT_DIR / 'strategy_evaluation_report.html', 'w') as f:
        f.write(report_html)
    print("Saved: strategy_evaluation_report.html")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for e in evaluations.values() if e['constraint_satisfied'])
    n_fail = len(evaluations) - n_pass

    print(f"\nComfort Threshold: {COMFORT_THRESHOLD}°C")
    print(f"Violation Limit: {VIOLATION_LIMIT * 100:.0f}%")
    print(f"\nResults: {n_pass} PASS, {n_fail} FAIL out of {len(evaluations)} strategies")

    if n_fail > 0:
        print("\nStrategies exceeding constraint:")
        for label, e in evaluations.items():
            if not e['constraint_satisfied']:
                print(f"  - {label}: {e['violation_pct']*100:.1f}% violation")

    print("\n" + "=" * 60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
