#!/usr/bin/env python3
"""
Phase 3: Weekly Model Decomposition Report

Generates a figure like fig3.01c for every week of available data,
compiled into a single HTML report for model exploration.

Usage:
    python src/phase3/05_weekly_decomposition.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PHASE3_DIR / 'weekly_decomposition'
OUTPUT_DIR.mkdir(exist_ok=True)

# Import from thermal model
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("thermal_model", PROJECT_ROOT / 'src' / 'phase3' / '01_thermal_model.py')
thermal_model = module_from_spec(spec)
spec.loader.exec_module(thermal_model)

# Key columns (from thermal model)
HK2_COL = thermal_model.HK2_COL
OUTDOOR_COL = thermal_model.OUTDOOR_COL
PV_COL = thermal_model.PV_COL
TARGET_SENSORS = thermal_model.TARGET_SENSORS
SENSOR_WEIGHTS = thermal_model.SENSOR_WEIGHTS


def load_data():
    """Load integrated dataset."""
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)
    print(f"  Dataset: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")
    return df


def load_model_params():
    """Load fitted model parameters from thermal model results."""
    results_path = PHASE3_DIR / 'thermal_model_results.csv'
    if not results_path.exists():
        print("ERROR: Run 01_thermal_model.py first to generate model parameters")
        return None

    results = pd.read_csv(results_path)
    # Get the primary sensor (highest weight)
    primary = results.loc[results['weight'].idxmax()]

    params = {
        'room': primary['room'] + '_temperature',
        'tau_out_h': primary['tau_outdoor_h'],
        'tau_eff_h': primary['tau_effort_h'],
        'tau_pv_h': primary['tau_pv_h'],
        'offset': primary['offset'],
        'gain_outdoor': primary['gain_outdoor'],
        'gain_effort': primary['gain_effort'],
        'gain_pv': primary['gain_pv'],
    }

    print(f"\nLoaded model parameters for {primary['room']}:")
    print(f"  τ_out={params['tau_out_h']}h, τ_eff={params['tau_eff_h']}h, τ_pv={params['tau_pv_h']}h")
    print(f"  g_out={params['gain_outdoor']:.3f}, g_eff={params['gain_effort']:.3f}, g_pv={params['gain_pv']:.3f}")

    return params


def load_heating_curve():
    """Load heating curve parameters."""
    hc_path = PHASE3_DIR / 'heating_curve.csv'
    if not hc_path.exists():
        print("ERROR: Run 01_thermal_model.py first")
        return None

    hc = pd.read_csv(hc_path).iloc[0]
    return {
        'baseline': hc['baseline'],
        'slope': hc['slope'],
    }


def get_weeks_with_data(df, room_col):
    """Get list of week start dates with sufficient data."""
    # Filter to valid data
    valid = df[[room_col, OUTDOOR_COL, PV_COL, HK2_COL]].dropna()

    # Group by week and count
    valid['week'] = valid.index.to_period('W').start_time
    week_counts = valid.groupby('week').size()

    # Require at least 2 days of data (davis_inside sensor has gaps)
    min_points = 2 * 96  # 2 days × 96 points/day
    valid_weeks = week_counts[week_counts >= min_points].index.tolist()

    print(f"\nFound {len(valid_weeks)} weeks with sufficient data (>= 2 days):")
    for w in sorted(valid_weeks):
        days = week_counts[w] / 96
        print(f"  {w.date()}: {days:.1f} days")
    return sorted(valid_weeks)


def plot_week_decomposition(df, effort, params, week_start, week_num, output_dir):
    """
    Create decomposition figure for a single week.

    Returns the figure path and week statistics.
    """
    # Get week data - handle timezone
    if df.index.tz is not None and week_start.tz is None:
        week_start = week_start.tz_localize(df.index.tz)
    week_end = week_start + pd.Timedelta(days=7)
    week_mask = (df.index >= week_start) & (df.index < week_end)
    df_week = df.loc[week_mask].copy()
    effort_week = effort.loc[week_mask]

    if len(df_week) < 96:  # Less than 1 day
        return None, None

    # Model parameters
    tau_out_h = params['tau_out_h']
    tau_eff_h = params['tau_eff_h']
    tau_pv_h = params['tau_pv_h']
    g_out = params['gain_outdoor']
    g_eff = params['gain_effort']
    g_pv = params['gain_pv']
    offset = params['offset']
    room_col = params['room']

    # Compute smoothed signals for full dataset (for proper initialization)
    out_smooth_full = thermal_model.exponential_smooth(df[OUTDOOR_COL].values, tau_out_h * 4)
    effort_smooth_full = thermal_model.exponential_smooth(effort.values, tau_eff_h * 4)
    pv_smooth_full = thermal_model.exponential_smooth(df[PV_COL].values, tau_pv_h * 4)

    # Extract week portion
    week_idx = df.index.get_indexer(df_week.index)
    out_smooth = out_smooth_full[week_idx]
    effort_smooth = effort_smooth_full[week_idx]
    pv_smooth = pv_smooth_full[week_idx]

    # Compute model terms
    term_outdoor = g_out * out_smooth
    term_effort = g_eff * effort_smooth
    term_pv = g_pv * pv_smooth
    y_pred = offset + term_outdoor + term_effort + term_pv

    # Actual room temperature
    y_actual = df_week[room_col].values

    # Calculate statistics
    valid_mask = ~np.isnan(y_actual)
    if valid_mask.sum() < 10:
        return None, None

    y_act_valid = y_actual[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    ss_tot = np.sum((y_act_valid - np.mean(y_act_valid))**2)
    ss_res = np.sum((y_act_valid - y_pred_valid)**2)
    r2_week = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse_week = np.sqrt(np.mean((y_act_valid - y_pred_valid)**2))

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Week label
    week_label = f"Week {week_num}: {week_start.strftime('%Y-%m-%d')} to {(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"

    # Panel 1: Room temperature actual vs predicted
    ax1 = axes[0]
    ax1.plot(df_week.index, y_actual, 'b-', linewidth=1, alpha=0.8, label='Actual')
    ax1.plot(df_week.index, y_pred, 'r-', linewidth=1, alpha=0.8, label='Predicted')
    ax1.fill_between(df_week.index, y_actual, y_pred, alpha=0.2, color='gray')
    ax1.set_ylabel('Temperature (°C)')
    room_name = room_col.replace('_temperature', '')
    ax1.set_title(f'{week_label} — {room_name}: R²={r2_week:.3f}, RMSE={rmse_week:.2f}°C')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Outdoor temperature and its contribution
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(df_week.index, df_week[OUTDOOR_COL], 'b-', linewidth=1, alpha=0.7, label='T_outdoor (raw)')
    ax2.plot(df_week.index, out_smooth, 'b--', linewidth=1.5, alpha=0.9, label=f'LPF(τ={tau_out_h}h)')
    ax2.set_ylabel('Outdoor Temp (°C)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2_twin.plot(df_week.index, term_outdoor, 'orange', linewidth=1.5, label=f'g_out×LPF')
    ax2_twin.set_ylabel('Contribution (°C)', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    ax2.set_title(f'Outdoor: g_out={g_out:+.3f}, τ={tau_out_h}h')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Heating effort and its contribution
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(df_week.index, effort_week, 'b-', linewidth=0.8, alpha=0.5, label='Effort (raw)')
    ax3.plot(df_week.index, effort_smooth, 'b-', linewidth=1.5, alpha=0.9, label=f'LPF(τ={tau_eff_h}h)')
    ax3.set_ylabel('Heating Effort (°C)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    ax3_twin.plot(df_week.index, term_effort, 'orange', linewidth=1.5, label=f'g_eff×LPF')
    ax3_twin.set_ylabel('Contribution (°C)', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')

    ax3.set_title(f'Heating Effort: g_eff={g_eff:+.3f}, τ={tau_eff_h}h')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: PV/Solar and its contribution
    ax4 = axes[3]
    ax4_twin = ax4.twinx()

    ax4.plot(df_week.index, df_week[PV_COL], 'b-', linewidth=0.8, alpha=0.5, label='PV (raw)')
    ax4.plot(df_week.index, pv_smooth, 'b-', linewidth=1.5, alpha=0.9, label=f'LPF(τ={tau_pv_h}h)')
    ax4.set_ylabel('PV (kWh)', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')

    ax4_twin.plot(df_week.index, term_pv, 'orange', linewidth=1.5, label=f'g_pv×LPF')
    ax4_twin.set_ylabel('Contribution (°C)', color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')

    ax4.set_title(f'Solar/PV: g_pv={g_pv:+.3f}, τ={tau_pv_h}h')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax4.set_xlabel('Date')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    fig_name = f'week_{week_num:02d}_{week_start.strftime("%Y%m%d")}.png'
    fig_path = output_dir / fig_name
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()

    # Return stats
    stats = {
        'week_num': week_num,
        'week_start': week_start,
        'week_end': week_end,
        'r2': r2_week,
        'rmse': rmse_week,
        'mean_outdoor': df_week[OUTDOOR_COL].mean(),
        'mean_room': np.nanmean(y_actual),
        'n_points': valid_mask.sum(),
        'fig_name': fig_name,
    }

    return fig_path, stats


def generate_html_report(all_stats, params, output_dir):
    """Generate HTML report with all weekly figures."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build figure sections
    figure_sections = ""
    for stats in all_stats:
        figure_sections += f"""
        <div class="week-section">
            <h3>Week {stats['week_num']}: {stats['week_start'].strftime('%b %d')} - {(stats['week_end'] - pd.Timedelta(days=1)).strftime('%b %d, %Y')}</h3>
            <table class="stats-table">
                <tr>
                    <td><strong>R²:</strong> {stats['r2']:.3f}</td>
                    <td><strong>RMSE:</strong> {stats['rmse']:.2f}°C</td>
                    <td><strong>Mean T<sub>out</sub>:</strong> {stats['mean_outdoor']:.1f}°C</td>
                    <td><strong>Mean T<sub>room</sub>:</strong> {stats['mean_room']:.1f}°C</td>
                    <td><strong>Points:</strong> {stats['n_points']:,}</td>
                </tr>
            </table>
            <figure>
                <img src="{stats['fig_name']}" alt="Week {stats['week_num']} decomposition">
            </figure>
        </div>
        """

    # Summary statistics
    r2_values = [s['r2'] for s in all_stats]
    rmse_values = [s['rmse'] for s in all_stats]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Weekly Model Decomposition Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #1a5f7a;
            border-bottom: 2px solid #1a5f7a;
            padding-bottom: 10px;
        }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        h3 {{
            color: #34495e;
            margin-bottom: 10px;
            border-left: 4px solid #1a5f7a;
            padding-left: 10px;
        }}
        .summary-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #1a5f7a;
            padding: 15px;
            margin: 20px 0;
        }}
        .week-section {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-table {{
            width: 100%;
            margin-bottom: 15px;
        }}
        .stats-table td {{
            padding: 5px 15px;
            text-align: center;
        }}
        figure {{
            margin: 0;
            text-align: center;
        }}
        figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table.summary {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        table.summary th, table.summary td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        table.summary th {{
            background-color: #f8f9fa;
        }}
        .model-info {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Weekly Model Decomposition Report</h1>
    <p><em>Generated: {timestamp}</em></p>

    <div class="summary-box">
        <h3 style="margin-top: 0; border: none; padding-left: 0;">Summary</h3>
        <p>This report shows the thermal model decomposition for each week of available data.
        Each figure breaks down the room temperature prediction into its component terms:</p>
        <ul>
            <li><strong>Panel 1:</strong> Actual vs predicted room temperature</li>
            <li><strong>Panel 2:</strong> Outdoor temperature contribution (g<sub>out</sub> × LPF(T<sub>out</sub>))</li>
            <li><strong>Panel 3:</strong> Heating effort contribution (g<sub>eff</sub> × LPF(Effort))</li>
            <li><strong>Panel 4:</strong> Solar/PV contribution (g<sub>pv</sub> × LPF(PV))</li>
        </ul>
    </div>

    <div class="model-info">
        <strong>Model Parameters:</strong><br>
        T<sub>room</sub> = {params['offset']:.1f} + {params['gain_outdoor']:.3f}×LPF(T<sub>out</sub>, τ={params['tau_out_h']}h)
        + {params['gain_effort']:.3f}×LPF(Effort, τ={params['tau_eff_h']}h)
        + {params['gain_pv']:.3f}×LPF(PV, τ={params['tau_pv_h']}h)
    </div>

    <h2>Overall Statistics</h2>
    <table class="summary">
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Min</th>
            <th>Max</th>
            <th>Std</th>
        </tr>
        <tr>
            <td>R²</td>
            <td>{np.mean(r2_values):.3f}</td>
            <td>{np.min(r2_values):.3f}</td>
            <td>{np.max(r2_values):.3f}</td>
            <td>{np.std(r2_values):.3f}</td>
        </tr>
        <tr>
            <td>RMSE (°C)</td>
            <td>{np.mean(rmse_values):.2f}</td>
            <td>{np.min(rmse_values):.2f}</td>
            <td>{np.max(rmse_values):.2f}</td>
            <td>{np.std(rmse_values):.2f}</td>
        </tr>
    </table>

    <h2>Weekly Decomposition ({len(all_stats)} weeks)</h2>

    {figure_sections}

    <p style="text-align: center; color: #666; margin-top: 40px;">
        <a href="../phase3_report.html">← Back to Phase 3 Report</a>
    </p>
</body>
</html>
"""

    report_path = output_dir / 'weekly_decomposition_report.html'
    report_path.write_text(html)
    print(f"\nSaved: {report_path}")
    return report_path


def main():
    """Generate weekly decomposition figures and report."""
    print("="*60)
    print("Phase 3: Weekly Model Decomposition")
    print("="*60)

    # Load data
    df = load_data()

    # Load model parameters
    params = load_model_params()
    if params is None:
        return

    # Load heating curve
    heating_curve = load_heating_curve()
    if heating_curve is None:
        return

    # Compute heating effort
    expected_hk2 = heating_curve['baseline'] + heating_curve['slope'] * df[OUTDOOR_COL]
    effort = df[HK2_COL] - expected_hk2

    # Get weeks with data
    room_col = params['room']
    weeks = get_weeks_with_data(df, room_col)

    if not weeks:
        print("ERROR: No weeks with sufficient data found")
        return

    # Generate figures for each week
    print(f"\nGenerating {len(weeks)} weekly decomposition figures...")
    all_stats = []

    for i, week_start in enumerate(weeks):
        week_num = i + 1
        fig_path, stats = plot_week_decomposition(
            df, effort, params, week_start, week_num, OUTPUT_DIR
        )

        if stats:
            all_stats.append(stats)
            print(f"  Week {week_num}: {week_start.strftime('%Y-%m-%d')} - R²={stats['r2']:.3f}, RMSE={stats['rmse']:.2f}°C")

    if not all_stats:
        print("ERROR: No figures generated")
        return

    # Generate HTML report
    report_path = generate_html_report(all_stats, params, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("WEEKLY DECOMPOSITION COMPLETE")
    print("="*60)
    print(f"\nGenerated {len(all_stats)} weekly figures")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report: {report_path.name}")

    r2_values = [s['r2'] for s in all_stats]
    print(f"\nOverall R²: {np.mean(r2_values):.3f} (range: {np.min(r2_values):.3f} - {np.max(r2_values):.3f})")


if __name__ == '__main__':
    main()
