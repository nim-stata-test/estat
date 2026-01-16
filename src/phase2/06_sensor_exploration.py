#!/usr/bin/env python3
"""
Phase 2, Step 6: Systematic Sensor Exploration

Analyzes all ~190 sensors to identify candidates that could improve thermal/COP models.

Key analyses:
1. Coverage analysis - data quality and availability for each sensor
2. Correlation with room temperature - direct predictive value
3. Correlation with thermal model residuals - captures missing dynamics
4. Correlation with daily COP - COP model improvement potential
5. Ranking by improvement potential

Focus sensors:
- Pressure: high_pressure_wp1, low_pressure_wp1 (thermodynamic COP)
- Temperature: hot_gas_temperature_wp1 (superheat)
- Wind: davis_wind_speed (infiltration)
- Status: evaporator_defrost, is_heating (filtering)

Output:
- output/phase2/fig_sensor_coverage.png
- output/phase2/fig_sensor_correlations.png
- output/phase2/sensor_exploration_rankings.csv
- output/phase2/sensor_exploration_report.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(exist_ok=True)

# Key sensors to highlight
FOCUS_SENSORS = [
    'stiebel_eltron_isg_high_pressure_wp1',
    'stiebel_eltron_isg_low_pressure_wp1',
    'stiebel_eltron_isg_hot_gas_temperature_wp1',
    'stiebel_eltron_isg_volume_stream_wp1',
    'stiebel_eltron_isg_evaporator_defrost',
    'stiebel_eltron_isg_is_heating',
    'stiebel_eltron_isg_compressor',
    'stiebel_eltron_isg_flow_temperature_wp1',
    'stiebel_eltron_isg_return_temperature_wp1',
    'davis_wind_speed',
    'davis_2_min_avg_wind_speed',
    'davis_outside_humidity',
    'davis_barometer',
]


def load_sensor_summary() -> pd.DataFrame:
    """Load the sensor summary CSV with counts and statistics."""
    print("Loading sensor summary...")
    summary = pd.read_csv(PROCESSED_DIR / 'sensor_summary.csv')
    print(f"  {len(summary)} sensors across {summary['category'].nunique()} categories")
    return summary


def load_integrated_dataset() -> pd.DataFrame:
    """Load the integrated dataset with all aligned sensors."""
    print("Loading integrated dataset...")
    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Period: {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_thermal_residuals() -> pd.Series:
    """Load thermal model residuals from 3-state model."""
    print("Loading thermal model residuals...")
    residuals_file = PHASE3_DIR / 'threestate_greybox_results.csv'

    if not residuals_file.exists():
        # Try stable greybox
        residuals_file = PHASE3_DIR / 'stable_greybox_results.csv'

    if residuals_file.exists():
        res_df = pd.read_csv(residuals_file)
        res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
        res_df['residual'] = res_df['T_room_actual'] - res_df['T_room_pred']
        residuals = res_df.set_index('timestamp')['residual']
        print(f"  Loaded {len(residuals):,} residual values")
        return residuals
    else:
        print("  WARNING: No thermal model residuals found")
        return None


def load_cop_data() -> pd.DataFrame:
    """Load daily COP data from heat pump model."""
    print("Loading COP data...")
    cop_file = PHASE3_DIR / 'heat_pump_daily_stats.csv'

    if cop_file.exists():
        cop_df = pd.read_csv(cop_file)
        cop_df['datetime'] = pd.to_datetime(cop_df['datetime'])
        cop_df['date'] = cop_df['datetime'].dt.date
        print(f"  Loaded {len(cop_df)} daily COP values")
        return cop_df
    else:
        print("  WARNING: No COP data found")
        return None


def compute_coverage_stats(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed coverage statistics for each sensor.

    Returns DataFrame with:
    - count, coverage_pct, missing_pct
    - overlap_coverage_pct (coverage during heating overlap period only)
    - mean, std, min, max
    - in_integrated (whether sensor is in integrated dataset)
    """
    print("\nComputing coverage statistics...")

    # Get heating period bounds
    heating_start = df.index.min()
    heating_end = df.index.max()
    total_intervals = len(df)

    # Define overlap period for heating analysis (Oct 2025 onward)
    overlap_start = pd.Timestamp('2025-10-28', tz='UTC')
    df_overlap = df[df.index >= overlap_start]
    overlap_intervals = len(df_overlap)

    stats_list = []

    for col in df.columns:
        if col in ['timestamp', 'date', 'datetime']:
            continue

        series = df[col]
        valid_count = series.notna().sum()

        # Also compute overlap-period coverage
        series_overlap = df_overlap[col] if col in df_overlap.columns else pd.Series()
        overlap_valid = series_overlap.notna().sum() if len(series_overlap) > 0 else 0
        overlap_coverage = 100 * overlap_valid / overlap_intervals if overlap_intervals > 0 else 0

        stats_list.append({
            'sensor': col,
            'count': valid_count,
            'coverage_pct': 100 * valid_count / total_intervals,
            'overlap_coverage_pct': overlap_coverage,
            'missing_pct': 100 * (1 - valid_count / total_intervals),
            'mean': series.mean() if valid_count > 0 else np.nan,
            'std': series.std() if valid_count > 0 else np.nan,
            'min': series.min() if valid_count > 0 else np.nan,
            'max': series.max() if valid_count > 0 else np.nan,
            'is_focus': col in FOCUS_SENSORS,
        })

    coverage_df = pd.DataFrame(stats_list)

    # Add category from summary
    sensor_to_category = dict(zip(summary['entity_id'], summary['category']))
    coverage_df['category'] = coverage_df['sensor'].map(
        lambda x: sensor_to_category.get(x, 'other')
    )

    # Sort by coverage
    coverage_df = coverage_df.sort_values('coverage_pct', ascending=False)

    print(f"  Analyzed {len(coverage_df)} sensors in integrated dataset")
    print(f"  High coverage (>80%): {(coverage_df['coverage_pct'] > 80).sum()} sensors")
    print(f"  High overlap coverage (>80%): {(coverage_df['overlap_coverage_pct'] > 80).sum()} sensors")
    print(f"  Focus sensors with data: {coverage_df['is_focus'].sum()}")

    return coverage_df


def compute_correlations(df: pd.DataFrame, target: pd.Series,
                         target_name: str) -> pd.DataFrame:
    """
    Compute correlations between all sensors and a target variable.

    Returns DataFrame with sensor, correlation, p_value, n_obs.
    """
    print(f"\nComputing correlations with {target_name}...")

    # Align index
    common_idx = df.index.intersection(target.index)
    if len(common_idx) == 0:
        print(f"  WARNING: No overlapping data with {target_name}")
        return pd.DataFrame(columns=['sensor', 'correlation', 'p_value', 'n_obs'])

    target_aligned = target.loc[common_idx]

    corr_list = []

    for col in df.columns:
        if col in ['timestamp', 'date', 'datetime']:
            continue

        series = df.loc[common_idx, col]

        # Need at least 30 valid pairs
        valid_mask = series.notna() & target_aligned.notna()
        n_valid = valid_mask.sum()

        if n_valid >= 30:
            try:
                corr, pval = stats.pearsonr(
                    series[valid_mask].values,
                    target_aligned[valid_mask].values
                )
                corr_list.append({
                    'sensor': col,
                    'correlation': corr,
                    'p_value': pval,
                    'n_obs': n_valid,
                })
            except Exception:
                pass

    corr_df = pd.DataFrame(corr_list)

    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
        print(f"  Computed {len(corr_df)} correlations")
        top_corr = corr_df.iloc[0]
        print(f"  Strongest: {top_corr['sensor']} (r={top_corr['correlation']:.3f})")

    return corr_df


def compute_cop_correlations(df: pd.DataFrame, cop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between sensors and daily COP.

    Resamples sensor data to daily before correlating.
    """
    print("\nComputing COP correlations (daily resolution)...")

    if cop_df is None:
        return pd.DataFrame(columns=['sensor', 'correlation', 'p_value', 'n_obs'])

    # Resample sensors to daily mean
    df_daily = df.resample('D').mean()

    # Convert index to date for alignment with COP data
    df_daily.index = df_daily.index.date

    # Align with COP data (date objects)
    cop_series = cop_df.set_index('date')['cop']

    return compute_correlations(df_daily, cop_series, 'daily COP')


def rank_sensors(coverage_df: pd.DataFrame,
                 room_corr: pd.DataFrame,
                 residual_corr: pd.DataFrame,
                 cop_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Rank sensors by improvement potential.

    Score = |correlation| × coverage_weight × interpretability

    - Higher residual correlation = captures dynamics model misses
    - Higher COP correlation = improves efficiency prediction
    """
    print("\nRanking sensors by improvement potential...")

    # Start with coverage
    rankings = coverage_df[['sensor', 'category', 'coverage_pct', 'overlap_coverage_pct', 'is_focus']].copy()

    # Add room temperature correlation
    room_dict = dict(zip(room_corr['sensor'], room_corr['correlation']))
    rankings['room_corr'] = rankings['sensor'].map(room_dict)

    # Add residual correlation (key metric!)
    if len(residual_corr) > 0:
        resid_dict = dict(zip(residual_corr['sensor'], residual_corr['correlation']))
        rankings['residual_corr'] = rankings['sensor'].map(resid_dict)
    else:
        rankings['residual_corr'] = np.nan

    # Add COP correlation
    if len(cop_corr) > 0:
        cop_dict = dict(zip(cop_corr['sensor'], cop_corr['correlation']))
        rankings['cop_corr'] = rankings['sensor'].map(cop_dict)
    else:
        rankings['cop_corr'] = np.nan

    # Use max of overall and overlap coverage for scoring
    # This is fair for heating sensors that only have data from Oct 2025
    rankings['effective_coverage'] = rankings[['coverage_pct', 'overlap_coverage_pct']].max(axis=1)

    # Coverage weight (0.5 at 50%, 1.0 at 100%)
    rankings['coverage_weight'] = np.clip(rankings['effective_coverage'] / 100, 0.1, 1.0)

    # Thermal improvement score (based on residual correlation)
    rankings['thermal_score'] = (
        rankings['residual_corr'].abs() * rankings['coverage_weight']
    ).fillna(0)

    # COP improvement score
    rankings['cop_score'] = (
        rankings['cop_corr'].abs() * rankings['coverage_weight']
    ).fillna(0)

    # Combined score (weighted average)
    rankings['combined_score'] = (
        0.5 * rankings['thermal_score'] +
        0.5 * rankings['cop_score']
    )

    # Boost focus sensors
    rankings.loc[rankings['is_focus'], 'combined_score'] *= 1.2

    # Sort by combined score
    rankings = rankings.sort_values('combined_score', ascending=False)

    print(f"  Top 5 sensors for thermal improvement:")
    for i, row in rankings.head(5).iterrows():
        resid_str = f"{row['residual_corr']:.3f}" if pd.notna(row['residual_corr']) else 'N/A'
        print(f"    {row['sensor']}: score={row['thermal_score']:.3f}, resid_corr={resid_str}")

    return rankings


def create_coverage_figure(coverage_df: pd.DataFrame) -> None:
    """Create coverage heatmap visualization."""
    print("\nCreating coverage visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Coverage by category
    ax1 = axes[0, 0]
    cat_stats = coverage_df.groupby('category').agg({
        'coverage_pct': 'mean',
        'sensor': 'count'
    }).rename(columns={'sensor': 'n_sensors'})

    colors = ['steelblue' if cat != 'heating' else 'coral'
              for cat in cat_stats.index]
    bars = ax1.barh(cat_stats.index, cat_stats['coverage_pct'], color=colors)
    ax1.set_xlabel('Mean Coverage (%)')
    ax1.set_title('Coverage by Sensor Category')
    ax1.set_xlim(0, 100)

    # Add sensor counts
    for bar, (idx, row) in zip(bars, cat_stats.iterrows()):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'n={int(row["n_sensors"])}', va='center', fontsize=9)

    # Panel 2: Top 30 sensors by coverage
    ax2 = axes[0, 1]
    top30 = coverage_df.head(30)
    colors = ['green' if f else 'steelblue' for f in top30['is_focus']]
    ax2.barh(range(len(top30)), top30['coverage_pct'].values, color=colors)
    ax2.set_yticks(range(len(top30)))
    ax2.set_yticklabels([s[:35] for s in top30['sensor']], fontsize=7)
    ax2.set_xlabel('Coverage (%)')
    ax2.set_title('Top 30 Sensors by Coverage')
    ax2.invert_yaxis()
    ax2.legend(handles=[
        plt.Rectangle((0,0), 1, 1, fc='green', label='Focus sensor'),
        plt.Rectangle((0,0), 1, 1, fc='steelblue', label='Other sensor'),
    ], loc='lower right', fontsize=8)

    # Panel 3: Focus sensors coverage
    ax3 = axes[1, 0]
    focus_df = coverage_df[coverage_df['is_focus']].copy()
    if len(focus_df) > 0:
        focus_df = focus_df.sort_values('coverage_pct', ascending=True)
        colors = ['green' if c > 50 else 'orange' if c > 10 else 'red'
                  for c in focus_df['coverage_pct']]
        ax3.barh(range(len(focus_df)), focus_df['coverage_pct'].values, color=colors)
        ax3.set_yticks(range(len(focus_df)))
        ax3.set_yticklabels([s.replace('stiebel_eltron_isg_', '').replace('davis_', '')
                            for s in focus_df['sensor']], fontsize=8)
        ax3.set_xlabel('Coverage (%)')
        ax3.set_title('Focus Sensors Coverage')
        ax3.axvline(50, color='green', linestyle='--', alpha=0.5, label='50% threshold')
        ax3.legend(fontsize=8)

    # Panel 4: Coverage distribution
    ax4 = axes[1, 1]
    ax4.hist(coverage_df['coverage_pct'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(50, color='red', linestyle='--', label='50% threshold')
    ax4.axvline(80, color='green', linestyle='--', label='80% threshold')
    ax4.set_xlabel('Coverage (%)')
    ax4.set_ylabel('Number of Sensors')
    ax4.set_title('Coverage Distribution')
    ax4.legend(fontsize=8)

    # Add summary stats
    n_high = (coverage_df['coverage_pct'] > 80).sum()
    n_medium = ((coverage_df['coverage_pct'] > 50) & (coverage_df['coverage_pct'] <= 80)).sum()
    n_low = (coverage_df['coverage_pct'] <= 50).sum()
    ax4.text(0.95, 0.95, f'High (>80%): {n_high}\nMedium (50-80%): {n_medium}\nLow (<50%): {n_low}',
             transform=ax4.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_sensor_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_sensor_coverage.png")


def create_correlation_figure(rankings: pd.DataFrame,
                              room_corr: pd.DataFrame,
                              residual_corr: pd.DataFrame,
                              cop_corr: pd.DataFrame,
                              df: pd.DataFrame) -> None:
    """Create correlation analysis visualization."""
    print("\nCreating correlation visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: Top 20 room temperature correlations
    ax1 = axes[0, 0]
    if len(room_corr) > 0:
        top20_room = room_corr.head(20)
        colors = ['green' if c > 0 else 'red' for c in top20_room['correlation']]
        ax1.barh(range(len(top20_room)), top20_room['correlation'].values, color=colors)
        ax1.set_yticks(range(len(top20_room)))
        ax1.set_yticklabels([s[:30] for s in top20_room['sensor']], fontsize=7)
        ax1.set_xlabel('Correlation')
        ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_title('Top 20: Room Temperature Correlation')
    ax1.invert_yaxis()

    # Panel 2: Top 20 residual correlations (KEY!)
    ax2 = axes[0, 1]
    if len(residual_corr) > 0:
        top20_res = residual_corr.head(20)
        colors = ['green' if c > 0 else 'red' for c in top20_res['correlation']]
        ax2.barh(range(len(top20_res)), top20_res['correlation'].values, color=colors)
        ax2.set_yticks(range(len(top20_res)))
        ax2.set_yticklabels([s[:30] for s in top20_res['sensor']], fontsize=7)
        ax2.set_xlabel('Correlation')
        ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_title('Top 20: Model Residual Correlation\n(Higher = Captures Missing Dynamics)')
    ax2.invert_yaxis()

    # Panel 3: Top 20 COP correlations
    ax3 = axes[0, 2]
    if len(cop_corr) > 0:
        top20_cop = cop_corr.head(20)
        colors = ['green' if c > 0 else 'red' for c in top20_cop['correlation']]
        ax3.barh(range(len(top20_cop)), top20_cop['correlation'].values, color=colors)
        ax3.set_yticks(range(len(top20_cop)))
        ax3.set_yticklabels([s[:30] for s in top20_cop['sensor']], fontsize=7)
        ax3.set_xlabel('Correlation')
        ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_title('Top 20: Daily COP Correlation')
    ax3.invert_yaxis()

    # Panel 4: Combined improvement scores
    ax4 = axes[1, 0]
    top20_score = rankings.head(20)
    colors = ['green' if f else 'steelblue' for f in top20_score['is_focus']]
    ax4.barh(range(len(top20_score)), top20_score['combined_score'].values, color=colors)
    ax4.set_yticks(range(len(top20_score)))
    ax4.set_yticklabels([s[:30] for s in top20_score['sensor']], fontsize=7)
    ax4.set_xlabel('Combined Improvement Score')
    ax4.set_title('Top 20: Overall Improvement Potential')
    ax4.invert_yaxis()

    # Panel 5: Scatter - pressure sensors vs COP (if available)
    ax5 = axes[1, 1]
    if 'stiebel_eltron_isg_high_pressure_wp1' in df.columns:
        high_p = df['stiebel_eltron_isg_high_pressure_wp1']
        low_p = df.get('stiebel_eltron_isg_low_pressure_wp1', pd.Series(dtype=float))

        valid = high_p.notna()
        if valid.sum() > 100:
            # Color by time
            ax5.scatter(high_p[valid].values[::10],
                       low_p[valid].values[::10] if len(low_p) > 0 else np.zeros(valid.sum()//10),
                       c=np.arange(valid.sum()//10), cmap='viridis', alpha=0.5, s=10)
            ax5.set_xlabel('High Pressure (bar)')
            ax5.set_ylabel('Low Pressure (bar)')
            ax5.set_title('Refrigerant Pressures (color = time)')
    else:
        ax5.text(0.5, 0.5, 'Pressure data not available\nin integrated dataset',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Refrigerant Pressures')

    # Panel 6: Focus sensors correlation summary
    ax6 = axes[1, 2]
    focus_rankings = rankings[rankings['is_focus']].copy()
    if len(focus_rankings) > 0:
        metrics = ['room_corr', 'residual_corr', 'cop_corr']
        x = np.arange(len(focus_rankings))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = focus_rankings[metric].fillna(0).values
            ax6.bar(x + i*width, values, width, label=metric.replace('_corr', ''))

        ax6.set_xticks(x + width)
        ax6.set_xticklabels([s.replace('stiebel_eltron_isg_', '').replace('davis_', '')[:12]
                           for s in focus_rankings['sensor']], rotation=45, ha='right', fontsize=7)
        ax6.set_ylabel('Correlation')
        ax6.legend(fontsize=8)
        ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_title('Focus Sensors: Correlation Summary')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_sensor_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_sensor_correlations.png")


def generate_report(coverage_df: pd.DataFrame,
                    rankings: pd.DataFrame,
                    room_corr: pd.DataFrame,
                    residual_corr: pd.DataFrame,
                    cop_corr: pd.DataFrame) -> str:
    """Generate HTML report for sensor exploration."""
    print("\nGenerating HTML report...")

    # Top sensors tables
    def df_to_html_table(df, columns, max_rows=15):
        rows = ""
        for _, row in df.head(max_rows).iterrows():
            cells = "".join(f"<td>{row[c]:.3f if isinstance(row[c], float) else row[c]}</td>"
                           for c in columns)
            rows += f"<tr>{cells}</tr>"
        headers = "".join(f"<th>{c}</th>" for c in columns)
        return f"<table><tr>{headers}</tr>{rows}</table>"

    # Focus sensors detail
    focus_detail = rankings[rankings['is_focus']][
        ['sensor', 'coverage_pct', 'room_corr', 'residual_corr', 'cop_corr', 'combined_score']
    ].copy()
    focus_detail['sensor'] = focus_detail['sensor'].apply(
        lambda x: x.replace('stiebel_eltron_isg_', '').replace('davis_', '')
    )

    html = f"""
    <section id="sensor-exploration">
    <h2>Sensor Exploration Analysis</h2>

    <h3>Summary</h3>
    <ul>
        <li><strong>Total sensors analyzed:</strong> {len(coverage_df)}</li>
        <li><strong>High coverage (>80%):</strong> {(coverage_df['coverage_pct'] > 80).sum()}</li>
        <li><strong>Focus sensors with data:</strong> {coverage_df['is_focus'].sum()}</li>
    </ul>

    <h3>Coverage by Category</h3>
    <table>
        <tr><th>Category</th><th>Sensors</th><th>Mean Coverage</th></tr>
        {''.join(f"<tr><td>{cat}</td><td>{len(g)}</td><td>{g['coverage_pct'].mean():.1f}%</td></tr>"
                for cat, g in coverage_df.groupby('category'))}
    </table>

    <h3>Focus Sensors Analysis</h3>
    <p>These sensors were identified as high-potential for model improvement:</p>
    {focus_detail.to_html(index=False, float_format='%.3f')}

    <h3>Top 10 for Thermal Model Improvement</h3>
    <p>Sensors with highest correlation to model residuals (captures dynamics the model misses):</p>
    {rankings.head(10)[['sensor', 'coverage_pct', 'residual_corr', 'thermal_score']].to_html(index=False, float_format='%.3f')}

    <h3>Top 10 for COP Model Improvement</h3>
    <p>Sensors with highest correlation to daily COP:</p>
    {rankings.nlargest(10, 'cop_score')[['sensor', 'coverage_pct', 'cop_corr', 'cop_score']].to_html(index=False, float_format='%.3f')}

    <h3>Key Findings</h3>
    <ul>
        <li><strong>Pressure sensors:</strong> {coverage_df[coverage_df['sensor'].str.contains('pressure', case=False)]['coverage_pct'].mean():.1f}% avg coverage - excellent for thermodynamic COP model</li>
        <li><strong>Wind sensors:</strong> {coverage_df[coverage_df['sensor'].str.contains('wind', case=False)]['coverage_pct'].mean():.1f}% avg coverage - potential for infiltration modeling</li>
        <li><strong>Defrost sensor:</strong> {coverage_df[coverage_df['sensor'].str.contains('defrost', case=False)]['coverage_pct'].values[0] if len(coverage_df[coverage_df['sensor'].str.contains('defrost', case=False)]) > 0 else 0:.1f}% coverage - useful for filtering</li>
    </ul>

    <h3>Recommendations</h3>
    <ol>
        <li><strong>Thermodynamic COP model:</strong> Use high/low pressure sensors (excellent coverage) to compute Carnot efficiency</li>
        <li><strong>Defrost filtering:</strong> Exclude periods with evaporator_defrost=1 to reduce noise</li>
        <li><strong>Wind infiltration:</strong> Add wind speed as thermal model input for infiltration losses</li>
    </ol>

    <figure>
        <img src="fig_sensor_coverage.png" alt="Sensor Coverage Analysis">
        <figcaption>Sensor coverage analysis by category and individual sensors.</figcaption>
    </figure>

    <figure>
        <img src="fig_sensor_correlations.png" alt="Sensor Correlation Analysis">
        <figcaption>Correlation analysis: sensors vs room temperature, model residuals, and COP.</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Run sensor exploration analysis."""
    print("=" * 60)
    print("Phase 2, Step 6: Sensor Exploration")
    print("=" * 60)

    # Load data
    summary = load_sensor_summary()
    df = load_integrated_dataset()
    residuals = load_thermal_residuals()
    cop_df = load_cop_data()

    # Room temperature target
    room_temp = df.get('davis_inside_temperature', pd.Series(dtype=float))

    # Compute coverage
    coverage_df = compute_coverage_stats(df, summary)

    # Compute correlations
    room_corr = compute_correlations(df, room_temp, 'room temperature')

    if residuals is not None:
        residual_corr = compute_correlations(df, residuals, 'model residuals')
    else:
        residual_corr = pd.DataFrame(columns=['sensor', 'correlation', 'p_value', 'n_obs'])

    cop_corr = compute_cop_correlations(df, cop_df)

    # Rank sensors
    rankings = rank_sensors(coverage_df, room_corr, residual_corr, cop_corr)

    # Visualizations
    create_coverage_figure(coverage_df)
    create_correlation_figure(rankings, room_corr, residual_corr, cop_corr, df)

    # Save rankings
    rankings.to_csv(OUTPUT_DIR / 'sensor_exploration_rankings.csv', index=False)
    print(f"\nSaved: sensor_exploration_rankings.csv")

    # Generate report
    report_html = generate_report(coverage_df, rankings, room_corr, residual_corr, cop_corr)
    with open(OUTPUT_DIR / 'sensor_exploration_report.html', 'w') as f:
        f.write(report_html)
    print("Saved: sensor_exploration_report.html")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nSensors analyzed: {len(coverage_df)}")
    print(f"High coverage (>80%): {(coverage_df['coverage_pct'] > 80).sum()}")

    print("\nTop 5 sensors for improvement:")
    for i, row in rankings.head(5).iterrows():
        print(f"  {i+1}. {row['sensor'][:40]}")
        print(f"     Coverage: {row['coverage_pct']:.1f}%, Score: {row['combined_score']:.3f}")


if __name__ == '__main__':
    main()
