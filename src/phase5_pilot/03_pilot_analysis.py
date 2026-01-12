#!/usr/bin/env python3
"""
Phase 5 Pilot: T_HK2-Based Thermal Response Analysis

Analyzes pilot experiment data to estimate the thermal response function:
    T_indoor = f(T_HK2_comfort, T_HK2_eco, T_outdoor)

Key insight: The heating curve model is deterministic and well-understood.
What we need to learn is how indoor temperature depends on T_HK2 history.

Two models are fit:
1. T_HK2-based model (PRIMARY): Uses T_HK2_comfort and T_HK2_eco as predictors
2. Raw parameter model (COMPARISON): Uses comfort_setpoint, eco_setpoint, curve_rise

Usage:
    python src/phase5_pilot/03_pilot_analysis.py
    python src/phase5_pilot/03_pilot_analysis.py --block 5  # Analyze through block 5

Outputs:
    output/phase5_pilot/pilot_analysis_results.csv
    output/phase5_pilot/pilot_model_coefficients.json
    output/phase5_pilot/pilot_analysis_report.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5_pilot'
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'

# Analysis constants
OCCUPIED_START = 8  # 08:00
OCCUPIED_END = 22   # 22:00
COMFORT_THRESHOLD = 18.5  # °C


def load_pilot_schedule() -> pd.DataFrame:
    """Load the pilot schedule."""
    schedule_path = OUTPUT_DIR / 'pilot_schedule.csv'
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    return pd.read_csv(schedule_path, parse_dates=['start_date', 'end_date',
                                                    'washout_end', 'measurement_start'])


def load_integrated_data() -> pd.DataFrame:
    """Load the integrated sensor dataset."""
    data_path = PHASE1_DIR / 'integrated_dataset.parquet'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {data_path}\n"
            "Run Phase 1 preprocessing first."
        )

    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def extract_block_metrics(
    data_df: pd.DataFrame,
    block_row: pd.Series,
) -> dict:
    """
    Extract metrics for a single block's measurement period.

    Args:
        data_df: Full integrated dataset
        block_row: Row from pilot_schedule

    Returns:
        Dictionary of metrics for the block
    """
    # Filter to measurement period only (exclude washout)
    mask = (
        (data_df['timestamp'] >= block_row['measurement_start']) &
        (data_df['timestamp'] <= block_row['end_date'] + pd.Timedelta(days=1))
    )
    block_data = data_df[mask].copy()

    if len(block_data) == 0:
        return None

    # Filter to occupied hours (08:00-22:00)
    block_data['hour'] = block_data['timestamp'].dt.hour
    occupied_mask = (
        (block_data['hour'] >= OCCUPIED_START) &
        (block_data['hour'] < OCCUPIED_END)
    )
    occupied_data = block_data[occupied_mask]

    # Extract T_weighted (davis_inside_temperature)
    t_weighted_col = 'davis_inside_temperature'
    if t_weighted_col not in block_data.columns:
        t_weighted_col = 'temphum_office1_temperature'  # Fallback

    t_weighted = occupied_data[t_weighted_col].dropna()

    # Calculate metrics
    metrics = {
        'block': block_row['block'],
        'design_point': block_row['design_point'],
        'measurement_start': block_row['measurement_start'],
        'measurement_end': block_row['end_date'],
        'n_readings': len(occupied_data),
        'n_t_weighted': len(t_weighted),

        # Parameters (for raw parameter model)
        'comfort_setpoint': block_row['comfort_setpoint'],
        'eco_setpoint': block_row['eco_setpoint'],
        'curve_rise': block_row['curve_rise'],
        'comfort_hours': block_row['comfort_hours'],

        # T_HK2 values (for thermal response model)
        'T_HK2_comfort': block_row.get('T_HK2_comfort', np.nan),
        'T_HK2_eco': block_row.get('T_HK2_eco', np.nan),

        # T_weighted metrics
        'T_weighted_mean': t_weighted.mean() if len(t_weighted) > 0 else np.nan,
        'T_weighted_min': t_weighted.min() if len(t_weighted) > 0 else np.nan,
        'T_weighted_max': t_weighted.max() if len(t_weighted) > 0 else np.nan,
        'T_weighted_std': t_weighted.std() if len(t_weighted) > 0 else np.nan,

        # Violation metrics
        'violation_pct': (
            (t_weighted < COMFORT_THRESHOLD).sum() / len(t_weighted) * 100
            if len(t_weighted) > 0 else np.nan
        ),
        'cold_hours': (
            (t_weighted < COMFORT_THRESHOLD).sum() * 0.25  # 15-min intervals
            if len(t_weighted) > 0 else np.nan
        ),
    }

    # Extract outdoor temperature for covariate
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    if outdoor_col in block_data.columns:
        outdoor = block_data[outdoor_col].dropna()
        metrics['outdoor_mean'] = outdoor.mean()
        metrics['outdoor_min'] = outdoor.min()
        metrics['outdoor_max'] = outdoor.max()

        # Heating degree days (base 18°C)
        metrics['HDD'] = max(0, 18 - outdoor.mean()) * len(block_data) / (24 * 4)

    # Extract COP if available
    cop_col = 'stiebel_eltron_isg_coefficient_of_performance'
    if cop_col in block_data.columns:
        cop = block_data[cop_col].dropna()
        metrics['COP_mean'] = cop.mean() if len(cop) > 0 else np.nan
        metrics['COP_std'] = cop.std() if len(cop) > 0 else np.nan

    # Extract grid import if available
    grid_col = 'external_energy_supply_kWh'
    if grid_col in block_data.columns:
        grid = block_data[grid_col].dropna()
        metrics['grid_kWh'] = grid.sum() if len(grid) > 0 else np.nan

    return metrics


def fit_rsm_model(
    df: pd.DataFrame,
    response: str,
    predictors: list,
    include_interactions: bool = True,
) -> dict:
    """
    Fit Response Surface Model for a given response variable.

    Args:
        df: DataFrame with block-level data
        response: Name of response variable
        predictors: List of predictor variable names
        include_interactions: Whether to include interaction terms

    Returns:
        Dictionary with model results
    """
    # Remove rows with missing values
    cols = [response] + predictors
    clean_df = df[cols].dropna()

    if len(clean_df) < len(predictors) + 1:
        return {
            'response': response,
            'n_obs': len(clean_df),
            'error': 'Insufficient data',
        }

    y = clean_df[response].values
    X = clean_df[predictors].values

    # Add intercept
    X_design = sm.add_constant(X)
    feature_names = ['intercept'] + predictors

    # Add interaction terms if requested
    if include_interactions and len(predictors) >= 2:
        # Add pairwise interactions
        for i in range(len(predictors)):
            for j in range(i + 1, len(predictors)):
                interaction = X[:, i] * X[:, j]
                X_design = np.column_stack([X_design, interaction])
                feature_names.append(f'{predictors[i]}:{predictors[j]}')

    # Fit OLS model
    model = sm.OLS(y, X_design)
    results = model.fit()

    # Extract coefficients and statistics
    coefficients = []
    for name, coef, se, t, p in zip(
        feature_names,
        results.params,
        results.bse,
        results.tvalues,
        results.pvalues
    ):
        coefficients.append({
            'term': name,
            'estimate': coef,
            'std_error': se,
            't_stat': t,
            'p_value': p,
            'significant': p < 0.05,
        })

    return {
        'response': response,
        'n_obs': len(clean_df),
        'r_squared': results.rsquared,
        'r_squared_adj': results.rsquared_adj,
        'f_stat': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'aic': results.aic,
        'bic': results.bic,
        'coefficients': coefficients,
    }


def analyze_pilot_data(max_block: int = None) -> dict:
    """
    Run full pilot analysis.

    Args:
        max_block: Maximum block number to include (None = all completed)

    Returns:
        Dictionary with all analysis results
    """
    print("Loading data...")
    schedule_df = load_pilot_schedule()

    try:
        data_df = load_integrated_data()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Analysis will be limited without sensor data.")
        data_df = None

    # Determine which blocks have data
    if max_block is not None:
        schedule_df = schedule_df[schedule_df['block'] <= max_block]

    print(f"Analyzing {len(schedule_df)} blocks...")

    # Extract metrics for each block
    block_metrics = []
    for _, row in schedule_df.iterrows():
        if data_df is not None:
            metrics = extract_block_metrics(data_df, row)
            if metrics is not None:
                block_metrics.append(metrics)
        else:
            # Add placeholder with just parameters
            block_metrics.append({
                'block': row['block'],
                'design_point': row['design_point'],
                'comfort_setpoint': row['comfort_setpoint'],
                'eco_setpoint': row['eco_setpoint'],
                'curve_rise': row['curve_rise'],
                'comfort_hours': row['comfort_hours'],
            })

    metrics_df = pd.DataFrame(block_metrics)

    print(f"Extracted metrics for {len(metrics_df)} blocks")

    # Fit models - both T_HK2-based (PRIMARY) and raw parameter (COMPARISON)
    models = {}

    # Check if T_HK2 values are available
    has_thk2 = (
        'T_HK2_comfort' in metrics_df.columns and
        metrics_df['T_HK2_comfort'].notna().sum() > 0
    )

    # T_HK2-based predictors (PRIMARY MODEL)
    if has_thk2:
        thk2_predictors = ['T_HK2_comfort', 'T_HK2_eco', 'comfort_hours']
        if 'outdoor_mean' in metrics_df.columns:
            thk2_predictors.append('outdoor_mean')

        # T_weighted model (T_HK2-based) - PRIMARY
        if 'T_weighted_mean' in metrics_df.columns:
            models['T_weighted_thk2'] = fit_rsm_model(
                metrics_df, 'T_weighted_mean', thk2_predictors, include_interactions=False
            )
            models['T_weighted_thk2']['model_type'] = 'T_HK2-based (PRIMARY)'

        # COP model (T_HK2-based)
        if 'COP_mean' in metrics_df.columns:
            models['COP_thk2'] = fit_rsm_model(
                metrics_df, 'COP_mean', thk2_predictors, include_interactions=False
            )
            models['COP_thk2']['model_type'] = 'T_HK2-based'

    # Raw parameter predictors (COMPARISON MODEL)
    raw_predictors = ['comfort_setpoint', 'eco_setpoint', 'curve_rise', 'comfort_hours']
    if 'outdoor_mean' in metrics_df.columns:
        raw_predictors.append('outdoor_mean')

    # T_weighted model (raw parameters) - COMPARISON
    if 'T_weighted_mean' in metrics_df.columns:
        models['T_weighted_raw'] = fit_rsm_model(
            metrics_df, 'T_weighted_mean', raw_predictors, include_interactions=True
        )
        models['T_weighted_raw']['model_type'] = 'Raw parameters (COMPARISON)'

    # COP model (raw parameters)
    if 'COP_mean' in metrics_df.columns:
        models['COP_raw'] = fit_rsm_model(
            metrics_df, 'COP_mean', raw_predictors, include_interactions=True
        )
        models['COP_raw']['model_type'] = 'Raw parameters'

    # Grid model (raw parameters only - not T_HK2 driven)
    if 'grid_kWh' in metrics_df.columns:
        models['grid'] = fit_rsm_model(
            metrics_df, 'grid_kWh', raw_predictors, include_interactions=True
        )

    results = {
        'generated': datetime.now().isoformat(),
        'n_blocks': len(metrics_df),
        'metrics': metrics_df.to_dict(orient='records'),
        'models': models,
    }

    return results, metrics_df


def print_model_summary(model: dict) -> None:
    """Print a summary of a fitted model."""
    if 'error' in model:
        print(f"  Error: {model['error']}")
        return

    model_type = model.get('model_type', '')
    if model_type:
        print(f"  Type: {model_type}")
    print(f"  R² = {model['r_squared']:.3f} (adj: {model['r_squared_adj']:.3f})")
    print(f"  F = {model['f_stat']:.2f}, p = {model['f_pvalue']:.4f}")
    print(f"  n = {model['n_obs']}")
    print(f"\n  Coefficients:")

    for coef in model['coefficients']:
        sig = '*' if coef['significant'] else ''
        print(f"    {coef['term']:30s}: {coef['estimate']:8.4f} "
              f"(SE={coef['std_error']:.4f}, p={coef['p_value']:.4f}){sig}")


def generate_html_report(results: dict, metrics_df: pd.DataFrame) -> str:
    """Generate HTML analysis report."""

    n_blocks = results['n_blocks']
    has_models = bool(results['models'])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Phase 5 Pilot: Analysis Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
        }}
        h1 {{ color: #2563eb; }}
        h2 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
        .card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; }}
        .significant {{ color: #16a34a; font-weight: bold; }}
        .not-significant {{ color: #6b7280; }}
        .metric {{ font-size: 1.5rem; font-weight: bold; color: #2563eb; }}
    </style>
</head>
<body>
    <h1>Phase 5 Pilot: Analysis Results</h1>
    <p>Generated: {results['generated']}</p>

    <h2>1. Summary</h2>
    <div class="card">
        <p><span class="metric">{n_blocks}</span> blocks analyzed</p>
    </div>
'''

    if not has_models or n_blocks < 3:
        html += '''
    <div class="card">
        <p><strong>Note:</strong> Insufficient data for model fitting.
        Run more blocks to enable regression analysis.</p>
    </div>
'''
    else:
        # T_HK2-based model (PRIMARY)
        if 'T_weighted_thk2' in results['models']:
            model = results['models']['T_weighted_thk2']
            model_type = model.get('model_type', 'T_HK2-based')
            html += f'''
    <h2>2. T_weighted Model: {model_type}</h2>
    <div class="card">
        <p><strong>This is the primary model</strong> - it estimates the thermal response function:
        T_indoor = f(T_HK2_comfort, T_HK2_eco, comfort_hours, outdoor_temp)</p>
        <p><strong>R² = {model['r_squared']:.3f}</strong> (adjusted: {model['r_squared_adj']:.3f})</p>
        <p>F-statistic: {model['f_stat']:.2f}, p-value: {model['f_pvalue']:.4f}</p>
        <table>
            <tr>
                <th>Term</th>
                <th>Estimate</th>
                <th>Std Error</th>
                <th>t</th>
                <th>p-value</th>
            </tr>
'''
            for coef in model['coefficients']:
                sig_class = 'significant' if coef['significant'] else 'not-significant'
                html += f'''
            <tr class="{sig_class}">
                <td>{coef['term']}</td>
                <td>{coef['estimate']:.4f}</td>
                <td>{coef['std_error']:.4f}</td>
                <td>{coef['t_stat']:.2f}</td>
                <td>{coef['p_value']:.4f}</td>
            </tr>
'''
            html += '''
        </table>
    </div>
'''

        # Raw parameter model (COMPARISON)
        if 'T_weighted_raw' in results['models']:
            model = results['models']['T_weighted_raw']
            model_type = model.get('model_type', 'Raw parameters')
            html += f'''
    <h2>3. T_weighted Model: {model_type}</h2>
    <div class="card">
        <p><em>Comparison model using raw parameters instead of T_HK2</em></p>
        <p><strong>R² = {model['r_squared']:.3f}</strong> (adjusted: {model['r_squared_adj']:.3f})</p>
        <table>
            <tr>
                <th>Term</th>
                <th>Estimate</th>
                <th>Std Error</th>
                <th>p-value</th>
            </tr>
'''
            for coef in model['coefficients']:
                sig_class = 'significant' if coef['significant'] else 'not-significant'
                html += f'''
            <tr class="{sig_class}">
                <td>{coef['term']}</td>
                <td>{coef['estimate']:.4f}</td>
                <td>{coef['std_error']:.4f}</td>
                <td>{coef['p_value']:.4f}</td>
            </tr>
'''
            html += '''
        </table>
    </div>
'''

        # COP T_HK2 model
        if 'COP_thk2' in results['models']:
            model = results['models']['COP_thk2']
            html += f'''
    <h2>4. COP Model (T_HK2-based)</h2>
    <div class="card">
        <p><strong>R² = {model['r_squared']:.3f}</strong> (adjusted: {model['r_squared_adj']:.3f})</p>
        <table>
            <tr>
                <th>Term</th>
                <th>Estimate</th>
                <th>p-value</th>
            </tr>
'''
            for coef in model['coefficients']:
                sig_class = 'significant' if coef['significant'] else 'not-significant'
                html += f'''
            <tr class="{sig_class}">
                <td>{coef['term']}</td>
                <td>{coef['estimate']:.4f}</td>
                <td>{coef['p_value']:.4f}</td>
            </tr>
'''
            html += '''
        </table>
    </div>
'''

    # Block metrics table
    html += '''
    <h2>5. Block-Level Metrics</h2>
    <div class="card">
        <table>
            <tr>
                <th>Block</th>
                <th>Design Pt</th>
                <th>T_HK2 C</th>
                <th>T_HK2 E</th>
                <th>Hours</th>
'''

    if 'T_weighted_mean' in metrics_df.columns:
        html += '<th>T_weighted</th>'
    if 'COP_mean' in metrics_df.columns:
        html += '<th>COP</th>'
    if 'outdoor_mean' in metrics_df.columns:
        html += '<th>Outdoor</th>'

    html += '</tr>'

    for _, row in metrics_df.iterrows():
        thk2_c = row.get('T_HK2_comfort', np.nan)
        thk2_e = row.get('T_HK2_eco', np.nan)
        html += f'''
            <tr>
                <td>{row['block']}</td>
                <td>{row['design_point']}</td>
                <td>{thk2_c:.1f}°C</td>
                <td>{thk2_e:.1f}°C</td>
                <td>{row['comfort_hours']:.0f}h</td>
'''
        if 'T_weighted_mean' in row and pd.notna(row.get('T_weighted_mean')):
            html += f'<td>{row["T_weighted_mean"]:.1f}°C</td>'
        elif 'T_weighted_mean' in metrics_df.columns:
            html += '<td>-</td>'

        if 'COP_mean' in row and pd.notna(row.get('COP_mean')):
            html += f'<td>{row["COP_mean"]:.2f}</td>'
        elif 'COP_mean' in metrics_df.columns:
            html += '<td>-</td>'

        if 'outdoor_mean' in row and pd.notna(row.get('outdoor_mean')):
            html += f'<td>{row["outdoor_mean"]:.1f}°C</td>'
        elif 'outdoor_mean' in metrics_df.columns:
            html += '<td>-</td>'

        html += '</tr>'

    html += '''
        </table>
        <p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">
            <strong>Note:</strong> T_HK2 C/E = Target flow temperature for comfort/eco mode (at ref outdoor temp 5°C).
            These are the design targets; actual T_HK2 depends on real outdoor temperature.
        </p>
    </div>
</body>
</html>'''

    return html


def save_results(results: dict, metrics_df: pd.DataFrame) -> None:
    """Save analysis results."""

    # CSV of block metrics
    csv_path = OUTPUT_DIR / 'pilot_analysis_results.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # JSON with full results
    json_path = OUTPUT_DIR / 'pilot_model_coefficients.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # HTML report
    html_path = OUTPUT_DIR / 'pilot_analysis_report.html'
    html = generate_html_report(results, metrics_df)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"Saved: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Phase 5 Pilot experiment data',
    )
    parser.add_argument(
        '--block', '-b',
        type=int,
        default=None,
        help='Analyze through this block number (default: all)',
    )

    args = parser.parse_args()

    print("Phase 5 Pilot: Response Surface Analysis")
    print("=" * 50)

    # Run analysis
    results, metrics_df = analyze_pilot_data(max_block=args.block)

    # Print summaries
    print(f"\nBlock metrics extracted: {len(metrics_df)}")

    if results['models']:
        for name, model in results['models'].items():
            print(f"\n{name} Model:")
            print_model_summary(model)
    else:
        print("\nNo models fitted (need more data).")

    # Save results
    save_results(results, metrics_df)

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("Review pilot_analysis_report.html for full results.")

    return 0


if __name__ == '__main__':
    exit(main())
