#!/usr/bin/env python3
"""
Phase 2, Step 6: HK2 Target vs Actual Temperature Analysis

Analyzes the relationship between HK2 target (heating curve setpoint) and HK2 actual
(measured flow) temperatures. This is important because:
- Thermal model uses actual HK2 as heating effort input
- Optimization controls target HK2 via setpoint and curve_rise parameters
- Understanding the lag/dynamics between target→actual enables better modeling

Analysis components:
1. Time series visualization of actual vs target over heating period
2. Scatter plot showing correlation and deviations
3. Lag model fitting: T_actual[k+1] = T_actual[k] + (1/τ)(T_target[k] - T_actual[k])
4. Statistical analysis of deviations

Output:
- output/phase2/fig17_hk2_target_actual.png (4-panel visualization)
- output/phase2/hk2_target_actual_stats.csv (statistics summary)
- output/phase2/hk2_target_actual_report_section.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(exist_ok=True)

# Sensor column names
HK2_TARGET_COL = 'stiebel_eltron_isg_target_temperature_hk_2'
HK2_ACTUAL_COL = 'stiebel_eltron_isg_actual_temperature_hk_2'
HK2_ACTUAL_ALT_COL = 'wp_anlage_hk2_ist'  # Alternative actual column
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'


def load_heating_data():
    """Load heating sensor data and pivot to wide format."""
    print("Loading heating sensor data...")
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)

    n_sensors = heating_raw['entity_id'].nunique()
    print(f"  Loaded {len(heating_raw):,} rows, {n_sensors} sensors")

    # Pivot to wide format
    heating_wide = heating_raw.pivot_table(
        index='datetime',
        columns='entity_id',
        values='value',
        aggfunc='mean'
    )

    # Resample to 15-minute intervals
    heating_15min = heating_wide.resample('15min').mean()

    print(f"  Wide format: {len(heating_15min):,} rows, {len(heating_15min.columns)} columns")
    print(f"  Date range: {heating_15min.index.min()} to {heating_15min.index.max()}")

    return heating_15min


def prepare_hk2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HK2 target and actual temperatures with consistent timestamps.

    Returns DataFrame with columns: target, actual, outdoor, deviation
    """
    print("\nPreparing HK2 data...")

    # Check available columns
    has_target = HK2_TARGET_COL in df.columns
    has_actual = HK2_ACTUAL_COL in df.columns
    has_actual_alt = HK2_ACTUAL_ALT_COL in df.columns
    has_outdoor = OUTDOOR_COL in df.columns

    print(f"  Target column ({HK2_TARGET_COL}): {'found' if has_target else 'NOT FOUND'}")
    print(f"  Actual column ({HK2_ACTUAL_COL}): {'found' if has_actual else 'NOT FOUND'}")
    print(f"  Actual alt column ({HK2_ACTUAL_ALT_COL}): {'found' if has_actual_alt else 'NOT FOUND'}")
    print(f"  Outdoor column ({OUTDOOR_COL}): {'found' if has_outdoor else 'NOT FOUND'}")

    if not has_target:
        raise ValueError(f"Target column {HK2_TARGET_COL} not found in data")

    # Select actual column (prefer primary, fallback to alternative)
    actual_col = HK2_ACTUAL_COL if has_actual else HK2_ACTUAL_ALT_COL
    print(f"  Using actual column: {actual_col}")

    # Build result DataFrame
    result = pd.DataFrame(index=df.index)
    result['target'] = df[HK2_TARGET_COL]
    result['actual'] = df[actual_col]

    if has_outdoor:
        result['outdoor'] = df[OUTDOOR_COL]

    # Drop rows with missing values
    result = result.dropna(subset=['target', 'actual'])

    # Calculate deviation
    result['deviation'] = result['actual'] - result['target']

    print(f"  Valid observations: {len(result):,}")
    print(f"  Period: {result.index.min().date()} to {result.index.max().date()}")

    return result


def analyze_deviation_statistics(df: pd.DataFrame) -> dict:
    """Compute statistics on actual-target deviation."""
    print("\nAnalyzing deviation statistics...")

    deviation = df['deviation']

    stats_dict = {
        'n_obs': len(deviation),
        'mean_deviation': deviation.mean(),
        'std_deviation': deviation.std(),
        'median_deviation': deviation.median(),
        'min_deviation': deviation.min(),
        'max_deviation': deviation.max(),
        'pct_actual_above_target': (deviation > 0).mean() * 100,
        'pct_within_1deg': ((deviation.abs() <= 1.0).mean() * 100),
        'pct_within_2deg': ((deviation.abs() <= 2.0).mean() * 100),
        'mean_target': df['target'].mean(),
        'mean_actual': df['actual'].mean(),
        'correlation': df['target'].corr(df['actual']),
    }

    print(f"  Mean deviation: {stats_dict['mean_deviation']:.2f}°C")
    print(f"  Std deviation: {stats_dict['std_deviation']:.2f}°C")
    print(f"  Actual above target: {stats_dict['pct_actual_above_target']:.1f}%")
    print(f"  Within ±1°C: {stats_dict['pct_within_1deg']:.1f}%")
    print(f"  Within ±2°C: {stats_dict['pct_within_2deg']:.1f}%")
    print(f"  Correlation: {stats_dict['correlation']:.3f}")

    return stats_dict


def fit_lag_model(df: pd.DataFrame) -> dict:
    """
    Fit a first-order lag model: T_actual[k+1] = T_actual[k] + (dt/τ)(T_target[k] - T_actual[k])

    This models the heat pump control dynamics as a simple exponential approach
    to the target setpoint.

    Returns dict with tau, r2, rmse, predictions
    """
    print("\nFitting first-order lag model...")

    target = df['target'].values
    actual = df['actual'].values
    dt = 0.25  # 15-minute timestep in hours

    def simulate_lag(tau, target, actual_init):
        """Forward simulate the lag model."""
        n = len(target)
        pred = np.zeros(n)
        pred[0] = actual_init

        for k in range(n - 1):
            # First-order dynamics
            pred[k + 1] = pred[k] + (dt / tau) * (target[k] - pred[k])

        return pred

    def objective(tau):
        """MSE objective for optimization."""
        if tau <= 0.01:  # Minimum tau
            return 1e10
        pred = simulate_lag(tau, target, actual[0])
        return np.mean((actual - pred) ** 2)

    # Grid search for initial tau estimate
    tau_range = np.arange(0.1, 10.0, 0.1)
    mse_values = [objective(tau) for tau in tau_range]
    tau_init = tau_range[np.argmin(mse_values)]

    # Fine-tune with optimization
    result = minimize_scalar(objective, bounds=(0.05, 20.0), method='bounded')
    tau_opt = result.x

    # Generate predictions with optimal tau
    pred = simulate_lag(tau_opt, target, actual[0])

    # One-step prediction (for comparison)
    pred_one_step = np.zeros_like(actual)
    pred_one_step[0] = actual[0]
    for k in range(len(actual) - 1):
        pred_one_step[k + 1] = actual[k] + (dt / tau_opt) * (target[k] - actual[k])

    # Calculate metrics
    r2_forward = r2_score(actual, pred)
    rmse_forward = np.sqrt(mean_squared_error(actual, pred))
    r2_one_step = r2_score(actual[1:], pred_one_step[1:])
    rmse_one_step = np.sqrt(mean_squared_error(actual[1:], pred_one_step[1:]))

    print(f"  Optimal time constant τ: {tau_opt:.2f} hours")
    print(f"  Forward simulation: R² = {r2_forward:.3f}, RMSE = {rmse_forward:.2f}°C")
    print(f"  One-step prediction: R² = {r2_one_step:.3f}, RMSE = {rmse_one_step:.2f}°C")

    return {
        'tau': tau_opt,
        'r2_forward': r2_forward,
        'rmse_forward': rmse_forward,
        'r2_one_step': r2_one_step,
        'rmse_one_step': rmse_one_step,
        'predictions_forward': pred,
        'predictions_one_step': pred_one_step,
    }


def analyze_cross_correlation(df: pd.DataFrame, max_lag_hours: float = 4.0) -> dict:
    """
    Analyze cross-correlation between target and actual to find optimal lag.

    Returns dict with lag analysis results.
    """
    print("\nAnalyzing cross-correlation...")

    target = df['target'].values
    actual = df['actual'].values

    # Normalize signals
    target_norm = (target - target.mean()) / target.std()
    actual_norm = (actual - actual.mean()) / actual.std()

    # Calculate cross-correlation for various lags (in 15-min steps)
    max_lag_steps = int(max_lag_hours * 4)  # 4 steps per hour
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(target_norm[:lag], actual_norm[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(target_norm[lag:], actual_norm[:-lag])[0, 1]
        else:
            corr = np.corrcoef(target_norm, actual_norm)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    lag_hours = lags * 0.25  # Convert to hours

    # Find peak correlation
    peak_idx = np.argmax(correlations)
    peak_lag_hours = lag_hours[peak_idx]
    peak_correlation = correlations[peak_idx]

    print(f"  Peak correlation: {peak_correlation:.3f} at lag {peak_lag_hours:.2f} hours")
    print(f"  Zero-lag correlation: {correlations[max_lag_steps]:.3f}")

    return {
        'lag_hours': lag_hours,
        'correlations': correlations,
        'peak_lag_hours': peak_lag_hours,
        'peak_correlation': peak_correlation,
        'zero_lag_correlation': correlations[max_lag_steps],
    }


def create_visualization(df: pd.DataFrame, lag_model: dict, xcorr: dict,
                        deviation_stats: dict) -> None:
    """Create 4-panel visualization of HK2 target vs actual analysis."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Time series (full period)
    ax = axes[0, 0]

    # Resample to hourly for cleaner plot
    df_hourly = df.resample('h').mean()

    ax.plot(df_hourly.index, df_hourly['target'], label='Target (setpoint)',
            color='blue', alpha=0.8, linewidth=1)
    ax.plot(df_hourly.index, df_hourly['actual'], label='Actual (measured)',
            color='red', alpha=0.8, linewidth=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('HK2 Flow Temperature: Target vs Actual')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)

    # Panel 2: Scatter plot
    ax = axes[0, 1]

    # Subsample for scatter plot (every 4th point = hourly)
    df_scatter = df.iloc[::4]

    ax.scatter(df_scatter['target'], df_scatter['actual'], alpha=0.3, s=10, c='steelblue')

    # Add y=x reference line
    lims = [
        min(df['target'].min(), df['actual'].min()) - 1,
        max(df['target'].max(), df['actual'].max()) + 1
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')

    # Add regression line
    slope, intercept, r_value, _, _ = stats.linregress(df['target'], df['actual'])
    x_line = np.array(lims)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', alpha=0.7,
            label=f'Fit: y = {slope:.2f}x + {intercept:.1f}\nR² = {r_value**2:.3f}')

    ax.set_xlabel('Target Temperature (°C)')
    ax.set_ylabel('Actual Temperature (°C)')
    ax.set_title('Actual vs Target Correlation')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # Panel 3: Deviation histogram
    ax = axes[1, 0]

    deviation = df['deviation']

    ax.hist(deviation, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Add normal distribution fit
    mu, sigma = deviation.mean(), deviation.std()
    x = np.linspace(deviation.min(), deviation.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'Normal fit\nμ = {mu:.2f}°C\nσ = {sigma:.2f}°C')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=mu, color='red', linestyle='-', alpha=0.7)

    ax.set_xlabel('Deviation: Actual - Target (°C)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Temperature Deviation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 4: Lag model fit
    ax = axes[1, 1]

    # Show a sample period (e.g., first 7 days)
    n_samples = min(7 * 24 * 4, len(df))  # 7 days at 15-min resolution
    sample = df.iloc[:n_samples]
    pred_sample = lag_model['predictions_forward'][:n_samples]

    ax.plot(range(n_samples), sample['target'].values, label='Target',
            color='blue', alpha=0.7, linewidth=1)
    ax.plot(range(n_samples), sample['actual'].values, label='Actual',
            color='red', alpha=0.7, linewidth=1)
    ax.plot(range(n_samples), pred_sample, label=f'Lag model (τ={lag_model["tau"]:.1f}h)',
            color='green', alpha=0.8, linewidth=1.5, linestyle='--')

    # X-axis in hours
    xticks = np.arange(0, n_samples, 24 * 4)  # Daily ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'Day {i+1}' for i in range(len(xticks))])

    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'First-Order Lag Model Fit (R² = {lag_model["r2_forward"]:.3f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig17_hk2_target_actual.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig17_hk2_target_actual.png")


def generate_report(deviation_stats: dict, lag_model: dict, xcorr: dict) -> str:
    """Generate HTML report section for HK2 target vs actual analysis."""

    tau = lag_model['tau']
    r2_forward = lag_model['r2_forward']
    rmse_forward = lag_model['rmse_forward']
    r2_one_step = lag_model['r2_one_step']
    rmse_one_step = lag_model['rmse_one_step']

    # Determine recommendation based on model fit
    if r2_forward > 0.9:
        recommendation = f"""
        <div class="summary-box">
        <strong>Recommendation:</strong> The first-order lag model fits well (R² = {r2_forward:.3f}).
        This suggests we could use the lag model to predict $T_{{HK2,actual}}$ from $T_{{HK2,target}}$
        in the grey-box thermal model, potentially improving forward simulation performance.
        </div>
        """
    elif r2_forward > 0.7:
        recommendation = f"""
        <div class="warning">
        <strong>Recommendation:</strong> The lag model achieves moderate fit (R² = {r2_forward:.3f}).
        Consider using actual HK2 temperature directly in thermal models rather than predicting
        from target. The dynamics may involve factors beyond simple first-order lag.
        </div>
        """
    else:
        recommendation = f"""
        <div class="warning">
        <strong>Recommendation:</strong> The lag model fit is poor (R² = {r2_forward:.3f}).
        The relationship between target and actual HK2 temperatures is complex. Continue using
        measured actual HK2 temperature in thermal models. Investigate additional factors
        (e.g., buffer tank state, compressor cycling) that affect actual flow temperature.
        </div>
        """

    html = f"""
    <section id="hk2-target-actual">
    <h2>2.7 HK2 Target vs Actual Temperature Analysis</h2>

    <h3>Overview</h3>
    <p>This analysis examines the relationship between the HK2 <strong>target</strong> temperature
    (heating curve setpoint, controllable) and <strong>actual</strong> temperature (measured flow).
    Understanding this relationship is critical because:</p>
    <ul>
        <li>The thermal model uses actual HK2 as the heating input signal</li>
        <li>Optimization controls target HK2 via setpoint and curve_rise parameters</li>
        <li>A gap between target and actual affects model predictions</li>
    </ul>

    <h3>Deviation Statistics</h3>
    <p>Analysis of the difference: $\\Delta T = T_{{actual}} - T_{{target}}$</p>
    <table>
        <tr><th>Statistic</th><th>Value</th></tr>
        <tr><td>Number of observations</td><td>{deviation_stats['n_obs']:,}</td></tr>
        <tr><td>Mean deviation ($\\bar{{\\Delta T}}$)</td><td><strong>{deviation_stats['mean_deviation']:.2f}°C</strong></td></tr>
        <tr><td>Std deviation ($\\sigma_{{\\Delta T}}$)</td><td>{deviation_stats['std_deviation']:.2f}°C</td></tr>
        <tr><td>Min deviation</td><td>{deviation_stats['min_deviation']:.1f}°C</td></tr>
        <tr><td>Max deviation</td><td>{deviation_stats['max_deviation']:.1f}°C</td></tr>
        <tr><td>Actual above target</td><td>{deviation_stats['pct_actual_above_target']:.1f}%</td></tr>
        <tr><td>Within ±1°C</td><td>{deviation_stats['pct_within_1deg']:.1f}%</td></tr>
        <tr><td>Within ±2°C</td><td>{deviation_stats['pct_within_2deg']:.1f}%</td></tr>
        <tr><td>Correlation ($r$)</td><td>{deviation_stats['correlation']:.3f}</td></tr>
    </table>

    <h3>First-Order Lag Model</h3>
    <p>We fit a first-order lag model to characterize the dynamics:</p>
    <div class="equation-box">
    $$T_{{actual}}[k+1] = T_{{actual}}[k] + \\frac{{\\Delta t}}{{\\tau}} \\left( T_{{target}}[k] - T_{{actual}}[k] \\right)$$
    </div>
    <p>where $\\tau$ is the time constant (how quickly actual approaches target).</p>

    <table>
        <tr><th>Model Parameter</th><th>Value</th></tr>
        <tr><td>Time constant ($\\tau$)</td><td><strong>{tau:.2f} hours</strong></td></tr>
        <tr><td>Forward simulation $R^2$</td><td>{r2_forward:.3f}</td></tr>
        <tr><td>Forward simulation RMSE</td><td>{rmse_forward:.2f}°C</td></tr>
        <tr><td>One-step prediction $R^2$</td><td>{r2_one_step:.3f}</td></tr>
        <tr><td>One-step prediction RMSE</td><td>{rmse_one_step:.2f}°C</td></tr>
    </table>

    <p><strong>Interpretation:</strong> A time constant of τ = {tau:.1f} hours means the actual
    temperature reaches ~63% of a step change in target within {tau:.1f} hours, and ~95% within
    {3*tau:.1f} hours (3τ).</p>

    <h3>Cross-Correlation Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Peak correlation</td><td>{xcorr['peak_correlation']:.3f}</td></tr>
        <tr><td>Peak lag</td><td>{xcorr['peak_lag_hours']:.2f} hours</td></tr>
        <tr><td>Zero-lag correlation</td><td>{xcorr['zero_lag_correlation']:.3f}</td></tr>
    </table>

    {recommendation}

    <h3>Implications for Thermal Modeling</h3>
    <ul>
        <li><strong>Transfer function model:</strong> Currently uses actual HK2 ($T_{{actual}}$) as input.
            This is appropriate since it reflects real heat delivery.</li>
        <li><strong>Grey-box model:</strong> Uses target HK2 ($T_{{target}}$) as input.
            {"Consider adding a lag pre-filter to predict actual from target." if r2_forward > 0.8 else "The mismatch between target and actual may explain forward simulation divergence."}</li>
        <li><strong>Optimization:</strong> Changes to setpoint/curve_rise affect target immediately
            but take ~{tau:.0f} hours to fully propagate to actual flow temperature.</li>
    </ul>

    <figure>
        <img src="fig17_hk2_target_actual.png" alt="HK2 Target vs Actual Analysis">
        <figcaption><strong>Figure 17:</strong> HK2 target vs actual temperature analysis:
        time series (top-left), scatter correlation (top-right), deviation distribution (bottom-left),
        lag model fit (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for HK2 target vs actual analysis."""
    print("="*60)
    print("Phase 2, Step 6: HK2 Target vs Actual Temperature Analysis")
    print("="*60)

    # Load data
    heating_df = load_heating_data()

    # Prepare HK2 data
    df = prepare_hk2_data(heating_df)

    # Analyze deviation statistics
    deviation_stats = analyze_deviation_statistics(df)

    # Fit lag model
    lag_model = fit_lag_model(df)

    # Analyze cross-correlation
    xcorr = analyze_cross_correlation(df)

    # Create visualization
    create_visualization(df, lag_model, xcorr, deviation_stats)

    # Save statistics
    stats_df = pd.DataFrame([{
        **deviation_stats,
        'lag_tau': lag_model['tau'],
        'lag_r2_forward': lag_model['r2_forward'],
        'lag_rmse_forward': lag_model['rmse_forward'],
        'lag_r2_one_step': lag_model['r2_one_step'],
        'xcorr_peak_lag': xcorr['peak_lag_hours'],
        'xcorr_peak_corr': xcorr['peak_correlation'],
    }])
    stats_df.to_csv(OUTPUT_DIR / 'hk2_target_actual_stats.csv', index=False)
    print("\nSaved: hk2_target_actual_stats.csv")

    # Generate HTML report
    report_html = generate_report(deviation_stats, lag_model, xcorr)
    with open(OUTPUT_DIR / 'hk2_target_actual_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: hk2_target_actual_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("HK2 TARGET VS ACTUAL ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nDeviation Statistics:")
    print(f"  Mean (actual - target): {deviation_stats['mean_deviation']:.2f}°C")
    print(f"  Std deviation: {deviation_stats['std_deviation']:.2f}°C")
    print(f"  Correlation: {deviation_stats['correlation']:.3f}")

    print(f"\nLag Model:")
    print(f"  Time constant τ: {lag_model['tau']:.2f} hours")
    print(f"  Forward R²: {lag_model['r2_forward']:.3f}")
    print(f"  Forward RMSE: {lag_model['rmse_forward']:.2f}°C")

    if lag_model['r2_forward'] > 0.9:
        print("\n✓ Lag model fits well - could improve grey-box thermal model")
    elif lag_model['r2_forward'] > 0.7:
        print("\n⚠ Lag model has moderate fit - consider additional factors")
    else:
        print("\n✗ Lag model fits poorly - use actual HK2 in thermal models")


if __name__ == '__main__':
    main()
