#!/usr/bin/env python3
"""
Phase 2, Step 2: Battery Degradation Analysis

Standalone analysis investigating whether the Feb-Mar 2025 deep-discharge event
(caused by a faulty inverter) significantly affected battery round-trip efficiency.

Outputs:
- battery_degradation_analysis.png - Figure with efficiency trends
- battery_degradation_report.html - Detailed HTML report with methods and results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(exist_ok=True)

# Event definition
EVENT_START = pd.Timestamp('2025-02-01')
EVENT_END = pd.Timestamp('2025-03-31')
EVENT_DESCRIPTION = "Deep discharge due to faulty inverter (no charging)"


def load_data():
    """Load required datasets for battery analysis."""
    print("Loading data...")

    data = {}

    # Energy balance - daily aggregates
    data['energy_daily'] = pd.read_parquet(PROCESSED_DIR / 'energy_balance_daily.parquet')
    data['energy_daily'].index = pd.to_datetime(data['energy_daily'].index)

    # Heating sensors for outdoor temperature
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)
    heating_raw = heating_raw.set_index('datetime')
    data['heating'] = heating_raw.pivot_table(
        values='value',
        index=heating_raw.index,
        columns='entity_id',
        aggfunc='mean'
    )

    print(f"  Energy data: {data['energy_daily'].index.min().date()} to {data['energy_daily'].index.max().date()}")
    print(f"  Heating sensors: {len(data['heating'])} rows")

    return data


def calculate_monthly_efficiency(data):
    """Calculate monthly battery efficiency metrics."""
    energy = data['energy_daily'].copy()

    # Monthly aggregates
    energy['year'] = energy.index.year
    energy['month'] = energy.index.month

    monthly = energy.groupby(['year', 'month']).agg({
        'battery_charging_kwh': 'sum',
        'battery_discharging_kwh': 'sum',
        'pv_generation_kwh': 'sum',
        'total_consumption_kwh': 'sum'
    }).reset_index()

    monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=15))
    monthly = monthly.set_index('date').sort_index()

    # Calculate efficiency (discharge / charge ratio)
    # Only for months with meaningful charging (> 10 kWh)
    monthly['efficiency'] = np.nan
    mask = monthly['battery_charging_kwh'] > 10
    monthly.loc[mask, 'efficiency'] = (
        monthly.loc[mask, 'battery_discharging_kwh'] /
        monthly.loc[mask, 'battery_charging_kwh']
    )

    # Add analysis variables
    monthly['months_since_start'] = np.arange(len(monthly))
    monthly['post_event'] = (monthly.index > EVENT_END).astype(int)
    monthly['is_event_period'] = (
        (monthly.index >= EVENT_START) & (monthly.index <= EVENT_END)
    )

    # Add seasonal variables
    monthly['month_num'] = monthly.index.month
    monthly['season'] = monthly['month_num'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    # Add outdoor temperature if available
    heating = data['heating']
    outdoor_temp_col = 'stiebel_eltron_isg_outdoor_temperature'
    if outdoor_temp_col in heating.columns:
        monthly_temp = heating[outdoor_temp_col].resample('ME').mean()
        monthly_temp.index = monthly_temp.index - pd.Timedelta(days=14)
        monthly_temp.index = monthly_temp.index.tz_localize(None)
        monthly = monthly.join(monthly_temp.rename('outdoor_temp'), how='left')

    return monthly


def run_statistical_analysis(monthly):
    """Run regression analysis and statistical tests."""
    results = {}

    # Descriptive statistics
    pre_event = monthly[(monthly.index < EVENT_START) & monthly['efficiency'].notna()]
    post_event = monthly[(monthly.index > EVENT_END) & monthly['efficiency'].notna()]

    results['pre_mean'] = pre_event['efficiency'].mean() * 100
    results['pre_std'] = pre_event['efficiency'].std() * 100
    results['pre_n'] = len(pre_event)
    results['post_mean'] = post_event['efficiency'].mean() * 100
    results['post_std'] = post_event['efficiency'].std() * 100
    results['post_n'] = len(post_event)
    results['diff'] = results['post_mean'] - results['pre_mean']

    # Regression analysis
    reg_data = monthly[monthly['efficiency'].notna() & ~monthly['is_event_period']].copy()

    if len(reg_data) > 10:
        try:
            import statsmodels.api as sm
            from scipy import stats

            # Prepare variables
            X = reg_data[['months_since_start', 'post_event']].copy()
            if 'outdoor_temp' in reg_data.columns and reg_data['outdoor_temp'].notna().sum() > 5:
                X['outdoor_temp'] = reg_data['outdoor_temp']
                results['has_temp'] = True
            else:
                results['has_temp'] = False

            X = sm.add_constant(X)
            y = reg_data['efficiency'] * 100

            # Fit OLS with robust standard errors
            model = sm.OLS(y, X, missing='drop')
            fit = model.fit(cov_type='HC3')

            results['r_squared'] = fit.rsquared
            results['coefficients'] = {}

            for var in fit.params.index:
                results['coefficients'][var] = {
                    'coef': fit.params[var],
                    'se': fit.bse[var],
                    'pvalue': fit.pvalues[var],
                    'ci_low': fit.conf_int().loc[var, 0],
                    'ci_high': fit.conf_int().loc[var, 1]
                }

            # Welch's t-test
            pre_eff = pre_event['efficiency'] * 100
            post_eff = post_event['efficiency'] * 100
            t_stat, t_pval = stats.ttest_ind(pre_eff, post_eff, equal_var=False)
            results['ttest_stat'] = t_stat
            results['ttest_pval'] = t_pval

            # --- Model 2: With seasonal controls ---
            # Create seasonal dummies (Summer as reference)
            season_dummies = pd.get_dummies(reg_data['season'], prefix='season', drop_first=False, dtype=float)
            # Drop Summer as reference category
            if 'season_Summer' in season_dummies.columns:
                season_dummies = season_dummies.drop('season_Summer', axis=1)

            X_seasonal = reg_data[['months_since_start', 'post_event']].copy().astype(float)
            X_seasonal = pd.concat([X_seasonal, season_dummies], axis=1)
            X_seasonal = sm.add_constant(X_seasonal)

            model_seasonal = sm.OLS(y, X_seasonal, missing='drop')
            fit_seasonal = model_seasonal.fit(cov_type='HC3')

            results['seasonal_model'] = {
                'r_squared': fit_seasonal.rsquared,
                'coefficients': {}
            }

            for var in fit_seasonal.params.index:
                results['seasonal_model']['coefficients'][var] = {
                    'coef': fit_seasonal.params[var],
                    'se': fit_seasonal.bse[var],
                    'pvalue': fit_seasonal.pvalues[var],
                    'ci_low': fit_seasonal.conf_int().loc[var, 0],
                    'ci_high': fit_seasonal.conf_int().loc[var, 1]
                }

            # --- Matched-month comparison (paired analysis) ---
            # Compare same calendar months before and after event
            pre_by_month = pre_event.groupby('month_num')['efficiency'].mean() * 100
            post_by_month = post_event.groupby('month_num')['efficiency'].mean() * 100

            # Find months that exist in both periods
            common_months = pre_by_month.index.intersection(post_by_month.index)
            if len(common_months) >= 3:
                pre_matched = pre_by_month[common_months]
                post_matched = post_by_month[common_months]

                # Paired t-test
                paired_t_stat, paired_t_pval = stats.ttest_rel(pre_matched, post_matched)
                paired_diff = (post_matched - pre_matched).mean()

                results['matched_month'] = {
                    'n_months': len(common_months),
                    'months': list(common_months),
                    'pre_mean': pre_matched.mean(),
                    'post_mean': post_matched.mean(),
                    'mean_diff': paired_diff,
                    't_stat': paired_t_stat,
                    'pvalue': paired_t_pval
                }
            else:
                results['matched_month'] = None

            results['success'] = True

        except ImportError as e:
            results['success'] = False
            results['error'] = str(e)
    else:
        results['success'] = False
        results['error'] = "Insufficient data for regression"

    return results


def create_figure(monthly, stats):
    """Create the battery degradation analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color scheme
    PRE_COLOR = '#2563eb'  # Blue
    POST_COLOR = '#dc2626'  # Red
    EVENT_COLOR = '#f97316'  # Orange

    # --- Plot 1: Efficiency time series with event highlighted ---
    ax = axes[0, 0]
    valid = monthly[monthly['efficiency'].notna() & ~monthly['is_event_period']].copy()

    # Pre-event points
    pre = valid[valid.index < EVENT_START]
    ax.scatter(pre.index, pre['efficiency'] * 100, color=PRE_COLOR,
               alpha=0.8, s=60, label=f'Pre-event (n={len(pre)})', zorder=3)

    # Post-event points
    post = valid[valid.index > EVENT_END]
    ax.scatter(post.index, post['efficiency'] * 100, color=POST_COLOR,
               alpha=0.8, s=60, label=f'Post-event (n={len(post)})', zorder=3)

    # Event period highlighting
    ax.axvspan(EVENT_START, EVENT_END, alpha=0.3, color=EVENT_COLOR,
               label='Deep discharge event', zorder=1)

    # Mean lines
    if len(pre) > 0:
        ax.axhline(y=stats['pre_mean'], color=PRE_COLOR, linestyle='--',
                   alpha=0.6, linewidth=2, zorder=2)
        ax.text(pre.index.min(), stats['pre_mean'] + 1,
                f"Pre: {stats['pre_mean']:.1f}%", color=PRE_COLOR, fontsize=10)
    if len(post) > 0:
        ax.axhline(y=stats['post_mean'], color=POST_COLOR, linestyle='--',
                   alpha=0.6, linewidth=2, zorder=2)
        ax.text(post.index.max(), stats['post_mean'] + 1,
                f"Post: {stats['post_mean']:.1f}%", color=POST_COLOR, fontsize=10, ha='right')

    ax.set_ylabel('Round-trip Efficiency (%)', fontsize=11)
    ax.set_title('Monthly Battery Efficiency Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # --- Plot 2: Charge/discharge volumes highlighting event ---
    ax = axes[0, 1]
    width = 20

    # Color bars by period
    bar_colors_charge = []
    bar_colors_discharge = []
    for date in monthly.index:
        if date >= EVENT_START and date <= EVENT_END:
            bar_colors_charge.append(EVENT_COLOR)
            bar_colors_discharge.append(EVENT_COLOR)
        elif date < EVENT_START:
            bar_colors_charge.append('#9333ea')  # Purple
            bar_colors_discharge.append('#f59e0b')  # Amber
        else:
            bar_colors_charge.append('#7c3aed')  # Darker purple
            bar_colors_discharge.append('#d97706')  # Darker amber

    ax.bar(monthly.index, monthly['battery_charging_kwh'], width=width,
           alpha=0.7, color=bar_colors_charge, label='Charging')
    ax.bar(monthly.index, -monthly['battery_discharging_kwh'], width=width,
           alpha=0.7, color=bar_colors_discharge, label='Discharging')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvspan(EVENT_START, EVENT_END, alpha=0.2, color=EVENT_COLOR)

    # Annotate event period
    mid_event = EVENT_START + (EVENT_END - EVENT_START) / 2
    ax.annotate('No charging\n(faulty inverter)', xy=(mid_event, 50),
                fontsize=9, ha='center', color=EVENT_COLOR, fontweight='bold')

    ax.set_ylabel('Energy (kWh/month)', fontsize=11)
    ax.set_title('Monthly Battery Activity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # --- Plot 3: Efficiency trend with regression line ---
    ax = axes[1, 0]
    valid_eff = monthly['efficiency'].dropna()

    if len(valid_eff) >= 3:
        # Rolling average
        rolling = valid_eff.rolling(window=3, center=True, min_periods=2).mean()
        ax.plot(rolling.index, rolling.values * 100, 'b-', linewidth=2,
                label='3-month rolling avg', alpha=0.8)

        # Original points
        ax.scatter(valid_eff.index, valid_eff.values * 100, alpha=0.4, s=40, color='gray')

        # Regression lines (separate for pre and post)
        trend_data = monthly[monthly['efficiency'].notna() & ~monthly['is_event_period']].copy()

        # Pre-event trend
        pre_trend = trend_data[trend_data.index < EVENT_START]
        if len(pre_trend) > 3:
            x = np.arange(len(pre_trend))
            y = pre_trend['efficiency'].values * 100
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(pre_trend.index, p(x), '--', color=PRE_COLOR, linewidth=2,
                    label=f'Pre-event trend: {z[0]:.2f}%/mo')

        # Post-event trend
        post_trend = trend_data[trend_data.index > EVENT_END]
        if len(post_trend) > 2:
            x = np.arange(len(post_trend))
            y = post_trend['efficiency'].values * 100
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(post_trend.index, p(x), '--', color=POST_COLOR, linewidth=2,
                    label=f'Post-event trend: {z[0]:.2f}%/mo')

    ax.axvspan(EVENT_START, EVENT_END, alpha=0.3, color=EVENT_COLOR)
    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title('Efficiency Trends Before and After Event', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # --- Plot 4: Pre/post comparison boxplot ---
    ax = axes[1, 1]
    pre_eff = monthly[(monthly.index < EVENT_START) & monthly['efficiency'].notna()]['efficiency'] * 100
    post_eff = monthly[(monthly.index > EVENT_END) & monthly['efficiency'].notna()]['efficiency'] * 100

    box_data = [pre_eff.values, post_eff.values]
    bp = ax.boxplot(box_data,
                    labels=[f'Pre-event\n(before Feb 2025)\nn={len(pre_eff)}',
                            f'Post-event\n(after Mar 2025)\nn={len(post_eff)}'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(PRE_COLOR)
    bp['boxes'][0].set_alpha(0.3)
    bp['boxes'][1].set_facecolor(POST_COLOR)
    bp['boxes'][1].set_alpha(0.3)
    bp['medians'][0].set_color(PRE_COLOR)
    bp['medians'][1].set_color(POST_COLOR)

    # Add individual points with jitter
    np.random.seed(42)
    jitter1 = 1 + np.random.normal(0, 0.04, len(pre_eff))
    jitter2 = 2 + np.random.normal(0, 0.04, len(post_eff))
    ax.scatter(jitter1, pre_eff, alpha=0.6, color=PRE_COLOR, s=40, zorder=3)
    ax.scatter(jitter2, post_eff, alpha=0.6, color=POST_COLOR, s=40, zorder=3)

    # Add significance annotation
    if stats.get('success') and 'post_event' in stats.get('coefficients', {}):
        pval = stats['coefficients']['post_event']['pvalue']
        effect = stats['coefficients']['post_event']['coef']
        sig_text = f"Effect: {effect:.1f}%\np = {pval:.4f}"
        if pval < 0.001:
            sig_text += " ***"
        elif pval < 0.01:
            sig_text += " **"
        elif pval < 0.05:
            sig_text += " *"
        ax.text(1.5, 95, sig_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title('Efficiency Distribution: Pre vs Post Event', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(65, 100)

    plt.tight_layout()

    # Save figure
    fig_path = OUTPUT_DIR / 'battery_degradation_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {fig_path}")
    return fig_path


def generate_html_report(monthly, stats, fig_path):
    """Generate HTML report with full analysis documentation."""
    import base64

    # Embed image as base64
    with open(fig_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    report_date = datetime.now().strftime('%Y-%m-%d')
    data_start = monthly.index.min().strftime('%Y-%m')
    data_end = monthly.index.max().strftime('%Y-%m')

    # Significance stars helper
    def sig_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        elif p < 0.1: return "."
        return ""

    # Get coefficient info
    if stats.get('success'):
        coef = stats['coefficients']
        event_effect = coef['post_event']['coef']
        event_pval = coef['post_event']['pvalue']
        event_ci_low = coef['post_event']['ci_low']
        event_ci_high = coef['post_event']['ci_high']
        time_trend = coef['months_since_start']['coef']
        time_pval = coef['months_since_start']['pvalue']
        r_squared = stats['r_squared']

    # Build HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battery Degradation Analysis Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --danger: #dc2626;
            --success: #16a34a;
            --warning: #f97316;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; font-size: 1.8rem; }}
        h2 {{ color: var(--text); margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); font-size: 1.4rem; }}
        h3 {{ color: var(--text-muted); margin: 1.5rem 0 0.75rem; font-size: 1.1rem; }}
        p {{ margin: 0.75rem 0; }}
        .meta {{ color: var(--text-muted); margin-bottom: 1.5rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        ul {{ margin: 0.5rem 0 0.5rem 1.5rem; }}
        li {{ margin: 0.3rem 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg); font-weight: 600; }}
        code {{ background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; font-family: 'SF Mono', Monaco, monospace; font-size: 0.85rem; }}
        pre {{ background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }}
        .figure {{ text-align: center; margin: 1.5rem 0; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .figure-caption {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 0.5rem; }}
        .highlight {{ background: #fef3c7; padding: 1rem; border-left: 4px solid var(--warning); margin: 1rem 0; }}
        .result-box {{ background: #f0fdf4; border: 1px solid #86efac; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        .sig {{ color: var(--danger); font-weight: bold; }}
        .summary-table {{ font-family: 'SF Mono', Monaco, monospace; }}
        .summary-table td:nth-child(2), .summary-table td:nth-child(3) {{ text-align: right; }}
    </style>
</head>
<body>
    <h1>Battery Degradation Analysis Report</h1>
    <p class="meta">Generated: {report_date} | Data period: {data_start} to {data_end}</p>

    <h2>1. Background</h2>

    <p>This analysis investigates whether a deep-discharge event affected the round-trip efficiency of the home battery storage system.</p>

    <h3>1.1 System Overview</h3>
    <p>The system consists of a solar PV installation with battery storage for self-consumption optimization. Battery efficiency is measured as the ratio of energy discharged to energy charged (round-trip efficiency), accounting for conversion losses in both directions.</p>

    <h3>1.2 The Deep-Discharge Event</h3>
    <div class="highlight">
        <p><strong>Event Period:</strong> February - March 2025</p>
        <p><strong>Cause:</strong> Faulty inverter prevented battery charging</p>
        <p><strong>Duration:</strong> Approximately 8 weeks</p>
        <p><strong>Consequence:</strong> Battery remained in a deeply discharged state for an extended period</p>
    </div>
    <p>Deep discharge events are known to potentially cause irreversible capacity loss and efficiency degradation in lithium-ion batteries, particularly when the battery remains at low state-of-charge for extended periods.</p>

    <h3>1.3 Research Question</h3>
    <p><em>Did the deep-discharge event in February-March 2025 significantly affect the battery's round-trip efficiency?</em></p>

    <h2>2. Methods</h2>

    <h3>2.1 Data</h3>
    <ul>
        <li>Monthly battery charging and discharging totals from {data_start} to {data_end}</li>
        <li>Pre-event observations: {stats['pre_n']} months</li>
        <li>Post-event observations: {stats['post_n']} months</li>
        <li>Event period (excluded from analysis): February-March 2025</li>
    </ul>

    <h3>2.2 Efficiency Metric</h3>
    <p>Monthly round-trip efficiency was calculated as:</p>
    <pre>Efficiency = Discharge (kWh) / Charge (kWh) × 100%</pre>
    <p>Months with less than 10 kWh of charging were excluded to avoid unreliable estimates.</p>

    <h3>2.3 Statistical Models</h3>

    <p><strong>Model 1 (Basic):</strong></p>
    <pre>Efficiency = β₀ + β₁(Time) + β₂(PostEvent) + ε</pre>

    <p><strong>Model 2 (With Seasonal Controls):</strong></p>
    <pre>Efficiency = β₀ + β₁(Time) + β₂(PostEvent) + β₃(Winter) + β₄(Spring) + β₅(Autumn) + ε</pre>

    <p>Where:</p>
    <ul>
        <li><strong>Time:</strong> Months since start of observation (captures natural degradation)</li>
        <li><strong>PostEvent:</strong> Binary indicator (1 if after March 2025, 0 otherwise)</li>
        <li><strong>Seasonal dummies:</strong> Winter, Spring, Autumn (Summer as reference)</li>
        <li><strong>β₂:</strong> The coefficient of interest - change in efficiency attributable to the event</li>
    </ul>
    <p>Both models use heteroskedasticity-robust standard errors (HC3).</p>

    <h3>2.4 Additional Tests</h3>
    <ul>
        <li><strong>Welch's t-test:</strong> Compares mean efficiency between pre- and post-event periods</li>
        <li><strong>Matched-month paired t-test:</strong> Compares same calendar months before and after (e.g., Nov 2024 vs Nov 2025)</li>
    </ul>

    <h2>3. Results</h2>

    <h3>3.1 Descriptive Statistics</h3>
    <div class="card">
        <table>
            <tr><th>Period</th><th>Mean</th><th>Std Dev</th><th>N</th></tr>
            <tr><td>Pre-event</td><td>{stats['pre_mean']:.1f}%</td><td>±{stats['pre_std']:.1f}%</td><td>{stats['pre_n']} months</td></tr>
            <tr><td>Post-event</td><td>{stats['post_mean']:.1f}%</td><td>±{stats['post_std']:.1f}%</td><td>{stats['post_n']} months</td></tr>
            <tr><td><strong>Difference</strong></td><td colspan="3"><strong>{stats['diff']:.1f} percentage points</strong></td></tr>
        </table>
    </div>
"""

    if stats.get('success'):
        html += f"""
    <h3>3.2 Model 1: Basic Regression</h3>
    <div class="card">
        <p><strong>R² = {r_squared:.3f}</strong></p>
        <table>
            <tr><th>Variable</th><th>Coefficient</th><th>Std.Err</th><th>p-value</th><th></th></tr>
            <tr><td>Intercept</td><td>{coef['const']['coef']:.2f}</td><td>{coef['const']['se']:.2f}</td><td>{coef['const']['pvalue']:.4f}</td><td>{sig_stars(coef['const']['pvalue'])}</td></tr>
            <tr><td>Time (months)</td><td>{time_trend:.3f}</td><td>{coef['months_since_start']['se']:.3f}</td><td>{time_pval:.4f}</td><td>{sig_stars(time_pval)}</td></tr>
            <tr><td><strong>Post-event</strong></td><td><strong>{event_effect:.2f}</strong></td><td>{coef['post_event']['se']:.2f}</td><td class="sig">{event_pval:.4f}</td><td class="sig">{sig_stars(event_pval)}</td></tr>
        </table>
        <p style="font-size: 0.8rem; color: var(--text-muted);">Significance: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05</p>
        <p><strong>Event effect:</strong> {event_effect:.2f} pp (95% CI: [{event_ci_low:.2f}, {event_ci_high:.2f}], p={event_pval:.4f})</p>
    </div>
"""

        # Seasonal model
        if 'seasonal_model' in stats:
            sm = stats['seasonal_model']
            sm_coef = sm['coefficients']
            sm_event = sm_coef.get('post_event', {})

            html += f"""
    <h3>3.3 Model 2: With Seasonal Controls</h3>
    <div class="card">
        <p><strong>R² = {sm['r_squared']:.3f}</strong></p>
        <table>
            <tr><th>Variable</th><th>Coefficient</th><th>Std.Err</th><th>p-value</th><th></th></tr>
            <tr><td>Intercept</td><td>{sm_coef['const']['coef']:.2f}</td><td>{sm_coef['const']['se']:.2f}</td><td>{sm_coef['const']['pvalue']:.4f}</td><td>{sig_stars(sm_coef['const']['pvalue'])}</td></tr>
            <tr><td>Time (months)</td><td>{sm_coef['months_since_start']['coef']:.3f}</td><td>{sm_coef['months_since_start']['se']:.3f}</td><td>{sm_coef['months_since_start']['pvalue']:.4f}</td><td>{sig_stars(sm_coef['months_since_start']['pvalue'])}</td></tr>
            <tr><td><strong>Post-event</strong></td><td><strong>{sm_event.get('coef', 0):.2f}</strong></td><td>{sm_event.get('se', 0):.2f}</td><td class="sig">{sm_event.get('pvalue', 1):.4f}</td><td class="sig">{sig_stars(sm_event.get('pvalue', 1))}</td></tr>
"""
            for season in ['season_Winter', 'season_Spring', 'season_Autumn']:
                if season in sm_coef:
                    label = season.replace('season_', '')
                    html += f"""            <tr><td>{label}</td><td>{sm_coef[season]['coef']:.2f}</td><td>{sm_coef[season]['se']:.2f}</td><td>{sm_coef[season]['pvalue']:.4f}</td><td>{sig_stars(sm_coef[season]['pvalue'])}</td></tr>
"""
            html += f"""        </table>
        <p style="font-size: 0.8rem; color: var(--text-muted);">Reference category: Summer</p>
        <p><strong>Event effect (seasonally adjusted):</strong> {sm_event.get('coef', 0):.2f} pp (95% CI: [{sm_event.get('ci_low', 0):.2f}, {sm_event.get('ci_high', 0):.2f}], p={sm_event.get('pvalue', 1):.4f})</p>
    </div>
"""

        # Matched-month comparison
        html += """
    <h3>3.4 Matched-Month Paired Comparison</h3>
"""
        if stats.get('matched_month'):
            mm = stats['matched_month']
            month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            months_str = ', '.join([month_names.get(m, str(m)) for m in sorted(mm['months'])])
            html += f"""    <div class="card">
        <p>Compares efficiency in the same calendar months before and after the event.</p>
        <ul>
            <li><strong>Matched months:</strong> {months_str} (n={mm['n_months']})</li>
            <li><strong>Pre-event mean:</strong> {mm['pre_mean']:.1f}%</li>
            <li><strong>Post-event mean:</strong> {mm['post_mean']:.1f}%</li>
            <li><strong>Mean difference:</strong> {mm['mean_diff']:.2f} pp</li>
        </ul>
        <p><strong>Paired t-test:</strong> t = {mm['t_stat']:.3f}, p = {mm['pvalue']:.4f} {sig_stars(mm['pvalue'])}</p>
    </div>
"""
        else:
            html += """    <p>Insufficient overlapping months for matched comparison.</p>
"""

        # Welch's t-test
        html += f"""
    <h3>3.5 Robustness Check: Welch's t-test</h3>
    <div class="card">
        <p><strong>t-statistic:</strong> {stats['ttest_stat']:.3f}</p>
        <p><strong>p-value:</strong> {stats['ttest_pval']:.4f} {sig_stars(stats['ttest_pval'])}</p>
    </div>

    <h3>3.6 Summary of All Tests</h3>
    <div class="card">
        <table class="summary-table">
            <tr><th>Test</th><th>Effect (pp)</th><th>p-value</th><th></th></tr>
            <tr><td>Model 1 (Basic)</td><td>{event_effect:.2f}</td><td>{event_pval:.4f}</td><td>{sig_stars(event_pval)}</td></tr>
"""
        if 'seasonal_model' in stats and 'post_event' in stats['seasonal_model']['coefficients']:
            sm_ev = stats['seasonal_model']['coefficients']['post_event']
            html += f"""            <tr><td>Model 2 (Seasonal)</td><td>{sm_ev['coef']:.2f}</td><td>{sm_ev['pvalue']:.4f}</td><td>{sig_stars(sm_ev['pvalue'])}</td></tr>
"""
        if stats.get('matched_month'):
            mm = stats['matched_month']
            html += f"""            <tr><td>Matched-Month</td><td>{mm['mean_diff']:.2f}</td><td>{mm['pvalue']:.4f}</td><td>{sig_stars(mm['pvalue'])}</td></tr>
"""
        html += f"""            <tr><td>Welch's t-test</td><td>{stats['diff']:.2f}</td><td>{stats['ttest_pval']:.4f}</td><td>{sig_stars(stats['ttest_pval'])}</td></tr>
        </table>
        <p style="font-size: 0.8rem; color: var(--text-muted);">pp = percentage points; *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05</p>
    </div>
"""

    # Conclusion
    html += """
    <h2>4. Conclusion</h2>
"""

    if stats.get('success'):
        # Check significance across tests
        seasonal_sig = False
        if 'seasonal_model' in stats and 'post_event' in stats['seasonal_model']['coefficients']:
            seasonal_sig = stats['seasonal_model']['coefficients']['post_event']['pvalue'] < 0.05
        matched_sig = stats.get('matched_month', {}).get('pvalue', 1) < 0.05 if stats.get('matched_month') else False
        n_sig = sum([event_pval < 0.05, seasonal_sig, matched_sig, stats['ttest_pval'] < 0.05])

        if n_sig >= 3 and event_effect < 0:
            html += f"""    <div class="result-box">
        <p><strong>Strong and consistent statistical evidence</strong> indicates that the deep-discharge event significantly degraded battery efficiency.</p>
        <ul>
            <li>Basic model: {abs(event_effect):.1f} pp reduction (p={event_pval:.4f})</li>
"""
            if seasonal_sig:
                sm_ev = stats['seasonal_model']['coefficients']['post_event']
                html += f"""            <li>Seasonal-adjusted model: {abs(sm_ev['coef']):.1f} pp reduction (p={sm_ev['pvalue']:.4f})</li>
"""
            if matched_sig:
                mm = stats['matched_month']
                html += f"""            <li>Matched-month comparison: {abs(mm['mean_diff']):.1f} pp reduction (p={mm['pvalue']:.4f})</li>
"""
            html += f"""        </ul>
        <p>The effect is <strong>robust across multiple specifications</strong>, including models that control for seasonal variation. This rules out the possibility that the observed decline is merely a seasonal artifact.</p>
        <p>Efficiency degraded from approximately <strong>{stats['pre_mean']:.0f}%</strong> to <strong>{stats['post_mean']:.0f}%</strong>.</p>
    </div>

    <h3>Recommendations</h3>
    <ul>
        <li>Monitor battery capacity and efficiency metrics going forward</li>
        <li>Consider professional battery health assessment</li>
        <li>Implement low-voltage protection to prevent future deep-discharge events</li>
        <li>Factor reduced efficiency into energy management calculations</li>
    </ul>
"""
        elif n_sig >= 1:
            html += f"""    <p>The analysis provides <strong>mixed evidence</strong> regarding the impact of the deep-discharge event.</p>
    <p>The basic model shows a reduction of {abs(event_effect):.1f} pp (p={event_pval:.4f}), but results vary across different specifications.</p>

    <h3>Recommendations</h3>
    <ul>
        <li>Continue monitoring battery performance</li>
        <li>Collect additional post-event data to increase statistical power</li>
        <li>Consider seasonal factors when interpreting efficiency metrics</li>
    </ul>
"""
        else:
            html += f"""    <p>The analysis does not provide statistically significant evidence of efficiency degradation after the deep-discharge event.</p>

    <h3>Recommendations</h3>
    <ul>
        <li>Continue monitoring battery performance over coming months</li>
        <li>Re-evaluate with additional post-event data points</li>
        <li>Implement preventive measures for future deep-discharge scenarios</li>
    </ul>
"""
    else:
        html += f"""    <p>Unable to draw conclusions due to analysis errors: {stats.get('error', 'Unknown error')}</p>
"""

    # Figure
    html += f"""
    <h2>5. Figure</h2>
    <div class="figure">
        <img src="data:image/png;base64,{img_b64}" alt="Battery Degradation Analysis">
        <p class="figure-caption"><strong>Figure 1:</strong> Battery Degradation Analysis. Top-left: Monthly efficiency time series with event period highlighted. Top-right: Monthly charge/discharge volumes. Bottom-left: Efficiency trends with regression lines. Bottom-right: Pre/post efficiency distributions.</p>
    </div>

</body>
</html>
"""

    # Write HTML file
    report_path = OUTPUT_DIR / 'battery_degradation_report.html'
    report_path.write_text(html)

    print(f"  Saved: {report_path}")
    return report_path


def main():
    """Run battery degradation analysis."""
    print("="*60)
    print("BATTERY DEGRADATION ANALYSIS")
    print("="*60)
    print(f"Event period: {EVENT_START.strftime('%Y-%m')} to {EVENT_END.strftime('%Y-%m')}")
    print(f"Event: {EVENT_DESCRIPTION}")
    print()

    # Load data
    data = load_data()

    # Calculate monthly efficiency
    print("\nCalculating monthly efficiency...")
    monthly = calculate_monthly_efficiency(data)
    valid_months = monthly['efficiency'].notna().sum()
    print(f"  Valid months: {valid_months}")

    # Run statistical analysis
    print("\nRunning statistical analysis...")
    stats = run_statistical_analysis(monthly)

    if stats.get('success'):
        print(f"\n  Pre-event efficiency:  {stats['pre_mean']:.1f}% +/- {stats['pre_std']:.1f}%")
        print(f"  Post-event efficiency: {stats['post_mean']:.1f}% +/- {stats['post_std']:.1f}%")

        if 'post_event' in stats.get('coefficients', {}):
            effect = stats['coefficients']['post_event']['coef']
            pval = stats['coefficients']['post_event']['pvalue']
            print(f"\n  Basic Model - Event effect: {effect:.2f} pp (p={pval:.4f})")

            if pval < 0.05:
                print(f"  Result: STATISTICALLY SIGNIFICANT {'degradation' if effect < 0 else 'improvement'}")
            else:
                print(f"  Result: No significant effect detected")

        # Seasonal model results
        if 'seasonal_model' in stats and 'post_event' in stats['seasonal_model']['coefficients']:
            sm_coef = stats['seasonal_model']['coefficients']['post_event']
            print(f"\n  Seasonal Model - Event effect: {sm_coef['coef']:.2f} pp (p={sm_coef['pvalue']:.4f})")
            print(f"  R-squared: {stats['seasonal_model']['r_squared']:.3f}")

        # Matched-month comparison
        if stats.get('matched_month'):
            mm = stats['matched_month']
            print(f"\n  Matched-Month Comparison (n={mm['n_months']} months):")
            print(f"    Mean difference: {mm['mean_diff']:.2f} pp (p={mm['pvalue']:.4f})")
    else:
        print(f"  Warning: {stats.get('error', 'Analysis failed')}")

    # Create figure
    print("\nGenerating figure...")
    fig_path = create_figure(monthly, stats)

    # Generate HTML report
    print("\nGenerating HTML report...")
    report_path = generate_html_report(monthly, stats, fig_path)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Figure: {fig_path}")
    print(f"  Report: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
