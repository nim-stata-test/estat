#!/usr/bin/env python3
"""
Battery Cost Savings Analysis (xtra)

Analyzes how much cost the battery saves compared to a hypothetical system
without a battery. The analysis considers:
- Time-varying purchase and feed-in tariffs
- 30% tax on feed-in income
- Battery round-trip efficiency (implicit in the data)

Logic:
- With battery (actual): We pay for grid imports and receive feed-in revenue
- Without battery: Battery charging energy would have been fed to grid,
  and battery discharging energy would have been imported from grid

Savings = battery_discharge × purchase_rate - battery_charge × feedin_rate × 0.70

Outputs:
- battery_savings_daily.csv - Daily cost savings data
- battery_savings_analysis.png - Cumulative savings visualization
- battery_savings_report.html - Summary report
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'xtra' / 'battery_savings'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tax rate on feed-in income
FEEDIN_TAX_RATE = 0.30


def load_data():
    """Load energy balance and tariff data."""
    print("Loading data...")

    # Load 15-minute energy balance data
    energy_path = PROCESSED_DIR / 'energy_balance_15min.parquet'
    energy = pd.read_parquet(energy_path)
    print(f"  Energy data: {len(energy):,} rows, {energy.index.min().date()} to {energy.index.max().date()}")

    # Load hourly tariff data
    tariff_path = PROCESSED_DIR / 'tariff_series_hourly.parquet'
    tariffs = pd.read_parquet(tariff_path)
    print(f"  Tariff data: {len(tariffs):,} rows, {tariffs.index.min().date()} to {tariffs.index.max().date()}")

    return energy, tariffs


def merge_tariffs_to_energy(energy, tariffs):
    """Merge tariff rates to 15-minute energy data."""
    print("Merging tariffs to energy data...")

    # Create hourly index for merging (floor to hour)
    energy['hour'] = energy.index.floor('H')

    # Select relevant tariff columns
    tariff_cols = ['purchase_rate_chf_kwh', 'feedin_rate_chf_kwh', 'is_high_tariff']
    tariffs_subset = tariffs[tariff_cols].copy()

    # Merge on hour
    merged = energy.merge(
        tariffs_subset,
        left_on='hour',
        right_index=True,
        how='left'
    )
    merged.index = energy.index
    merged = merged.drop(columns=['hour'])

    # Check for missing tariff data
    missing = merged['purchase_rate_chf_kwh'].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} rows missing tariff data, will be excluded")
        merged = merged.dropna(subset=['purchase_rate_chf_kwh'])

    print(f"  Merged data: {len(merged):,} rows")
    return merged


def calculate_costs(df):
    """Calculate actual costs and hypothetical costs without battery."""
    print("Calculating costs...")

    # Net feed-in rate after 30% tax
    feedin_rate_net = df['feedin_rate_chf_kwh'] * (1 - FEEDIN_TAX_RATE)

    # Actual system (with battery)
    # Cost = grid_import × purchase_rate - grid_feedin × feedin_rate_net
    df['cost_actual_chf'] = (
        df['external_supply_kwh'] * df['purchase_rate_chf_kwh'] -
        df['grid_feedin_kwh'] * feedin_rate_net
    )

    # Hypothetical system (without battery)
    # Grid import would increase by battery discharge
    # Grid export would increase by battery charge
    grid_import_no_battery = df['external_supply_kwh'] + df['battery_discharging_kwh']
    grid_export_no_battery = df['grid_feedin_kwh'] + df['battery_charging_kwh']

    df['cost_no_battery_chf'] = (
        grid_import_no_battery * df['purchase_rate_chf_kwh'] -
        grid_export_no_battery * feedin_rate_net
    )

    # Battery savings (positive = battery saves money)
    df['battery_savings_chf'] = df['cost_no_battery_chf'] - df['cost_actual_chf']

    # Also calculate component breakdown for analysis
    # Savings from avoided grid purchases
    df['savings_avoided_purchase_chf'] = df['battery_discharging_kwh'] * df['purchase_rate_chf_kwh']
    # Cost of foregone feed-in revenue
    df['cost_foregone_feedin_chf'] = df['battery_charging_kwh'] * feedin_rate_net

    return df


def aggregate_daily(df):
    """Aggregate 15-minute data to daily totals."""
    print("Aggregating to daily data...")

    daily = df.resample('D').agg({
        # Energy quantities
        'external_supply_kwh': 'sum',
        'grid_feedin_kwh': 'sum',
        'battery_charging_kwh': 'sum',
        'battery_discharging_kwh': 'sum',
        'pv_generation_kwh': 'sum',
        'total_consumption_kwh': 'sum',
        # Costs
        'cost_actual_chf': 'sum',
        'cost_no_battery_chf': 'sum',
        'battery_savings_chf': 'sum',
        'savings_avoided_purchase_chf': 'sum',
        'cost_foregone_feedin_chf': 'sum',
        # Tariff info (average for the day)
        'purchase_rate_chf_kwh': 'mean',
        'feedin_rate_chf_kwh': 'mean',
    })

    # Remove incomplete days (first and last)
    daily = daily.iloc[1:-1]

    # Add cumulative savings
    daily['cumulative_savings_chf'] = daily['battery_savings_chf'].cumsum()

    # Add battery round-trip efficiency
    daily['battery_efficiency'] = np.where(
        daily['battery_charging_kwh'] > 0,
        daily['battery_discharging_kwh'] / daily['battery_charging_kwh'],
        np.nan
    )

    print(f"  Daily data: {len(daily)} days")
    return daily


def create_visualizations(daily):
    """Create visualization of battery cost savings."""
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Cumulative savings over time
    ax1 = axes[0, 0]
    ax1.fill_between(daily.index, 0, daily['cumulative_savings_chf'],
                     alpha=0.3, color='green', label='Cumulative savings')
    ax1.plot(daily.index, daily['cumulative_savings_chf'],
             color='green', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Savings (CHF)')
    ax1.set_title('Cumulative Battery Cost Savings')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Add total savings annotation
    total_savings = daily['cumulative_savings_chf'].iloc[-1]
    ax1.annotate(f'Total: CHF {total_savings:,.0f}',
                 xy=(daily.index[-1], total_savings),
                 xytext=(-80, 10), textcoords='offset points',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: Monthly savings breakdown
    ax2 = axes[0, 1]
    monthly = daily.resample('M').agg({
        'battery_savings_chf': 'sum',
        'savings_avoided_purchase_chf': 'sum',
        'cost_foregone_feedin_chf': 'sum',
    })

    x = range(len(monthly))
    width = 0.8
    ax2.bar(x, monthly['savings_avoided_purchase_chf'], width,
            label='Avoided grid purchase', color='green', alpha=0.7)
    ax2.bar(x, -monthly['cost_foregone_feedin_chf'], width,
            label='Foregone feed-in revenue', color='red', alpha=0.7)
    ax2.plot(x, monthly['battery_savings_chf'], 'ko-',
             markersize=4, linewidth=1.5, label='Net savings')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Amount (CHF)')
    ax2.set_title('Monthly Savings Breakdown')
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index[::3]],
                        rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Daily savings distribution
    ax3 = axes[1, 0]
    ax3.hist(daily['battery_savings_chf'], bins=50, color='steelblue',
             alpha=0.7, edgecolor='white')
    ax3.axvline(x=daily['battery_savings_chf'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Mean: CHF {daily["battery_savings_chf"].mean():.2f}')
    ax3.axvline(x=daily['battery_savings_chf'].median(), color='orange',
                linestyle='--', linewidth=2, label=f'Median: CHF {daily["battery_savings_chf"].median():.2f}')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Daily Savings (CHF)')
    ax3.set_ylabel('Frequency (days)')
    ax3.set_title('Distribution of Daily Battery Savings')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Add negative days count
    neg_days = (daily['battery_savings_chf'] < 0).sum()
    total_days = len(daily)
    ax3.annotate(f'Days with negative savings: {neg_days} ({100*neg_days/total_days:.1f}%)',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 4: Savings vs tariff spread
    ax4 = axes[1, 1]
    # Calculate effective tariff spread (purchase - net feedin)
    daily['tariff_spread'] = daily['purchase_rate_chf_kwh'] - daily['feedin_rate_chf_kwh'] * (1 - FEEDIN_TAX_RATE)

    # Color by season
    colors = ['steelblue' if m in [11, 12, 1, 2, 3] else 'orange'
              for m in daily.index.month]
    ax4.scatter(daily['tariff_spread'] * 100, daily['battery_savings_chf'],
                c=colors, alpha=0.5, s=20)

    # Add trend line
    z = np.polyfit(daily['tariff_spread'], daily['battery_savings_chf'], 1)
    p = np.poly1d(z)
    spread_range = np.linspace(daily['tariff_spread'].min(), daily['tariff_spread'].max(), 100)
    ax4.plot(spread_range * 100, p(spread_range), 'r--', linewidth=2,
             label=f'Trend: {z[0]*100:.2f} CHF per Rp spread')

    ax4.set_xlabel('Tariff Spread (Rp/kWh)\n(Purchase rate - Net feed-in rate)')
    ax4.set_ylabel('Daily Savings (CHF)')
    ax4.set_title('Battery Savings vs Tariff Spread')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Add season legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.5, label='Winter (Nov-Mar)'),
                       Patch(facecolor='orange', alpha=0.5, label='Summer (Apr-Oct)')]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'battery_savings_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return fig_path


def generate_report(daily):
    """Generate HTML summary report."""
    print("Generating report...")

    # Calculate summary statistics
    total_days = len(daily)
    total_savings = daily['battery_savings_chf'].sum()
    avg_daily_savings = daily['battery_savings_chf'].mean()
    median_daily_savings = daily['battery_savings_chf'].median()

    # Days with positive/negative savings
    pos_days = (daily['battery_savings_chf'] > 0).sum()
    neg_days = (daily['battery_savings_chf'] < 0).sum()

    # Energy statistics
    total_battery_charge = daily['battery_charging_kwh'].sum()
    total_battery_discharge = daily['battery_discharging_kwh'].sum()
    avg_efficiency = total_battery_discharge / total_battery_charge if total_battery_charge > 0 else 0

    # Component breakdown
    total_avoided_purchase = daily['savings_avoided_purchase_chf'].sum()
    total_foregone_feedin = daily['cost_foregone_feedin_chf'].sum()

    # Monthly statistics
    monthly = daily.resample('M')['battery_savings_chf'].sum()
    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()

    # Yearly totals
    yearly = daily.resample('Y')['battery_savings_chf'].sum()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Battery Cost Savings Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #27ae60; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #d5f5e3; font-weight: bold; }}
        .negative {{ color: #c0392b; }}
        .positive {{ color: #27ae60; }}
        .summary-box {{ background-color: #eafaf1; border-left: 4px solid #27ae60;
                       padding: 15px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .methodology {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Battery Cost Savings Analysis</h1>

    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>

    <div class="summary-box">
        <h3>Key Finding</h3>
        <p>The battery has saved a total of <strong class="positive">CHF {total_savings:,.2f}</strong>
        over {total_days:,} days ({total_days/365:.1f} years),
        averaging <strong>CHF {avg_daily_savings:.2f}</strong> per day.</p>
    </div>

    <h2>Summary Statistics</h2>

    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Analysis period</td><td>{daily.index.min().date()} to {daily.index.max().date()}</td></tr>
        <tr><td>Total days</td><td>{total_days:,}</td></tr>
        <tr class="highlight"><td>Total battery savings</td><td class="positive">CHF {total_savings:,.2f}</td></tr>
        <tr><td>Average daily savings</td><td>CHF {avg_daily_savings:.2f}</td></tr>
        <tr><td>Median daily savings</td><td>CHF {median_daily_savings:.2f}</td></tr>
        <tr><td>Days with positive savings</td><td>{pos_days:,} ({100*pos_days/total_days:.1f}%)</td></tr>
        <tr><td>Days with negative savings</td><td class="negative">{neg_days:,} ({100*neg_days/total_days:.1f}%)</td></tr>
    </table>

    <h2>Savings Breakdown</h2>

    <table>
        <tr><th>Component</th><th>Amount (CHF)</th><th>Description</th></tr>
        <tr>
            <td>Avoided grid purchases</td>
            <td class="positive">+{total_avoided_purchase:,.2f}</td>
            <td>Value of battery discharge at purchase tariff</td>
        </tr>
        <tr>
            <td>Foregone feed-in revenue</td>
            <td class="negative">-{total_foregone_feedin:,.2f}</td>
            <td>Revenue lost by storing instead of selling (after {FEEDIN_TAX_RATE*100:.0f}% tax)</td>
        </tr>
        <tr class="highlight">
            <td><strong>Net savings</strong></td>
            <td><strong>CHF {total_savings:,.2f}</strong></td>
            <td>Total benefit from having the battery</td>
        </tr>
    </table>

    <h2>Battery Energy Statistics</h2>

    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total energy charged</td><td>{total_battery_charge:,.0f} kWh</td></tr>
        <tr><td>Total energy discharged</td><td>{total_battery_discharge:,.0f} kWh</td></tr>
        <tr><td>Average round-trip efficiency</td><td>{100*avg_efficiency:.1f}%</td></tr>
        <tr><td>Energy lost to inefficiency</td><td>{total_battery_charge - total_battery_discharge:,.0f} kWh</td></tr>
    </table>

    <h2>Yearly Totals</h2>

    <table>
        <tr><th>Year</th><th>Savings (CHF)</th></tr>
"""

    for year, savings in yearly.items():
        html += f"        <tr><td>{year.year}</td><td>CHF {savings:,.2f}</td></tr>\n"

    html += f"""    </table>

    <h2>Monthly Performance</h2>

    <table>
        <tr><th>Statistic</th><th>Month</th><th>Savings</th></tr>
        <tr><td>Best month</td><td>{best_month.strftime('%Y-%m')}</td><td class="positive">CHF {monthly[best_month]:,.2f}</td></tr>
        <tr><td>Worst month</td><td>{worst_month.strftime('%Y-%m')}</td><td>CHF {monthly[worst_month]:,.2f}</td></tr>
    </table>

    <h2>Visualization</h2>

    <img src="battery_savings_analysis.png" alt="Battery Savings Analysis">

    <div class="methodology">
        <h3>Methodology</h3>
        <p>This analysis compares the actual electricity costs with a hypothetical scenario
        where no battery exists:</p>
        <ul>
            <li><strong>With battery (actual):</strong> Grid costs based on actual imports and exports</li>
            <li><strong>Without battery (hypothetical):</strong> Battery charging energy would have been
            exported to the grid; battery discharging energy would have been imported from the grid</li>
        </ul>
        <p><strong>Battery savings = </strong>
        (Battery discharge × Purchase rate) - (Battery charge × Feed-in rate × 0.70)</p>
        <p>The 0.70 factor accounts for the {FEEDIN_TAX_RATE*100:.0f}% income tax on feed-in revenue.</p>
        <p>Time-varying tariffs (high/low periods) are applied at 15-minute resolution for accuracy.</p>
    </div>

</body>
</html>
"""

    report_path = OUTPUT_DIR / 'battery_savings_report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {report_path}")

    return report_path


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Battery Cost Savings Analysis")
    print("=" * 60)

    # Load data
    energy, tariffs = load_data()

    # Merge tariffs to energy data
    df = merge_tariffs_to_energy(energy, tariffs)

    # Calculate costs
    df = calculate_costs(df)

    # Aggregate to daily
    daily = aggregate_daily(df)

    # Save daily data
    csv_path = OUTPUT_DIR / 'battery_savings_daily.csv'
    daily.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Create visualizations
    create_visualizations(daily)

    # Generate report
    generate_report(daily)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_savings = daily['battery_savings_chf'].sum()
    avg_savings = daily['battery_savings_chf'].mean()
    print(f"Total battery savings: CHF {total_savings:,.2f}")
    print(f"Average daily savings: CHF {avg_savings:.2f}")
    print(f"Analysis period: {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"Total days: {len(daily):,}")
    print("=" * 60)


if __name__ == '__main__':
    main()
