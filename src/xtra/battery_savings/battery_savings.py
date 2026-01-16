#!/usr/bin/env python3
"""
Battery Cost Savings Analysis (xtra)

Analyzes how much cost the battery saves compared to a hypothetical system
without a battery. The analysis considers:
- Time-varying purchase tariffs (Hochtarif/Niedertarif)
- Feed-in tariffs with HKN bonus (not base rate)
- 30% tax on feed-in income
- Battery round-trip efficiency (implicit in the data)

Logic:
- With battery (actual): We pay for grid imports and receive feed-in revenue
- Without battery: Battery charging energy would have been fed to grid,
  and battery discharging energy would have been imported from grid

Savings = battery_discharge × purchase_rate - battery_charge × feedin_rate_hkn × 0.70

Outputs:
- battery_savings_daily.csv - Daily cost savings by tariff period (high/low)
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

# Target savings for projection
TARGET_SAVINGS_CHF = 10000

# HKN bonus rates (CHF/kWh) - added to base feed-in rate
# From CLAUDE.md: HKN bonus is 1.5 Rp until Dec 2024, then 2.5 Rp from Jan 2025
HKN_BONUS = {
    # (start_date, end_date): bonus in CHF/kWh
    (pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-31')): 0.015,  # 1.5 Rp
    (pd.Timestamp('2025-01-01'), pd.Timestamp('2099-12-31')): 0.025,  # 2.5 Rp
}


def get_hkn_bonus(timestamp):
    """Get HKN bonus for a given timestamp."""
    for (start, end), bonus in HKN_BONUS.items():
        if start <= timestamp <= end:
            return bonus
    return 0.015  # Default to 1.5 Rp


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

    # Select relevant tariff columns - use the time-specific purchase rate
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

    # Add HKN bonus to feed-in rate
    print("  Adding HKN bonus to feed-in rates...")
    merged['hkn_bonus_chf_kwh'] = merged.index.map(get_hkn_bonus)
    merged['feedin_rate_hkn_chf_kwh'] = merged['feedin_rate_chf_kwh'] + merged['hkn_bonus_chf_kwh']

    # Check for missing tariff data
    missing = merged['purchase_rate_chf_kwh'].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} rows missing tariff data, will be excluded")
        merged = merged.dropna(subset=['purchase_rate_chf_kwh'])

    # Verify HKN rates
    print("  Feed-in rates with HKN by period:")
    monthly_rates = merged.groupby(merged.index.to_period('M'))['feedin_rate_hkn_chf_kwh'].first()
    for period in ['2023-03', '2023-07', '2024-01', '2024-07', '2025-01']:
        if period in monthly_rates.index.astype(str).values:
            rate = monthly_rates[monthly_rates.index.astype(str) == period].iloc[0]
            print(f"    {period}: {rate:.3f} CHF/kWh ({rate*100:.1f} Rp)")

    print(f"  Merged data: {len(merged):,} rows")
    return merged


def calculate_costs(df):
    """Calculate actual costs and hypothetical costs without battery."""
    print("Calculating costs...")

    # Net feed-in rate after 30% tax (using HKN rate)
    feedin_rate_net = df['feedin_rate_hkn_chf_kwh'] * (1 - FEEDIN_TAX_RATE)

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


def aggregate_daily_by_tariff(df):
    """Aggregate 15-minute data to daily totals, split by tariff period."""
    print("Aggregating to daily data by tariff period...")

    # Add date and tariff type columns
    df['date'] = df.index.date
    df['tariff_type'] = df['is_high_tariff'].map({True: 'high', False: 'low'})

    # Group by date and tariff type
    grouped = df.groupby(['date', 'tariff_type']).agg({
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
        # Tariff rates (should be constant within tariff type for a day)
        'purchase_rate_chf_kwh': 'first',
        'feedin_rate_hkn_chf_kwh': 'first',
    }).reset_index()

    # Convert date to datetime for proper indexing
    grouped['date'] = pd.to_datetime(grouped['date'])

    # Remove incomplete days (first and last)
    min_date = grouped['date'].min() + pd.Timedelta(days=1)
    max_date = grouped['date'].max() - pd.Timedelta(days=1)
    grouped = grouped[(grouped['date'] >= min_date) & (grouped['date'] <= max_date)]

    # Sort by date and tariff type
    grouped = grouped.sort_values(['date', 'tariff_type'])

    print(f"  Daily data by tariff: {len(grouped)} records ({len(grouped)//2} days × 2 tariff types)")
    return grouped


def aggregate_daily_totals(df):
    """Aggregate 15-minute data to daily totals (for visualization)."""
    print("Aggregating to daily totals...")

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

    print(f"  Daily totals: {len(daily)} days")
    return daily


def calculate_projection(daily, daily_by_tariff):
    """
    Project when savings will reach the target amount.

    Uses 2025 energy data but recalculates savings assuming the CURRENT feed-in
    rate of 13 Rp (Apr 2025+), since earlier months had the higher 15.5 Rp rate.

    Savings formula: discharge × purchase_rate - charge × feedin_rate × 0.70
    """
    print(f"Calculating projection to CHF {TARGET_SAVINGS_CHF:,}...")

    # Current feed-in rate (Apr 2025+): 10.5 Rp base + 2.5 Rp HKN = 13 Rp
    CURRENT_FEEDIN_RATE = 0.13  # CHF/kWh

    # Current cumulative savings (actual, with historical rates)
    current_savings = daily['battery_savings_chf'].sum()
    remaining = TARGET_SAVINGS_CHF - current_savings

    if remaining <= 0:
        print(f"  Target already reached!")
        return {
            'current_savings': current_savings,
            'target_savings': TARGET_SAVINGS_CHF,
            'remaining': 0,
            'target_reached': True,
            'target_date': daily.index.max(),
        }

    # Use 2025 data for energy patterns, but recalculate with current tariff
    recent_data = daily[daily.index.year >= 2025].copy()
    if len(recent_data) < 30:
        recent_data = daily.iloc[-365:].copy()
        projection_basis = "last 365 days"
    else:
        projection_basis = "2025 energy patterns"

    # Recalculate savings using current 13 Rp feed-in rate
    # Savings = avoided_purchase - foregone_feedin
    # avoided_purchase stays the same (purchase rates unchanged)
    # foregone_feedin = charge × feedin_rate × 0.70 (after tax)
    recent_data['projected_foregone_feedin'] = (
        recent_data['battery_charging_kwh'] * CURRENT_FEEDIN_RATE * (1 - FEEDIN_TAX_RATE)
    )
    recent_data['projected_savings'] = (
        recent_data['savings_avoided_purchase_chf'] - recent_data['projected_foregone_feedin']
    )

    # Calculate average daily savings at current tariff
    avg_daily_savings = recent_data['projected_savings'].mean()
    daily_std = recent_data['projected_savings'].std()

    # Compare with actual 2025 savings to show the difference
    actual_avg_savings = recent_data['battery_savings_chf'].mean()

    # Calculate tariff breakdown from recent period (recalculated)
    recent_by_tariff = daily_by_tariff[daily_by_tariff['date'].dt.year >= 2025].copy()
    if len(recent_by_tariff) < 30:
        recent_by_tariff = daily_by_tariff.iloc[-730:].copy()

    # Recalculate with current feed-in rate
    recent_by_tariff['projected_foregone_feedin'] = (
        recent_by_tariff['battery_charging_kwh'] * CURRENT_FEEDIN_RATE * (1 - FEEDIN_TAX_RATE)
    )
    recent_by_tariff['projected_savings'] = (
        recent_by_tariff['savings_avoided_purchase_chf'] - recent_by_tariff['projected_foregone_feedin']
    )

    tariff_breakdown = recent_by_tariff.groupby('tariff_type').agg({
        'projected_savings': 'sum',
        'battery_discharging_kwh': 'sum',
        'battery_charging_kwh': 'sum',
    })
    tariff_breakdown = tariff_breakdown.rename(columns={'projected_savings': 'battery_savings_chf'})

    # Days to target
    days_to_target = remaining / avg_daily_savings if avg_daily_savings > 0 else float('inf')
    years_to_target = days_to_target / 365.25

    # Target date
    last_date = daily.index.max()
    target_date = last_date + pd.Timedelta(days=days_to_target)

    # Calculate confidence interval using recent data variability
    savings_rate_se = daily_std / np.sqrt(len(recent_data))

    # Pessimistic and optimistic estimates (±2 SE on savings rate)
    rate_low = max(0.01, avg_daily_savings - 2 * savings_rate_se)
    rate_high = avg_daily_savings + 2 * savings_rate_se

    days_pessimistic = remaining / rate_low if rate_low > 0 else float('inf')
    days_optimistic = remaining / rate_high if rate_high > 0 else float('inf')

    target_date_pessimistic = last_date + pd.Timedelta(days=days_pessimistic)
    target_date_optimistic = last_date + pd.Timedelta(days=days_optimistic)

    projection = {
        'current_savings': current_savings,
        'target_savings': TARGET_SAVINGS_CHF,
        'remaining': remaining,
        'target_reached': False,
        'projection_basis': projection_basis,
        'projection_days': len(recent_data),
        'current_feedin_rate': CURRENT_FEEDIN_RATE,
        'avg_daily_savings': avg_daily_savings,
        'actual_avg_savings': actual_avg_savings,
        'daily_savings_std': daily_std,
        'days_to_target': days_to_target,
        'years_to_target': years_to_target,
        'target_date': target_date,
        'target_date_optimistic': target_date_optimistic,
        'target_date_pessimistic': target_date_pessimistic,
        'tariff_breakdown': tariff_breakdown,
        'high_tariff_pct': tariff_breakdown.loc['high', 'battery_savings_chf'] / tariff_breakdown['battery_savings_chf'].sum() * 100 if 'high' in tariff_breakdown.index else 0,
    }

    print(f"  Projection basis: {projection_basis} ({len(recent_data)} days)")
    print(f"  Current feed-in rate: {CURRENT_FEEDIN_RATE*100:.1f} Rp (Apr 2025+)")
    print(f"  Actual 2025 avg savings: CHF {actual_avg_savings:.2f}/day (mixed rates)")
    print(f"  Projected avg savings:   CHF {avg_daily_savings:.2f}/day (at 13 Rp)")
    print(f"  Current savings: CHF {current_savings:,.2f}")
    print(f"  Remaining to target: CHF {remaining:,.2f}")
    print(f"  Days to target: {days_to_target:,.0f} ({years_to_target:.1f} years)")
    print(f"  Estimated target date: {target_date.strftime('%Y-%m')}")
    print(f"  Range: {target_date_optimistic.strftime('%Y-%m')} to {target_date_pessimistic.strftime('%Y-%m')}")

    return projection


def create_visualizations(daily, projection=None):
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

    # Add projection to target if provided
    if projection and not projection.get('target_reached', False):
        target_date = projection['target_date']
        target_savings = projection['target_savings']

        # Extend the x-axis to include projection
        ax1.axhline(y=target_savings, color='gold', linestyle='--', linewidth=2,
                    label=f'Target: CHF {target_savings:,}')

        # Add projection line
        proj_dates = pd.date_range(daily.index[-1], target_date, freq='M')
        proj_savings = np.linspace(total_savings, target_savings, len(proj_dates))
        ax1.plot(proj_dates, proj_savings, 'g--', linewidth=1.5, alpha=0.7,
                 label='Projection')

        # Add target date annotation
        ax1.annotate(f'Target: {target_date.strftime("%Y-%m")}\n({projection["years_to_target"]:.1f} years)',
                     xy=(target_date, target_savings),
                     xytext=(-100, -30), textcoords='offset points',
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='gold'),
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax1.legend(loc='upper left', fontsize=9)

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

    # Panel 4: Savings by tariff type (monthly)
    ax4 = axes[1, 1]

    # We need to recalculate from 15-min data split by tariff
    # For now, show yearly savings trend
    yearly = daily.resample('Y').agg({
        'battery_savings_chf': 'sum',
        'battery_charging_kwh': 'sum',
        'battery_discharging_kwh': 'sum',
    })
    yearly['savings_per_kwh_discharged'] = yearly['battery_savings_chf'] / yearly['battery_discharging_kwh']

    years = [d.year for d in yearly.index]
    x = range(len(years))

    bars = ax4.bar(x, yearly['battery_savings_chf'], color='green', alpha=0.7)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Annual Savings (CHF)')
    ax4.set_title('Annual Battery Savings')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, yearly['battery_savings_chf'])):
        ax4.annotate(f'CHF {val:.0f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    fig_path = OUTPUT_DIR / 'battery_savings_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return fig_path


def generate_report(daily, daily_by_tariff, projection=None):
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

    # Tariff breakdown
    tariff_summary = daily_by_tariff.groupby('tariff_type').agg({
        'battery_savings_chf': 'sum',
        'savings_avoided_purchase_chf': 'sum',
        'cost_foregone_feedin_chf': 'sum',
        'battery_charging_kwh': 'sum',
        'battery_discharging_kwh': 'sum',
    })

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

    <h2>Savings by Tariff Period</h2>

    <table>
        <tr><th>Tariff</th><th>Savings (CHF)</th><th>Avoided Purchase</th><th>Foregone Feed-in</th><th>Discharge (kWh)</th></tr>
        <tr>
            <td>Hochtarif (high)</td>
            <td>CHF {tariff_summary.loc['high', 'battery_savings_chf']:,.2f}</td>
            <td>CHF {tariff_summary.loc['high', 'savings_avoided_purchase_chf']:,.2f}</td>
            <td>CHF {tariff_summary.loc['high', 'cost_foregone_feedin_chf']:,.2f}</td>
            <td>{tariff_summary.loc['high', 'battery_discharging_kwh']:,.0f}</td>
        </tr>
        <tr>
            <td>Niedertarif (low)</td>
            <td>CHF {tariff_summary.loc['low', 'battery_savings_chf']:,.2f}</td>
            <td>CHF {tariff_summary.loc['low', 'savings_avoided_purchase_chf']:,.2f}</td>
            <td>CHF {tariff_summary.loc['low', 'cost_foregone_feedin_chf']:,.2f}</td>
            <td>{tariff_summary.loc['low', 'battery_discharging_kwh']:,.0f}</td>
        </tr>
        <tr class="highlight">
            <td><strong>Total</strong></td>
            <td><strong>CHF {total_savings:,.2f}</strong></td>
            <td>CHF {total_avoided_purchase:,.2f}</td>
            <td>CHF {total_foregone_feedin:,.2f}</td>
            <td>{total_battery_discharge:,.0f}</td>
        </tr>
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
            <td>Revenue lost by storing instead of selling (after {FEEDIN_TAX_RATE*100:.0f}% tax, using HKN rate)</td>
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
"""

    # Add projection section if available
    if projection and not projection.get('target_reached', False):
        html += f"""
    <h2>Projection to CHF {projection['target_savings']:,}</h2>

    <div class="summary-box">
        <h3>Estimated Target Date: {projection['target_date'].strftime('%B %Y')}</h3>
        <p>Based on {projection['projection_basis']} recalculated at the current feed-in rate of
        <strong>{projection['current_feedin_rate']*100:.1f} Rp</strong>, the battery is expected to reach
        <strong>CHF {projection['target_savings']:,}</strong> in cumulative savings by
        <strong>{projection['target_date'].strftime('%B %Y')}</strong>
        ({projection['years_to_target']:.1f} years from now).</p>
        <p>Range: {projection['target_date_optimistic'].strftime('%B %Y')} (optimistic)
        to {projection['target_date_pessimistic'].strftime('%B %Y')} (pessimistic)</p>
    </div>

    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Current cumulative savings</td><td>CHF {projection['current_savings']:,.2f}</td></tr>
        <tr><td>Remaining to target</td><td>CHF {projection['remaining']:,.2f}</td></tr>
        <tr><td>Current feed-in rate (Apr 2025+)</td><td>{projection['current_feedin_rate']*100:.1f} Rp</td></tr>
        <tr><td>Actual 2025 avg savings (mixed rates)</td><td>CHF {projection['actual_avg_savings']:.2f}/day</td></tr>
        <tr class="highlight"><td>Projected avg savings (at {projection['current_feedin_rate']*100:.0f} Rp)</td><td>CHF {projection['avg_daily_savings']:.2f}/day</td></tr>
        <tr><td>Days to target</td><td>{projection['days_to_target']:,.0f}</td></tr>
        <tr class="highlight"><td>Years to target</td><td>{projection['years_to_target']:.1f}</td></tr>
        <tr><td>Estimated target date</td><td>{projection['target_date'].strftime('%Y-%m')}</td></tr>
        <tr><td>High tariff contribution</td><td>{projection['high_tariff_pct']:.1f}%</td></tr>
    </table>

    <div class="methodology">
        <h4>Projection Methodology</h4>
        <ul>
            <li>Uses 2025 energy patterns (charging/discharging volumes)</li>
            <li><strong>Recalculates savings at the current 13 Rp feed-in rate</strong> (not the mixed 2025 rates)</li>
            <li>Feed-in changed from 15.5 Rp (Jan-Mar 2025) to 13 Rp (Apr 2025+)</li>
            <li>Purchase rates assumed constant (~32.6 Rp average)</li>
            <li>Range estimates use ±2 standard errors on the daily savings rate</li>
        </ul>
    </div>
"""

    html += f"""
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
        (Battery discharge × Purchase rate) - (Battery charge × Feed-in rate with HKN × 0.70)</p>
        <p>The 0.70 factor accounts for the {FEEDIN_TAX_RATE*100:.0f}% income tax on feed-in revenue.</p>
        <p><strong>Feed-in rates use HKN bonus:</strong> Base rate + 1.5 Rp (2023-2024) or + 2.5 Rp (2025+)</p>
        <p>Time-varying tariffs (Hochtarif/Niedertarif) are applied at 15-minute resolution.</p>
        <p><strong>Output format:</strong> Daily CSV contains separate rows for high and low tariff periods.</p>
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

    # Merge tariffs to energy data (adds HKN bonus)
    df = merge_tariffs_to_energy(energy, tariffs)

    # Calculate costs
    df = calculate_costs(df)

    # Aggregate to daily by tariff type (for CSV output)
    daily_by_tariff = aggregate_daily_by_tariff(df)

    # Aggregate to daily totals (for visualization)
    daily = aggregate_daily_totals(df)

    # Save daily data by tariff
    csv_path = OUTPUT_DIR / 'battery_savings_daily.csv'
    daily_by_tariff.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Calculate projection to CHF 10,000
    projection = calculate_projection(daily, daily_by_tariff)

    # Create visualizations
    create_visualizations(daily, projection)

    # Generate report
    generate_report(daily, daily_by_tariff, projection)

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

    # Tariff breakdown
    tariff_summary = daily_by_tariff.groupby('tariff_type')['battery_savings_chf'].sum()
    print(f"\nBy tariff:")
    print(f"  Hochtarif:   CHF {tariff_summary['high']:,.2f}")
    print(f"  Niedertarif: CHF {tariff_summary['low']:,.2f}")

    # Projection
    if not projection.get('target_reached', False):
        print(f"\nProjection to CHF {projection['target_savings']:,}:")
        print(f"  Avg daily savings (recent): CHF {projection['avg_daily_savings']:.2f}")
        print(f"  Years to target: {projection['years_to_target']:.1f}")
        print(f"  Target date: {projection['target_date'].strftime('%Y-%m')}")
        print(f"  Range: {projection['target_date_optimistic'].strftime('%Y-%m')} to {projection['target_date_pessimistic'].strftime('%Y-%m')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
