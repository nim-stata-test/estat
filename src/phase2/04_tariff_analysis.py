#!/usr/bin/env python3
"""
Phase 2, Step 4: Electricity Tariff Analysis

Analyzes electricity tariffs for cost modeling visualization:
- Purchase tariffs (high/low rates over time)
- Feed-in tariffs (with HKN only, base rates excluded)
- Tariff time window distribution
- Cost implications for optimization strategies

Note: Only feed-in tariffs WITH HKN (Herkunftsnachweis) are included.
Base-only feed-in rates are excluded as the installation uses HKN.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(exist_ok=True)

# Figure style
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (14, 10)


def load_tariff_data():
    """Load tariff data from Phase 1 preprocessing."""
    print("Loading tariff data...")

    data = {}

    # Tariff schedule
    schedule_path = PROCESSED_DIR / 'tariff_schedule.csv'
    if schedule_path.exists():
        data['schedule'] = pd.read_csv(schedule_path)
        data['schedule']['valid_from'] = pd.to_datetime(data['schedule']['valid_from'])
        data['schedule']['valid_to'] = pd.to_datetime(data['schedule']['valid_to'])
        print(f"  Schedule: {len(data['schedule'])} entries")
    else:
        print(f"  WARNING: {schedule_path} not found")
        data['schedule'] = pd.DataFrame()

    # Hourly tariff flags
    flags_path = PROCESSED_DIR / 'tariff_flags_hourly.parquet'
    if flags_path.exists():
        data['flags'] = pd.read_parquet(flags_path)
        print(f"  Flags: {len(data['flags'])} hourly records")
    else:
        print(f"  WARNING: {flags_path} not found")
        data['flags'] = pd.DataFrame()

    # Hourly tariff series
    series_path = PROCESSED_DIR / 'tariff_series_hourly.parquet'
    if series_path.exists():
        data['series'] = pd.read_parquet(series_path)
        print(f"  Series: {len(data['series'])} hourly records")
    else:
        print(f"  WARNING: {series_path} not found")
        data['series'] = pd.DataFrame()

    return data


def filter_hkn_tariffs(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Filter tariff schedule to only include relevant tariffs.

    For feed-in: Only HKN tariffs (total_standard), excluding base-only rates.
    For purchase: All rates included.

    Note: Base-only feed-in rates are excluded because the installation
    participates in the HKN (Herkunftsnachweis) program, receiving the
    additional HKN bonus on top of the base rate.
    """
    if schedule.empty:
        return schedule

    # Keep all purchase tariffs
    purchase = schedule[schedule['tariff_type'] == 'purchase'].copy()

    # For feed-in, only keep total_standard (includes HKN) and minimum_guarantee
    # Exclude 'base' rate_type as it's not applicable with HKN participation
    feedin = schedule[schedule['tariff_type'] == 'feedin'].copy()
    feedin = feedin[feedin['rate_type'].isin(['total_standard', 'minimum_guarantee'])]

    filtered = pd.concat([purchase, feedin], ignore_index=True)

    print(f"\nFiltered tariffs (HKN-only for feed-in):")
    print(f"  Purchase: {len(purchase)} entries")
    print(f"  Feed-in (with HKN): {len(feedin)} entries")
    print(f"  Excluded: {len(schedule) - len(filtered)} base-only entries")

    return filtered


def create_tariff_timeline_figure(schedule: pd.DataFrame) -> plt.Figure:
    """Create combined timeline visualization of purchase and feed-in tariffs."""
    fig, ax = plt.subplots(figsize=(14, 6))

    labels_shown = set()

    # Purchase tariffs (solid lines)
    purchase = schedule[schedule['tariff_type'] == 'purchase']
    # Note: single tariff not used (energy-only component, excludes network charges)
    purchase_colors = {'high': '#dc2626', 'low': '#16a34a', 'average_estimate': '#6b7280'}

    for _, row in purchase.iterrows():
        rate_type = row['rate_type']
        color = purchase_colors.get(rate_type, '#6b7280')
        label = f"Purchase: {rate_type.replace('_', ' ').title()}"

        show_label = label not in labels_shown
        if show_label:
            labels_shown.add(label)

        ax.hlines(y=row['rate_rp_kwh'], xmin=row['valid_from'], xmax=row['valid_to'],
                  colors=color, linewidth=3, linestyle='-', label=label if show_label else '')

    # Feed-in tariffs (dashed lines)
    feedin = schedule[schedule['tariff_type'] == 'feedin']
    feedin_colors = {'total_standard': '#f59e0b', 'minimum_guarantee': '#ef4444'}

    for _, row in feedin.iterrows():
        rate_type = row['rate_type']
        color = feedin_colors.get(rate_type, '#6b7280')
        label = 'Feed-in: With HKN' if rate_type == 'total_standard' else 'Feed-in: Min Guarantee'

        show_label = label not in labels_shown
        if show_label:
            labels_shown.add(label)

        ax.hlines(y=row['rate_rp_kwh'], xmin=row['valid_from'], xmax=row['valid_to'],
                  colors=color, linewidth=3, linestyle='--', label=label if show_label else '')

    ax.set_ylabel('Rate (Rp/kWh)')
    ax.set_xlabel('Date')
    ax.set_title('Electricity Tariffs: Purchase (solid) vs Feed-in (dashed)')
    ax.legend(loc='upper right', ncol=2)

    # Set y-axis to show full range
    all_rates = schedule['rate_rp_kwh']
    ax.set_ylim(0, max(all_rates.max() * 1.15, 40))
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


def create_tariff_windows_figure(flags: pd.DataFrame) -> plt.Figure:
    """Create visualization of tariff time windows."""
    if flags.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No tariff flag data available', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    # 1. Weekly distribution (heatmap-style)
    ax1 = axes[0, 0]
    flags_copy = flags.copy()
    flags_copy['hour'] = flags_copy.index.hour
    flags_copy['dayofweek'] = flags_copy.index.dayofweek

    # Pivot for heatmap
    pivot = flags_copy.groupby(['dayofweek', 'hour'])['is_high_tariff'].mean().unstack()

    im = ax1.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax1.set_xticks(range(0, 24, 3))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Day of Week')
    ax1.set_title('High Tariff Probability by Day/Hour')
    plt.colorbar(im, ax=ax1, label='P(High Tariff)')

    # 2. Hourly profile (average)
    ax2 = axes[0, 1]
    hourly_avg = flags_copy.groupby('hour')['is_high_tariff'].mean()
    colors = ['#dc2626' if v > 0.5 else '#16a34a' for v in hourly_avg]
    ax2.bar(hourly_avg.index, hourly_avg.values, color=colors, alpha=0.8)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Probability of High Tariff')
    ax2.set_title('Average High Tariff Probability by Hour')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_ylim(0, 1)

    # 3. Monthly distribution
    ax3 = axes[1, 0]
    flags_copy['month'] = flags_copy.index.month
    monthly = flags_copy.groupby('month').agg({
        'is_high_tariff': 'sum',
        'is_low_tariff': 'sum'
    })
    monthly.plot(kind='bar', ax=ax3, color=['#dc2626', '#16a34a'], alpha=0.8)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Hours')
    ax3.set_title('High vs Low Tariff Hours by Month')
    ax3.legend(['High Tariff', 'Low Tariff'])
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_hours = len(flags)
    high_hours = flags['is_high_tariff'].sum()
    low_hours = flags['is_low_tariff'].sum()
    holiday_hours = flags['is_holiday'].sum() if 'is_holiday' in flags.columns else 0

    summary_text = f"""
    Tariff Time Window Summary
    ══════════════════════════════════

    Total Hours Analyzed: {total_hours:,}

    High Tariff (Hochtarif):
      • Hours: {high_hours:,} ({100*high_hours/total_hours:.1f}%)
      • Mon-Fri 06:00-21:00
      • Sat 06:00-12:00

    Low Tariff (Niedertarif):
      • Hours: {low_hours:,} ({100*low_hours/total_hours:.1f}%)
      • Mon-Fri 21:00-06:00
      • Sat 12:00 - Mon 06:00
      • Federal holidays (all day)

    Holiday Hours: {holiday_hours:,}

    ══════════════════════════════════
    Note: Low tariff is more favorable
    for grid import (cheaper purchase).
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#e2e8f0'))

    plt.tight_layout()
    return fig


def create_cost_implications_figure(schedule: pd.DataFrame) -> plt.Figure:
    """Create figure showing cost implications of tariff structure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Feed-in tariff trend
    ax1 = axes[0]
    feedin = schedule[(schedule['tariff_type'] == 'feedin') &
                      (schedule['rate_type'] == 'total_standard')].copy()
    feedin = feedin.sort_values('valid_from')

    if not feedin.empty:
        # Create step plot
        dates = []
        rates = []
        for _, row in feedin.iterrows():
            dates.extend([row['valid_from'], row['valid_to']])
            rates.extend([row['rate_rp_kwh'], row['rate_rp_kwh']])

        ax1.fill_between(dates, rates, alpha=0.3, color='#f59e0b', step='pre')
        ax1.step(dates, rates, where='pre', color='#f59e0b', linewidth=2, label='Feed-in (with HKN)')

        # Add minimum guarantee line
        ax1.axhline(9.0, color='#ef4444', linestyle='--', linewidth=2, label='Minimum Guarantee')

        ax1.set_ylabel('Feed-in Rate (Rp/kWh)')
        ax1.set_xlabel('Date')
        ax1.set_title('Feed-in Tariff Trend (2023-2025)')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Annotate decline
        if len(feedin) >= 2:
            first_rate = feedin.iloc[0]['rate_rp_kwh']
            last_rate = feedin.iloc[-1]['rate_rp_kwh']
            decline_pct = (first_rate - last_rate) / first_rate * 100
            ax1.annotate(f'{decline_pct:.0f}% decline',
                         xy=(feedin.iloc[-1]['valid_from'], last_rate),
                         xytext=(feedin.iloc[-1]['valid_from'], last_rate + 3),
                         fontsize=10, ha='center',
                         arrowprops=dict(arrowstyle='->', color='gray'))

    # 2. Cost comparison scenarios
    ax2 = axes[1]

    # Hypothetical scenarios (1000 kWh)
    scenarios = {
        'Grid Import\n(High Tariff)': 32.8,  # 2024 estimate
        'Grid Import\n(Low Tariff)': 28.0,   # Estimated ~15% lower
        'Feed-in\n(Jan 2023)': 21.5,         # Peak with HKN
        'Feed-in\n(Apr 2025)': 13.0,         # Current with HKN
        'Feed-in\n(Minimum)': 9.0,           # Guarantee
    }

    colors = ['#dc2626', '#16a34a', '#f59e0b', '#f59e0b', '#ef4444']
    bars = ax2.bar(scenarios.keys(), scenarios.values(), color=colors, alpha=0.8)
    ax2.set_ylabel('Rate (Rp/kWh)')
    ax2.set_title('Tariff Rate Comparison')
    ax2.axhline(0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, scenarios.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def generate_tariff_tables(schedule: pd.DataFrame) -> dict:
    """Generate HTML tables for tariff data."""
    tables = {}

    # Purchase tariffs table
    purchase = schedule[schedule['tariff_type'] == 'purchase'].copy()
    purchase = purchase.sort_values('valid_from')
    purchase['period'] = purchase['valid_from'].dt.strftime('%Y-%m-%d') + ' to ' + purchase['valid_to'].dt.strftime('%Y-%m-%d')
    purchase_display = purchase[['period', 'rate_type', 'rate_rp_kwh', 'notes']]
    tables['purchase'] = purchase_display.to_html(index=False, classes='data-table')

    # Feed-in tariffs table (HKN only)
    feedin = schedule[schedule['tariff_type'] == 'feedin'].copy()
    feedin = feedin.sort_values('valid_from')
    feedin['period'] = feedin['valid_from'].dt.strftime('%Y-%m-%d') + ' to ' + feedin['valid_to'].dt.strftime('%Y-%m-%d')
    feedin_display = feedin[['period', 'rate_type', 'rate_rp_kwh', 'notes']]
    tables['feedin'] = feedin_display.to_html(index=False, classes='data-table')

    return tables


def generate_report_section(schedule: pd.DataFrame, flags: pd.DataFrame) -> str:
    """Generate HTML report section for tariffs."""
    tables = generate_tariff_tables(schedule)

    # Calculate summary stats
    total_hours = len(flags) if not flags.empty else 0
    high_hours = int(flags['is_high_tariff'].sum()) if not flags.empty else 0
    low_hours = int(flags['is_low_tariff'].sum()) if not flags.empty else 0

    feedin_hkn = schedule[(schedule['tariff_type'] == 'feedin') &
                          (schedule['rate_type'] == 'total_standard')]
    if not feedin_hkn.empty:
        feedin_max = feedin_hkn['rate_rp_kwh'].max()
        feedin_min = feedin_hkn['rate_rp_kwh'].min()
        feedin_decline = (feedin_max - feedin_min) / feedin_max * 100
    else:
        feedin_max = feedin_min = feedin_decline = 0

    html = f"""
    <h2 id="tariffs">10. Electricity Tariffs</h2>

    <div class="card">
        <h4>Data Source Note</h4>
        <p><strong>Important:</strong> Only feed-in tariffs <em>with HKN</em> (Herkunftsnachweis) are shown.
        Base-only feed-in rates are excluded because this installation participates in the HKN program,
        receiving the additional HKN bonus (1.5-4.5 Rp/kWh) on top of the base rate.</p>
    </div>

    <div class="grid">
        <div class="stat-box">
            <div class="value">{feedin_max:.1f}</div>
            <div class="label">Rp/kWh Peak Feed-in (HKN)</div>
        </div>
        <div class="stat-box">
            <div class="value">{feedin_min:.1f}</div>
            <div class="label">Rp/kWh Current Feed-in (HKN)</div>
        </div>
        <div class="stat-box">
            <div class="value">{feedin_decline:.0f}%</div>
            <div class="label">Feed-in Rate Decline</div>
        </div>
        <div class="stat-box">
            <div class="value">{100*high_hours/total_hours:.0f}%</div>
            <div class="label">High Tariff Hours</div>
        </div>
    </div>

    <h3>Tariff Time Windows</h3>
    <div class="card">
        <table>
            <tr><th>Tariff</th><th>Time Windows</th><th>Hours</th></tr>
            <tr>
                <td><strong>High Tariff</strong> (Hochtarif)</td>
                <td>Mon-Fri 06:00-21:00, Sat 06:00-12:00</td>
                <td>{high_hours:,} ({100*high_hours/total_hours:.1f}%)</td>
            </tr>
            <tr>
                <td><strong>Low Tariff</strong> (Niedertarif)</td>
                <td>Mon-Fri 21:00-06:00, Sat 12:00-Mon 06:00, Holidays</td>
                <td>{low_hours:,} ({100*low_hours/total_hours:.1f}%)</td>
            </tr>
        </table>
    </div>

    <h3>Purchase Tariffs (Grid Import)</h3>
    <div class="card">
        {tables['purchase']}
    </div>

    <h3>Feed-in Tariffs (Grid Export) - With HKN Only</h3>
    <div class="card">
        {tables['feedin']}
        <p class="note" style="margin-top: 1rem; font-size: 0.85rem; color: #64748b;">
        HKN = Herkunftsnachweis (certificate of origin). Minimum guarantee of 9 Rp/kWh applies through 2028.
        </p>
    </div>

    <h3>Key Insights</h3>
    <div class="card">
        <ul>
            <li><strong>Feed-in rate decline:</strong> From {feedin_max:.1f} Rp/kWh (Jan 2023) to {feedin_min:.1f} Rp/kWh (Apr 2025) - a {feedin_decline:.0f}% reduction</li>
            <li><strong>Minimum guarantee:</strong> 9 Rp/kWh floor until 2028 provides investment security</li>
            <li><strong>Grid import cost:</strong> ~32-33 Rp/kWh (2024-2025), roughly 2.5x the current feed-in rate</li>
            <li><strong>Self-consumption value:</strong> Avoiding grid import saves ~20 Rp/kWh more than feeding in</li>
            <li><strong>Low tariff opportunity:</strong> {100*low_hours/total_hours:.0f}% of hours are low-tariff - shifting consumption here reduces costs</li>
        </ul>
    </div>

    <h3>Tariff Timeline</h3>
    <div class="figure">
        <img src="fig14_tariff_timeline.png" alt="Tariff Timeline" style="max-width: 100%;">
        <p class="caption"><strong>Figure 14:</strong> Purchase and feed-in tariff rates over time (2023-2025).
        Top panel shows grid import rates, bottom panel shows feed-in rates with HKN (excluding base-only rates).</p>
    </div>

    <h3>Tariff Time Windows Distribution</h3>
    <div class="figure">
        <img src="fig15_tariff_windows.png" alt="Tariff Windows" style="max-width: 100%;">
        <p class="caption"><strong>Figure 15:</strong> Distribution of high/low tariff periods.
        Shows weekly heatmap, hourly probability, monthly distribution, and summary statistics.</p>
    </div>

    <h3>Cost Comparison</h3>
    <div class="figure">
        <img src="fig16_tariff_costs.png" alt="Tariff Costs" style="max-width: 100%;">
        <p class="caption"><strong>Figure 16:</strong> Tariff rate comparison showing feed-in trend (left)
        and rate comparison across different scenarios (right). Self-consumption value = import rate - feed-in rate.</p>
    </div>
    """

    return html


def main():
    """Main tariff analysis function."""
    print("=" * 60)
    print("Phase 2, Step 4: Electricity Tariff Analysis")
    print("=" * 60)

    # Load data
    data = load_tariff_data()

    if data['schedule'].empty:
        print("\nERROR: No tariff schedule data found. Run Phase 1 Step 4 first.")
        return 1

    # Filter to HKN-only feed-in tariffs
    schedule_filtered = filter_hkn_tariffs(data['schedule'])

    # Create figures
    print("\nGenerating figures...")

    # Figure 1: Tariff timeline
    print("  Creating tariff timeline figure...")
    fig1 = create_tariff_timeline_figure(schedule_filtered)
    fig1.savefig(OUTPUT_DIR / 'fig14_tariff_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Tariff windows
    print("  Creating tariff windows figure...")
    fig2 = create_tariff_windows_figure(data['flags'])
    fig2.savefig(OUTPUT_DIR / 'fig15_tariff_windows.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Figure 3: Cost implications
    print("  Creating cost implications figure...")
    fig3 = create_cost_implications_figure(schedule_filtered)
    fig3.savefig(OUTPUT_DIR / 'fig16_tariff_costs.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # Generate report section
    print("\nGenerating report section...")
    report_section = generate_report_section(schedule_filtered, data['flags'])
    report_path = OUTPUT_DIR / 'tariff_report_section.html'
    report_path.write_text(report_section)
    print(f"  Saved: {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Tariff Analysis Complete")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - fig14_tariff_timeline.png")
    print("  - fig15_tariff_windows.png")
    print("  - fig16_tariff_costs.png")
    print("  - tariff_report_section.html")

    return 0


if __name__ == "__main__":
    exit(main())
