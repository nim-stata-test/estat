#!/usr/bin/env python3
"""
Phase 3, Step 3: Energy System Model

Models the solar/battery system:
- PV generation prediction from historical patterns
- Battery efficiency and state-of-charge dynamics
- Grid interaction patterns
- Self-sufficiency optimization potential

Key inputs from Phase 2:
- Battery efficiency: ~85% pre-event, ~75-80% post-event (degraded)
- Daily PV generation: mean 55.9 kWh
- Self-sufficiency: 44%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load energy balance data for system modeling."""
    print("Loading energy data...")

    # Load 15-minute energy data (from data/daily/)
    energy_15min = pd.read_parquet(PROCESSED_DIR / 'energy_balance_15min.parquet')
    energy_15min.index = pd.to_datetime(energy_15min.index)

    print(f"  15-min data: {len(energy_15min):,} rows ({energy_15min.index.min().date()} to {energy_15min.index.max().date()})")

    return energy_15min


def analyze_pv_patterns(energy_15min: pd.DataFrame) -> dict:
    """
    Analyze PV generation patterns for prediction.

    Models:
    - Seasonal patterns (monthly averages)
    - Daily patterns (hourly profiles)
    - Year-over-year trends
    """
    print("\nAnalyzing PV generation patterns...")

    results = {}

    # Column name
    pv_col = 'pv_generation_kwh'
    if pv_col not in energy_15min.columns:
        print(f"  Warning: {pv_col} not found")
        return results

    # Daily totals
    daily_pv = energy_15min[pv_col].resample('D').sum()
    daily_pv = daily_pv[daily_pv > 0]  # Exclude zeros

    results['daily_stats'] = {
        'mean': daily_pv.mean(),
        'std': daily_pv.std(),
        'max': daily_pv.max(),
        'min': daily_pv[daily_pv > 1].min()  # Min of production days
    }

    print(f"  Daily PV generation:")
    print(f"    Mean: {daily_pv.mean():.1f} kWh")
    print(f"    Max: {daily_pv.max():.1f} kWh")
    print(f"    Std: {daily_pv.std():.1f} kWh")

    # Monthly patterns
    monthly_pv = daily_pv.resample('ME').mean()
    monthly_by_month = daily_pv.groupby(daily_pv.index.month).mean()

    results['monthly_pattern'] = monthly_by_month.to_dict()

    print(f"  Monthly averages (kWh/day):")
    for month, value in monthly_by_month.items():
        print(f"    Month {month}: {value:.1f}")

    # Hourly profile
    hourly_profile = energy_15min[pv_col].groupby(energy_15min.index.hour).mean() * 4  # Convert to hourly rate
    results['hourly_profile'] = hourly_profile.to_dict()

    # Peak hours
    peak_hours = hourly_profile[hourly_profile > hourly_profile.max() * 0.5].index.tolist()
    results['peak_hours'] = peak_hours
    print(f"  Peak generation hours: {peak_hours[0]:02d}:00 - {peak_hours[-1]:02d}:00")

    # Year-over-year trend
    yearly_pv = daily_pv.resample('YE').sum()
    if len(yearly_pv) >= 2:
        years = range(len(yearly_pv))
        slope, intercept, r_value, _, _ = stats.linregress(years, yearly_pv.values)
        results['yearly_trend'] = {
            'slope': slope,
            'r2': r_value**2,
            'latest_year': yearly_pv.values[-1]
        }
        print(f"  Yearly trend: {slope:.0f} kWh/year (R²={r_value**2:.2f})")

    return results


def analyze_battery_dynamics(energy_15min: pd.DataFrame) -> dict:
    """
    Analyze battery charging/discharging patterns and efficiency.

    Key metrics:
    - Round-trip efficiency
    - Daily charge/discharge cycles
    - Self-consumption contribution
    """
    print("\nAnalyzing battery dynamics...")

    results = {}

    charge_col = 'battery_charging_kwh'
    discharge_col = 'battery_discharging_kwh'

    if charge_col not in energy_15min.columns or discharge_col not in energy_15min.columns:
        print("  Battery columns not found")
        return results

    # Daily totals
    daily_charge = energy_15min[charge_col].resample('D').sum()
    daily_discharge = energy_15min[discharge_col].resample('D').sum()

    # Filter days with activity
    mask = (daily_charge > 0.1) & (daily_discharge > 0.1)
    active_charge = daily_charge[mask]
    active_discharge = daily_discharge[mask]

    results['daily_charge'] = {
        'mean': active_charge.mean(),
        'max': active_charge.max()
    }

    results['daily_discharge'] = {
        'mean': active_discharge.mean(),
        'max': active_discharge.max()
    }

    print(f"  Daily charging: mean {active_charge.mean():.1f} kWh, max {active_charge.max():.1f} kWh")
    print(f"  Daily discharge: mean {active_discharge.mean():.1f} kWh, max {active_discharge.max():.1f} kWh")

    # Round-trip efficiency by month
    monthly_charge = daily_charge.resample('ME').sum()
    monthly_discharge = daily_discharge.resample('ME').sum()

    monthly_efficiency = monthly_discharge / monthly_charge
    monthly_efficiency = monthly_efficiency[(monthly_efficiency > 0.5) & (monthly_efficiency < 1.1)]

    results['monthly_efficiency'] = monthly_efficiency.to_dict()

    # Overall efficiency
    total_charge = daily_charge.sum()
    total_discharge = daily_discharge.sum()
    overall_efficiency = total_discharge / total_charge if total_charge > 0 else 0

    results['overall_efficiency'] = overall_efficiency
    print(f"  Overall round-trip efficiency: {100*overall_efficiency:.1f}%")

    # Charging/discharging patterns by hour
    hourly_charge = energy_15min[charge_col].groupby(energy_15min.index.hour).mean() * 4
    hourly_discharge = energy_15min[discharge_col].groupby(energy_15min.index.hour).mean() * 4

    results['hourly_charge_profile'] = hourly_charge.to_dict()
    results['hourly_discharge_profile'] = hourly_discharge.to_dict()

    # Identify charging hours (solar) vs discharging hours (evening/night)
    charge_hours = hourly_charge[hourly_charge > hourly_charge.max() * 0.3].index.tolist()
    discharge_hours = hourly_discharge[hourly_discharge > hourly_discharge.max() * 0.3].index.tolist()

    print(f"  Primary charging hours: {charge_hours}")
    print(f"  Primary discharge hours: {discharge_hours}")

    return results


def analyze_grid_interaction(energy_15min: pd.DataFrame) -> dict:
    """
    Analyze grid import/export patterns.

    Key metrics:
    - Import/export balance
    - Self-sufficiency ratio
    - Times of grid dependency
    """
    print("\nAnalyzing grid interaction patterns...")

    results = {}

    grid_import_col = 'external_supply_kwh'  # Grid import
    grid_export_col = 'grid_feedin_kwh'      # Grid export
    consumption_col = 'total_consumption_kwh'
    direct_col = 'direct_consumption_kwh'    # Direct from PV

    # Check columns
    cols = [grid_import_col, grid_export_col, consumption_col, direct_col]
    available = [c for c in cols if c in energy_15min.columns]

    if len(available) < 3:
        print(f"  Missing columns: {set(cols) - set(available)}")
        return results

    # Daily totals
    daily_import = energy_15min[grid_import_col].resample('D').sum()
    daily_export = energy_15min[grid_export_col].resample('D').sum()
    daily_consumption = energy_15min[consumption_col].resample('D').sum()

    if direct_col in energy_15min.columns:
        daily_direct = energy_15min[direct_col].resample('D').sum()
    else:
        daily_direct = daily_consumption - daily_import

    results['daily_stats'] = {
        'import_mean': daily_import.mean(),
        'export_mean': daily_export.mean(),
        'consumption_mean': daily_consumption.mean(),
        'direct_mean': daily_direct.mean()
    }

    print(f"  Daily averages:")
    print(f"    Grid import: {daily_import.mean():.1f} kWh")
    print(f"    Grid export: {daily_export.mean():.1f} kWh")
    print(f"    Consumption: {daily_consumption.mean():.1f} kWh")
    print(f"    Direct solar: {daily_direct.mean():.1f} kWh")

    # Self-sufficiency
    # = (consumption - grid_import) / consumption
    # = 1 - grid_import / consumption
    self_sufficiency = 1 - (daily_import / daily_consumption)
    self_sufficiency = self_sufficiency[(self_sufficiency >= 0) & (self_sufficiency <= 1)]

    results['self_sufficiency'] = {
        'mean': self_sufficiency.mean(),
        'std': self_sufficiency.std()
    }

    print(f"  Self-sufficiency: {100*self_sufficiency.mean():.1f}% ± {100*self_sufficiency.std():.1f}%")

    # Hourly grid dependency
    hourly_import = energy_15min[grid_import_col].groupby(energy_15min.index.hour).mean() * 4
    hourly_export = energy_15min[grid_export_col].groupby(energy_15min.index.hour).mean() * 4

    results['hourly_import'] = hourly_import.to_dict()
    results['hourly_export'] = hourly_export.to_dict()

    # Peak grid dependency hours
    import_hours = hourly_import[hourly_import > hourly_import.mean()].index.tolist()
    print(f"  High grid import hours: {import_hours}")

    # Monthly patterns
    monthly_import = daily_import.resample('ME').sum()
    monthly_export = daily_export.resample('ME').sum()
    monthly_net = monthly_export - monthly_import

    results['monthly_net'] = monthly_net.to_dict()

    net_producer_months = (monthly_net > 0).sum()
    print(f"  Net producer months: {net_producer_months}/{len(monthly_net)}")

    return results


def model_self_sufficiency_potential(energy_15min: pd.DataFrame, pv_results: dict,
                                     battery_results: dict) -> dict:
    """
    Model potential self-sufficiency improvements.

    Scenarios:
    - Baseline (current)
    - With load shifting (heating to solar hours)
    - With increased battery capacity
    """
    print("\nModeling self-sufficiency potential...")

    results = {}

    pv_col = 'pv_generation_kwh'
    consumption_col = 'total_consumption_kwh'
    battery_charge_col = 'battery_charging_kwh'
    battery_discharge_col = 'battery_discharging_kwh'
    grid_import_col = 'external_supply_kwh'

    # Calculate hourly averages
    hourly_pv = energy_15min[pv_col].groupby(energy_15min.index.hour).mean() * 4
    hourly_consumption = energy_15min[consumption_col].groupby(energy_15min.index.hour).mean() * 4
    hourly_import = energy_15min[grid_import_col].groupby(energy_15min.index.hour).mean() * 4

    # Current self-sufficiency by hour
    hourly_self_sufficiency = 1 - (hourly_import / hourly_consumption)
    hourly_self_sufficiency = hourly_self_sufficiency.clip(0, 1)

    results['hourly_self_sufficiency'] = hourly_self_sufficiency.to_dict()

    # Overall baseline
    total_consumption = energy_15min[consumption_col].sum()
    total_import = energy_15min[grid_import_col].sum()
    baseline_ss = 1 - (total_import / total_consumption)

    results['baseline_self_sufficiency'] = baseline_ss
    print(f"  Baseline self-sufficiency: {100*baseline_ss:.1f}%")

    # Scenario 1: Shift 20% of non-solar consumption to solar hours
    # Identify solar hours (PV > 0.1 kWh/15min average)
    solar_hours = hourly_pv[hourly_pv > 0.5].index.tolist()
    non_solar_hours = [h for h in range(24) if h not in solar_hours]

    # Assume 20% of evening/night consumption could be shifted
    shiftable_load = hourly_consumption[non_solar_hours].sum() * 0.20
    shifted_import_reduction = shiftable_load * 0.7  # 70% of shifted load avoids grid

    scenario1_import = total_import - shifted_import_reduction * len(energy_15min.index.normalize().unique())
    scenario1_ss = 1 - (scenario1_import / total_consumption)

    results['scenario1_load_shift'] = {
        'self_sufficiency': max(scenario1_ss, baseline_ss),
        'improvement': max(scenario1_ss - baseline_ss, 0)
    }
    print(f"  Scenario 1 (20% load shift): {100*max(scenario1_ss, baseline_ss):.1f}% (+{100*max(scenario1_ss-baseline_ss, 0):.1f}pp)")

    # Scenario 2: Double battery capacity
    battery_efficiency = battery_results.get('overall_efficiency', 0.80)
    current_daily_discharge = battery_results.get('daily_discharge', {}).get('mean', 10)

    # Double discharge capacity (roughly)
    additional_discharge = current_daily_discharge * battery_efficiency
    scenario2_import_reduction = additional_discharge * len(energy_15min.index.normalize().unique())
    scenario2_import = max(total_import - scenario2_import_reduction, 0)
    scenario2_ss = 1 - (scenario2_import / total_consumption)

    results['scenario2_battery'] = {
        'self_sufficiency': min(scenario2_ss, 1.0),
        'improvement': min(scenario2_ss - baseline_ss, 1.0 - baseline_ss)
    }
    print(f"  Scenario 2 (2x battery): {100*min(scenario2_ss, 1.0):.1f}% (+{100*min(scenario2_ss-baseline_ss, 1.0-baseline_ss):.1f}pp)")

    # Scenario 3: Combined (load shift + battery)
    scenario3_import = max(scenario2_import - shifted_import_reduction * len(energy_15min.index.normalize().unique()), 0)
    scenario3_ss = 1 - (scenario3_import / total_consumption)

    results['scenario3_combined'] = {
        'self_sufficiency': min(scenario3_ss, 1.0),
        'improvement': min(scenario3_ss - baseline_ss, 1.0 - baseline_ss)
    }
    print(f"  Scenario 3 (combined): {100*min(scenario3_ss, 1.0):.1f}% (+{100*min(scenario3_ss-baseline_ss, 1.0-baseline_ss):.1f}pp)")

    return results


def create_energy_system_plots(energy_15min: pd.DataFrame, pv_results: dict,
                              battery_results: dict, grid_results: dict,
                              ss_potential: dict) -> None:
    """Create visualization of energy system model results."""
    print("\nCreating energy system plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Hourly PV and Consumption Profile
    ax = axes[0, 0]

    pv_col = 'pv_generation_kwh'
    consumption_col = 'total_consumption_kwh'

    if pv_col in energy_15min.columns:
        hourly_pv = energy_15min[pv_col].groupby(energy_15min.index.hour).mean() * 4
        ax.fill_between(hourly_pv.index, 0, hourly_pv.values, alpha=0.5, color='yellow', label='PV Generation')

    if consumption_col in energy_15min.columns:
        hourly_cons = energy_15min[consumption_col].groupby(energy_15min.index.hour).mean() * 4
        ax.plot(hourly_cons.index, hourly_cons.values, 'b-', linewidth=2, label='Consumption')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Daily Energy Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    # Panel 2: Battery Charge/Discharge Profile
    ax = axes[0, 1]

    if 'hourly_charge_profile' in battery_results:
        charge = pd.Series(battery_results['hourly_charge_profile'])
        discharge = pd.Series(battery_results['hourly_discharge_profile'])

        ax.fill_between(charge.index, 0, charge.values, alpha=0.5, color='green', label='Charging')
        ax.fill_between(discharge.index, 0, -discharge.values, alpha=0.5, color='red', label='Discharging')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Battery Charge/Discharge Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlim(0, 23)

    # Panel 3: Monthly Energy Balance
    ax = axes[1, 0]

    grid_import_col = 'external_supply_kwh'
    grid_export_col = 'grid_feedin_kwh'

    if grid_import_col in energy_15min.columns and grid_export_col in energy_15min.columns:
        monthly_import = energy_15min[grid_import_col].resample('ME').sum()
        monthly_export = energy_15min[grid_export_col].resample('ME').sum()

        x = range(len(monthly_import))
        width = 0.35

        ax.bar([i - width/2 for i in x], monthly_import.values, width, label='Grid Import', color='red', alpha=0.7)
        ax.bar([i + width/2 for i in x], monthly_export.values, width, label='Grid Export', color='green', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly_import.index], rotation=45, ha='right')

    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Monthly Grid Interaction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Self-Sufficiency Scenarios
    ax = axes[1, 1]

    scenarios = ['Baseline', 'Load Shift', '2x Battery', 'Combined']
    values = [
        ss_potential.get('baseline_self_sufficiency', 0) * 100,
        ss_potential.get('scenario1_load_shift', {}).get('self_sufficiency', 0) * 100,
        ss_potential.get('scenario2_battery', {}).get('self_sufficiency', 0) * 100,
        ss_potential.get('scenario3_combined', {}).get('self_sufficiency', 0) * 100
    ]

    colors = ['blue', 'orange', 'green', 'red']
    bars = ax.bar(scenarios, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Self-Sufficiency (%)')
    ax.set_title('Self-Sufficiency Optimization Scenarios')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig20_energy_system_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig20_energy_system_model.png")


def generate_report(pv_results: dict, battery_results: dict,
                   grid_results: dict, ss_potential: dict) -> str:
    """Generate HTML report section for energy system model."""

    # Extract key values
    daily_pv = pv_results.get('daily_stats', {}).get('mean', 0)
    daily_pv_max = pv_results.get('daily_stats', {}).get('max', 0)
    peak_hours = pv_results.get('peak_hours', [])

    battery_eff = battery_results.get('overall_efficiency', 0) * 100
    daily_discharge = battery_results.get('daily_discharge', {}).get('mean', 0)

    daily_import = grid_results.get('daily_stats', {}).get('import_mean', 0)
    daily_export = grid_results.get('daily_stats', {}).get('export_mean', 0)
    self_suff = grid_results.get('self_sufficiency', {}).get('mean', 0) * 100

    baseline_ss = ss_potential.get('baseline_self_sufficiency', 0) * 100
    scenario1_ss = ss_potential.get('scenario1_load_shift', {}).get('self_sufficiency', 0) * 100
    scenario2_ss = ss_potential.get('scenario2_battery', {}).get('self_sufficiency', 0) * 100
    scenario3_ss = ss_potential.get('scenario3_combined', {}).get('self_sufficiency', 0) * 100

    html = f"""
    <section id="energy-system-model">
    <h2>3.3 Energy System Model</h2>

    <h3>PV Generation Patterns</h3>
    <table>
        <tr><th>Metric</th><th>Symbol</th><th>Value</th></tr>
        <tr><td>Mean daily generation</td><td>P̄<sub>pv</sub></td><td><strong>{daily_pv:.1f} kWh</strong></td></tr>
        <tr><td>Peak daily generation</td><td>P<sub>pv,max</sub></td><td>{daily_pv_max:.1f} kWh</td></tr>
        <tr><td>Peak generation hours</td><td>—</td><td>{peak_hours[0] if peak_hours else 'N/A'}:00 - {peak_hours[-1] if peak_hours else 'N/A'}:00</td></tr>
    </table>

    <h3>Battery Performance</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Notes</th></tr>
        <tr>
            <td>Round-trip efficiency (η<sub>bat</sub>)</td>
            <td><strong>{battery_eff:.1f}%</strong></td>
            <td>{"Below expected (degraded)" if battery_eff < 82 else "Within normal range"}</td>
        </tr>
        <tr>
            <td>Mean daily discharge (E<sub>discharge</sub>)</td>
            <td>{daily_discharge:.1f} kWh</td>
            <td>Energy supplied to home from battery</td>
        </tr>
    </table>

    <h3>Grid Interaction</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean daily import (E<sub>import</sub>)</td><td>{daily_import:.1f} kWh</td></tr>
        <tr><td>Mean daily export (E<sub>export</sub>)</td><td>{daily_export:.1f} kWh</td></tr>
        <tr><td>Net export</td><td>{daily_export - daily_import:.1f} kWh/day</td></tr>
        <tr><td>Current self-sufficiency (η<sub>ss</sub>)</td><td><strong>{self_suff:.1f}%</strong></td></tr>
    </table>
    <p>Self-sufficiency definition:</p>
    <div class="equation-box">
    η<sub>ss</sub> = 1 − E<sub>import</sub>/E<sub>consumption</sub> = (E<sub>direct</sub> + E<sub>battery</sub>)/E<sub>consumption</sub>
    </div>

    <h3>Self-Sufficiency Optimization Scenarios</h3>
    <table>
        <tr><th>Scenario</th><th>Self-Sufficiency</th><th>Improvement</th><th>Description</th></tr>
        <tr>
            <td>Baseline (current)</td>
            <td>{baseline_ss:.1f}%</td>
            <td>—</td>
            <td>Current system operation</td>
        </tr>
        <tr>
            <td>Load Shifting</td>
            <td>{scenario1_ss:.1f}%</td>
            <td>+{scenario1_ss - baseline_ss:.1f}pp</td>
            <td>Shift 20% of evening load to solar hours</td>
        </tr>
        <tr>
            <td>2× Battery Capacity</td>
            <td>{scenario2_ss:.1f}%</td>
            <td>+{scenario2_ss - baseline_ss:.1f}pp</td>
            <td>Double battery storage capacity</td>
        </tr>
        <tr>
            <td>Combined</td>
            <td>{scenario3_ss:.1f}%</td>
            <td>+{scenario3_ss - baseline_ss:.1f}pp</td>
            <td>Load shifting + larger battery</td>
        </tr>
    </table>

    <h3>Recommendations</h3>
    <ul>
        <li><strong>Heating timing</strong>: Schedule comfort mode start during solar hours
            ({peak_hours[0] if peak_hours else 8}:00 onwards) to maximize direct PV consumption.</li>
        <li><strong>Buffer tank pre-heating</strong>: Charge buffer tank during peak PV
            ({peak_hours[len(peak_hours)//2] if peak_hours else 12}:00-{peak_hours[-1] if peak_hours else 15}:00)
            to store thermal energy for evening.</li>
        <li><strong>Battery considerations</strong>: Current efficiency ({battery_eff:.0f}%) is
            {"degraded from the Feb-Mar 2025 event" if battery_eff < 82 else "acceptable"}.
            Account for this in optimization.</li>
        <li><strong>Grid export value</strong>: Net exporter ({daily_export - daily_import:.0f} kWh/day).
            Tariff optimization could improve financial return on exports.</li>
    </ul>

    <figure>
        <img src="fig20_energy_system_model.png" alt="Energy System Model">
        <figcaption><strong>Figure 20:</strong> Energy system analysis: daily profile (top-left),
        battery patterns (top-right), monthly grid balance (bottom-left),
        self-sufficiency scenarios (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for energy system modeling."""
    print("="*60)
    print("Phase 3, Step 3: Energy System Model")
    print("="*60)

    # Load data
    energy_15min = load_data()

    # Analyze PV patterns
    pv_results = analyze_pv_patterns(energy_15min)

    # Analyze battery dynamics
    battery_results = analyze_battery_dynamics(energy_15min)

    # Analyze grid interaction
    grid_results = analyze_grid_interaction(energy_15min)

    # Model self-sufficiency potential
    ss_potential = model_self_sufficiency_potential(energy_15min, pv_results, battery_results)

    # Create visualizations
    create_energy_system_plots(energy_15min, pv_results, battery_results, grid_results, ss_potential)

    # Generate report section
    report_html = generate_report(pv_results, battery_results, grid_results, ss_potential)
    with open(OUTPUT_DIR / 'energy_system_model_report_section.html', 'w') as f:
        f.write(report_html)
    print("\nSaved: energy_system_model_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("ENERGY SYSTEM MODEL SUMMARY")
    print("="*60)

    print(f"\nPV Generation:")
    print(f"  Mean daily: {pv_results.get('daily_stats', {}).get('mean', 0):.1f} kWh")
    print(f"  Peak hours: {pv_results.get('peak_hours', [])}")

    print(f"\nBattery:")
    print(f"  Round-trip efficiency: {100*battery_results.get('overall_efficiency', 0):.1f}%")

    print(f"\nSelf-Sufficiency:")
    print(f"  Current: {100*ss_potential.get('baseline_self_sufficiency', 0):.1f}%")
    print(f"  Potential with optimization: {100*ss_potential.get('scenario3_combined', {}).get('self_sufficiency', 0):.1f}%")


if __name__ == '__main__':
    main()
