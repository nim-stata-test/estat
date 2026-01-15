#!/usr/bin/env python3
"""
Phase 3, Step 4: Tariff Cost Model

Models electricity costs using time-of-use tariffs:
- Historical cost analysis (daily/monthly breakdown)
- High vs low tariff cost contribution
- Cost forecast model (predict from weather/HDD)
- Cost optimization scenarios (shift timing)

Inputs:
- energy_balance_15min.parquet (grid import/export data)
- tariff_series_hourly.parquet (purchase and feed-in rates)
- integrated_overlap_only.parquet (for weather correlation)
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
    """Load energy and tariff data for cost modeling."""
    print("Loading data...")

    # Load 15-minute energy data
    energy_15min = pd.read_parquet(PROCESSED_DIR / 'energy_balance_15min.parquet')
    energy_15min.index = pd.to_datetime(energy_15min.index)

    # Load hourly tariff data
    tariff_series = pd.read_parquet(PROCESSED_DIR / 'tariff_series_hourly.parquet')
    tariff_series.index = pd.to_datetime(tariff_series.index)

    # Load integrated dataset (for weather/HDD correlation)
    integrated_path = PROCESSED_DIR / 'integrated_overlap_only.parquet'
    if integrated_path.exists():
        integrated = pd.read_parquet(integrated_path)
        integrated.index = pd.to_datetime(integrated.index)
    else:
        integrated = None

    print(f"  Energy data: {len(energy_15min):,} rows ({energy_15min.index.min().date()} to {energy_15min.index.max().date()})")
    print(f"  Tariff data: {len(tariff_series):,} hours")
    if integrated is not None:
        print(f"  Integrated data: {len(integrated):,} rows (for weather correlation)")

    return energy_15min, tariff_series, integrated


def calculate_hourly_costs(energy_15min: pd.DataFrame, tariff_series: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hourly electricity costs using time-of-use rates.

    Returns DataFrame with columns:
    - grid_import_kwh: energy imported from grid
    - grid_export_kwh: energy exported to grid
    - purchase_rate: applicable purchase rate (Rp/kWh)
    - feedin_rate: applicable feed-in rate (Rp/kWh)
    - grid_cost_rp: cost of grid import (Rappen)
    - feedin_revenue_rp: revenue from feed-in (Rappen)
    - net_cost_rp: grid_cost - feedin_revenue
    - is_high_tariff: boolean flag
    """
    print("\nCalculating hourly costs...")

    # Resample energy to hourly
    grid_import_col = 'external_supply_kwh'
    grid_export_col = 'grid_feedin_kwh'

    hourly_import = energy_15min[grid_import_col].resample('h').sum()
    hourly_export = energy_15min[grid_export_col].resample('h').sum()

    # Create result DataFrame
    result = pd.DataFrame(index=hourly_import.index)
    result['grid_import_kwh'] = hourly_import
    result['grid_export_kwh'] = hourly_export

    # Merge tariff rates (align by hour)
    result = result.join(tariff_series[['purchase_rate_rp_kwh', 'purchase_rate_high_rp_kwh',
                                        'purchase_rate_low_rp_kwh', 'feedin_rate_rp_kwh',
                                        'is_high_tariff']], how='left')

    # Forward fill missing rates (for edges)
    result['purchase_rate_rp_kwh'] = result['purchase_rate_rp_kwh'].ffill().bfill()
    result['feedin_rate_rp_kwh'] = result['feedin_rate_rp_kwh'].ffill().bfill()
    result['is_high_tariff'] = result['is_high_tariff'].ffill().bfill()

    # Calculate costs
    result['grid_cost_rp'] = result['grid_import_kwh'] * result['purchase_rate_rp_kwh']
    result['feedin_revenue_rp'] = result['grid_export_kwh'] * result['feedin_rate_rp_kwh']
    result['net_cost_rp'] = result['grid_cost_rp'] - result['feedin_revenue_rp']

    # Convert to CHF
    result['grid_cost_chf'] = result['grid_cost_rp'] / 100
    result['feedin_revenue_chf'] = result['feedin_revenue_rp'] / 100
    result['net_cost_chf'] = result['net_cost_rp'] / 100

    print(f"  Calculated costs for {len(result):,} hours")
    print(f"  Total grid cost: CHF {result['grid_cost_chf'].sum():,.2f}")
    print(f"  Total feed-in revenue: CHF {result['feedin_revenue_chf'].sum():,.2f}")
    print(f"  Total net cost: CHF {result['net_cost_chf'].sum():,.2f}")

    return result


def analyze_cost_patterns(hourly_costs: pd.DataFrame) -> dict:
    """
    Analyze electricity cost patterns over time.

    Returns:
    - Daily, monthly, yearly cost aggregations
    - High vs low tariff breakdown
    - Seasonal patterns
    """
    print("\nAnalyzing cost patterns...")

    results = {}

    # Daily aggregation
    daily = hourly_costs.resample('D').agg({
        'grid_import_kwh': 'sum',
        'grid_export_kwh': 'sum',
        'grid_cost_chf': 'sum',
        'feedin_revenue_chf': 'sum',
        'net_cost_chf': 'sum'
    })
    daily['net_export'] = daily['grid_export_kwh'] > daily['grid_import_kwh']

    results['daily_stats'] = {
        'mean_cost': daily['grid_cost_chf'].mean(),
        'mean_revenue': daily['feedin_revenue_chf'].mean(),
        'mean_net': daily['net_cost_chf'].mean(),
        'std_net': daily['net_cost_chf'].std(),
        'net_producer_days': daily['net_export'].sum(),
        'total_days': len(daily)
    }

    print(f"  Daily averages:")
    print(f"    Grid cost: CHF {daily['grid_cost_chf'].mean():.2f}")
    print(f"    Feed-in revenue: CHF {daily['feedin_revenue_chf'].mean():.2f}")
    print(f"    Net cost: CHF {daily['net_cost_chf'].mean():.2f}")
    print(f"    Net producer days: {daily['net_export'].sum()} / {len(daily)}")

    # Monthly aggregation
    monthly = hourly_costs.resample('ME').agg({
        'grid_import_kwh': 'sum',
        'grid_export_kwh': 'sum',
        'grid_cost_chf': 'sum',
        'feedin_revenue_chf': 'sum',
        'net_cost_chf': 'sum'
    })

    results['monthly_costs'] = monthly['net_cost_chf'].to_dict()
    results['monthly_stats'] = {
        'mean_monthly_cost': monthly['grid_cost_chf'].mean(),
        'mean_monthly_revenue': monthly['feedin_revenue_chf'].mean(),
        'mean_monthly_net': monthly['net_cost_chf'].mean()
    }

    print(f"\n  Monthly averages:")
    print(f"    Grid cost: CHF {monthly['grid_cost_chf'].mean():.2f}")
    print(f"    Feed-in revenue: CHF {monthly['feedin_revenue_chf'].mean():.2f}")
    print(f"    Net cost: CHF {monthly['net_cost_chf'].mean():.2f}")

    # High vs Low tariff breakdown
    high_tariff = hourly_costs[hourly_costs['is_high_tariff'] == True]
    low_tariff = hourly_costs[hourly_costs['is_high_tariff'] == False]

    ht_cost = high_tariff['grid_cost_chf'].sum()
    lt_cost = low_tariff['grid_cost_chf'].sum()
    total_cost = ht_cost + lt_cost

    results['tariff_breakdown'] = {
        'high_tariff_cost': ht_cost,
        'low_tariff_cost': lt_cost,
        'high_tariff_pct': ht_cost / total_cost if total_cost > 0 else 0,
        'high_tariff_import_kwh': high_tariff['grid_import_kwh'].sum(),
        'low_tariff_import_kwh': low_tariff['grid_import_kwh'].sum()
    }

    print(f"\n  Tariff breakdown:")
    print(f"    High tariff cost: CHF {ht_cost:.2f} ({100*ht_cost/total_cost:.1f}%)")
    print(f"    Low tariff cost: CHF {lt_cost:.2f} ({100*lt_cost/total_cost:.1f}%)")

    # Seasonal patterns
    hourly_costs['month'] = hourly_costs.index.month
    hourly_costs['is_winter'] = hourly_costs['month'].isin([10, 11, 12, 1, 2, 3])

    winter = hourly_costs[hourly_costs['is_winter']]
    summer = hourly_costs[~hourly_costs['is_winter']]

    results['seasonal'] = {
        'winter_daily_cost': winter.resample('D')['net_cost_chf'].sum().mean() if len(winter) > 0 else 0,
        'summer_daily_cost': summer.resample('D')['net_cost_chf'].sum().mean() if len(summer) > 0 else 0
    }

    if len(winter) > 0 and len(summer) > 0:
        print(f"\n  Seasonal patterns:")
        print(f"    Winter daily net cost: CHF {results['seasonal']['winter_daily_cost']:.2f}")
        print(f"    Summer daily net cost: CHF {results['seasonal']['summer_daily_cost']:.2f}")

    # Annual projection
    days_in_data = (hourly_costs.index.max() - hourly_costs.index.min()).days + 1
    total_net_cost = hourly_costs['net_cost_chf'].sum()
    annual_projection = total_net_cost * 365 / days_in_data if days_in_data > 0 else 0

    results['annual_projection'] = annual_projection
    print(f"\n  Annual projection: CHF {annual_projection:,.2f}")

    return results, daily


def build_cost_forecast_model(daily_costs: pd.DataFrame, integrated: pd.DataFrame = None) -> dict:
    """
    Build a regression model to predict daily costs from weather/HDD.

    Model: daily_cost = a + b*HDD + c*month + d*is_weekend
    """
    print("\nBuilding cost forecast model...")

    results = {}

    # Add features to daily costs
    daily = daily_costs.copy()
    daily['month'] = daily.index.month
    daily['dayofweek'] = daily.index.dayofweek
    daily['is_weekend'] = daily['dayofweek'] >= 5
    daily['day_of_year'] = daily.index.dayofyear

    # Simple seasonal model (no weather data)
    # Use month and weekend as predictors
    X = pd.DataFrame({
        'month_sin': np.sin(2 * np.pi * daily['month'] / 12),
        'month_cos': np.cos(2 * np.pi * daily['month'] / 12),
        'is_weekend': daily['is_weekend'].astype(int)
    })
    y = daily['net_cost_chf'].values

    # Remove NaN
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(y_clean) < 10:
        print("  Insufficient data for model")
        return results

    # Fit model
    model = LinearRegression()
    model.fit(X_clean, y_clean)

    y_pred = model.predict(X_clean)
    r2 = r2_score(y_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))

    results['seasonal_model'] = {
        'r2': r2,
        'rmse': rmse,
        'coefficients': {
            'intercept': model.intercept_,
            'month_sin': model.coef_[0],
            'month_cos': model.coef_[1],
            'is_weekend': model.coef_[2]
        }
    }

    print(f"  Seasonal model:")
    print(f"    R² = {r2:.3f}")
    print(f"    RMSE = CHF {rmse:.2f}")
    print(f"    Weekend effect: CHF {model.coef_[2]:.2f}")

    # If we have weather data, build HDD model
    if integrated is not None and 'stiebel_eltron_isg_outdoor_temperature' in integrated.columns:
        print("\n  Building HDD-based model...")

        # Calculate daily HDD from outdoor temperature
        outdoor_temp = integrated['stiebel_eltron_isg_outdoor_temperature'].resample('D').mean()

        # HDD = max(0, 18 - T_outdoor)
        hdd = (18 - outdoor_temp).clip(lower=0)

        # Handle timezone mismatch - convert to naive datetime for join
        if hdd.index.tz is not None:
            hdd.index = hdd.index.tz_localize(None)
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        # Merge with daily costs
        daily_with_hdd = daily.join(hdd.rename('hdd'), how='inner')

        if len(daily_with_hdd) > 10:
            X_hdd = pd.DataFrame({
                'hdd': daily_with_hdd['hdd'],
                'month_sin': np.sin(2 * np.pi * daily_with_hdd['month'] / 12),
                'month_cos': np.cos(2 * np.pi * daily_with_hdd['month'] / 12),
                'is_weekend': daily_with_hdd['is_weekend'].astype(int)
            })
            y_hdd = daily_with_hdd['net_cost_chf'].values

            mask = ~np.isnan(y_hdd) & ~np.isnan(X_hdd['hdd'])
            X_hdd_clean = X_hdd[mask]
            y_hdd_clean = y_hdd[mask]

            model_hdd = LinearRegression()
            model_hdd.fit(X_hdd_clean, y_hdd_clean)

            y_pred_hdd = model_hdd.predict(X_hdd_clean)
            r2_hdd = r2_score(y_hdd_clean, y_pred_hdd)
            rmse_hdd = np.sqrt(mean_squared_error(y_hdd_clean, y_pred_hdd))

            results['hdd_model'] = {
                'r2': r2_hdd,
                'rmse': rmse_hdd,
                'hdd_coefficient': model_hdd.coef_[0],
                'coefficients': {
                    'intercept': model_hdd.intercept_,
                    'hdd': model_hdd.coef_[0],
                    'month_sin': model_hdd.coef_[1],
                    'month_cos': model_hdd.coef_[2],
                    'is_weekend': model_hdd.coef_[3]
                }
            }

            print(f"    R² = {r2_hdd:.3f}")
            print(f"    RMSE = CHF {rmse_hdd:.2f}")
            print(f"    HDD coefficient: CHF {model_hdd.coef_[0]:.2f} per HDD")

    return results


def simulate_cost_scenarios(hourly_costs: pd.DataFrame, cost_patterns: dict) -> dict:
    """
    Simulate cost under different timing scenarios.

    Scenarios:
    - Baseline: current operation
    - Shift to low tariff: move 20% of high-tariff consumption to low tariff
    - Maximize feed-in: export more during high feed-in periods
    """
    print("\nSimulating cost optimization scenarios...")

    results = {}

    # Baseline
    baseline_cost = hourly_costs['grid_cost_chf'].sum()
    baseline_revenue = hourly_costs['feedin_revenue_chf'].sum()
    baseline_net = baseline_cost - baseline_revenue

    results['baseline'] = {
        'grid_cost': baseline_cost,
        'feedin_revenue': baseline_revenue,
        'net_cost': baseline_net
    }

    print(f"  Baseline:")
    print(f"    Grid cost: CHF {baseline_cost:,.2f}")
    print(f"    Feed-in revenue: CHF {baseline_revenue:,.2f}")
    print(f"    Net cost: CHF {baseline_net:,.2f}")

    # Scenario 1: Shift 20% of high-tariff import to low-tariff hours
    high_tariff_mask = hourly_costs['is_high_tariff'] == True
    ht_import = hourly_costs.loc[high_tariff_mask, 'grid_import_kwh'].sum()
    shift_amount = ht_import * 0.20

    # Calculate savings from shift
    avg_ht_rate = hourly_costs.loc[high_tariff_mask, 'purchase_rate_rp_kwh'].mean()
    avg_lt_rate = hourly_costs.loc[~high_tariff_mask, 'purchase_rate_rp_kwh'].mean()
    rate_differential = (avg_ht_rate - avg_lt_rate) / 100  # CHF/kWh

    shift_savings = shift_amount * rate_differential
    scenario1_net = baseline_net - shift_savings

    # Calculate reduction percentage - handle negative baseline (net income)
    if baseline_net > 0:
        reduction_pct = shift_savings / baseline_net * 100
    else:
        # If baseline is negative (net income), show improvement as % of absolute value
        reduction_pct = shift_savings / abs(baseline_net) * 100 if baseline_net != 0 else 0

    results['shift_to_low_tariff'] = {
        'shifted_kwh': shift_amount,
        'rate_differential': rate_differential * 100,  # Rp/kWh
        'savings': shift_savings,
        'net_cost': scenario1_net,
        'reduction_pct': reduction_pct
    }

    print(f"\n  Scenario 1 (shift 20% to low tariff):")
    print(f"    Shifted: {shift_amount:,.1f} kWh")
    print(f"    Rate differential: {rate_differential*100:.1f} Rp/kWh")
    print(f"    Savings: CHF {shift_savings:,.2f}")
    improvement_label = "cost reduction" if baseline_net > 0 else "income increase"
    print(f"    Net cost: CHF {scenario1_net:,.2f} ({reduction_pct:.1f}% {improvement_label})")

    # Scenario 2: Shift 30% to low tariff (more aggressive)
    shift_amount_30 = ht_import * 0.30
    shift_savings_30 = shift_amount_30 * rate_differential
    scenario2_net = baseline_net - shift_savings_30

    if baseline_net > 0:
        reduction_pct_30 = shift_savings_30 / baseline_net * 100
    else:
        reduction_pct_30 = shift_savings_30 / abs(baseline_net) * 100 if baseline_net != 0 else 0

    results['aggressive_shift'] = {
        'shifted_kwh': shift_amount_30,
        'savings': shift_savings_30,
        'net_cost': scenario2_net,
        'reduction_pct': reduction_pct_30
    }

    print(f"\n  Scenario 2 (shift 30% to low tariff):")
    print(f"    Savings: CHF {shift_savings_30:,.2f}")
    print(f"    Net cost: CHF {scenario2_net:,.2f} ({reduction_pct_30:.1f}% {improvement_label})")

    # Scenario 3: Combined - shift load + optimize battery
    # Assume battery can shift additional 10% of consumption
    additional_shift = ht_import * 0.10
    battery_savings = additional_shift * rate_differential * 0.85  # Account for battery efficiency
    scenario3_net = scenario1_net - battery_savings

    total_savings = shift_savings + battery_savings
    if baseline_net > 0:
        reduction_pct_combined = total_savings / baseline_net * 100
    else:
        reduction_pct_combined = total_savings / abs(baseline_net) * 100 if baseline_net != 0 else 0

    results['combined_optimization'] = {
        'load_shift_savings': shift_savings,
        'battery_shift_savings': battery_savings,
        'total_savings': total_savings,
        'net_cost': scenario3_net,
        'reduction_pct': reduction_pct_combined
    }

    print(f"\n  Scenario 3 (combined - load shift + battery):")
    print(f"    Total savings: CHF {total_savings:,.2f}")
    print(f"    Net cost: CHF {scenario3_net:,.2f} ({reduction_pct_combined:.1f}% {improvement_label})")

    return results


def create_cost_analysis_plots(hourly_costs: pd.DataFrame, daily_costs: pd.DataFrame,
                               cost_patterns: dict, scenarios: dict) -> None:
    """Create visualization of cost model results."""
    print("\nCreating cost analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Monthly costs over time
    ax = axes[0, 0]

    monthly = hourly_costs.resample('ME').agg({
        'grid_cost_chf': 'sum',
        'feedin_revenue_chf': 'sum',
        'net_cost_chf': 'sum'
    })

    x = range(len(monthly))
    width = 0.35

    ax.bar([i - width/2 for i in x], monthly['grid_cost_chf'].values, width,
           label='Grid Cost', color='red', alpha=0.7)
    ax.bar([i + width/2 for i in x], monthly['feedin_revenue_chf'].values, width,
           label='Feed-in Revenue', color='green', alpha=0.7)
    ax.plot(x, monthly['net_cost_chf'].values, 'b-o', label='Net Cost', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=45, ha='right')
    ax.set_ylabel('Cost (CHF)')
    ax.set_title('Monthly Electricity Costs')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: High vs Low tariff breakdown
    ax = axes[0, 1]

    breakdown = cost_patterns['tariff_breakdown']
    labels = ['High Tariff', 'Low Tariff']
    sizes = [breakdown['high_tariff_cost'], breakdown['low_tariff_cost']]
    colors = ['#ff9999', '#99ff99']
    explode = (0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Grid Cost by Tariff Period')

    # Panel 3: Hourly cost profile
    ax = axes[1, 0]

    hourly_profile = hourly_costs.groupby(hourly_costs.index.hour).agg({
        'grid_cost_chf': 'mean',
        'feedin_revenue_chf': 'mean',
        'net_cost_chf': 'mean'
    })

    ax.fill_between(hourly_profile.index, 0, hourly_profile['grid_cost_chf'].values,
                    alpha=0.5, color='red', label='Grid Cost')
    ax.fill_between(hourly_profile.index, 0, -hourly_profile['feedin_revenue_chf'].values,
                    alpha=0.5, color='green', label='Feed-in Revenue')
    ax.plot(hourly_profile.index, hourly_profile['net_cost_chf'].values, 'b-',
            linewidth=2, label='Net Cost')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Hourly Cost (CHF)')
    ax.set_title('Daily Cost Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    # Panel 4: Optimization scenarios comparison
    ax = axes[1, 1]

    scenario_names = ['Baseline', '20% Shift', '30% Shift', 'Combined']
    scenario_values = [
        scenarios['baseline']['net_cost'],
        scenarios['shift_to_low_tariff']['net_cost'],
        scenarios['aggressive_shift']['net_cost'],
        scenarios['combined_optimization']['net_cost']
    ]

    # Annualize for comparison
    days_in_data = (hourly_costs.index.max() - hourly_costs.index.min()).days + 1
    annual_factor = 365 / days_in_data if days_in_data > 0 else 1
    scenario_values_annual = [v * annual_factor for v in scenario_values]

    colors = ['blue', 'orange', 'red', 'green']
    bars = ax.bar(scenario_names, scenario_values_annual, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, scenario_values_annual):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'CHF {val:,.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Annual Net Cost (CHF)')
    ax.set_title('Cost Optimization Scenarios')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig21_tariff_cost_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig21_tariff_cost_model.png")


def generate_report(cost_patterns: dict, forecast_results: dict,
                   scenarios: dict, hourly_costs: pd.DataFrame) -> str:
    """Generate HTML report section for tariff cost model."""

    # Calculate annual projection
    days_in_data = (hourly_costs.index.max() - hourly_costs.index.min()).days + 1
    annual_factor = 365 / days_in_data if days_in_data > 0 else 1

    daily = cost_patterns['daily_stats']
    tariff = cost_patterns['tariff_breakdown']
    seasonal = cost_patterns.get('seasonal', {})

    baseline = scenarios['baseline']
    shift_20 = scenarios['shift_to_low_tariff']
    shift_30 = scenarios['aggressive_shift']
    combined = scenarios['combined_optimization']

    seasonal_model = forecast_results.get('seasonal_model', {})
    hdd_model = forecast_results.get('hdd_model', {})

    html = f"""
    <section id="tariff-cost-model">
    <h2>3.4 Tariff Cost Model</h2>

    <h3>Cost Definitions</h3>
    <p>Electricity cost components:</p>
    <div class="equation-box">
    $$C_{{grid}} = E_{{import}} \\times p_{{purchase}}$$
    $$R_{{feedin}} = E_{{export}} \\times p_{{feedin}}$$
    $$C_{{net}} = C_{{grid}} - R_{{feedin}}$$
    </div>
    <p>where $E_{{import}}$ is grid import (kWh), $E_{{export}}$ is grid export (kWh),
    $p_{{purchase}}$ is purchase rate (CHF/kWh), $p_{{feedin}}$ is feed-in rate (CHF/kWh).</p>

    <h3>Historical Cost Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Symbol</th><th>Daily Average</th><th>Annual Projection</th></tr>
        <tr>
            <td>Grid purchase cost</td>
            <td>$C_{{grid}}$</td>
            <td>CHF {daily['mean_cost']:.2f}</td>
            <td>CHF {daily['mean_cost'] * 365:,.0f}</td>
        </tr>
        <tr>
            <td>Feed-in revenue</td>
            <td>$R_{{feedin}}$</td>
            <td>CHF {daily['mean_revenue']:.2f}</td>
            <td>CHF {daily['mean_revenue'] * 365:,.0f}</td>
        </tr>
        <tr>
            <td><strong>Net cost</strong></td>
            <td>$C_{{net}}$</td>
            <td><strong>CHF {daily['mean_net']:.2f}</strong></td>
            <td><strong>CHF {daily['mean_net'] * 365:,.0f}</strong></td>
        </tr>
    </table>

    <h3>High vs Low Tariff Breakdown</h3>
    <table>
        <tr><th>Tariff Period</th><th>Grid Import (kWh)</th><th>Cost (CHF)</th><th>Share</th></tr>
        <tr>
            <td>High Tariff (HT)</td>
            <td>{tariff['high_tariff_import_kwh']:,.1f}</td>
            <td>{tariff['high_tariff_cost']:,.2f}</td>
            <td>{tariff['high_tariff_pct']*100:.1f}%</td>
        </tr>
        <tr>
            <td>Low Tariff (NT)</td>
            <td>{tariff['low_tariff_import_kwh']:,.1f}</td>
            <td>{tariff['low_tariff_cost']:,.2f}</td>
            <td>{(1-tariff['high_tariff_pct'])*100:.1f}%</td>
        </tr>
    </table>

    <p><strong>Insight:</strong> {tariff['high_tariff_pct']*100:.0f}% of grid costs ($C_{{grid}}$) occur during high-tariff
    periods. Shifting consumption to low-tariff hours (21:00-06:00 weekdays, weekends) can reduce costs.</p>

    <h3>Seasonal Patterns</h3>
    <table>
        <tr><th>Season</th><th>Daily Net Cost</th><th>Notes</th></tr>
        <tr>
            <td>Winter (Oct-Mar)</td>
            <td>CHF {seasonal.get('winter_daily_cost', 0):.2f}</td>
            <td>Higher heating demand, less PV</td>
        </tr>
        <tr>
            <td>Summer (Apr-Sep)</td>
            <td>CHF {seasonal.get('summer_daily_cost', 0):.2f}</td>
            <td>Higher PV, often net producer</td>
        </tr>
    </table>

    <h3>Cost Forecast Model</h3>
    <p>Seasonal model with heating degree-days:</p>
    <div class="equation-box">
    $$C_{{net}}(t) = \\beta_0 + \\beta_{{HDD}} \\cdot \\text{{HDD}}(t) + \\beta_{{sin}} \\cdot \\sin\\left(\\frac{{2\\pi m}}{{12}}\\right) + \\beta_{{cos}} \\cdot \\cos\\left(\\frac{{2\\pi m}}{{12}}\\right) + \\beta_{{we}} \\cdot \\mathbf{{1}}_{{weekend}}$$
    </div>
    <p>where $\\text{{HDD}} = \\max(0, 18 - T_{{out}})$ is heating degree-days, $m$ is month, and $\\mathbf{{1}}_{{weekend}}$ is weekend indicator.</p>
    <table>
        <tr><th>Model</th><th>$R^2$</th><th>RMSE</th><th>Key Coefficient</th></tr>
        <tr>
            <td>Seasonal model</td>
            <td>{seasonal_model.get('r2', 0):.3f}</td>
            <td>CHF {seasonal_model.get('rmse', 0):.2f}</td>
            <td>$\\beta_{{we}}$ = CHF {seasonal_model.get('coefficients', {}).get('is_weekend', 0):.2f}</td>
        </tr>
        {"<tr><td>HDD model</td><td>" + f"{hdd_model.get('r2', 0):.3f}</td><td>CHF {hdd_model.get('rmse', 0):.2f}</td><td>$\\beta_{{HDD}}$ = CHF {hdd_model.get('hdd_coefficient', 0):.2f}/HDD</td></tr>" if hdd_model else ""}
    </table>

    <h3>Cost Optimization Scenarios</h3>
    <p>Load shifting potential based on moving high-tariff consumption ($E_{{HT}}$) to low-tariff periods:</p>
    <div class="equation-box">
    $$\\Delta C = E_{{shift}} \\times (p_{{HT}} - p_{{NT}})$$
    </div>
    <table>
        <tr><th>Scenario</th><th>$C_{{net}}$</th><th>Annual</th><th>$\\Delta C$</th><th>Reduction</th></tr>
        <tr>
            <td>Baseline (current)</td>
            <td>CHF {baseline['net_cost']:,.2f}</td>
            <td>CHF {baseline['net_cost'] * annual_factor:,.0f}</td>
            <td>—</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Shift 20% to low tariff</td>
            <td>CHF {shift_20['net_cost']:,.2f}</td>
            <td>CHF {shift_20['net_cost'] * annual_factor:,.0f}</td>
            <td>CHF {shift_20['savings'] * annual_factor:,.0f}/yr</td>
            <td>{shift_20['reduction_pct']:.1f}%</td>
        </tr>
        <tr>
            <td>Shift 30% to low tariff</td>
            <td>CHF {shift_30['net_cost']:,.2f}</td>
            <td>CHF {shift_30['net_cost'] * annual_factor:,.0f}</td>
            <td>CHF {shift_30['savings'] * annual_factor:,.0f}/yr</td>
            <td>{shift_30['reduction_pct']:.1f}%</td>
        </tr>
        <tr>
            <td><strong>Combined (load + battery)</strong></td>
            <td><strong>CHF {combined['net_cost']:,.2f}</strong></td>
            <td><strong>CHF {combined['net_cost'] * annual_factor:,.0f}</strong></td>
            <td><strong>CHF {combined['total_savings'] * annual_factor:,.0f}/yr</strong></td>
            <td><strong>{combined['reduction_pct']:.1f}%</strong></td>
        </tr>
    </table>

    <h3>Recommendations</h3>
    <ul>
        <li><strong>Load shifting</strong>: Schedule heating comfort mode to start during solar hours
            and extend into low-tariff evening periods.</li>
        <li><strong>High-tariff avoidance</strong>: Reduce $E_{{import}}$ during 06:00-21:00 weekdays
            by pre-heating with PV or using stored heat/battery.</li>
        <li><strong>Potential savings</strong>: {combined['reduction_pct']:.0f}% reduction (~CHF {combined['total_savings'] * annual_factor:,.0f}/year)
            achievable through combined load shifting and battery optimization.</li>
        <li><strong>Rate differential</strong>: $(p_{{HT}} - p_{{NT}}) \\approx$ {shift_20['rate_differential']:.1f} Rp/kWh
            provides economic incentive for time-shifting.</li>
    </ul>

    <figure>
        <img src="fig21_tariff_cost_model.png" alt="Tariff Cost Model">
        <figcaption><strong>Figure 21:</strong> Cost analysis: monthly costs (top-left), tariff breakdown (top-right),
        daily profile (bottom-left), optimization scenarios (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for tariff cost modeling."""
    print("="*60)
    print("Phase 3, Step 4: Tariff Cost Model")
    print("="*60)

    # Load data
    energy_15min, tariff_series, integrated = load_data()

    # Calculate hourly costs
    hourly_costs = calculate_hourly_costs(energy_15min, tariff_series)

    # Analyze cost patterns
    cost_patterns, daily_costs = analyze_cost_patterns(hourly_costs)

    # Build forecast model
    forecast_results = build_cost_forecast_model(daily_costs, integrated)

    # Simulate optimization scenarios
    scenarios = simulate_cost_scenarios(hourly_costs, cost_patterns)

    # Create visualizations
    create_cost_analysis_plots(hourly_costs, daily_costs, cost_patterns, scenarios)

    # Save daily stats
    daily_costs.to_csv(OUTPUT_DIR / 'cost_model_daily_stats.csv')
    print("\nSaved: cost_model_daily_stats.csv")

    # Save forecast model
    if forecast_results:
        import json
        with open(OUTPUT_DIR / 'cost_forecast_model.json', 'w') as f:
            json.dump(forecast_results, f, indent=2, default=str)
        print("Saved: cost_forecast_model.json")

    # Generate report section
    report_html = generate_report(cost_patterns, forecast_results, scenarios, hourly_costs)
    with open(OUTPUT_DIR / 'tariff_cost_model_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: tariff_cost_model_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("TARIFF COST MODEL SUMMARY")
    print("="*60)

    daily = cost_patterns['daily_stats']
    print(f"\nDaily Costs:")
    print(f"  Grid cost: CHF {daily['mean_cost']:.2f}")
    print(f"  Feed-in revenue: CHF {daily['mean_revenue']:.2f}")
    print(f"  Net cost: CHF {daily['mean_net']:.2f}")

    print(f"\nAnnual Projection: CHF {cost_patterns['annual_projection']:,.0f}")

    print(f"\nOptimization Potential:")
    print(f"  Combined scenario: {scenarios['combined_optimization']['reduction_pct']:.1f}% cost reduction")


if __name__ == '__main__':
    main()
