#!/usr/bin/env python3
"""
Phase 3, Step 2: Heat Pump Model

Extends the Phase 2 heating curve analysis with:
- COP vs flow temperature relationship
- COP vs outdoor temperature relationship
- Capacity constraints and modulation behavior
- Buffer tank charging/discharging dynamics

Builds on existing findings:
- Mean COP: 3.55 (range 2.49-5.18)
- Heating curve: T_target = T_setpoint + 1.08 × (T_ref - T_outdoor)
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
    """Load sensor data for heat pump analysis."""
    print("Loading data for heat pump modeling...")

    # Load heating sensors - long format
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)

    # Pivot to wide format
    heating = heating_raw.pivot_table(
        values='value',
        index='datetime',
        columns='entity_id',
        aggfunc='mean'
    )

    # Resample to 15-min intervals for cleaner analysis
    heating = heating.resample('15min').mean()

    print(f"  Heating sensors: {len(heating):,} rows, {heating.shape[1]} columns")
    print(f"  Date range: {heating.index.min()} to {heating.index.max()}")

    return heating


def calculate_cop(heating: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate COP (Coefficient of Performance) at different time scales.

    COP = Heat Produced / Electricity Consumed

    Uses cumulative energy counters with proper differencing.
    """
    print("\nCalculating COP...")

    # Key columns
    consumed_col = 'stiebel_eltron_isg_consumed_heating_today'
    produced_col = 'stiebel_eltron_isg_produced_heating_today'
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    flow_col = 'stiebel_eltron_isg_flow_temperature_wp1'
    buffer_col = 'stiebel_eltron_isg_actual_temperature_buffer'
    compressor_col = 'stiebel_eltron_isg_compressor'

    # Check available columns
    available = []
    for col in [consumed_col, produced_col, outdoor_col, flow_col, buffer_col]:
        if col in heating.columns:
            available.append(col)
        else:
            print(f"  Warning: {col} not found")

    if consumed_col not in heating.columns or produced_col not in heating.columns:
        print("  Cannot calculate COP without energy counters")
        return pd.DataFrame()

    df = heating[available].copy()

    # Daily energy values (counters reset daily, take max per day)
    daily = df.resample('D').agg({
        consumed_col: 'max',
        produced_col: 'max',
    })

    if outdoor_col in df.columns:
        daily['T_outdoor'] = df[outdoor_col].resample('D').mean()

    if flow_col in df.columns:
        daily['T_flow'] = df[flow_col].resample('D').mean()

    if buffer_col in df.columns:
        daily['T_buffer'] = df[buffer_col].resample('D').mean()

    # Calculate daily COP
    daily['consumed'] = daily[consumed_col]
    daily['produced'] = daily[produced_col]

    # COP only valid when both consumed and produced are positive
    mask = (daily['consumed'] > 0.5) & (daily['produced'] > 0.5)
    daily.loc[mask, 'cop'] = daily.loc[mask, 'produced'] / daily.loc[mask, 'consumed']

    # Remove outliers (COP should be 1-7 for heat pumps)
    daily.loc[(daily['cop'] < 1.0) | (daily['cop'] > 7.0), 'cop'] = np.nan

    valid_days = daily['cop'].notna().sum()
    print(f"  Valid COP days: {valid_days}")
    if valid_days > 0:
        print(f"  Mean COP: {daily['cop'].mean():.2f}")
        print(f"  COP range: {daily['cop'].min():.2f} - {daily['cop'].max():.2f}")

    return daily


def analyze_cop_relationships(daily: pd.DataFrame) -> dict:
    """
    Analyze COP vs temperature relationships.

    Key relationships:
    - COP vs outdoor temperature (Carnot limit effect)
    - COP vs flow temperature (delivery temperature effect)
    """
    print("\nAnalyzing COP relationships...")

    results = {}

    # COP vs Outdoor Temperature
    if 'T_outdoor' in daily.columns:
        mask = daily['cop'].notna() & daily['T_outdoor'].notna()
        if mask.sum() >= 10:
            X = daily.loc[mask, 'T_outdoor'].values.reshape(-1, 1)
            y = daily.loc[mask, 'cop'].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            results['cop_vs_outdoor'] = {
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'r2': r2_score(y, y_pred),
                'n_points': len(y)
            }

            print(f"  COP vs T_outdoor:")
            print(f"    Slope: {reg.coef_[0]:.4f} COP/°C")
            print(f"    Base COP (at 0°C): {reg.intercept_:.2f}")
            print(f"    R²: {r2_score(y, y_pred):.3f}")

    # COP vs Flow Temperature
    if 'T_flow' in daily.columns:
        mask = daily['cop'].notna() & daily['T_flow'].notna()
        if mask.sum() >= 10:
            X = daily.loc[mask, 'T_flow'].values.reshape(-1, 1)
            y = daily.loc[mask, 'cop'].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            results['cop_vs_flow'] = {
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'r2': r2_score(y, y_pred),
                'n_points': len(y)
            }

            print(f"  COP vs T_flow:")
            print(f"    Slope: {reg.coef_[0]:.4f} COP/°C")
            print(f"    Base COP (at 0°C): {reg.intercept_:.2f}")
            print(f"    R²: {r2_score(y, y_pred):.3f}")

    # Multi-variable: COP = f(T_outdoor, T_flow)
    if 'T_outdoor' in daily.columns and 'T_flow' in daily.columns:
        mask = daily['cop'].notna() & daily['T_outdoor'].notna() & daily['T_flow'].notna()
        if mask.sum() >= 15:
            X = daily.loc[mask, ['T_outdoor', 'T_flow']].values
            y = daily.loc[mask, 'cop'].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            results['cop_multivar'] = {
                'outdoor_coef': reg.coef_[0],
                'flow_coef': reg.coef_[1],
                'intercept': reg.intercept_,
                'r2': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'n_points': len(y)
            }

            print(f"  COP = {reg.intercept_:.2f} + {reg.coef_[0]:.4f}×T_outdoor + {reg.coef_[1]:.4f}×T_flow")
            print(f"    R²: {r2_score(y, y_pred):.3f}")

    return results


def analyze_capacity_modulation(heating: pd.DataFrame) -> dict:
    """
    Analyze heat pump capacity and modulation behavior.

    Looks at:
    - Maximum heating capacity (kWh/day)
    - Compressor run time
    - Heating power during operation
    """
    print("\nAnalyzing capacity and modulation...")

    results = {}

    # Daily heating energy
    consumed_col = 'stiebel_eltron_isg_consumed_heating_today'
    produced_col = 'stiebel_eltron_isg_produced_heating_today'

    if consumed_col in heating.columns:
        daily_consumed = heating[consumed_col].resample('D').max()
        valid = daily_consumed[daily_consumed > 0]

        results['daily_consumed'] = {
            'mean': valid.mean(),
            'max': valid.max(),
            'min': valid.min(),
            'std': valid.std()
        }

        print(f"  Daily electricity consumed:")
        print(f"    Mean: {valid.mean():.1f} kWh")
        print(f"    Max: {valid.max():.1f} kWh")
        print(f"    Min: {valid.min():.1f} kWh")

    if produced_col in heating.columns:
        daily_produced = heating[produced_col].resample('D').max()
        valid = daily_produced[daily_produced > 0]

        results['daily_produced'] = {
            'mean': valid.mean(),
            'max': valid.max(),
            'min': valid.min(),
            'std': valid.std()
        }

        print(f"  Daily heat produced:")
        print(f"    Mean: {valid.mean():.1f} kWh")
        print(f"    Max: {valid.max():.1f} kWh")
        print(f"    Min: {valid.min():.1f} kWh")

    # Compressor run time analysis
    compressor_col = 'stiebel_eltron_isg_compressor'
    if compressor_col in heating.columns:
        # Compressor is typically 0/1
        heating_on = heating[compressor_col] > 0.5
        daily_run_hours = heating_on.resample('D').sum() * 0.25  # 15-min intervals

        valid = daily_run_hours[daily_run_hours > 0]

        results['compressor_runtime'] = {
            'mean_hours': valid.mean(),
            'max_hours': valid.max(),
            'duty_cycle': valid.mean() / 24.0  # Fraction of day
        }

        print(f"  Compressor runtime:")
        print(f"    Mean: {valid.mean():.1f} h/day")
        print(f"    Max: {valid.max():.1f} h/day")
        print(f"    Duty cycle: {100*valid.mean()/24:.1f}%")

    return results


def analyze_buffer_tank(heating: pd.DataFrame) -> dict:
    """
    Analyze buffer tank behavior.

    Looks at:
    - Temperature range and variability
    - Charging/discharging patterns
    - Relationship with heating demand
    """
    print("\nAnalyzing buffer tank dynamics...")

    buffer_col = 'stiebel_eltron_isg_actual_temperature_buffer'
    flow_col = 'stiebel_eltron_isg_flow_temperature_wp1'
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'

    if buffer_col not in heating.columns:
        print("  Buffer tank sensor not found")
        return {}

    results = {}

    buffer_temp = heating[buffer_col].dropna()

    results['buffer_stats'] = {
        'mean': buffer_temp.mean(),
        'min': buffer_temp.min(),
        'max': buffer_temp.max(),
        'std': buffer_temp.std()
    }

    print(f"  Buffer tank temperature:")
    print(f"    Mean: {buffer_temp.mean():.1f}°C")
    print(f"    Range: {buffer_temp.min():.1f} - {buffer_temp.max():.1f}°C")
    print(f"    Std: {buffer_temp.std():.2f}°C")

    # Temperature change rate (charging/discharging)
    buffer_rate = buffer_temp.diff() * 4  # Per hour (from 15-min intervals)
    results['buffer_dynamics'] = {
        'mean_rate': buffer_rate.mean(),
        'max_charging': buffer_rate.max(),
        'max_discharging': abs(buffer_rate.min()),
        'std_rate': buffer_rate.std()
    }

    print(f"  Buffer dynamics:")
    print(f"    Max charging rate: {buffer_rate.max():.2f}°C/h")
    print(f"    Max discharging rate: {abs(buffer_rate.min()):.2f}°C/h")

    # Buffer temperature vs outdoor temperature
    if outdoor_col in heating.columns:
        df = heating[[buffer_col, outdoor_col]].dropna()
        if len(df) > 50:
            corr = df.corr().iloc[0, 1]
            results['buffer_outdoor_corr'] = corr
            print(f"  Buffer-Outdoor correlation: {corr:.3f}")

    return results


def create_heat_pump_plots(daily: pd.DataFrame, heating: pd.DataFrame,
                          cop_results: dict, capacity_results: dict,
                          buffer_results: dict) -> None:
    """Create visualization of heat pump model results."""
    print("\nCreating heat pump model plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: COP vs Outdoor Temperature
    ax = axes[0, 0]
    if 'T_outdoor' in daily.columns and daily['cop'].notna().any():
        mask = daily['cop'].notna() & daily['T_outdoor'].notna()
        ax.scatter(daily.loc[mask, 'T_outdoor'], daily.loc[mask, 'cop'],
                   alpha=0.6, s=50, c='blue')

        if 'cop_vs_outdoor' in cop_results:
            r = cop_results['cop_vs_outdoor']
            x_range = np.array([daily['T_outdoor'].min(), daily['T_outdoor'].max()])
            y_line = r['intercept'] + r['slope'] * x_range
            ax.plot(x_range, y_line, 'r-', linewidth=2,
                    label=f"COP = {r['intercept']:.2f} + {r['slope']:.3f}×T_out (R²={r['r2']:.2f})")
            ax.legend()

    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('COP')
    ax.set_title('COP vs Outdoor Temperature')
    ax.grid(True, alpha=0.3)

    # Panel 2: COP vs Flow Temperature
    ax = axes[0, 1]
    if 'T_flow' in daily.columns and daily['cop'].notna().any():
        mask = daily['cop'].notna() & daily['T_flow'].notna()
        ax.scatter(daily.loc[mask, 'T_flow'], daily.loc[mask, 'cop'],
                   alpha=0.6, s=50, c='green')

        if 'cop_vs_flow' in cop_results:
            r = cop_results['cop_vs_flow']
            x_range = np.array([daily['T_flow'].min(), daily['T_flow'].max()])
            y_line = r['intercept'] + r['slope'] * x_range
            ax.plot(x_range, y_line, 'r-', linewidth=2,
                    label=f"COP = {r['intercept']:.2f} + {r['slope']:.3f}×T_flow (R²={r['r2']:.2f})")
            ax.legend()

    ax.set_xlabel('Flow Temperature (°C)')
    ax.set_ylabel('COP')
    ax.set_title('COP vs Flow Temperature')
    ax.grid(True, alpha=0.3)

    # Panel 3: Daily Energy Production
    ax = axes[1, 0]
    if 'consumed' in daily.columns and 'produced' in daily.columns:
        valid = daily[['consumed', 'produced']].dropna()
        dates = valid.index

        ax.bar(dates, valid['consumed'], alpha=0.7, label='Electricity Consumed', color='orange')
        ax.bar(dates, valid['produced'], alpha=0.4, label='Heat Produced', color='red')

        ax.set_ylabel('Energy (kWh/day)')
        ax.set_title('Daily Heating Energy')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    ax.grid(True, alpha=0.3)

    # Panel 4: Buffer Tank Temperature
    ax = axes[1, 1]
    buffer_col = 'stiebel_eltron_isg_actual_temperature_buffer'
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'

    if buffer_col in heating.columns:
        # Hourly averages for cleaner plot
        hourly = heating[[buffer_col]].resample('h').mean()
        ax.plot(hourly.index, hourly[buffer_col], label='Buffer Tank', color='red', alpha=0.8)

        if outdoor_col in heating.columns:
            hourly_out = heating[outdoor_col].resample('h').mean()
            ax.plot(hourly_out.index, hourly_out, label='Outdoor', color='blue', alpha=0.6)

        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Buffer Tank vs Outdoor Temperature')
        ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig14_heat_pump_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig14_heat_pump_model.png")


def generate_report(daily: pd.DataFrame, cop_results: dict,
                   capacity_results: dict, buffer_results: dict) -> str:
    """Generate HTML report section for heat pump model."""

    # Extract key values
    mean_cop = daily['cop'].mean() if 'cop' in daily.columns else 0
    cop_range = (daily['cop'].min(), daily['cop'].max()) if 'cop' in daily.columns else (0, 0)

    cop_outdoor_slope = cop_results.get('cop_vs_outdoor', {}).get('slope', 0)
    cop_flow_slope = cop_results.get('cop_vs_flow', {}).get('slope', 0)

    multivar = cop_results.get('cop_multivar', {})
    cop_formula = ""
    if multivar:
        cop_formula = f"COP = {multivar['intercept']:.2f} + {multivar['outdoor_coef']:.4f}×T_out + {multivar['flow_coef']:.4f}×T_flow"

    daily_consumed = capacity_results.get('daily_consumed', {})
    daily_produced = capacity_results.get('daily_produced', {})
    runtime = capacity_results.get('compressor_runtime', {})

    buffer_stats = buffer_results.get('buffer_stats', {})

    html = f"""
    <section id="heat-pump-model">
    <h2>3.2 Heat Pump Model</h2>

    <h3>COP Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Notes</th></tr>
        <tr>
            <td>Mean COP</td>
            <td><strong>{mean_cop:.2f}</strong></td>
            <td>Good efficiency for air-source heat pump</td>
        </tr>
        <tr>
            <td>COP Range</td>
            <td>{cop_range[0]:.2f} – {cop_range[1]:.2f}</td>
            <td>Varies with outdoor/flow temperature</td>
        </tr>
        <tr>
            <td>COP sensitivity to outdoor temp</td>
            <td>{cop_outdoor_slope:.4f} COP/°C</td>
            <td>{"Increases" if cop_outdoor_slope > 0 else "Decreases"} with warmer outdoor</td>
        </tr>
        <tr>
            <td>COP sensitivity to flow temp</td>
            <td>{cop_flow_slope:.4f} COP/°C</td>
            <td>{"Increases" if cop_flow_slope > 0 else "Decreases"} with higher flow temp</td>
        </tr>
    </table>

    <h3>COP Model</h3>
    <p>Multi-variable regression model:</p>
    <pre>{cop_formula if cop_formula else "Insufficient data for multi-variable model"}</pre>
    <p>R² = {multivar.get('r2', 0):.3f}, RMSE = {multivar.get('rmse', 0):.3f}</p>

    <h3>Capacity Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Mean</th><th>Max</th><th>Min</th></tr>
        <tr>
            <td>Daily electricity consumed</td>
            <td>{daily_consumed.get('mean', 0):.1f} kWh</td>
            <td>{daily_consumed.get('max', 0):.1f} kWh</td>
            <td>{daily_consumed.get('min', 0):.1f} kWh</td>
        </tr>
        <tr>
            <td>Daily heat produced</td>
            <td>{daily_produced.get('mean', 0):.1f} kWh</td>
            <td>{daily_produced.get('max', 0):.1f} kWh</td>
            <td>{daily_produced.get('min', 0):.1f} kWh</td>
        </tr>
        <tr>
            <td>Compressor runtime</td>
            <td>{runtime.get('mean_hours', 0):.1f} h/day</td>
            <td>{runtime.get('max_hours', 0):.1f} h/day</td>
            <td>—</td>
        </tr>
    </table>

    <h3>Buffer Tank Dynamics</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean temperature</td><td>{buffer_stats.get('mean', 0):.1f}°C</td></tr>
        <tr><td>Temperature range</td><td>{buffer_stats.get('min', 0):.1f} – {buffer_stats.get('max', 0):.1f}°C</td></tr>
        <tr><td>Temperature variability (std)</td><td>{buffer_stats.get('std', 0):.2f}°C</td></tr>
    </table>

    <h3>Implications for Optimization</h3>
    <ul>
        <li><strong>COP optimization</strong>: Lower flow temperatures improve COP.
            With slope {cop_flow_slope:.4f}, reducing flow by 5°C improves COP by ~{abs(cop_flow_slope*5):.2f}.</li>
        <li><strong>Timing strategy</strong>: Run heat pump during warmest outdoor temps (daytime/solar hours)
            for better COP. Each +1°C outdoor improves COP by ~{cop_outdoor_slope:.3f}.</li>
        <li><strong>Capacity headroom</strong>: Max observed {daily_consumed.get('max', 0):.0f} kWh/day
            suggests capacity is sufficient for current heating demand.</li>
        <li><strong>Buffer utilization</strong>: Buffer tank (mean {buffer_stats.get('mean', 0):.0f}°C)
            provides thermal storage for load shifting.</li>
    </ul>

    <figure>
        <img src="fig14_heat_pump_model.png" alt="Heat Pump Model Analysis">
        <figcaption>Heat pump analysis: COP vs outdoor temperature (top-left),
        COP vs flow temperature (top-right), daily energy (bottom-left),
        buffer tank dynamics (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for heat pump modeling."""
    print("="*60)
    print("Phase 3, Step 2: Heat Pump Model")
    print("="*60)

    # Load data
    heating = load_data()

    # Calculate COP
    daily = calculate_cop(heating)

    if daily.empty:
        print("ERROR: Could not calculate COP")
        return

    # Analyze COP relationships
    cop_results = analyze_cop_relationships(daily)

    # Analyze capacity and modulation
    capacity_results = analyze_capacity_modulation(heating)

    # Analyze buffer tank
    buffer_results = analyze_buffer_tank(heating)

    # Create visualizations
    create_heat_pump_plots(daily, heating, cop_results, capacity_results, buffer_results)

    # Save results
    daily.to_csv(OUTPUT_DIR / 'heat_pump_daily_stats.csv')
    print("\nSaved: heat_pump_daily_stats.csv")

    # Generate report section
    report_html = generate_report(daily, cop_results, capacity_results, buffer_results)
    with open(OUTPUT_DIR / 'heat_pump_model_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: heat_pump_model_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("HEAT PUMP MODEL SUMMARY")
    print("="*60)

    print(f"\nCOP Analysis:")
    print(f"  Mean COP: {daily['cop'].mean():.2f}")
    print(f"  COP range: {daily['cop'].min():.2f} - {daily['cop'].max():.2f}")

    if 'cop_multivar' in cop_results:
        mv = cop_results['cop_multivar']
        print(f"\nCOP Model (R²={mv['r2']:.3f}):")
        print(f"  COP = {mv['intercept']:.2f} + {mv['outdoor_coef']:.4f}×T_out + {mv['flow_coef']:.4f}×T_flow")

    print(f"\nCapacity:")
    if 'daily_produced' in capacity_results:
        dp = capacity_results['daily_produced']
        print(f"  Max daily heat: {dp['max']:.1f} kWh")
        print(f"  Mean daily heat: {dp['mean']:.1f} kWh")


if __name__ == '__main__':
    main()
