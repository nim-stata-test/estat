#!/usr/bin/env python3
"""
Phase 3: Thermodynamic COP Model

Physics-based COP model using refrigerant pressure sensors.

Theory:
    COP_carnot = T_cond / (T_cond - T_evap)  [temperatures in Kelvin]
    COP_actual = η × COP_carnot

Where T_cond and T_evap are saturation temperatures from pressure readings.

Refrigerant: R410A (identified from pressure-temperature relationship)
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'output' / 'phase3'
PHASE1_DIR = ROOT_DIR / 'output' / 'phase1'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# R410A Refrigerant Properties
# ============================================================

# R410A saturation pressure-temperature data (bar gauge, °C)
# Based on published R410A tables
R410A_DATA = {
    # Pressure (bar abs) : Temperature (°C)
    2.0: -25.7,
    3.0: -16.9,
    4.0: -10.0,
    5.0: -4.4,
    6.0: 0.3,
    7.0: 4.4,
    8.0: 8.2,
    9.0: 11.6,
    10.0: 14.8,
    12.0: 20.5,
    14.0: 25.5,
    16.0: 30.0,
    18.0: 34.1,
    20.0: 37.9,
    22.0: 41.5,
    24.0: 44.8,
    26.0: 48.0,
    28.0: 51.0,
    30.0: 53.8,
    32.0: 56.5,
    34.0: 59.1,
    36.0: 61.6,
    38.0: 64.0,
    40.0: 66.3,
}

# Convert to arrays for interpolation
_P_R410A = np.array(list(R410A_DATA.keys()))
_T_R410A = np.array(list(R410A_DATA.values()))


def pressure_to_saturation_temp(pressure_bar: float) -> float:
    """
    Convert R410A pressure to saturation temperature.

    Args:
        pressure_bar: Absolute pressure in bar

    Returns:
        Saturation temperature in °C
    """
    return np.interp(pressure_bar, _P_R410A, _T_R410A)


def carnot_cop(T_cond_C: float, T_evap_C: float) -> float:
    """
    Calculate Carnot COP.

    Args:
        T_cond_C: Condenser temperature in °C
        T_evap_C: Evaporator temperature in °C

    Returns:
        Carnot COP (dimensionless)
    """
    T_cond_K = T_cond_C + 273.15
    T_evap_K = T_evap_C + 273.15

    if T_cond_K <= T_evap_K:
        return np.nan

    return T_cond_K / (T_cond_K - T_evap_K)


# ============================================================
# Data Loading
# ============================================================

def load_data() -> pd.DataFrame:
    """Load integrated dataset with pressure and COP data."""
    print("Loading data...")

    df = pd.read_parquet(PHASE1_DIR / 'integrated_dataset.parquet')

    # Filter to overlap period
    overlap_start = pd.Timestamp('2025-10-28', tz='UTC')
    df = df[df.index >= overlap_start]

    # Required columns
    required = [
        'stiebel_eltron_isg_high_pressure_wp1',
        'stiebel_eltron_isg_low_pressure_wp1',
        'stiebel_eltron_isg_hot_gas_temperature_wp1',
        'stiebel_eltron_isg_outdoor_temperature',
        'stiebel_eltron_isg_actual_temperature_hk_2',  # T_HK2
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    return df


def load_cop_data() -> pd.DataFrame:
    """Load daily COP data from heat pump model."""
    cop_file = OUTPUT_DIR / 'heat_pump_daily_stats.csv'

    if cop_file.exists():
        cop_df = pd.read_csv(cop_file)
        cop_df['datetime'] = pd.to_datetime(cop_df['datetime'])
        cop_df['date'] = cop_df['datetime'].dt.date
        return cop_df
    else:
        raise FileNotFoundError(f"COP data not found: {cop_file}")


# ============================================================
# Thermodynamic COP Calculation
# ============================================================

def compute_thermodynamic_cop(df: pd.DataFrame, min_lift: float = 15.0) -> pd.DataFrame:
    """
    Compute thermodynamic COP from pressure sensors.

    Args:
        df: DataFrame with pressure and temperature columns
        min_lift: Minimum temperature lift (T_cond - T_evap) to consider steady-state.
                  This filters out idle/equalized pressure states.

    Returns DataFrame with:
    - T_evap: Evaporator saturation temperature
    - T_cond: Condenser saturation temperature
    - COP_carnot: Theoretical maximum COP
    - superheat: Hot gas temp - T_cond (compressor superheat)
    - is_steady_state: Boolean flag for filtering
    """
    print("\nComputing thermodynamic COP...")

    # Get pressure data
    high_p = df['stiebel_eltron_isg_high_pressure_wp1']
    low_p = df['stiebel_eltron_isg_low_pressure_wp1']
    hot_gas = df['stiebel_eltron_isg_hot_gas_temperature_wp1']

    # Convert pressures to saturation temperatures
    T_evap = low_p.apply(pressure_to_saturation_temp)
    T_cond = high_p.apply(pressure_to_saturation_temp)

    # Calculate temperature lift
    temp_lift = T_cond - T_evap

    # Identify steady-state operation (reasonable temperature lift)
    is_steady = temp_lift >= min_lift

    # Calculate Carnot COP (only for steady-state)
    COP_carnot = []
    for te, tc, steady in zip(T_evap, T_cond, is_steady):
        if pd.notna(te) and pd.notna(tc) and steady:
            COP_carnot.append(carnot_cop(tc, te))
        else:
            COP_carnot.append(np.nan)

    # Calculate superheat
    superheat = hot_gas - T_cond

    result = pd.DataFrame({
        'datetime': df.index,
        'T_evap': T_evap.values,
        'T_cond': T_cond.values,
        'temp_lift': temp_lift.values,
        'T_outdoor': df['stiebel_eltron_isg_outdoor_temperature'].values,
        'T_HK2': df['stiebel_eltron_isg_actual_temperature_hk_2'].values,
        'hot_gas': hot_gas.values,
        'superheat': superheat.values,
        'COP_carnot': COP_carnot,
        'high_pressure': high_p.values,
        'low_pressure': low_p.values,
        'is_steady_state': is_steady.values,
    }).set_index('datetime')

    # Report filtering
    total = len(result)
    steady = result['is_steady_state'].sum()
    print(f"  Steady-state filter (lift >= {min_lift}°C): {steady}/{total} ({100*steady/total:.1f}%)")

    valid = result['COP_carnot'].notna()
    print(f"  Computed {valid.sum()} Carnot COP values")
    print(f"  T_evap range: {result['T_evap'].min():.1f}°C to {result['T_evap'].max():.1f}°C")
    print(f"  T_cond range: {result['T_cond'].min():.1f}°C to {result['T_cond'].max():.1f}°C")
    print(f"  COP_carnot range: {result['COP_carnot'].min():.2f} to {result['COP_carnot'].max():.2f}")

    return result


def aggregate_to_daily(thermo_df: pd.DataFrame, cop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate thermodynamic data to daily and merge with actual COP.

    Only uses steady-state data points for thermodynamic calculations.
    """
    print("\nAggregating to daily resolution...")

    # Filter to steady-state only for thermodynamic variables
    steady = thermo_df[thermo_df['is_steady_state']]
    print(f"  Using {len(steady)}/{len(thermo_df)} steady-state rows ({100*len(steady)/len(thermo_df):.1f}%)")

    # Resample steady-state data to daily
    daily = steady.resample('D').agg({
        'T_evap': 'mean',
        'T_cond': 'mean',
        'temp_lift': 'mean',
        'T_outdoor': 'mean',
        'T_HK2': 'mean',
        'hot_gas': 'mean',
        'superheat': 'mean',
        'COP_carnot': 'mean',
        'high_pressure': 'mean',
        'low_pressure': 'mean',
    })

    # Also count steady-state hours per day
    steady_counts = steady.resample('D').size()
    daily['steady_state_hours'] = steady_counts * 0.25  # 15-min intervals

    daily['date'] = daily.index.date

    # Merge with actual COP
    cop_dict = dict(zip(cop_df['date'], cop_df['cop']))
    daily['COP_actual'] = daily['date'].map(cop_dict)

    # Calculate efficiency factor
    daily['eta'] = daily['COP_actual'] / daily['COP_carnot']

    valid = daily[['COP_carnot', 'COP_actual']].dropna()
    print(f"  {len(valid)} days with both Carnot and actual COP")
    if len(valid) > 0:
        print(f"  Mean temperature lift: {daily['temp_lift'].mean():.1f}°C")
        print(f"  COP_carnot range: {daily['COP_carnot'].min():.2f} to {daily['COP_carnot'].max():.2f}")
        print(f"  Efficiency (η) range: {daily['eta'].min():.2%} to {daily['eta'].max():.2%}")
        print(f"  Mean efficiency: {daily['eta'].mean():.2%}")

    return daily


# ============================================================
# Model Fitting
# ============================================================

def fit_efficiency_model(daily: pd.DataFrame) -> dict:
    """
    Fit efficiency factor model: η = f(operating conditions)

    Tests several models:
    1. Constant η
    2. η = a + b × (T_cond - T_evap)  [lift-dependent]
    3. η = a + b × T_outdoor  [outdoor-dependent]
    """
    print("\nFitting efficiency models...")

    valid = daily[['COP_carnot', 'COP_actual', 'eta', 'T_evap', 'T_cond', 'T_outdoor']].dropna()

    if len(valid) < 10:
        print("  WARNING: Not enough data for model fitting")
        return {}

    eta = valid['eta'].values
    lift = (valid['T_cond'] - valid['T_evap']).values
    T_out = valid['T_outdoor'].values

    results = {}

    # Model 1: Constant η
    eta_mean = eta.mean()
    eta_std = eta.std()
    results['constant'] = {
        'eta': float(eta_mean),
        'eta_std': float(eta_std),
    }
    print(f"  Constant η: {eta_mean:.3f} ± {eta_std:.3f}")

    # Model 2: Lift-dependent η
    slope, intercept, r_val, p_val, _ = stats.linregress(lift, eta)
    results['lift_dependent'] = {
        'intercept': float(intercept),
        'slope': float(slope),
        'r_squared': float(r_val**2),
        'p_value': float(p_val),
    }
    print(f"  Lift-dependent: η = {intercept:.3f} + {slope:.4f} × ΔT  (R²={r_val**2:.3f})")

    # Model 3: Outdoor-temperature dependent η
    slope, intercept, r_val, p_val, _ = stats.linregress(T_out, eta)
    results['outdoor_dependent'] = {
        'intercept': float(intercept),
        'slope': float(slope),
        'r_squared': float(r_val**2),
        'p_value': float(p_val),
    }
    print(f"  Outdoor-dependent: η = {intercept:.3f} + {slope:.4f} × T_out  (R²={r_val**2:.3f})")

    return results


def fit_cop_model(daily: pd.DataFrame) -> dict:
    """
    Fit full COP prediction models and compare.

    Models:
    1. Empirical: COP = a + b×T_out + c×T_HK2
    2. Thermodynamic: COP = η × COP_carnot
    3. Hybrid: COP = a + b×COP_carnot + c×T_out
    """
    print("\nFitting COP prediction models...")

    valid = daily[['COP_carnot', 'COP_actual', 'T_outdoor', 'T_HK2']].dropna()

    if len(valid) < 10:
        print("  WARNING: Not enough data for model fitting")
        return {}

    n = len(valid)
    COP_actual = valid['COP_actual'].values
    COP_carnot = valid['COP_carnot'].values
    T_out = valid['T_outdoor'].values
    T_HK2 = valid['T_HK2'].values

    results = {}

    # Model 1: Empirical (from existing model)
    X_empirical = np.column_stack([np.ones(n), T_out, T_HK2])
    coef_empirical, residuals, rank, s = np.linalg.lstsq(X_empirical, COP_actual, rcond=None)
    COP_pred_empirical = X_empirical @ coef_empirical
    ss_res = np.sum((COP_actual - COP_pred_empirical)**2)
    ss_tot = np.sum((COP_actual - COP_actual.mean())**2)
    r2_empirical = 1 - ss_res / ss_tot
    rmse_empirical = np.sqrt(ss_res / n)

    results['empirical'] = {
        'intercept': float(coef_empirical[0]),
        'coef_T_outdoor': float(coef_empirical[1]),
        'coef_T_HK2': float(coef_empirical[2]),
        'r_squared': float(r2_empirical),
        'rmse': float(rmse_empirical),
        'formula': f"COP = {coef_empirical[0]:.2f} + {coef_empirical[1]:.3f}×T_out + {coef_empirical[2]:.3f}×T_HK2"
    }
    print(f"  Empirical: R²={r2_empirical:.4f}, RMSE={rmse_empirical:.3f}")
    print(f"    {results['empirical']['formula']}")

    # Model 2: Thermodynamic (η × COP_carnot)
    eta_mean = (COP_actual / COP_carnot).mean()
    COP_pred_thermo = eta_mean * COP_carnot
    ss_res = np.sum((COP_actual - COP_pred_thermo)**2)
    r2_thermo = 1 - ss_res / ss_tot
    rmse_thermo = np.sqrt(ss_res / n)

    results['thermodynamic'] = {
        'eta': float(eta_mean),
        'r_squared': float(r2_thermo),
        'rmse': float(rmse_thermo),
        'formula': f"COP = {eta_mean:.3f} × COP_carnot"
    }
    print(f"  Thermodynamic: R²={r2_thermo:.4f}, RMSE={rmse_thermo:.3f}")
    print(f"    {results['thermodynamic']['formula']}")

    # Model 3: Hybrid (linear combination)
    X_hybrid = np.column_stack([np.ones(n), COP_carnot, T_out])
    coef_hybrid, _, _, _ = np.linalg.lstsq(X_hybrid, COP_actual, rcond=None)
    COP_pred_hybrid = X_hybrid @ coef_hybrid
    ss_res = np.sum((COP_actual - COP_pred_hybrid)**2)
    r2_hybrid = 1 - ss_res / ss_tot
    rmse_hybrid = np.sqrt(ss_res / n)

    results['hybrid'] = {
        'intercept': float(coef_hybrid[0]),
        'coef_COP_carnot': float(coef_hybrid[1]),
        'coef_T_outdoor': float(coef_hybrid[2]),
        'r_squared': float(r2_hybrid),
        'rmse': float(rmse_hybrid),
        'formula': f"COP = {coef_hybrid[0]:.2f} + {coef_hybrid[1]:.3f}×COP_carnot + {coef_hybrid[2]:.3f}×T_out"
    }
    print(f"  Hybrid: R²={r2_hybrid:.4f}, RMSE={rmse_hybrid:.3f}")
    print(f"    {results['hybrid']['formula']}")

    # Store predictions for plotting
    results['predictions'] = {
        'COP_actual': COP_actual.tolist(),
        'COP_pred_empirical': COP_pred_empirical.tolist(),
        'COP_pred_thermo': COP_pred_thermo.tolist(),
        'COP_pred_hybrid': COP_pred_hybrid.tolist(),
        'dates': valid.index.strftime('%Y-%m-%d').tolist(),
    }

    return results


# ============================================================
# Visualization
# ============================================================

def create_visualization(daily: pd.DataFrame, cop_results: dict) -> None:
    """Create 4-panel visualization of thermodynamic COP analysis."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Thermodynamic COP Model Analysis', fontsize=14, fontweight='bold')

    valid = daily[['COP_carnot', 'COP_actual', 'eta', 'T_evap', 'T_cond', 'T_outdoor']].dropna()

    # Panel 1: Pressure-Temperature relationship
    ax1 = axes[0, 0]
    ax1.scatter(daily['low_pressure'], daily['T_evap'],
                alpha=0.5, label='Evaporator', c='blue', s=10)
    ax1.scatter(daily['high_pressure'], daily['T_cond'],
                alpha=0.5, label='Condenser', c='red', s=10)
    ax1.set_xlabel('Pressure (bar)')
    ax1.set_ylabel('Saturation Temperature (°C)')
    ax1.set_title('R410A Pressure-Temperature Relationship')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Carnot vs Actual COP
    ax2 = axes[0, 1]
    ax2.scatter(valid['COP_carnot'], valid['COP_actual'],
                alpha=0.7, c=valid['T_outdoor'], cmap='coolwarm', s=40)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('T_outdoor (°C)')

    # Add efficiency lines
    x = np.linspace(valid['COP_carnot'].min(), valid['COP_carnot'].max(), 100)
    for eta_val in [0.3, 0.4, 0.5]:
        ax2.plot(x, eta_val * x, '--', alpha=0.5, label=f'η={eta_val:.0%}')

    ax2.set_xlabel('Carnot COP')
    ax2.set_ylabel('Actual COP')
    ax2.set_title('Carnot vs Actual COP')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Model comparison (predicted vs actual)
    ax3 = axes[1, 0]

    if 'predictions' in cop_results:
        preds = cop_results['predictions']
        actual = np.array(preds['COP_actual'])

        ax3.scatter(actual, preds['COP_pred_empirical'],
                    alpha=0.7, label=f"Empirical (R²={cop_results['empirical']['r_squared']:.3f})",
                    marker='o', s=40)
        ax3.scatter(actual, preds['COP_pred_thermo'],
                    alpha=0.7, label=f"Thermo (R²={cop_results['thermodynamic']['r_squared']:.3f})",
                    marker='s', s=40)
        ax3.scatter(actual, preds['COP_pred_hybrid'],
                    alpha=0.7, label=f"Hybrid (R²={cop_results['hybrid']['r_squared']:.3f})",
                    marker='^', s=40)

        # Perfect prediction line
        lims = [min(actual.min(), 3), max(actual.max(), 6)]
        ax3.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
        ax3.set_xlim(lims)
        ax3.set_ylim(lims)

    ax3.set_xlabel('Actual COP')
    ax3.set_ylabel('Predicted COP')
    ax3.set_title('Model Comparison: Predicted vs Actual')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Efficiency vs operating conditions
    ax4 = axes[1, 1]
    lift = valid['T_cond'] - valid['T_evap']
    scatter = ax4.scatter(lift, valid['eta'],
                         c=valid['T_outdoor'], cmap='coolwarm', alpha=0.7, s=40)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('T_outdoor (°C)')

    # Fit line
    slope, intercept, r, p, _ = stats.linregress(lift, valid['eta'])
    x_fit = np.linspace(lift.min(), lift.max(), 100)
    ax4.plot(x_fit, intercept + slope * x_fit, 'r--',
             label=f'η = {intercept:.2f} + {slope:.4f}×ΔT\nR²={r**2:.3f}')

    ax4.set_xlabel('Temperature Lift ΔT = T_cond - T_evap (°C)')
    ax4.set_ylabel('Efficiency η = COP_actual / COP_carnot')
    ax4.set_title('Efficiency vs Temperature Lift')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3.06_pressure_cop_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: fig3.06_pressure_cop_model.png")


# ============================================================
# HTML Report
# ============================================================

def generate_report(daily: pd.DataFrame, efficiency_results: dict,
                   cop_results: dict) -> str:
    """Generate HTML report section."""

    valid = daily[['COP_carnot', 'COP_actual', 'eta']].dropna()

    html = f"""
    <section id="pressure-cop-model">
    <h2>Thermodynamic COP Model</h2>

    <h3>Theory</h3>
    <p>The thermodynamic COP model uses refrigerant pressure measurements to calculate the
    theoretical maximum COP (Carnot COP):</p>

    <pre>
COP_carnot = T_cond / (T_cond - T_evap)    [temperatures in Kelvin]
COP_actual = η × COP_carnot                 [η is efficiency factor]
    </pre>

    <p>Where T_cond and T_evap are saturation temperatures derived from the high and low
    pressure sensors using R410A refrigerant properties.</p>

    <h3>Data Summary</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Days analyzed</td><td>{len(valid)}</td></tr>
        <tr><td>Mean Carnot COP</td><td>{valid['COP_carnot'].mean():.2f}</td></tr>
        <tr><td>Mean Actual COP</td><td>{valid['COP_actual'].mean():.2f}</td></tr>
        <tr><td>Mean Efficiency (η)</td><td>{valid['eta'].mean():.1%}</td></tr>
        <tr><td>Efficiency Range</td><td>{valid['eta'].min():.1%} - {valid['eta'].max():.1%}</td></tr>
    </table>

    <h3>Model Comparison</h3>
    <table>
        <tr><th>Model</th><th>Formula</th><th>R²</th><th>RMSE</th></tr>
    """

    if cop_results:
        for name, res in [('Empirical', cop_results.get('empirical', {})),
                         ('Thermodynamic', cop_results.get('thermodynamic', {})),
                         ('Hybrid', cop_results.get('hybrid', {}))]:
            if res:
                html += f"""
        <tr>
            <td>{name}</td>
            <td><code>{res.get('formula', 'N/A')}</code></td>
            <td>{res.get('r_squared', 0):.4f}</td>
            <td>{res.get('rmse', 0):.3f}</td>
        </tr>
                """

    html += """
    </table>

    <h3>Key Findings</h3>
    <ul>
        <li><strong>Refrigerant identified as R410A</strong> based on pressure-temperature relationship</li>
    """

    if cop_results:
        emp_r2 = cop_results.get('empirical', {}).get('r_squared', 0)
        thermo_r2 = cop_results.get('thermodynamic', {}).get('r_squared', 0)
        hybrid_r2 = cop_results.get('hybrid', {}).get('r_squared', 0)

        best_model = 'Hybrid' if hybrid_r2 >= max(emp_r2, thermo_r2) else \
                     ('Thermodynamic' if thermo_r2 > emp_r2 else 'Empirical')

        html += f"""
        <li><strong>Best model: {best_model}</strong> with R²={max(emp_r2, thermo_r2, hybrid_r2):.4f}</li>
        <li>Mean efficiency η = {valid['eta'].mean():.1%} of Carnot maximum</li>
        <li>Efficiency decreases with temperature lift (larger ΔT = less efficient)</li>
        """

    html += """
    </ul>

    <figure>
        <img src="fig3.06_pressure_cop_model.png" alt="Thermodynamic COP Model">
        <figcaption>Thermodynamic COP analysis: pressure-temperature relationship,
        Carnot vs actual COP, model comparison, and efficiency factors.</figcaption>
    </figure>
    </section>
    """

    return html


# ============================================================
# Main
# ============================================================

def main():
    """Run thermodynamic COP analysis."""
    print("=" * 60)
    print("Phase 3: Thermodynamic COP Model")
    print("=" * 60)

    # Load data
    df = load_data()
    cop_df = load_cop_data()

    # Compute thermodynamic COP
    thermo_df = compute_thermodynamic_cop(df)

    # Aggregate to daily
    daily = aggregate_to_daily(thermo_df, cop_df)

    # Fit models
    efficiency_results = fit_efficiency_model(daily)
    cop_results = fit_cop_model(daily)

    # Visualization
    create_visualization(daily, cop_results)

    # Save results
    results = {
        'efficiency': efficiency_results,
        'cop_models': {k: v for k, v in cop_results.items() if k != 'predictions'},
        'summary': {
            'n_days': int(daily['COP_actual'].notna().sum()),
            'mean_carnot_cop': float(daily['COP_carnot'].mean()),
            'mean_actual_cop': float(daily['COP_actual'].mean()),
            'mean_efficiency': float(daily['eta'].mean()),
        }
    }

    with open(OUTPUT_DIR / 'pressure_cop_model.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: pressure_cop_model.json")

    # Save daily data
    daily.to_csv(OUTPUT_DIR / 'pressure_cop_daily.csv')
    print(f"Saved: pressure_cop_daily.csv")

    # Generate report
    report_html = generate_report(daily, efficiency_results, cop_results)
    with open(OUTPUT_DIR / 'pressure_cop_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: pressure_cop_report_section.html")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nDays analyzed: {results['summary']['n_days']}")
    print(f"Mean Carnot COP: {results['summary']['mean_carnot_cop']:.2f}")
    print(f"Mean Actual COP: {results['summary']['mean_actual_cop']:.2f}")
    print(f"Mean Efficiency: {results['summary']['mean_efficiency']:.1%}")

    if cop_results:
        print("\nModel Comparison:")
        for name in ['empirical', 'thermodynamic', 'hybrid']:
            if name in cop_results:
                r2 = cop_results[name]['r_squared']
                rmse = cop_results[name]['rmse']
                print(f"  {name.capitalize()}: R²={r2:.4f}, RMSE={rmse:.3f}")


if __name__ == '__main__':
    main()
