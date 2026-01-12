#!/usr/bin/env python3
"""
Phase 3, Step 1: Building Thermal Model

Estimates building thermal characteristics using a transfer function approach:
1. Model heating curve: HK2 = f(T_outdoor)
2. Calculate heating effort: deviation from heating curve
3. Model room temps: T_room = f(outdoor_smooth, effort_smooth, pv_smooth)

Each room has individual parameters for:
- τ_outdoor: Time constant for outdoor temperature response
- τ_effort: Time constant for heating effort response
- τ_pv: Time constant for solar gain response
- gain_outdoor, gain_effort, gain_pv: Response magnitudes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(exist_ok=True)

# Target sensor for thermal model (single sensor for simplicity)
TARGET_SENSORS = [
    'davis_inside_temperature',  # 100% - primary living area, least noise
]

# Single sensor for indoor temperature objective
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 1.0,
}

# Key sensor columns
HK2_COL = 'wp_anlage_hk2_ist'
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
PV_COL = 'pv_generation_kwh'


def exponential_smooth(x: np.ndarray, tau_steps: float) -> np.ndarray:
    """
    Apply exponential smoothing (first-order low-pass filter).

    Args:
        x: Input signal
        tau_steps: Time constant in number of time steps

    Returns:
        Smoothed signal
    """
    if tau_steps < 1:
        return x.copy()

    alpha = 1 - np.exp(-1/tau_steps)
    result = np.zeros_like(x, dtype=float)

    # Initialize with first valid value or mean
    first_valid = x[~np.isnan(x)][0] if any(~np.isnan(x)) else 0
    result[0] = x[0] if not np.isnan(x[0]) else first_valid

    for i in range(1, len(x)):
        if np.isnan(x[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]

    return result


def load_data() -> pd.DataFrame:
    """Load integrated dataset for thermal modeling."""
    print("Loading data for thermal modeling...")

    # Load full integrated dataset (more data coverage)
    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)

    print(f"  Dataset: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")

    return df


def load_heating_curve_params() -> dict:
    """
    Load parametric heating curve from Phase 2 analysis.

    The Phase 2 heating curve model accounts for controllable parameters:
    T_flow = setpoint + curve_rise × (T_ref - T_outdoor)

    Where:
    - setpoint: comfort or eco temperature setting
    - curve_rise: heating curve slope setting on heat pump
    - T_ref: reference temperature (estimated from data)

    Returns:
        dict with t_ref_comfort, t_ref_eco, and model statistics
    """
    import json

    params_file = PROCESSED_DIR.parent / 'phase2' / 'heating_curve_params.json'

    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        print(f"\nLoaded Phase 2 heating curve parameters:")
        print(f"  T_ref (comfort): {params['t_ref_comfort']:.2f}°C")
        print(f"  T_ref (eco): {params['t_ref_eco']:.2f}°C")
        print(f"  Model R² (normal): {params['normal_r_squared']:.3f}")
        print(f"  Model RMSE: {params['normal_rmse']:.2f}°C")
        return params
    else:
        print(f"\nWARNING: Phase 2 heating curve params not found at {params_file}")
        print("  Run src/phase2/03_heating_curve_analysis.py first")
        print("  Using default values...")
        return {
            't_ref_comfort': 21.32,
            't_ref_eco': 19.18,
            'normal_r_squared': 0.98,
            'normal_rmse': 0.57,
        }


def fit_heating_curve(df: pd.DataFrame) -> dict:
    """
    Fit simple heating curve for reference: HK2 = baseline + slope × T_outdoor

    NOTE: This is a simplified model that ignores controllable parameters.
    For optimization, use the parametric model from Phase 2:
    T_flow = setpoint + curve_rise × (T_ref - T_outdoor)

    Returns:
        dict with baseline, slope, r2, and phase2_params
    """
    print("\nFitting heating curve (simplified reference model)...")

    clean = df[[HK2_COL, OUTDOOR_COL]].dropna()

    if len(clean) < 100:
        print("  ERROR: Insufficient data for heating curve")
        return {}

    X = clean[OUTDOOR_COL].values.reshape(-1, 1)
    y = clean[HK2_COL].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Load Phase 2 parametric model for comparison
    phase2_params = load_heating_curve_params()

    result = {
        'baseline': model.intercept_,
        'slope': model.coef_[0],
        'r2': r2,
        'n_points': len(clean),
        'phase2_params': phase2_params,  # Include parametric model
    }

    print(f"\n  Simple model: HK2 = {result['baseline']:.1f} + {result['slope']:.3f} × T_outdoor")
    print(f"  Simple R² = {r2:.3f} ({len(clean):,} points)")
    print(f"\n  Phase 2 parametric model (R²={phase2_params['normal_r_squared']:.3f}):")
    print(f"  T_flow = setpoint + curve_rise × (T_ref - T_outdoor)")
    print(f"  Use this for optimization (accounts for setpoint/curve_rise changes)")

    return result


def compute_heating_effort(df: pd.DataFrame, heating_curve: dict) -> pd.Series:
    """
    Compute heating effort as deviation from heating curve.

    Heating effort = HK2_actual - HK2_expected

    Positive effort: system delivering MORE heat than curve suggests
    Negative effort: system delivering LESS heat than curve suggests
    """
    expected_hk2 = heating_curve['baseline'] + heating_curve['slope'] * df[OUTDOOR_COL]
    effort = df[HK2_COL] - expected_hk2
    return effort


def fit_room_model(df: pd.DataFrame, room_col: str, effort: pd.Series,
                   tau_out_h: float, tau_effort_h: float, tau_pv_h: float) -> dict:
    """
    Fit room temperature model with given time constants.

    Model: T_room = offset + g_out×LPF(T_out,τ_out) + g_eff×LPF(effort,τ_eff) + g_pv×LPF(PV,τ_pv)

    Args:
        df: DataFrame with sensor data
        room_col: Room temperature column name
        effort: Heating effort series
        tau_out_h: Time constant for outdoor response (hours)
        tau_effort_h: Time constant for heating effort response (hours)
        tau_pv_h: Time constant for solar response (hours)

    Returns:
        dict with model parameters and fit statistics
    """
    # Prepare data
    data = pd.DataFrame({
        'room': df[room_col],
        'outdoor': df[OUTDOOR_COL],
        'effort': effort,
        'pv': df[PV_COL]
    }).dropna()

    if len(data) < 300:
        return None

    # Apply low-pass filtering (convert hours to 15-min steps)
    out_smooth = exponential_smooth(data['outdoor'].values, tau_out_h * 4)
    effort_smooth = exponential_smooth(data['effort'].values, tau_effort_h * 4)
    pv_smooth = exponential_smooth(data['pv'].values, tau_pv_h * 4)

    # Fit linear model
    X = np.column_stack([out_smooth, effort_smooth, pv_smooth])
    y = data['room'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    return {
        'tau_out_h': tau_out_h,
        'tau_effort_h': tau_effort_h,
        'tau_pv_h': tau_pv_h,
        'offset': model.intercept_,
        'gain_outdoor': model.coef_[0],
        'gain_effort': model.coef_[1],
        'gain_pv': model.coef_[2],
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'n_points': len(y),
        'y_actual': y,
        'y_pred': y_pred,
        'index': data.index
    }


def grid_search_room_model(df: pd.DataFrame, room_col: str, effort: pd.Series) -> dict:
    """
    Find optimal time constants for room model via grid search.

    Returns:
        Best model parameters
    """
    tau_out_range = [24, 48, 72, 96, 120]
    tau_effort_range = [2, 4, 8, 12, 24, 48]
    tau_pv_range = [1, 2, 4, 8, 12, 24]

    best_r2 = -1
    best_result = None

    for tau_out in tau_out_range:
        for tau_effort in tau_effort_range:
            for tau_pv in tau_pv_range:
                result = fit_room_model(df, room_col, effort, tau_out, tau_effort, tau_pv)
                if result and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_result = result

    return best_result


def compute_weighted_temperature(df: pd.DataFrame) -> pd.Series:
    """
    Compute weighted average indoor temperature from target sensors.
    """
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for sensor, weight in SENSOR_WEIGHTS.items():
        if sensor in df.columns:
            valid_mask = df[sensor].notna()
            weighted_sum[valid_mask] += df.loc[valid_mask, sensor] * weight
            weight_sum[valid_mask] += weight

    T_weighted = weighted_sum / weight_sum
    T_weighted[weight_sum == 0] = np.nan

    return T_weighted


def plot_thermal_analysis(results: list, heating_curve: dict, df: pd.DataFrame) -> None:
    """Create visualization of thermal model results."""
    print("\nCreating thermal model visualization...")

    # Adaptive layout based on number of sensors
    n_sensors = len(results)

    if n_sensors == 1:
        # Single sensor: 1x3 layout
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 1: Heating curve with both models
        ax1 = axes[0]
        clean = df[[HK2_COL, OUTDOOR_COL]].dropna()
        ax1.scatter(clean[OUTDOOR_COL], clean[HK2_COL], alpha=0.2, s=3, label='Data')
        x_line = np.linspace(clean[OUTDOOR_COL].min(), clean[OUTDOOR_COL].max(), 100)

        # Simple linear model (diagnostic)
        y_simple = heating_curve['baseline'] + heating_curve['slope'] * x_line
        ax1.plot(x_line, y_simple, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Simple: R²={heating_curve["r2"]:.2f}')

        # Phase 2 parametric model (for optimization) - show comfort mode example
        p2 = heating_curve['phase2_params']
        setpoint_example = 20.5  # typical comfort setpoint
        curve_rise_example = 1.08  # typical curve rise
        y_parametric = setpoint_example + curve_rise_example * (p2['t_ref_comfort'] - x_line)
        ax1.plot(x_line, y_parametric, 'g-', linewidth=2,
                 label=f'Parametric: R²={p2["normal_r_squared"]:.2f}')

        ax1.set_xlabel('Outdoor Temperature (°C)')
        ax1.set_ylabel('Flow Temperature (°C)')
        ax1.set_title('Heating Curve Models')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.02, 'Parametric model used\nfor optimization',
                 transform=ax1.transAxes, fontsize=7, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Panel 2: Actual vs Predicted scatter
        r = results[0]
        ax2 = axes[1]
        n = len(r['y_actual'])
        step = max(1, n // 500)
        ax2.scatter(r['y_actual'][::step], r['y_pred'][::step], alpha=0.4, s=10)
        temp_range = [r['y_actual'].min(), r['y_actual'].max()]
        ax2.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')
        room_name = r['room'].replace('_temperature', '')
        ax2.set_xlabel('Actual Temperature (°C)')
        ax2.set_ylabel('Predicted Temperature (°C)')
        ax2.set_title(f'{room_name}: R²={r["r2"]:.3f}, RMSE={r["rmse"]:.2f}°C')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Time series (last 2 weeks)
        ax3 = axes[2]
        idx = r['index']
        recent = idx >= idx.max() - pd.Timedelta(days=14)
        ax3.plot(idx[recent], r['y_actual'][recent], 'b-', alpha=0.7, linewidth=0.8, label='Actual')
        ax3.plot(idx[recent], r['y_pred'][recent], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title(f'{room_name}: Last 2 Weeks')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    else:
        # Multiple sensors: 2x3 layout
        fig = plt.figure(figsize=(16, 12))

        # Panel 1: Heating curve with both models
        ax1 = fig.add_subplot(2, 3, 1)
        clean = df[[HK2_COL, OUTDOOR_COL]].dropna()
        ax1.scatter(clean[OUTDOOR_COL], clean[HK2_COL], alpha=0.2, s=3, label='Data')
        x_line = np.linspace(clean[OUTDOOR_COL].min(), clean[OUTDOOR_COL].max(), 100)

        # Simple linear model (diagnostic)
        y_simple = heating_curve['baseline'] + heating_curve['slope'] * x_line
        ax1.plot(x_line, y_simple, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Simple: R²={heating_curve["r2"]:.2f}')

        # Phase 2 parametric model (for optimization)
        p2 = heating_curve['phase2_params']
        setpoint_example = 20.5
        curve_rise_example = 1.08
        y_parametric = setpoint_example + curve_rise_example * (p2['t_ref_comfort'] - x_line)
        ax1.plot(x_line, y_parametric, 'g-', linewidth=2,
                 label=f'Parametric: R²={p2["normal_r_squared"]:.2f}')

        ax1.set_xlabel('Outdoor Temperature (°C)')
        ax1.set_ylabel('Flow Temperature (°C)')
        ax1.set_title('Heating Curve Models')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.02, 'Parametric model used\nfor optimization',
                 transform=ax1.transAxes, fontsize=7, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Panels 2-5: Room model fits (actual vs predicted)
        for i, r in enumerate(results[:4]):
            ax = fig.add_subplot(2, 3, i + 2)
            n = len(r['y_actual'])
            step = max(1, n // 500)
            ax.scatter(r['y_actual'][::step], r['y_pred'][::step], alpha=0.4, s=10)
            temp_range = [r['y_actual'].min(), r['y_actual'].max()]
            ax.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')
            room_name = r['room'].replace('_temperature', '')
            ax.set_xlabel('Actual Temperature (°C)')
            ax.set_ylabel('Predicted Temperature (°C)')
            ax.set_title(f'{room_name}\nR²={r["r2"]:.3f}, RMSE={r["rmse"]:.2f}°C')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel 6: Time series for best room
        ax6 = fig.add_subplot(2, 3, 6)
        best = max(results, key=lambda x: x['r2'])
        idx = best['index']
        recent = idx >= idx.max() - pd.Timedelta(days=14)
        ax6.plot(idx[recent], best['y_actual'][recent], 'b-', alpha=0.7, linewidth=0.8, label='Actual')
        ax6.plot(idx[recent], best['y_pred'][recent], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')
        room_name = best['room'].replace('_temperature', '')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_title(f'{room_name}: Last 2 Weeks (R²={best["r2"]:.3f})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig17_thermal_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig17_thermal_model.png")

    # Figure 17b: Time series for all rooms (only if multiple sensors)
    if n_sensors >= 2:
        n_rows = (n_sensors + 1) // 2
        fig2, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_sensors > 2 else [axes[0], axes[1]]

        sorted_results = sorted(results, key=lambda x: SENSOR_WEIGHTS.get(x['room'], 0), reverse=True)

        for i, r in enumerate(sorted_results):
            ax = axes[i]
            idx = r['index']
            recent = idx >= idx.max() - pd.Timedelta(days=14)
            ax.plot(idx[recent], r['y_actual'][recent], 'b-', alpha=0.7, linewidth=0.8, label='Actual')
            ax.plot(idx[recent], r['y_pred'][recent], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')
            room_name = r['room'].replace('_temperature', '')
            weight = SENSOR_WEIGHTS.get(r['room'], 0)
            ax.set_xlabel('Date')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'{room_name} ({weight:.0%} weight): R²={r["r2"]:.3f}, RMSE={r["rmse"]:.2f}°C')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Hide unused subplots
        for j in range(len(sorted_results), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Room Temperature Models: Actual vs Predicted (Last 2 Weeks)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig17b_room_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: fig17b_room_timeseries.png")


def generate_report(results: list, heating_curve: dict, weighted_r2: float) -> str:
    """Generate HTML report section for thermal model."""

    # Build results table
    results_table = ""
    for r in results:
        room_name = r['room'].replace('_temperature', '')
        weight = SENSOR_WEIGHTS.get(r['room'], 0)
        results_table += f"""
        <tr>
            <td>{room_name}</td>
            <td>{weight:.0%}</td>
            <td>{r['n_points']:,}</td>
            <td>{r['tau_out_h']}h</td>
            <td>{r['tau_effort_h']}h</td>
            <td>{r['tau_pv_h']}h</td>
            <td>{r['gain_outdoor']:+.3f}</td>
            <td>{r['gain_effort']:+.3f}</td>
            <td>{r['gain_pv']:+.3f}</td>
            <td>{r['r2']:.3f}</td>
            <td>{r['rmse']:.2f}°C</td>
        </tr>
        """

    # Weights description
    weights_desc = ", ".join([f"{k.replace('_temperature', '')}: {v:.0%}"
                              for k, v in SENSOR_WEIGHTS.items()])

    html = f"""
    <section id="thermal-model">
    <h2>3.1 Building Thermal Model</h2>

    <h3>Methodology</h3>
    <p>The thermal model uses a <strong>transfer function approach</strong> that separates the
    heating system behavior from building thermal response:</p>

    <ol>
        <li><strong>Heating Curve</strong>: Model HK2 = f(T_outdoor) to capture how the heat pump
            adjusts flow temperature based on outdoor conditions</li>
        <li><strong>Heating Effort</strong>: Calculate deviation from heating curve as the actual
            heating input signal</li>
        <li><strong>Room Response</strong>: Model each room's temperature as a function of
            smoothed outdoor temp, heating effort, and solar radiation</li>
    </ol>

    <h3>Heating Curve Model</h3>

    <h4>Parametric Model (from Phase 2 - used for optimization)</h4>
    <pre>T_flow = setpoint + curve_rise × (T_ref - T_outdoor)   (R² = {heating_curve['phase2_params']['normal_r_squared']:.3f})</pre>
    <p>Where:</p>
    <ul>
        <li>T_ref (comfort) = {heating_curve['phase2_params']['t_ref_comfort']:.2f}°C</li>
        <li>T_ref (eco) = {heating_curve['phase2_params']['t_ref_eco']:.2f}°C</li>
        <li>RMSE = {heating_curve['phase2_params']['normal_rmse']:.2f}°C</li>
    </ul>
    <p><strong>This is the model used in Phase 4 optimization.</strong> It accounts for controllable
    parameters (setpoint, curve_rise) that affect T_flow → COP → energy consumption.</p>

    <h4>Simple Reference Model (diagnostic only)</h4>
    <pre>HK2 = {heating_curve['baseline']:.1f} {heating_curve['slope']:+.3f} × T_outdoor   (R² = {heating_curve['r2']:.3f})</pre>
    <p>This simplified model ignores controllable parameters and is used only for computing
    "heating effort" as a diagnostic signal for thermal response analysis.</p>

    <h3>Room Temperature Model</h3>
    <pre>T_room = offset + g_out × LPF(T_outdoor, τ_out) + g_eff × LPF(effort, τ_eff) + g_pv × LPF(PV, τ_pv)</pre>
    <p>Where LPF = low-pass filter (exponential smoothing with time constant τ)</p>

    <h4>Model Parameters</h4>
    <ul>
        <li><strong>τ_outdoor</strong>: How slowly room tracks outdoor temperature changes (hours)</li>
        <li><strong>τ_effort</strong>: How quickly room responds to heating effort (hours)</li>
        <li><strong>τ_pv</strong>: How quickly room responds to solar radiation (hours)</li>
        <li><strong>gain_outdoor</strong>: °C room change per °C outdoor change</li>
        <li><strong>gain_effort</strong>: °C room change per °C heating effort</li>
        <li><strong>gain_pv</strong>: °C room change per kWh PV generation</li>
    </ul>

    <h3>Results by Room</h3>
    <p><strong>Weighted temperature sensors:</strong> {weights_desc}</p>

    <table>
        <tr>
            <th>Room</th>
            <th>Weight</th>
            <th>Points</th>
            <th>τ_out</th>
            <th>τ_eff</th>
            <th>τ_pv</th>
            <th>g_out</th>
            <th>g_eff</th>
            <th>g_pv</th>
            <th>R²</th>
            <th>RMSE</th>
        </tr>
        {results_table}
    </table>

    <h3>Physical Interpretation</h3>

    <h4>Heating Response</h4>
    <p>The <code>gain_effort</code> coefficient shows how much each room responds to additional
    heating beyond the baseline heating curve:</p>
    <ul>
    """

    # Sort by heating response
    sorted_by_effort = sorted(results, key=lambda x: x['gain_effort'], reverse=True)
    for r in sorted_by_effort:
        room_name = r['room'].replace('_temperature', '')
        html += f"<li><strong>{room_name}</strong>: {r['gain_effort']:+.3f} °C per °C effort"
        if r['gain_effort'] > 0.5:
            html += " (strong response)"
        elif r['gain_effort'] < 0.3:
            html += " (weak response)"
        html += "</li>\n"

    html += """
    </ul>

    <h4>Solar Response</h4>
    <p>The <code>gain_pv</code> coefficient shows how much each room heats up from solar radiation
    (using PV generation as a proxy for irradiance):</p>
    <ul>
    """

    # Sort by solar response
    sorted_by_pv = sorted(results, key=lambda x: x['gain_pv'], reverse=True)
    for r in sorted_by_pv:
        room_name = r['room'].replace('_temperature', '')
        if r['gain_pv'] > 0:
            html += f"<li><strong>{room_name}</strong>: {r['gain_pv']:+.3f} °C per kWh PV</li>\n"
        else:
            html += f"<li><strong>{room_name}</strong>: {r['gain_pv']:+.3f} °C per kWh PV (anomalous - possibly north-facing)</li>\n"

    html += f"""
    </ul>

    <h4>Time Constants</h4>
    <ul>
        <li><strong>τ_outdoor</strong>: 24-120h - rooms respond slowly to outdoor changes (3-5 days)</li>
        <li><strong>τ_effort</strong>: 4-48h - rooms respond faster to heating changes</li>
        <li><strong>τ_pv</strong>: ~24h for all rooms - consistent solar response time</li>
    </ul>

    <h3>Weighted Average Model Performance</h3>
    <p>Overall weighted R² = <strong>{weighted_r2:.3f}</strong></p>

    <h3>Implications for Optimization</h3>
    <ul>
        <li><strong>Pre-heating timing</strong>: With τ_effort of 4-48h, rooms need advance notice
            to reach target temperature</li>
        <li><strong>Solar preheating</strong>: All rooms (except atelier) benefit from solar gain.
            Schedule comfort periods during/after sunny periods.</li>
        <li><strong>Room variation</strong>: Different rooms respond differently to heating.
            simlab and studio respond strongly; atelier responds weakly.</li>
    </ul>

    <figure>
        <img src="fig17_thermal_model.png" alt="Thermal Model Analysis">
        <figcaption><strong>Figure 17:</strong> Thermal model: heating curve (left),
        actual vs predicted scatter (middle), time series validation (right).</figcaption>
    </figure>
    </section>
    """

    # Add fig17b only if multiple sensors
    if len(results) >= 2:
        html = html.replace('</section>', f"""
    <figure>
        <img src="fig17b_room_timeseries.png" alt="Room Temperature Time Series">
        <figcaption><strong>Figure 17b:</strong> Actual vs predicted temperature for all rooms
        in the weighted temperature objective (last 2 weeks).</figcaption>
    </figure>
    </section>
    """)

    return html


def main():
    """Main function for thermal model estimation."""
    print("="*60)
    print("Phase 3, Step 1: Building Thermal Model")
    print("="*60)

    # Load data
    df = load_data()

    # Check required columns
    required = [HK2_COL, OUTDOOR_COL, PV_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    # Step 1: Fit heating curve
    heating_curve = fit_heating_curve(df)
    if not heating_curve:
        print("ERROR: Could not fit heating curve")
        return

    # Step 2: Compute heating effort
    effort = compute_heating_effort(df, heating_curve)
    print(f"\nHeating effort: mean={effort.mean():.2f}°C, std={effort.std():.2f}°C")

    # Step 3: Fit models for each room
    print("\n" + "="*60)
    print("Fitting room-specific thermal models")
    print("="*60)

    results = []
    for room_col in TARGET_SENSORS:
        if room_col not in df.columns:
            print(f"\n{room_col}: Not found in data")
            continue

        valid_count = df[room_col].notna().sum()
        if valid_count < 500:
            print(f"\n{room_col}: Only {valid_count} points (need 500+)")
            continue

        room_name = room_col.replace('_temperature', '')
        print(f"\n{room_name} ({valid_count:,} points):")

        # Grid search for best parameters
        best = grid_search_room_model(df, room_col, effort)

        if best:
            best['room'] = room_col
            results.append(best)

            print(f"  τ_out={best['tau_out_h']}h, τ_eff={best['tau_effort_h']}h, τ_pv={best['tau_pv_h']}h")
            print(f"  R² = {best['r2']:.3f}, RMSE = {best['rmse']:.2f}°C")
            print(f"  g_out={best['gain_outdoor']:+.3f}, g_eff={best['gain_effort']:+.3f}, g_pv={best['gain_pv']:+.3f}")

    if not results:
        print("ERROR: No room models fitted successfully")
        return

    # Calculate weighted R²
    weighted_r2 = sum(r['r2'] * SENSOR_WEIGHTS.get(r['room'], 0) for r in results)
    weighted_rmse = sum(r['rmse'] * SENSOR_WEIGHTS.get(r['room'], 0) for r in results)

    # Create visualizations
    plot_thermal_analysis(results, heating_curve, df)

    # Save results
    results_df = pd.DataFrame([{
        'room': r['room'].replace('_temperature', ''),
        'weight': SENSOR_WEIGHTS.get(r['room'], 0),
        'tau_outdoor_h': r['tau_out_h'],
        'tau_effort_h': r['tau_effort_h'],
        'tau_pv_h': r['tau_pv_h'],
        'offset': r['offset'],
        'gain_outdoor': r['gain_outdoor'],
        'gain_effort': r['gain_effort'],
        'gain_pv': r['gain_pv'],
        'r2': r['r2'],
        'rmse': r['rmse'],
        'n_points': r['n_points']
    } for r in results])

    results_df.to_csv(OUTPUT_DIR / 'thermal_model_results.csv', index=False)
    print(f"\nSaved: thermal_model_results.csv")

    # Save heating curve
    hc_df = pd.DataFrame([heating_curve])
    hc_df.to_csv(OUTPUT_DIR / 'heating_curve.csv', index=False)
    print("Saved: heating_curve.csv")

    # Generate report
    report_html = generate_report(results, heating_curve, weighted_r2)
    with open(OUTPUT_DIR / 'thermal_model_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: thermal_model_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("THERMAL MODEL SUMMARY")
    print("="*60)

    print(f"\nHeating Curve: HK2 = {heating_curve['baseline']:.1f} {heating_curve['slope']:+.3f} × T_outdoor")
    print(f"  R² = {heating_curve['r2']:.3f}")

    print(f"\nRoom Models (weighted R² = {weighted_r2:.3f}, RMSE = {weighted_rmse:.2f}°C):")
    print(f"{'Room':<15} {'Weight':>6} {'R²':>6} {'τ_out':>6} {'τ_eff':>6} {'τ_pv':>5} {'g_eff':>7} {'g_pv':>7}")
    print("-"*70)
    for r in sorted(results, key=lambda x: SENSOR_WEIGHTS.get(x['room'], 0), reverse=True):
        room_name = r['room'].replace('_temperature', '')
        weight = SENSOR_WEIGHTS.get(r['room'], 0)
        print(f"{room_name:<15} {weight:>5.0%} {r['r2']:>6.3f} {r['tau_out_h']:>5}h {r['tau_effort_h']:>5}h "
              f"{r['tau_pv_h']:>4}h {r['gain_effort']:>+7.3f} {r['gain_pv']:>+7.3f}")


if __name__ == '__main__':
    main()
