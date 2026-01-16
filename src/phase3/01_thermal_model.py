#!/usr/bin/env python3
"""
Phase 3, Step 1: Building Thermal Model

Estimates building thermal characteristics using a transfer function approach:
1. Model heating curve: HK2 = f(T_outdoor)
2. Calculate heating effort: deviation from heating curve
3. Model room temps: T_room = f(outdoor_smooth, effort_smooth, pv_smooth)

Each room has individual parameters for:
- τ_out: Time constant for outdoor temperature response
- τ_eff: Time constant for heating effort response
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
HK2_ACTUAL_COL = 'wp_anlage_hk2_ist'
HK2_TARGET_COL = 'stiebel_eltron_isg_target_temperature_hk_2'
HK2_COL = HK2_TARGET_COL  # Use TARGET for optimization (we control this)
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
PV_COL = 'pv_generation_kwh'


def exponential_smooth(x: np.ndarray, tau_steps: float) -> np.ndarray:
    """
    Apply first-order exponential smoothing (low-pass filter).

    Implements a discrete-time first-order IIR filter:

        y[n] = α × x[n] + (1 - α) × y[n-1]

    where α = 1 - exp(-1/τ) is the smoothing factor.

    Parameters
    ----------
    x : np.ndarray
        Input signal (e.g., temperature, power measurements).
        NaN values are handled by holding the last valid output.

    tau_steps : float
        Time constant in number of timesteps (not hours).
        For 15-minute data: tau_steps = tau_hours × 4
        Example: 24h time constant → tau_steps = 96

    Returns
    -------
    np.ndarray
        Smoothed signal with same shape as input.

    Physical Interpretation
    -----------------------
    The time constant τ determines how quickly the filter responds:

    - After 1×τ: output reaches 63.2% of step change
    - After 2×τ: output reaches 86.5% of step change
    - After 3×τ: output reaches 95.0% of step change
    - After 5×τ: output reaches 99.3% (essentially steady state)

    For building thermal modeling:
    - τ_outdoor ~ 24-120h: Building mass slowly tracks outdoor changes
    - τ_effort ~ 2-48h: Room responds to heating within hours
    - τ_pv ~ 1-24h: Solar gain response (depends on window area/orientation)

    Examples
    --------
    >>> # 24-hour smoothing of outdoor temperature (15-min data)
    >>> outdoor_smooth = exponential_smooth(outdoor_temp, tau_steps=96)

    >>> # 4-hour smoothing of heating effort
    >>> effort_smooth = exponential_smooth(effort, tau_steps=16)

    Notes
    -----
    - Initial value is set to first non-NaN input value
    - Equivalent to scipy.signal.lfilter([alpha], [1, -(1-alpha)], x)
    - The transfer function in z-domain is: H(z) = α / (1 - (1-α)z⁻¹)
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


def plot_lpf_visualization(output_dir: Path) -> None:
    """
    Create visualization of LPF (exponential smoothing) behavior.

    Shows:
    1. Step response for different tau values
    2. Impulse response (decay curves)
    3. Example with actual building data
    """
    print("\nCreating LPF visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Time axis (in hours, assuming 15-min timesteps)
    n_steps = 200  # 50 hours
    t_hours = np.arange(n_steps) * 0.25  # Convert to hours

    # Tau values to demonstrate (in hours)
    tau_values_h = [4, 12, 24, 48]
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

    # Panel 1: Step response
    ax1 = axes[0]
    step_input = np.ones(n_steps)
    step_input[:10] = 0  # Step at t=2.5h

    for tau_h, color in zip(tau_values_h, colors):
        tau_steps = tau_h * 4
        response = exponential_smooth(step_input, tau_steps)
        ax1.plot(t_hours, response, color=color, linewidth=2, label=f'τ = {tau_h}h')

    ax1.axhline(y=0.632, color='gray', linestyle='--', alpha=0.5, label='63.2% (1τ)')
    ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95% (3τ)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Response')
    ax1.set_title('Step Response')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-0.05, 1.05)

    # Panel 2: Impulse response (decay)
    ax2 = axes[1]
    impulse_input = np.zeros(n_steps)
    impulse_input[0] = 1  # Impulse at t=0

    for tau_h, color in zip(tau_values_h, colors):
        tau_steps = tau_h * 4
        response = exponential_smooth(impulse_input, tau_steps)
        ax2.plot(t_hours, response, color=color, linewidth=2, label=f'τ = {tau_h}h')

    ax2.axhline(y=0.368, color='gray', linestyle='--', alpha=0.5, label='36.8% (1τ)')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Response')
    ax2.set_title('Impulse Response (Decay)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(-0.05, 1.05)

    # Panel 3: Formula and key facts
    ax3 = axes[2]
    ax3.axis('off')

    info_text = """
    Low-Pass Filter (Exponential Smoothing)

    Difference equation:
        y[n] = α·x[n] + (1-α)·y[n-1]

    where:
        α = 1 - exp(-1/τ)
        τ = time constant (in timesteps)

    Time constant interpretation:
        1τ → 63.2% of final value
        2τ → 86.5% of final value
        3τ → 95.0% of final value
        5τ → 99.3% of final value

    For 15-min data:
        τ_steps = τ_hours × 4

    Typical values in thermal model:
        τ_outdoor = 24-120h (slow)
        τ_effort = 2-48h (medium)
        τ_pv = 1-24h (fast)
    """
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig18a_lpf_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig18a_lpf_visualization.png")


def select_representative_week(df: pd.DataFrame, room_col: str) -> tuple:
    """
    Select a representative winter week with typical heating behavior.

    Criteria:
    - Full 7 days of data (no significant gaps)
    - Typical outdoor temperature range (near median)
    - Complete heating cycles visible

    Returns:
        tuple of (start_date, end_date) as pd.Timestamp
    """
    # Get valid data range
    valid_data = df[[room_col, OUTDOOR_COL, PV_COL]].dropna()
    if len(valid_data) < 7 * 96:  # Need at least 7 days
        return valid_data.index[0], valid_data.index[-1]

    # Find weeks with complete data
    weekly_counts = valid_data.resample('W').count()[room_col]
    complete_weeks = weekly_counts[weekly_counts >= 7 * 96 * 0.9]  # 90% complete

    if len(complete_weeks) == 0:
        # Fallback: use most recent 7 days
        end = valid_data.index[-1]
        start = end - pd.Timedelta(days=7)
        return start, end

    # Find week closest to median outdoor temperature
    median_outdoor = df[OUTDOOR_COL].median()
    best_week = None
    best_diff = float('inf')

    for week_end in complete_weeks.index:
        week_start = week_end - pd.Timedelta(days=7)
        week_data = df.loc[week_start:week_end, OUTDOOR_COL]
        week_mean = week_data.mean()
        diff = abs(week_mean - median_outdoor)
        if diff < best_diff:
            best_diff = diff
            best_week = (week_start, week_end)

    return best_week


def plot_model_decomposition(result: dict, df: pd.DataFrame, effort: pd.Series,
                              output_dir: Path) -> None:
    """
    Create 4-panel figure showing model term decomposition for one week.

    Panel layout (all full width, stacked vertically):
    1. T_room actual vs predicted (representative week)
    2. T_outdoor raw + second term: g_out × LPF(T_out, τ_out)
    3. Heating effort raw + third term: g_eff × LPF(E, τ_eff)
    4. PV generation raw + final term: g_pv × LPF(PV, τ_pv)
    """
    print("\nCreating model decomposition figure...")

    # Get model parameters
    tau_out_h = result['tau_out_h']
    tau_eff_h = result['tau_effort_h']
    tau_pv_h = result['tau_pv_h']
    g_out = result['gain_outdoor']
    g_eff = result['gain_effort']
    g_pv = result['gain_pv']
    offset = result['offset']

    # Select representative week
    room_col = result['room']
    start_date, end_date = select_representative_week(df, room_col)
    print(f"  Representative week: {start_date.date()} to {end_date.date()}")

    # Prepare data for the week
    week_mask = (df.index >= start_date) & (df.index <= end_date)
    df_week = df.loc[week_mask].copy()
    effort_week = effort.loc[week_mask]

    # Compute smoothed signals for full dataset first (for proper filter initialization)
    out_smooth_full = exponential_smooth(df[OUTDOOR_COL].values, tau_out_h * 4)
    effort_smooth_full = exponential_smooth(effort.values, tau_eff_h * 4)
    pv_smooth_full = exponential_smooth(df[PV_COL].values, tau_pv_h * 4)

    # Extract week portion
    week_idx = df.index.get_indexer(df_week.index)
    out_smooth = out_smooth_full[week_idx]
    effort_smooth = effort_smooth_full[week_idx]
    pv_smooth = pv_smooth_full[week_idx]

    # Compute model terms
    term_outdoor = g_out * out_smooth
    term_effort = g_eff * effort_smooth
    term_pv = g_pv * pv_smooth
    y_pred = offset + term_outdoor + term_effort + term_pv

    # Actual room temperature
    y_actual = df_week[room_col].values

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Room temperature actual vs predicted
    ax1 = axes[0]
    ax1.plot(df_week.index, y_actual, 'b-', linewidth=1, alpha=0.8, label='Actual')
    ax1.plot(df_week.index, y_pred, 'r-', linewidth=1, alpha=0.8, label='Predicted')
    ax1.fill_between(df_week.index, y_actual, y_pred, alpha=0.2, color='gray')
    ax1.set_ylabel('Temperature (°C)')
    room_name = room_col.replace('_temperature', '')
    r2_week = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)
    rmse_week = np.sqrt(np.mean((y_actual - y_pred)**2))
    ax1.set_title(f'Panel 1: Room Temperature ({room_name}) — R²={r2_week:.3f}, RMSE={rmse_week:.2f}°C')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Outdoor temperature and its contribution
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(df_week.index, df_week[OUTDOOR_COL], 'b-', linewidth=1, alpha=0.7, label='T_outdoor (raw)')
    ax2.plot(df_week.index, out_smooth, 'b--', linewidth=1.5, alpha=0.9, label=f'LPF(T_outdoor, τ={tau_out_h}h)')
    ax2.set_ylabel('Outdoor Temperature (°C)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2_twin.plot(df_week.index, term_outdoor, 'orange', linewidth=1.5, label=f'g_out×LPF = {g_out:.3f}×LPF')
    ax2_twin.set_ylabel('Contribution to T_room (°C)', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    ax2.set_title(f'Panel 2: Outdoor Temperature Contribution (g_out={g_out:+.3f}, τ={tau_out_h}h)')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Heating effort and its contribution
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(df_week.index, effort_week, 'b-', linewidth=0.8, alpha=0.5, label='Effort (raw)')
    ax3.plot(df_week.index, effort_smooth, 'b-', linewidth=1.5, alpha=0.9, label=f'LPF(Effort, τ={tau_eff_h}h)')
    ax3.set_ylabel('Heating Effort (°C)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    ax3_twin.plot(df_week.index, term_effort, 'orange', linewidth=1.5, label=f'g_eff×LPF = {g_eff:.3f}×LPF')
    ax3_twin.set_ylabel('Contribution to T_room (°C)', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')

    ax3.set_title(f'Panel 3: Heating Effort Contribution (g_eff={g_eff:+.3f}, τ={tau_eff_h}h)')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: PV/Solar and its contribution
    ax4 = axes[3]
    ax4_twin = ax4.twinx()

    ax4.plot(df_week.index, df_week[PV_COL], 'b-', linewidth=0.8, alpha=0.5, label='PV (raw)')
    ax4.plot(df_week.index, pv_smooth, 'b-', linewidth=1.5, alpha=0.9, label=f'LPF(PV, τ={tau_pv_h}h)')
    ax4.set_ylabel('PV Generation (kWh)', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')

    ax4_twin.plot(df_week.index, term_pv, 'orange', linewidth=1.5, label=f'g_pv×LPF = {g_pv:.3f}×LPF')
    ax4_twin.set_ylabel('Contribution to T_room (°C)', color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')

    ax4.set_title(f'Panel 4: Solar/PV Contribution (g_pv={g_pv:+.3f}, τ={tau_pv_h}h)')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax4.set_xlabel('Date')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add overall title
    fig.suptitle(f'Model Term Decomposition: T_room = {offset:.1f} + g_out×LPF(T_out) + g_eff×LPF(E) + g_pv×LPF(PV)',
                 fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig18c_model_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig18c_model_decomposition.png")


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
    Compute heating effort as deviation from heating curve baseline.

    Heating effort = HK2_target - HK2_baseline

    Using TARGET (not actual) because:
    - We control target via setpoint and curve_rise parameters
    - Actual follows target with ~1.5°C bias (absorbed into model coefficients)
    - This makes the model directly useful for optimization

    Positive effort: target temp HIGHER than baseline curve
    Negative effort: target temp LOWER than baseline curve
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
        # Single sensor: 2 rows - top row has 2 panels, bottom row spans full width
        fig = plt.figure(figsize=(14, 8))

        # Row 1: Two panels side by side
        ax1 = fig.add_subplot(2, 2, 1)  # Top left
        ax2 = fig.add_subplot(2, 2, 2)  # Top right

        # Row 2: Full-width time series
        ax3 = fig.add_subplot(2, 1, 2)  # Bottom, spans full width

        # Panel 1: Heating curve with both models
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

        # Panel 3: Time series (FULL heating period) - full width
        idx = r['index']
        # Downsample for readability (every 2 hours = 8 timesteps for better resolution)
        step = 8
        n_days = len(idx) // 96
        ax3.plot(idx[::step], r['y_actual'][::step], 'b-', alpha=0.7, linewidth=0.8, label='Actual')
        ax3.plot(idx[::step], r['y_pred'][::step], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title(f'{room_name}: Full {n_days} Days - Predicted vs Actual')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # Add R² annotation
        ax3.text(0.01, 0.95, f'R² = {r["r2"]:.3f}\nRMSE = {r["rmse"]:.2f}°C',
                 transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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

        # Panel 6: Time series for best room (FULL heating period)
        ax6 = fig.add_subplot(2, 3, 6)
        best = max(results, key=lambda x: x['r2'])
        idx = best['index']
        # Downsample for readability (every 4 hours = 16 timesteps)
        step = 16
        n_days = len(idx) // 96
        ax6.plot(idx[::step], best['y_actual'][::step], 'b-', alpha=0.7, linewidth=0.8, label='Actual')
        ax6.plot(idx[::step], best['y_pred'][::step], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')
        room_name = best['room'].replace('_temperature', '')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_title(f'{room_name}: Full {n_days} Days (R²={best["r2"]:.3f})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # Add R² annotation
        ax6.text(0.02, 0.98, f'R² = {best["r2"]:.3f}', transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig18_thermal_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig18_thermal_model.png")

    # Figure 18b: Time series for all rooms (only if multiple sensors)
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
        plt.savefig(OUTPUT_DIR / 'fig18b_room_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: fig18b_room_timeseries.png")


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
        <li><strong>Heating Curve</strong>: Model T<sub>HK2</sub> = f(T<sub>out</sub>) to capture how the heat pump
            adjusts flow temperature based on outdoor conditions</li>
        <li><strong>Heating Effort</strong>: Calculate deviation from heating curve as the actual
            heating input signal</li>
        <li><strong>Room Response</strong>: Model each room's temperature as a function of
            smoothed outdoor temp, heating effort, and solar radiation</li>
    </ol>

    <h3>Heating Curve Model</h3>

    <h4>Parametric Model (from Phase 2 - used for optimization)</h4>
    <div class="equation-box">
    $$T_{{flow}} = T_{{setpoint}} + k_{{curve}} \\times (T_{{ref}} - T_{{out}})$$
    </div>
    <p>Where R² = {heating_curve['phase2_params']['normal_r_squared']:.3f} and:</p>
    <ul>
        <li>T<sub>ref,comfort</sub> = {heating_curve['phase2_params']['t_ref_comfort']:.2f}°C</li>
        <li>T<sub>ref,eco</sub> = {heating_curve['phase2_params']['t_ref_eco']:.2f}°C</li>
        <li>RMSE = {heating_curve['phase2_params']['normal_rmse']:.2f}°C</li>
    </ul>
    <p><strong>This is the model used in Phase 4 optimization.</strong> It accounts for controllable
    parameters (<em>T<sub>setpoint</sub></em>, <em>k<sub>curve</sub></em>) that affect T<sub>flow</sub> → COP → energy consumption.</p>

    <h4>Simple Reference Model (diagnostic only)</h4>
    <div class="equation-box">
    $$T_{{HK2}} = {heating_curve['baseline']:.1f} {heating_curve['slope']:+.3f} \\times T_{{out}} \\quad (R^2 = {heating_curve['r2']:.3f})$$
    </div>
    <p>This simplified model ignores controllable parameters and is used only for computing
    "heating effort" as a diagnostic signal for thermal response analysis.</p>

    <h3>Room Temperature Model</h3>
    <div class="equation-box">
    $$T_{{room}} = c_0 + g_{{out}} \\cdot \\text{{LPF}}(T_{{out}}, \\tau_{{out}}) + g_{{eff}} \\cdot \\text{{LPF}}(E, \\tau_{{eff}}) + g_{{pv}} \\cdot \\text{{LPF}}(P_{{pv}}, \\tau_{{pv}})$$
    </div>
    <p>Where LPF(<em>x</em>, <em>τ</em>) = low-pass filter (exponential smoothing with time constant <em>τ</em>)</p>

    <h4>Model Parameters</h4>
    <table>
        <tr><th>Symbol</th><th>Parameter</th><th>Description</th></tr>
        <tr><td>τ<sub>out</sub></td><td>Outdoor time constant</td><td>How slowly room tracks outdoor temperature changes (hours)</td></tr>
        <tr><td>τ<sub>eff</sub></td><td>Effort time constant</td><td>How quickly room responds to heating effort (hours)</td></tr>
        <tr><td>τ<sub>pv</sub></td><td>Solar time constant</td><td>How quickly room responds to solar radiation (hours)</td></tr>
        <tr><td>g<sub>out</sub></td><td>Outdoor gain</td><td>°C room change per °C outdoor change</td></tr>
        <tr><td>g<sub>eff</sub></td><td>Effort gain</td><td>°C room change per °C heating effort</td></tr>
        <tr><td>g<sub>pv</sub></td><td>Solar gain</td><td>°C room change per kWh PV generation</td></tr>
    </table>

    <h3>Low-Pass Filter (LPF) Details</h3>
    <p>The model uses first-order exponential smoothing to capture thermal inertia.
    This is a discrete-time IIR filter with the difference equation:</p>
    <div class="equation-box">
    $$y[n] = \\alpha \\cdot x[n] + (1-\\alpha) \\cdot y[n-1]$$
    </div>
    <p>where <em>α</em> = 1 − e<sup>−1/τ</sup> and <em>τ</em> is the time constant in timesteps
    (multiply hours by 4 for 15-min data).</p>

    <h4>Time Constant Interpretation</h4>
    <table>
        <tr><th>Time</th><th>Response</th><th>Physical Meaning</th></tr>
        <tr><td>1τ</td><td>63.2%</td><td>Most of the response has occurred</td></tr>
        <tr><td>2τ</td><td>86.5%</td><td>Nearly at steady state</td></tr>
        <tr><td>3τ</td><td>95.0%</td><td>Effectively at steady state</td></tr>
        <tr><td>5τ</td><td>99.3%</td><td>Full response complete</td></tr>
    </table>

    <figure>
        <img src="fig18a_lpf_visualization.png" alt="Low-Pass Filter Visualization">
        <figcaption><strong>Figure 18a:</strong> Low-pass filter behavior: step response (left),
        impulse response/decay (middle), and filter equations (right).</figcaption>
    </figure>

    <h3>Results by Room</h3>
    <p><strong>Weighted temperature sensors:</strong> {weights_desc}</p>

    <table>
        <tr>
            <th>Room</th>
            <th>Weight</th>
            <th>Points</th>
            <th>τ<sub>out</sub></th>
            <th>τ<sub>eff</sub></th>
            <th>τ<sub>pv</sub></th>
            <th>g<sub>out</sub></th>
            <th>g<sub>eff</sub></th>
            <th>g<sub>pv</sub></th>
            <th>R²</th>
            <th>RMSE</th>
        </tr>
        {results_table}
    </table>

    <h3>Physical Interpretation</h3>

    <h4>Heating Response</h4>
    <p>The g<sub>eff</sub> coefficient shows how much each room responds to additional
    heating beyond the baseline heating curve:</p>
    <ul>
    """

    # Sort by heating response
    sorted_by_effort = sorted(results, key=lambda x: x['gain_effort'], reverse=True)
    for r in sorted_by_effort:
        room_name = r['room'].replace('_temperature', '')
        html += f"<li><strong>{room_name}</strong>: g<sub>eff</sub> = {r['gain_effort']:+.3f} °C per °C effort"
        if r['gain_effort'] > 0.5:
            html += " (strong response)"
        elif r['gain_effort'] < 0.3:
            html += " (weak response)"
        html += "</li>\n"

    html += """
    </ul>

    <h4>Solar Response</h4>
    <p>The g<sub>pv</sub> coefficient shows how much each room heats up from solar radiation
    (using PV generation as a proxy for irradiance):</p>
    <ul>
    """

    # Sort by solar response
    sorted_by_pv = sorted(results, key=lambda x: x['gain_pv'], reverse=True)
    for r in sorted_by_pv:
        room_name = r['room'].replace('_temperature', '')
        if r['gain_pv'] > 0:
            html += f"<li><strong>{room_name}</strong>: g<sub>pv</sub> = {r['gain_pv']:+.3f} °C per kWh PV</li>\n"
        else:
            html += f"<li><strong>{room_name}</strong>: g<sub>pv</sub> = {r['gain_pv']:+.3f} °C per kWh PV (anomalous)</li>\n"

    html += f"""
    </ul>

    <h4>Time Constants</h4>
    <ul>
        <li>τ<sub>out</sub>: 24-120h — rooms respond slowly to outdoor changes (3-5 days)</li>
        <li>τ<sub>eff</sub>: 4-48h — rooms respond faster to heating changes</li>
        <li>τ<sub>pv</sub>: ~24h for all rooms — consistent solar response time</li>
    </ul>

    <h3>Weighted Average Model Performance</h3>
    <p>Overall weighted R² = <strong>{weighted_r2:.3f}</strong></p>

    <h3>Implications for Optimization</h3>
    <ul>
        <li><strong>Pre-heating timing</strong>: With τ<sub>eff</sub> of 4-48h, rooms need advance notice
            to reach target temperature</li>
        <li><strong>Solar preheating</strong>: Positive g<sub>pv</sub> means rooms benefit from solar gain.
            Schedule comfort periods during/after sunny periods.</li>
        <li><strong>Room variation</strong>: Different rooms respond differently to heating based on g<sub>eff</sub>.</li>
    </ul>

    <figure>
        <img src="fig18_thermal_model.png" alt="Thermal Model Analysis">
        <figcaption><strong>Figure 18:</strong> Thermal model: heating curve (left),
        actual vs predicted scatter (middle), time series validation (right).</figcaption>
    </figure>

    <h3>Model Term Decomposition</h3>
    <p>The following figure shows how each input term contributes to the predicted room temperature
    over a representative one-week period. Each panel shows the raw input signal (blue, left axis)
    and its contribution to the room temperature prediction (orange, right axis):</p>

    <figure>
        <img src="fig18c_model_decomposition.png" alt="Model Term Decomposition">
        <figcaption><strong>Figure 18c:</strong> Model term decomposition for a representative week.
        Panel 1: Actual vs predicted room temperature. Panel 2: Outdoor temperature contribution.
        Panel 3: Heating effort contribution. Panel 4: Solar/PV contribution.</figcaption>
    </figure>
    </section>
    """

    # Add fig18b only if multiple sensors
    if len(results) >= 2:
        html = html.replace('<h3>Model Term Decomposition</h3>', f"""
    <figure>
        <img src="fig18b_room_timeseries.png" alt="Room Temperature Time Series">
        <figcaption><strong>Figure 18b:</strong> Actual vs predicted temperature for all rooms
        in the weighted temperature objective (last 2 weeks).</figcaption>
    </figure>

    <h3>Model Term Decomposition</h3>""")

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

    # Create LPF visualization
    plot_lpf_visualization(OUTPUT_DIR)

    # Create model decomposition figure (using best/primary result)
    best_result = max(results, key=lambda x: SENSOR_WEIGHTS.get(x['room'], 0))
    plot_model_decomposition(best_result, df, effort, OUTPUT_DIR)

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
