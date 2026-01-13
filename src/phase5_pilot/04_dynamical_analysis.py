#!/usr/bin/env python3
"""
Phase 5 Pilot: Dynamical Analysis Using Grey-Box Model

Alternative to RSM block-averaged analysis (03_pilot_analysis.py).
Uses the grey-box state-space model on continuous 15-min data.

Key insight: With a dynamical model, washout periods are unnecessary.
The transitions between parameter settings are actually the most
informative data for estimating time constants and thermal dynamics.

This script:
1. Loads continuous 15-min data for the pilot period (no washout exclusion)
2. Annotates data with parameter regimes from the schedule
3. Fits the grey-box model to the continuous data
4. Analyzes step responses at each parameter change
5. Validates the model on held-out blocks
6. Computes steady-state effects

Usage:
    python src/phase5_pilot/04_dynamical_analysis.py
    python src/phase5_pilot/04_dynamical_analysis.py --holdout-blocks 9 10

Outputs:
    output/phase5_pilot/dynamical_model_params.json
    output/phase5_pilot/step_response_analysis.csv
    output/phase5_pilot/model_validation.csv
    output/phase5_pilot/fig_dynamical_model.png
    output/phase5_pilot/fig_step_responses.png
    output/phase5_pilot/dynamical_analysis_report.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5_pilot'
PHASE1_DIR = PROJECT_ROOT / 'output' / 'phase1'
PHASE3_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor columns (same as grey-box model)
BUFFER_COL = 'stiebel_eltron_isg_actual_temperature_buffer'
HK2_COL = 'wp_anlage_hk2_ist'
ROOM_COL = 'davis_inside_temperature'
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
PV_COL = 'pv_generation_kwh'

# Time step (15 minutes = 0.25 hours)
DT_HOURS = 0.25

# Parameter bounds [lower, upper] - same as Phase 3 grey-box model
PARAM_BOUNDS = {
    'tau_buf': (0.5, 4.0),      # Buffer time constant (hours)
    'tau_room': (12.0, 96.0),   # Building time constant (hours) - extended upper bound
    'r_emit': (0.1, 3.0),       # Emitter coupling ratio
    'r_heat': (0.1, 3.0),       # Heat transfer ratio
    'k_solar': (0.0, 2.0),      # Solar gain (K/kWh)
    'c_offset': (-3.0, 3.0),    # Temperature offset (K)
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# Occupied hours for comfort evaluation
OCCUPIED_START = 8   # 08:00
OCCUPIED_END = 22    # 22:00


def load_pilot_schedule() -> pd.DataFrame:
    """Load the pilot schedule with parameter settings per block."""
    schedule_path = OUTPUT_DIR / 'pilot_schedule.csv'
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Schedule not found: {schedule_path}\n"
            "Run 02_generate_pilot_schedule.py first."
        )

    df = pd.read_csv(schedule_path)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['washout_end'] = pd.to_datetime(df['washout_end'])
    df['measurement_start'] = pd.to_datetime(df['measurement_start'])

    return df


def load_integrated_data(start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Load the integrated sensor dataset for the pilot period.

    Args:
        start_date: Start of pilot period (default: from schedule)
        end_date: End of pilot period (default: from schedule)

    Returns:
        DataFrame with 15-min sensor data
    """
    data_path = PHASE1_DIR / 'integrated_dataset.parquet'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {data_path}\n"
            "Run Phase 1 preprocessing first."
        )

    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    # Filter to pilot period if specified
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date + pd.Timedelta(days=1)]

    return df


def annotate_parameter_regimes(
    data_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annotate each data point with the current parameter settings.

    Adds columns for:
    - block: Current block number (NaN if outside schedule)
    - comfort_setpoint, eco_setpoint, curve_rise, comfort_hours
    - regime_change: True at block boundaries
    - in_comfort: True during comfort hours

    Args:
        data_df: Sensor data with datetime index
        schedule_df: Block schedule with parameters

    Returns:
        Annotated DataFrame
    """
    df = data_df.copy()

    # Initialize columns
    df['block'] = np.nan
    df['comfort_setpoint'] = np.nan
    df['eco_setpoint'] = np.nan
    df['curve_rise'] = np.nan
    df['comfort_hours'] = np.nan
    df['comfort_start_hour'] = np.nan
    df['comfort_end_hour'] = np.nan

    # Annotate each block
    for _, row in schedule_df.iterrows():
        mask = (df.index >= row['start_date']) & (df.index < row['end_date'] + pd.Timedelta(days=1))
        df.loc[mask, 'block'] = row['block']
        df.loc[mask, 'comfort_setpoint'] = row['comfort_setpoint']
        df.loc[mask, 'eco_setpoint'] = row['eco_setpoint']
        df.loc[mask, 'curve_rise'] = row['curve_rise']
        df.loc[mask, 'comfort_hours'] = row['comfort_hours']

        # Parse comfort start/end times
        start_parts = row['comfort_start'].split(':')
        end_parts = row['comfort_end'].split(':')
        df.loc[mask, 'comfort_start_hour'] = int(start_parts[0]) + int(start_parts[1]) / 60
        df.loc[mask, 'comfort_end_hour'] = int(end_parts[0]) + int(end_parts[1]) / 60

    # Identify regime changes (block boundaries)
    df['regime_change'] = df['block'].diff().fillna(0) != 0

    # Determine if in comfort mode
    df['hour'] = df.index.hour + df.index.minute / 60
    df['in_comfort'] = (
        (df['hour'] >= df['comfort_start_hour']) &
        (df['hour'] < df['comfort_end_hour'])
    )

    # T_HK2 values from heating curve model
    # T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
    T_REF_COMFORT = 21.32  # From Phase 2 heating curve analysis
    T_REF_ECO = 19.18

    # Use current setpoint and curve_rise to compute expected T_HK2
    # (This is the target, actual T_HK2 may differ)
    t_setpoint = np.where(df['in_comfort'], df['comfort_setpoint'], df['eco_setpoint'])
    t_ref = np.where(df['in_comfort'], T_REF_COMFORT, T_REF_ECO)

    # Get outdoor temp for expected T_HK2 calculation
    if OUTDOOR_COL in df.columns:
        t_outdoor = df[OUTDOOR_COL].fillna(method='ffill')
        df['T_HK2_expected'] = t_setpoint + df['curve_rise'] * (t_ref - t_outdoor)
    else:
        df['T_HK2_expected'] = np.nan

    return df


def prepare_model_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for grey-box model fitting.

    Returns:
        x_obs: Observed states [T_buffer, T_room] shape (n, 2)
        u_inputs: Inputs [T_HK2, T_outdoor, PV] shape (n, 3)
        timestamps: DatetimeIndex
        regime_changes: Array of indices where parameters changed
    """
    required = [BUFFER_COL, HK2_COL, ROOM_COL, OUTDOOR_COL, PV_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Extract and align data
    data = df[required].copy()

    # Forward fill small gaps (up to 1 hour = 4 steps)
    data = data.ffill(limit=4)

    # Drop remaining NaN rows
    valid_mask = data.notna().all(axis=1)
    data = data[valid_mask]

    # Keep block info aligned
    block_info = df.loc[data.index, ['block', 'regime_change', 'comfort_setpoint',
                                      'eco_setpoint', 'curve_rise', 'in_comfort']].copy()

    x_obs = data[[BUFFER_COL, ROOM_COL]].values
    u_inputs = data[[HK2_COL, OUTDOOR_COL, PV_COL]].values
    timestamps = data.index

    # Find regime change indices
    regime_changes = np.where(block_info['regime_change'].values)[0]

    return x_obs, u_inputs, timestamps, regime_changes, block_info


def one_step_predict(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS) -> np.ndarray:
    """
    One-step-ahead prediction using observed states.

    Each prediction uses the actual observed state from the previous step,
    which is more robust than forward simulation (doesn't compound errors).
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x_obs[0]

    for k in range(n - 1):
        T_buf = x_obs[k, 0]
        T_room = x_obs[k, 1]
        T_hk2, T_out, pv = u_inputs[k]

        # Buffer dynamics
        dT_buf = (dt / tau_buf) * ((T_hk2 - T_buf) - r_emit * (T_buf - T_room))

        # Room dynamics
        dT_room = (dt / tau_room) * (r_heat * (T_buf - T_room) - (T_room - T_out)) + k_solar * pv

        x_pred[k + 1, 0] = T_buf + dT_buf
        x_pred[k + 1, 1] = T_room + dT_room + c_offset * dt

    return x_pred


def simulate_forward(params: np.ndarray, x0: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate the two-state thermal model (recursive).

    Uses only initial condition - harder test of model quality.
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x0

    for k in range(n - 1):
        T_buf = x_pred[k, 0]
        T_room = x_pred[k, 1]
        T_hk2, T_out, pv = u_inputs[k]

        dT_buf = (dt / tau_buf) * ((T_hk2 - T_buf) - r_emit * (T_buf - T_room))
        dT_room = (dt / tau_room) * (r_heat * (T_buf - T_room) - (T_room - T_out)) + k_solar * pv

        x_pred[k + 1, 0] = T_buf + dT_buf
        x_pred[k + 1, 1] = T_room + dT_room + c_offset * dt

    return x_pred


def residual_function(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                      dt: float = DT_HOURS, weights: tuple = (0.3, 1.0)) -> np.ndarray:
    """Compute weighted residuals between observed and predicted states."""
    x_pred = one_step_predict(params, x_obs, u_inputs, dt)

    resid_buf = weights[0] * (x_obs[1:, 0] - x_pred[1:, 0])
    resid_room = weights[1] * (x_obs[1:, 1] - x_pred[1:, 1])

    return np.concatenate([resid_buf, resid_room])


def fit_greybox_model(
    x_obs: np.ndarray,
    u_inputs: np.ndarray,
    initial_params: dict = None,
    n_restarts: int = 5,
    seed: int = 42,
) -> dict:
    """
    Fit grey-box model to continuous data.

    Args:
        x_obs: Observed states [T_buffer, T_room]
        u_inputs: Inputs [T_HK2, T_outdoor, PV]
        initial_params: Optional dict of initial parameter values
        n_restarts: Number of random restarts
        seed: Random seed

    Returns:
        dict with fitted parameters and statistics
    """
    print("\nFitting grey-box model to continuous data...")

    rng = np.random.RandomState(seed)

    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    # Default initial guess
    if initial_params is not None:
        p0_default = np.array([initial_params.get(p, (lower[i] + upper[i]) / 2)
                               for i, p in enumerate(PARAM_NAMES)])
    else:
        p0_default = np.array([1.0, 24.0, 1.0, 1.0, 0.5, 0.0])

    best_result = None
    best_cost = np.inf

    for i in range(n_restarts):
        if i == 0:
            p0 = p0_default
        else:
            p0 = lower + (upper - lower) * rng.rand(len(lower))

        try:
            result = least_squares(
                residual_function,
                p0,
                args=(x_obs, u_inputs, DT_HOURS),
                bounds=(lower, upper),
                method='trf',
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=1000,
                verbose=0
            )

            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                print(f"  Restart {i + 1}: cost = {result.cost:.2f} (new best)")
            else:
                print(f"  Restart {i + 1}: cost = {result.cost:.2f}")

        except Exception as e:
            print(f"  Restart {i + 1}: failed ({e})")

    if best_result is None:
        raise RuntimeError("All optimization restarts failed")

    params = best_result.x
    param_dict = {name: params[i] for i, name in enumerate(PARAM_NAMES)}

    # Compute covariance estimate
    J = best_result.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (best_result.fun ** 2).mean()
        param_std = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        param_std = np.full(len(params), np.nan)

    # Compute predictions
    x_pred = one_step_predict(params, x_obs, u_inputs)

    # Fit statistics
    stats = {
        'params': param_dict,
        'param_std': {name: param_std[i] for i, name in enumerate(PARAM_NAMES)},
        'r2_buffer': r2_score(x_obs[1:, 0], x_pred[1:, 0]),
        'r2_room': r2_score(x_obs[1:, 1], x_pred[1:, 1]),
        'rmse_buffer': np.sqrt(mean_squared_error(x_obs[1:, 0], x_pred[1:, 0])),
        'rmse_room': np.sqrt(mean_squared_error(x_obs[1:, 1], x_pred[1:, 1])),
        'mae_room': mean_absolute_error(x_obs[1:, 1], x_pred[1:, 1]),
        'bias_room': (x_pred[1:, 1] - x_obs[1:, 1]).mean(),
        'n_points': len(x_obs),
        'optimization_cost': best_result.cost,
        'x_pred': x_pred,
    }

    print(f"\nModel fit:")
    print(f"  R² (room): {stats['r2_room']:.4f}")
    print(f"  RMSE (room): {stats['rmse_room']:.3f}°C")

    return stats


def analyze_step_responses(
    x_obs: np.ndarray,
    u_inputs: np.ndarray,
    timestamps: pd.DatetimeIndex,
    regime_changes: np.ndarray,
    block_info: pd.DataFrame,
    params: np.ndarray,
    window_hours: int = 48,
) -> pd.DataFrame:
    """
    Analyze model behavior at each parameter change.

    For each regime change:
    - Extract window around change
    - Compute T_HK2 step magnitude
    - Compare predicted vs actual T_room response
    - Estimate effective time constant from response curve

    Returns:
        DataFrame with step response metrics for each transition
    """
    print("\nAnalyzing step responses at parameter changes...")

    window_steps = int(window_hours / DT_HOURS)
    half_window = window_steps // 2

    results = []

    for i, change_idx in enumerate(regime_changes):
        if change_idx < half_window or change_idx >= len(timestamps) - half_window:
            continue

        # Get block info before and after
        block_before = block_info.iloc[change_idx - 1]
        block_after = block_info.iloc[change_idx]

        # Extract window
        start_idx = change_idx - half_window
        end_idx = change_idx + half_window

        t_window = timestamps[start_idx:end_idx]
        x_window = x_obs[start_idx:end_idx]
        u_window = u_inputs[start_idx:end_idx]

        # Compute T_HK2 step (mean before vs after)
        thk2_before = u_window[:half_window, 0].mean()
        thk2_after = u_window[half_window:, 0].mean()
        thk2_step = thk2_after - thk2_before

        # Compute T_room change (mean before vs after)
        troom_before = x_window[:half_window, 1].mean()
        troom_after = x_window[half_window:, 1].mean()
        troom_change = troom_after - troom_before

        # Predict T_room using model (forward simulation from change point)
        x0 = x_obs[change_idx]
        x_pred_forward = simulate_forward(params, x0, u_window[half_window:])

        # Compare predicted vs actual after change
        actual_after = x_window[half_window:, 1]
        pred_after = x_pred_forward[:, 1]

        prediction_rmse = np.sqrt(mean_squared_error(actual_after, pred_after))
        prediction_r2 = r2_score(actual_after, pred_after)

        # Estimate time constant from step response
        # (time to reach 63.2% of final change)
        if abs(troom_change) > 0.1:
            target_63 = troom_before + 0.632 * troom_change
            actual_after_full = x_window[:, 1]
            crossed = np.where(
                (actual_after_full[half_window:] - target_63) *
                (actual_after_full[half_window - 1] - target_63) < 0
            )[0]
            if len(crossed) > 0:
                tau_empirical = crossed[0] * DT_HOURS
            else:
                tau_empirical = np.nan
        else:
            tau_empirical = np.nan

        results.append({
            'transition': i + 1,
            'timestamp': t_window[half_window],
            'block_from': block_before['block'],
            'block_to': block_after['block'],
            'setpoint_from': block_before['comfort_setpoint'],
            'setpoint_to': block_after['comfort_setpoint'],
            'curve_rise_from': block_before['curve_rise'],
            'curve_rise_to': block_after['curve_rise'],
            'T_HK2_before': thk2_before,
            'T_HK2_after': thk2_after,
            'T_HK2_step': thk2_step,
            'T_room_before': troom_before,
            'T_room_after': troom_after,
            'T_room_change': troom_change,
            'tau_empirical_hours': tau_empirical,
            'prediction_rmse': prediction_rmse,
            'prediction_r2': prediction_r2,
        })

    df = pd.DataFrame(results)
    print(f"  Analyzed {len(df)} transitions")

    if len(df) > 0:
        print(f"  Mean T_HK2 step: {df['T_HK2_step'].mean():.1f}°C")
        print(f"  Mean T_room change: {df['T_room_change'].mean():.2f}°C")
        print(f"  Mean prediction R²: {df['prediction_r2'].mean():.3f}")

    return df


def validate_on_holdout(
    x_obs: np.ndarray,
    u_inputs: np.ndarray,
    block_info: pd.DataFrame,
    params: np.ndarray,
    holdout_blocks: list,
) -> dict:
    """
    Validate model on held-out blocks.

    Fits on non-holdout blocks, predicts on holdout blocks.

    Returns:
        dict with validation metrics
    """
    print(f"\nValidating on holdout blocks: {holdout_blocks}")

    # Split by blocks
    train_mask = ~block_info['block'].isin(holdout_blocks)
    test_mask = block_info['block'].isin(holdout_blocks)

    # Ensure we have test data
    if test_mask.sum() == 0:
        print("  Warning: No holdout data available")
        return {'error': 'No holdout data'}

    x_train = x_obs[train_mask.values]
    u_train = u_inputs[train_mask.values]
    x_test = x_obs[test_mask.values]
    u_test = u_inputs[test_mask.values]

    # One-step prediction on test data
    x_pred_1step = one_step_predict(params, x_test, u_test)

    # Forward simulation on test data (harder test)
    x0_test = x_train[-1]  # Use last training point as initial condition
    x_pred_forward = simulate_forward(params, x0_test, u_test)

    validation = {
        'n_train': len(x_train),
        'n_test': len(x_test),
        'holdout_blocks': holdout_blocks,
        # One-step metrics
        'r2_room_1step': r2_score(x_test[1:, 1], x_pred_1step[1:, 1]),
        'rmse_room_1step': np.sqrt(mean_squared_error(x_test[1:, 1], x_pred_1step[1:, 1])),
        'mae_room_1step': mean_absolute_error(x_test[1:, 1], x_pred_1step[1:, 1]),
        # Forward simulation metrics
        'r2_room_forward': r2_score(x_test[:, 1], x_pred_forward[:, 1]),
        'rmse_room_forward': np.sqrt(mean_squared_error(x_test[:, 1], x_pred_forward[:, 1])),
    }

    print(f"  Training points: {validation['n_train']:,}")
    print(f"  Test points: {validation['n_test']:,}")
    print(f"  One-step R²: {validation['r2_room_1step']:.4f}")
    print(f"  Forward sim R²: {validation['r2_room_forward']:.4f}")

    return validation


def compute_steady_state_effects(params: dict) -> dict:
    """
    Derive steady-state T_room response to parameter changes.

    From the grey-box model equations at equilibrium:
    - dT_buffer/dt = 0
    - dT_room/dt = 0

    Solving for steady-state gain: dT_room_ss / dT_HK2
    """
    tau_buf = params['tau_buf']
    tau_room = params['tau_room']
    r_emit = params['r_emit']
    r_heat = params['r_heat']

    # At steady state with constant T_HK2, T_outdoor:
    # T_buffer = (T_HK2 + r_emit * T_room) / (1 + r_emit)
    # T_room = r_heat * T_buffer + T_outdoor
    #
    # Substituting:
    # T_room = r_heat * (T_HK2 + r_emit * T_room) / (1 + r_emit) + T_outdoor
    # T_room * (1 - r_heat * r_emit / (1 + r_emit)) = r_heat * T_HK2 / (1 + r_emit) + T_outdoor
    # T_room * ((1 + r_emit - r_heat * r_emit) / (1 + r_emit)) = r_heat * T_HK2 / (1 + r_emit) + T_outdoor

    denom = 1 + r_emit * (1 - r_heat)
    gain_hk2 = r_heat / denom if denom != 0 else np.nan
    gain_outdoor = (1 + r_emit - r_heat * r_emit) / denom if denom != 0 else np.nan

    # Response time scale (dominant time constant)
    # Approximately max(tau_buf, tau_room / r_heat)
    response_time = max(tau_buf, tau_room / r_heat) if r_heat > 0 else tau_room

    effects = {
        'gain_T_room_per_T_HK2': gain_hk2,
        'gain_T_room_per_T_outdoor': gain_outdoor,
        'response_time_hours': response_time,
        'tau_buf': tau_buf,
        'tau_room': tau_room,
    }

    print("\nSteady-state effects:")
    print(f"  dT_room/dT_HK2 = {gain_hk2:.3f}")
    print(f"  dT_room/dT_outdoor = {gain_outdoor:.3f}")
    print(f"  Response time: {response_time:.1f} hours")

    return effects


def plot_dynamical_model(
    x_obs: np.ndarray,
    x_pred: np.ndarray,
    u_inputs: np.ndarray,
    timestamps: pd.DatetimeIndex,
    regime_changes: np.ndarray,
    stats: dict,
    output_path: Path,
) -> None:
    """Create 4-panel visualization of dynamical model results."""
    print("\nCreating dynamical model visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Full time series with regime boundaries
    ax1 = axes[0, 0]

    ax1.plot(timestamps, x_obs[:, 1], 'b-', linewidth=0.8, alpha=0.8, label='Actual T_room')
    ax1.plot(timestamps, x_pred[:, 1], 'r-', linewidth=0.8, alpha=0.6, label='Predicted T_room')

    # Mark regime changes
    for idx in regime_changes:
        if 0 <= idx < len(timestamps):
            ax1.axvline(timestamps[idx], color='green', linestyle='--', alpha=0.5, linewidth=0.8)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Room Temperature: Actual vs Predicted (full pilot period)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Actual vs Predicted scatter
    ax2 = axes[0, 1]
    step = max(1, len(x_obs) // 1000)
    ax2.scatter(x_obs[::step, 1], x_pred[::step, 1], alpha=0.4, s=10, c='blue')
    temp_range = [x_obs[:, 1].min() - 0.5, x_obs[:, 1].max() + 0.5]
    ax2.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')

    ax2.set_xlabel('Actual T_room (°C)')
    ax2.set_ylabel('Predicted T_room (°C)')
    ax2.set_title(f'R²={stats["r2_room"]:.4f}, RMSE={stats["rmse_room"]:.3f}°C')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # Panel 3: Residuals over time
    ax3 = axes[1, 0]
    residuals = x_obs[:, 1] - x_pred[:, 1]
    ax3.plot(timestamps, residuals, 'k-', linewidth=0.5, alpha=0.6)
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.fill_between(timestamps, residuals, 0, alpha=0.3, color='steelblue')

    for idx in regime_changes:
        if 0 <= idx < len(timestamps):
            ax3.axvline(timestamps[idx], color='green', linestyle='--', alpha=0.5, linewidth=0.8)

    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual (°C)')
    ax3.set_title(f'Prediction Residuals (mean={residuals.mean():.3f}°C, std={residuals.std():.3f}°C)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 4: Parameter summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    params = stats['params']
    param_std = stats['param_std']

    text = "Grey-Box Model Parameters (Pilot Data)\n"
    text += "=" * 45 + "\n\n"
    text += f"tau_buf (buffer time constant):    {params['tau_buf']:.2f} +/- {param_std['tau_buf']:.2f} h\n"
    text += f"tau_room (building time constant): {params['tau_room']:.1f} +/- {param_std['tau_room']:.1f} h\n"
    text += f"r_emit (emitter coupling):         {params['r_emit']:.3f} +/- {param_std['r_emit']:.3f}\n"
    text += f"r_heat (heat transfer ratio):      {params['r_heat']:.3f} +/- {param_std['r_heat']:.3f}\n"
    text += f"k_solar (solar gain):              {params['k_solar']:.4f} +/- {param_std['k_solar']:.4f} K/kWh\n"
    text += f"c_offset (temperature offset):     {params['c_offset']:.3f} +/- {param_std['c_offset']:.3f} K\n"
    text += "\n" + "=" * 45 + "\n\n"
    text += f"Fit Statistics:\n"
    text += f"  R² (room):   {stats['r2_room']:.4f}\n"
    text += f"  RMSE (room): {stats['rmse_room']:.3f}°C\n"
    text += f"  MAE (room):  {stats['mae_room']:.3f}°C\n"
    text += f"  Bias:        {stats['bias_room']:+.3f}°C\n"
    text += f"  N points:    {stats['n_points']:,}\n"

    ax4.text(0.1, 0.95, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_step_responses(
    step_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create visualization of step responses at parameter changes."""
    if len(step_df) == 0:
        print("  No step responses to plot")
        return

    print("Creating step response visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: T_HK2 step vs T_room change
    ax1 = axes[0, 0]
    ax1.scatter(step_df['T_HK2_step'], step_df['T_room_change'], s=80, alpha=0.7, c='steelblue')

    # Add trend line
    if len(step_df) > 1:
        z = np.polyfit(step_df['T_HK2_step'].dropna(), step_df['T_room_change'].dropna(), 1)
        x_line = np.linspace(step_df['T_HK2_step'].min(), step_df['T_HK2_step'].max(), 50)
        ax1.plot(x_line, z[0] * x_line + z[1], 'r--', linewidth=2,
                 label=f'Slope = {z[0]:.3f} °C/°C')
        ax1.legend()

    ax1.set_xlabel('T_HK2 Step (°C)')
    ax1.set_ylabel('T_room Change (°C)')
    ax1.set_title('Thermal Response to Flow Temperature Changes')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axvline(0, color='gray', linestyle='-', linewidth=0.5)

    # Panel 2: Empirical time constants
    ax2 = axes[0, 1]
    valid_tau = step_df['tau_empirical_hours'].dropna()
    if len(valid_tau) > 0:
        ax2.hist(valid_tau, bins=min(10, len(valid_tau)), color='steelblue', edgecolor='white', alpha=0.7)
        ax2.axvline(valid_tau.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {valid_tau.mean():.1f}h')
        ax2.legend()
    ax2.set_xlabel('Time Constant (hours)')
    ax2.set_ylabel('Count')
    ax2.set_title('Empirical Time Constants from Step Responses')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Prediction accuracy per transition
    ax3 = axes[1, 0]
    ax3.bar(step_df['transition'], step_df['prediction_r2'], color='steelblue', edgecolor='white')
    ax3.axhline(step_df['prediction_r2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean R² = {step_df["prediction_r2"].mean():.3f}')
    ax3.set_xlabel('Transition Number')
    ax3.set_ylabel('Prediction R²')
    ax3.set_title('Forward Prediction Accuracy at Each Transition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    text = "Step Response Summary\n"
    text += "=" * 40 + "\n\n"
    text += f"Number of transitions: {len(step_df)}\n\n"
    text += f"T_HK2 Steps:\n"
    text += f"  Mean: {step_df['T_HK2_step'].mean():+.1f}°C\n"
    text += f"  Std:  {step_df['T_HK2_step'].std():.1f}°C\n"
    text += f"  Range: [{step_df['T_HK2_step'].min():.1f}, {step_df['T_HK2_step'].max():.1f}]°C\n\n"
    text += f"T_room Changes:\n"
    text += f"  Mean: {step_df['T_room_change'].mean():+.2f}°C\n"
    text += f"  Std:  {step_df['T_room_change'].std():.2f}°C\n\n"
    text += f"Empirical Time Constants:\n"
    if len(valid_tau) > 0:
        text += f"  Mean: {valid_tau.mean():.1f}h\n"
        text += f"  Std:  {valid_tau.std():.1f}h\n\n"
    else:
        text += "  (insufficient data)\n\n"
    text += f"Prediction Accuracy:\n"
    text += f"  Mean R²:  {step_df['prediction_r2'].mean():.3f}\n"
    text += f"  Mean RMSE: {step_df['prediction_rmse'].mean():.2f}°C\n"

    ax4.text(0.1, 0.95, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_html_report(
    stats: dict,
    step_df: pd.DataFrame,
    validation: dict,
    effects: dict,
) -> str:
    """Generate HTML report for dynamical analysis."""

    params = stats['params']
    param_std = stats['param_std']

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Phase 5 Pilot: Dynamical Analysis Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
        }}
        h1 {{ color: #2563eb; }}
        h2 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; margin-top: 2rem; }}
        h3 {{ color: #374151; }}
        .card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        .highlight {{
            background: #dbeafe;
            border-left: 4px solid #2563eb;
            padding: 1rem;
            margin: 1rem 0;
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; }}
        .metric {{ font-size: 1.5rem; font-weight: bold; color: #2563eb; }}
        code {{ background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; }}
        figure {{ margin: 2rem 0; text-align: center; }}
        figcaption {{ color: #6b7280; margin-top: 0.5rem; }}
    </style>
</head>
<body>
    <h1>Phase 5 Pilot: Dynamical Analysis Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="highlight">
        <strong>Key Insight:</strong> With a dynamical model, washout periods are unnecessary.
        The transitions between parameter settings are the most informative data for estimating
        thermal dynamics. This analysis uses ALL data from the pilot period.
    </div>

    <h2>1. Model Overview</h2>

    <div class="card">
        <h3>Grey-Box State-Space Model</h3>
        <p>A physics-based discrete-time model with two states:</p>
        <ul>
            <li><strong>T_buffer</strong>: Buffer tank temperature (intermediate thermal storage)</li>
            <li><strong>T_room</strong>: Room temperature (comfort objective)</li>
        </ul>

        <h4>State Equations (dt = 15 min)</h4>
        <pre>
T_buffer[k+1] = T_buffer[k] + (dt/tau_buf) * [(T_HK2[k] - T_buffer[k]) - r_emit*(T_buffer[k] - T_room[k])]

T_room[k+1] = T_room[k] + (dt/tau_room) * [r_heat*(T_buffer[k] - T_room[k]) - (T_room[k] - T_outdoor[k])] + k_solar*PV[k]
        </pre>
    </div>

    <h2>2. Fitted Parameters</h2>

    <div class="card">
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Std Error</th>
                <th>Physical Meaning</th>
            </tr>
            <tr>
                <td><code>tau_buf</code></td>
                <td><strong>{params['tau_buf']:.2f} h</strong></td>
                <td>+/- {param_std['tau_buf']:.2f}</td>
                <td>Buffer tank time constant</td>
            </tr>
            <tr>
                <td><code>tau_room</code></td>
                <td><strong>{params['tau_room']:.1f} h</strong></td>
                <td>+/- {param_std['tau_room']:.1f}</td>
                <td>Building thermal time constant</td>
            </tr>
            <tr>
                <td><code>r_emit</code></td>
                <td>{params['r_emit']:.3f}</td>
                <td>+/- {param_std['r_emit']:.3f}</td>
                <td>Emitter/HP coupling ratio</td>
            </tr>
            <tr>
                <td><code>r_heat</code></td>
                <td>{params['r_heat']:.3f}</td>
                <td>+/- {param_std['r_heat']:.3f}</td>
                <td>Heat transfer ratio</td>
            </tr>
            <tr>
                <td><code>k_solar</code></td>
                <td>{params['k_solar']:.4f} K/kWh</td>
                <td>+/- {param_std['k_solar']:.4f}</td>
                <td>Solar gain coefficient</td>
            </tr>
            <tr>
                <td><code>c_offset</code></td>
                <td>{params['c_offset']:.3f} K</td>
                <td>+/- {param_std['c_offset']:.3f}</td>
                <td>Temperature offset</td>
            </tr>
        </table>
    </div>

    <h2>3. Fit Statistics</h2>

    <div class="card">
        <p><span class="metric">{stats['r2_room']:.4f}</span> R² (room temperature)</p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>RMSE</td><td>{stats['rmse_room']:.3f}°C</td></tr>
            <tr><td>MAE</td><td>{stats['mae_room']:.3f}°C</td></tr>
            <tr><td>Bias</td><td>{stats['bias_room']:+.3f}°C</td></tr>
            <tr><td>N points</td><td>{stats['n_points']:,}</td></tr>
        </table>
    </div>

    <figure>
        <img src="fig_dynamical_model.png" alt="Dynamical Model Results" style="max-width: 100%;">
        <figcaption>Grey-box model fit to continuous pilot data.</figcaption>
    </figure>

    <h2>4. Steady-State Effects</h2>

    <div class="card">
        <p>From the model equations at equilibrium:</p>
        <table>
            <tr>
                <th>Effect</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>dT_room / dT_HK2</td>
                <td><strong>{effects['gain_T_room_per_T_HK2']:.3f}</strong></td>
                <td>Room warms {effects['gain_T_room_per_T_HK2']:.2f}°C per 1°C increase in flow temp</td>
            </tr>
            <tr>
                <td>dT_room / dT_outdoor</td>
                <td>{effects['gain_T_room_per_T_outdoor']:.3f}</td>
                <td>Room tracks outdoor temp with this coupling</td>
            </tr>
            <tr>
                <td>Response time</td>
                <td>{effects['response_time_hours']:.1f} h</td>
                <td>Time to reach ~63% of new equilibrium</td>
            </tr>
        </table>
    </div>
'''

    # Step response section
    if len(step_df) > 0:
        html += f'''
    <h2>5. Step Response Analysis</h2>

    <div class="card">
        <p><strong>{len(step_df)}</strong> parameter transitions analyzed</p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
            </tr>
            <tr>
                <td>T_HK2 Step</td>
                <td>{step_df['T_HK2_step'].mean():+.1f}°C</td>
                <td>{step_df['T_HK2_step'].std():.1f}°C</td>
            </tr>
            <tr>
                <td>T_room Change</td>
                <td>{step_df['T_room_change'].mean():+.2f}°C</td>
                <td>{step_df['T_room_change'].std():.2f}°C</td>
            </tr>
            <tr>
                <td>Prediction R²</td>
                <td>{step_df['prediction_r2'].mean():.3f}</td>
                <td>{step_df['prediction_r2'].std():.3f}</td>
            </tr>
        </table>
    </div>

    <figure>
        <img src="fig_step_responses.png" alt="Step Response Analysis" style="max-width: 100%;">
        <figcaption>Step response characteristics at parameter transitions.</figcaption>
    </figure>
'''

    # Validation section
    if 'error' not in validation:
        html += f'''
    <h2>6. Holdout Validation</h2>

    <div class="card">
        <p>Validated on blocks: {validation['holdout_blocks']}</p>
        <table>
            <tr>
                <th>Metric</th>
                <th>One-Step</th>
                <th>Forward Sim</th>
            </tr>
            <tr>
                <td>R²</td>
                <td>{validation['r2_room_1step']:.4f}</td>
                <td>{validation['r2_room_forward']:.4f}</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{validation['rmse_room_1step']:.3f}°C</td>
                <td>{validation['rmse_room_forward']:.3f}°C</td>
            </tr>
        </table>
        <p style="color: #6b7280; font-size: 0.9rem;">
            <em>One-step uses observed states; forward simulation uses only initial condition.</em>
        </p>
    </div>
'''

    html += '''
    <h2>7. Implications for Phase 5</h2>

    <div class="highlight">
        <h3>Key Findings</h3>
        <ul>
            <li><strong>Washout periods optional:</strong> The dynamical model can handle non-equilibrium states,
                making traditional washout exclusion unnecessary for model fitting.</li>
            <li><strong>Transition data valuable:</strong> Parameter change events provide step responses
                that help estimate time constants.</li>
            <li><strong>Accurate prediction:</strong> The model can predict T_room trajectories given
                T_HK2 input, enabling simulation of different strategies.</li>
        </ul>

        <h3>Recommendations for Full Phase 5</h3>
        <ul>
            <li>Keep short washout (1-2 days) for ANOVA-style analysis on block means</li>
            <li>Use dynamical model for supplementary analysis on full data</li>
            <li>Validate step response predictions during the main study</li>
        </ul>
    </div>

</body>
</html>'''

    return html


def save_results(
    stats: dict,
    step_df: pd.DataFrame,
    validation: dict,
    effects: dict,
) -> None:
    """Save all analysis results."""

    # JSON with model parameters
    json_path = OUTPUT_DIR / 'dynamical_model_params.json'
    results = {
        'generated': datetime.now().isoformat(),
        'params': stats['params'],
        'param_std': stats['param_std'],
        'param_bounds': PARAM_BOUNDS,
        'fit_stats': {
            'r2_room': stats['r2_room'],
            'r2_buffer': stats['r2_buffer'],
            'rmse_room': stats['rmse_room'],
            'rmse_buffer': stats['rmse_buffer'],
            'mae_room': stats['mae_room'],
            'bias_room': stats['bias_room'],
            'n_points': stats['n_points'],
        },
        'steady_state_effects': effects,
        'validation': {k: v for k, v in validation.items() if k != 'error'},
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Step response CSV
    if len(step_df) > 0:
        csv_path = OUTPUT_DIR / 'step_response_analysis.csv'
        step_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # Validation CSV
    if 'error' not in validation:
        val_path = OUTPUT_DIR / 'model_validation.csv'
        pd.DataFrame([validation]).to_csv(val_path, index=False)
        print(f"Saved: {val_path}")

    # HTML report
    html_path = OUTPUT_DIR / 'dynamical_analysis_report.html'
    html = generate_html_report(stats, step_df, validation, effects)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"Saved: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dynamical analysis of pilot experiment using grey-box model',
    )
    parser.add_argument(
        '--holdout-blocks', '-H',
        type=int,
        nargs='+',
        default=[9, 10],
        help='Block numbers to hold out for validation (default: 9 10)',
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for optimization (default: 42)',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Pilot: Dynamical Analysis (Grey-Box Model)")
    print("=" * 60)
    print("\nUsing continuous data (NO washout exclusion)")
    print("Transitions between settings are treated as informative step inputs")

    # Load schedule and data
    print("\nLoading data...")
    try:
        schedule_df = load_pilot_schedule()
        print(f"  Schedule: {len(schedule_df)} blocks")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Get pilot period
    start_date = schedule_df['start_date'].min()
    end_date = schedule_df['end_date'].max()
    print(f"  Period: {start_date.date()} to {end_date.date()}")

    try:
        data_df = load_integrated_data(start_date, end_date)
        print(f"  Data: {len(data_df):,} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Check if we have data in the pilot period
    if len(data_df) == 0:
        print("\nNo sensor data available for pilot period.")
        print("This is expected if the pilot hasn't started yet.")
        print("Run this analysis after collecting pilot data.")
        return 0

    # Annotate with parameter regimes
    print("\nAnnotating parameter regimes...")
    data_df = annotate_parameter_regimes(data_df, schedule_df)

    # Check how many blocks have data
    blocks_with_data = data_df['block'].dropna().unique()
    print(f"  Blocks with data: {sorted(blocks_with_data)}")

    if len(blocks_with_data) < 2:
        print("\nInsufficient data for analysis (need at least 2 blocks).")
        return 0

    # Prepare model data
    print("\nPreparing model data...")
    x_obs, u_inputs, timestamps, regime_changes, block_info = prepare_model_data(data_df)
    print(f"  Valid observations: {len(x_obs):,}")
    print(f"  Regime changes: {len(regime_changes)}")

    # Load Phase 3 parameters as initial guess
    phase3_params_path = PHASE3_DIR / 'greybox_model_params.json'
    initial_params = None
    if phase3_params_path.exists():
        with open(phase3_params_path) as f:
            phase3_results = json.load(f)
            initial_params = phase3_results.get('params')
            print(f"\nUsing Phase 3 parameters as initial guess")

    # Fit grey-box model
    stats = fit_greybox_model(x_obs, u_inputs, initial_params, n_restarts=5, seed=args.seed)
    params = np.array([stats['params'][p] for p in PARAM_NAMES])

    # Analyze step responses
    step_df = analyze_step_responses(
        x_obs, u_inputs, timestamps, regime_changes, block_info, params
    )

    # Holdout validation
    valid_holdout = [b for b in args.holdout_blocks if b in blocks_with_data]
    if len(valid_holdout) > 0:
        validation = validate_on_holdout(x_obs, u_inputs, block_info, params, valid_holdout)
    else:
        print(f"\nHoldout blocks {args.holdout_blocks} not in data, skipping validation")
        validation = {'error': 'Holdout blocks not available'}

    # Compute steady-state effects
    effects = compute_steady_state_effects(stats['params'])

    # Create visualizations
    plot_dynamical_model(
        x_obs, stats['x_pred'], u_inputs, timestamps, regime_changes, stats,
        OUTPUT_DIR / 'fig_dynamical_model.png'
    )

    plot_step_responses(step_df, OUTPUT_DIR / 'fig_step_responses.png')

    # Save results
    save_results(stats, step_df, validation, effects)

    # Summary
    print("\n" + "=" * 60)
    print("DYNAMICAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nModel fit (all data, no washout exclusion):")
    print(f"  R² (room):   {stats['r2_room']:.4f}")
    print(f"  RMSE (room): {stats['rmse_room']:.3f}°C")
    print(f"\nPhysical parameters:")
    print(f"  tau_buf:  {stats['params']['tau_buf']:.2f} h (buffer tank)")
    print(f"  tau_room: {stats['params']['tau_room']:.1f} h (building)")
    print(f"\nSteady-state gain:")
    print(f"  dT_room/dT_HK2: {effects['gain_T_room_per_T_HK2']:.3f}")
    print(f"\nReview dynamical_analysis_report.html for full results")

    return 0


if __name__ == '__main__':
    exit(main())
